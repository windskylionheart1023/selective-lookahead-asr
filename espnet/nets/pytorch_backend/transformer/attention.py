#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import logging
import math
import os

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.nets_utils import (
    ChunkedMaskConfig,
    ResolvedChunkedMaskConfig,
    build_flex_blockmask_with_future_chunks,
    build_flex_block_mask_for_encoder,
)

try:
    from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb_torch
except Exception as e:
    print(f"Failed to import Flash Attention, using ESPnet default: {e}")

from typing import Optional, Union
from torch.nn.attention.flex_attention import flex_attention


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        qk_norm (bool): Normalize q and k before dot product.
        use_flash_attn (bool): Use flash_attn implementation.
        causal (bool): Apply causal attention.
        cross_attn (bool): Cross attention instead of self attention.
        use_sdpa (bool): Use PyTorch's scaled dot product attention.
        use_kvcache (bool): Enable the decoder inference-time KV cache.
        kvcache_maxlen (int): Maximum number of positions in the KV cache.
        use_rope (bool): Apply rotary positional encoding to q/k.
        use_flex_attention (bool): Opt-in FlexAttention fast path for
            encoder self-attention during training.

    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        qk_norm=False,
        use_flash_attn=False,
        causal=False,
        cross_attn=False,
        use_sdpa=False,
        use_kvcache=False,
        kvcache_maxlen=1024,
        use_rope=False,
        use_flex_attention=False,
    ):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()

        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = (
            nn.Dropout(p=dropout_rate) if not use_flash_attn else nn.Identity()
        )
        self.dropout_rate = dropout_rate

        # LayerNorm for q and k
        self.q_norm = LayerNorm(self.d_k) if qk_norm else nn.Identity()
        self.k_norm = LayerNorm(self.d_k) if qk_norm else nn.Identity()

        self.use_flash_attn = use_flash_attn
        self.causal = causal  # only used with flash_attn
        self.cross_attn = cross_attn  # only used with flash_attn

        self.use_sdpa = use_sdpa

        # KV cache
        self.use_kvcache = use_kvcache
        self.k_cache = None  # Key cache of shape (B, T, H, D)
        self.v_cache = None  # Value cache of shape (B, T, H, D)
        self._kvcache_curlen = None # int
        self.kvcache_maxlen = kvcache_maxlen

        # Stage B: encoder self-attn streaming KV cache (separate from decoder KV cache above).
        # Shape: (1, H, T_cached*C, d_k) — stored post-RoPE, beam-shared (batch=1).
        # Updated per chunk: oldest-tail chunk's K/V appended on each streaming call.
        self.enc_stream_k_cache = None
        self.enc_stream_v_cache = None

        # RoPE
        self.use_rope = use_rope
        self.rotary_emb = None

        # For KV cache validation mode
        self.validation_mode = False

        # FlexAttention (opt-in, training encoder self-attn only).
        # On any runtime failure, this flag is flipped off for the rest of the
        # run and the module falls through to the existing default path.
        self.use_flex_attention = use_flex_attention
        self._logged_flex_success = False

    def _clear_kvcache(self):
        """Clear key-value cache used for inference-time attention.

        This method deletes the stored key and value tensors along with their
        associated length tracking variables. It is typically used to reset
        the internal state before starting a new sequence or utterance.
        """
        del self.k_cache
        del self.v_cache
        self.k_cache = None
        self.v_cache = None
        self._kvcache_curlen = 0

    def reset_enc_stream_cache(self):
        """Clear Stage B encoder streaming KV cache between utterances."""
        self.enc_stream_k_cache = None
        self.enc_stream_v_cache = None

    def _forward_encoder_streaming(
        self,
        x_tail,        # (1, T_tail, C, D)
        mask_flat,     # (1, T_tail*C, (T_cached+T_tail)*C) or broadcast — bool, True=mask
        n_finalized_chunks,
        chunk_size,
        num_left_chunks,
    ):
        """Stage B encoder self-attn streaming path.

        Q/K/V computed fresh for the tail only. Past chunks' K/V come from
        ``self.enc_stream_k_cache`` / ``enc_stream_v_cache`` (stored post-RoPE).
        After computing tail output, the oldest tail chunk's fresh K/V is
        appended to the cache (it is now finalized).

        Args:
            x_tail: tail input at this layer, shape (1, T_tail, C, D).
            mask_flat: attention mask over (T_tail*C) queries and
                ((T_cached + T_tail) * C) keys. True = mask.
            n_finalized_chunks: how many chunks are already in the cache
                (used for RoPE absolute position offset).
            chunk_size: encoder-rate frames per chunk.
            num_left_chunks: sliding window bound (<= 0 means unlimited).

        Returns:
            (1, T_tail, C, D) tail output.
        """
        B = 1
        _, T_tail, C, D = x_tail.shape
        H = self.h
        d_k = self.d_k

        # Flatten C-axis for linear projection: (1, T_tail, C, D) -> (1, T_tail*C, D)
        x_flat = x_tail.reshape(B, T_tail * C, D)

        # Fresh Q/K/V for tail. Shape: (1, H, T_tail*C, d_k).
        fresh_q = self.linear_q(x_flat).view(B, T_tail * C, H, d_k).transpose(1, 2)
        fresh_k = self.linear_k(x_flat).view(B, T_tail * C, H, d_k).transpose(1, 2)
        fresh_v = self.linear_v(x_flat).view(B, T_tail * C, H, d_k).transpose(1, 2)

        fresh_q = self.q_norm(fresh_q)
        fresh_k = self.k_norm(fresh_k)

        # RoPE at absolute positions. Queries and fresh keys start at position
        # n_finalized_chunks * chunk_size. Cached K is already post-RoPE (no
        # re-rotation).
        if self.use_rope:
            if self.rotary_emb is None:
                self.rotary_emb = RotaryEmbedding(d_k, device=x_tail.device)
            abs_offset = n_finalized_chunks * chunk_size
            fresh_q, fresh_k, fresh_v = self._RotaryPositionalEncoding(
                fresh_q, fresh_k, fresh_v, C=C,
                q_offset=abs_offset,
                kv_offset=abs_offset,
            )

        # Concat cached K/V (finalized prefix) with fresh K/V (tail).
        if self.enc_stream_k_cache is not None:
            total_k = torch.cat([self.enc_stream_k_cache, fresh_k], dim=2)
            total_v = torch.cat([self.enc_stream_v_cache, fresh_v], dim=2)
        else:
            total_k = fresh_k
            total_v = fresh_v

        # SDPA. mask_flat: (1, T_tail*C, (T_cached+T_tail)*C) with True=mask.
        # SDPA expects attention-bias-or-bool shape broadcast to (B, H, Lq, Lk);
        # follow the existing convention: unsqueeze head dim, ~mask convention
        # matches the rest of this file.
        if mask_flat is not None:
            attn_mask = mask_flat.unsqueeze(1)  # (1, 1, Lq, Lk)
        else:
            attn_mask = None

        out = torch.nn.functional.scaled_dot_product_attention(
            fresh_q, total_k, total_v,
            attn_mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
        )  # (1, H, T_tail*C, d_k)

        # Reshape back to (1, T_tail, C, D)
        out = out.transpose(1, 2)                             # (1, T_tail*C, H, d_k)
        out = out.reshape(B, T_tail, C, H * d_k)              # (1, T_tail, C, D_model)
        out = self.linear_out(out)                            # (1, T_tail, C, D)

        # FINALIZE: append oldest tail chunk's fresh K/V to the permanent cache.
        # Flat layout (B, H, T*C, D) has outer T, inner C, so positions [0, chunk_size*C)
        # correspond to the oldest tail chunk (all C slots for chunk positions 0..chunk_size-1).
        oldest_k = fresh_k[:, :, : chunk_size * C, :]
        oldest_v = fresh_v[:, :, : chunk_size * C, :]

        if self.enc_stream_k_cache is None:
            self.enc_stream_k_cache = oldest_k
            self.enc_stream_v_cache = oldest_v
        else:
            self.enc_stream_k_cache = torch.cat([self.enc_stream_k_cache, oldest_k], dim=2)
            self.enc_stream_v_cache = torch.cat([self.enc_stream_v_cache, oldest_v], dim=2)

        # Sliding-window trim (drop oldest if cache grew beyond num_left_chunks).
        # L >= 0 is finite (L=0 -> own-chunk-only, trims to an empty cache),
        # consistent with the full path and n_in_cache in conformer_encoder.py;
        # L=-1 (unlimited) skips the trim.
        if num_left_chunks is not None and num_left_chunks >= 0:
            max_cached_T_with_C = num_left_chunks * chunk_size * C
            if self.enc_stream_k_cache.shape[2] > max_cached_T_with_C:
                drop = self.enc_stream_k_cache.shape[2] - max_cached_T_with_C
                self.enc_stream_k_cache = self.enc_stream_k_cache[:, :, drop:, :].contiguous()
                self.enc_stream_v_cache = self.enc_stream_v_cache[:, :, drop:, :].contiguous()

        return out

    def _extend_cross_kvcache(
        self,
        new_memory: torch.Tensor,
        max_cache_frames: int = -1,
    ):
        """Extend cross-attention KV cache with new encoder frames.

        This is used in streaming decoding where new encoder chunks arrive
        incrementally. The existing cached K/V for old encoder frames are
        preserved, and K/V for the new frames are appended.

        When max_cache_frames > 0, a sliding window is enforced: after
        appending, the oldest frames are dropped so the cache never exceeds
        max_cache_frames.

        Args:
            new_memory: New encoder frames (B, T_new, D) to append.
                        Uses [:1] internally since encoder output is shared
                        across beams.
            max_cache_frames: Maximum number of encoder frames to keep in
                the cache.  -1 means unlimited (keep all).
        """
        if not self.cross_attn or self.k_cache is None:
            # Early-return when the cache is not yet materialized is correct,
            # not a missed extension: the first forward materializes the
            # cache from the FULL memory passed at that point, which already
            # includes these frames. Extending here would duplicate them.
            return

        B = self.k_cache.shape[0]
        H, D = self.h, self.d_k
        old_len = self._kvcache_curlen
        T_new = new_memory.shape[1]

        # Project new frames through K, V linear layers
        # Encoder output is same for all beams, so use [:1] and expand
        new_k = self.k_norm(
            self.linear_k(new_memory[:1]).view(1, -1, H, D).transpose(1, 2)
        )  # (1, H, T_new, D)
        new_v = self.linear_v(new_memory[:1]).view(1, -1, H, D).transpose(1, 2)

        # Store as (B, T_new, H, D) and append
        new_k_stored = new_k.expand(B, -1, -1, -1).transpose(1, 2)
        new_v_stored = new_v.expand(B, -1, -1, -1).transpose(1, 2)

        self.k_cache = torch.cat([self.k_cache, new_k_stored], dim=1)
        self.v_cache = torch.cat([self.v_cache, new_v_stored], dim=1)
        self._kvcache_curlen += T_new

        # Sliding window: drop oldest frames if cache exceeds limit
        if max_cache_frames > 0 and self._kvcache_curlen > max_cache_frames:
            drop = self._kvcache_curlen - max_cache_frames
            self.k_cache = self.k_cache[:, drop:].contiguous()
            self.v_cache = self.v_cache[:, drop:].contiguous()
            self._kvcache_curlen = max_cache_frames
            logging.debug(
                f"[Cross-attn KV] Sliding window: dropped {drop} old frames, "
                f"kept {max_cache_frames}"
            )

        logging.debug(
            f"[Cross-attn KV] Extended cache: "
            f"old_len={old_len}, new_frames={T_new}, "
            f"total_len={self._kvcache_curlen}, "
            f"k_cache_shape={self.k_cache.shape}, "
            f"v_cache_shape={self.v_cache.shape}"
        )

    def _update_hyp_order(self, order):
        """Reorder cached key-value pairs according to new hypothesis order.

        This is used in beam search decoding where hypotheses are reordered
        after pruning or scoring. The internal key and value caches are
        updated to reflect the new beam order.

        Args:
            order (torch.LongTensor): A tensor of shape (B,) containing the new
                indices for each hypothesis in the beam. This is used to permute
                the current cache tensors accordingly.
        """
        B = order.numel()

        # Expand cache batch dimension if the beam grew (e.g. from 1 to beam_size)
        if self.k_cache is not None and self.k_cache.shape[0] < B:
            old_k = self.k_cache
            self.k_cache = old_k.new_zeros(B, *old_k.shape[1:])
            self.k_cache[: old_k.shape[0]] = old_k
        if self.v_cache is not None and self.v_cache.shape[0] < B:
            old_v = self.v_cache
            self.v_cache = old_v.new_zeros(B, *old_v.shape[1:])
            self.v_cache[: old_v.shape[0]] = old_v

        # rearrange cache. Advanced indexing self.k_cache[order] already returns a
        # fresh reordered copy, so the historical .clone() was a redundant 2nd copy.
        # Default-on (verified byte-identical on GPU, canary 16881): drop the redundant
        # clone and skip no-op (identity) reorders. SLOW_HYP_REORDER=1 restores legacy.
        if not os.environ.get("SLOW_HYP_REORDER"):
            if not torch.equal(order, torch.arange(B, device=order.device)):
                if self.k_cache is not None:
                    self.k_cache = self.k_cache[order]
                if self.v_cache is not None:
                    self.v_cache = self.v_cache[order]
        else:
            if self.k_cache is not None:
                self.k_cache = self.k_cache[order].clone()
            if self.v_cache is not None:
                self.v_cache = self.v_cache[order].clone()

    def _RotaryPositionalEncoding(
        self,
        q: torch.Tensor,   # (B, H, Tq*C, D)
        k: torch.Tensor,   # (B, H, Tk*C, D)
        v: torch.Tensor,   # (B, H, Tk*C, D)
        C: int,
        q_offset: int = 0,
        kv_offset: int = 0,
    ):
        """Apply RoPE such that position (t, c) gets rotation offset (q_offset + t).

        The upstream flat reshape is T-major, C-minor: ``query.reshape(B, T*C, D)``
        on ``(B, T, C, D)`` puts position (t, c) at flat index ``t*C + c``. This
        function interprets the flat dim accordingly — ``.view(B, H, Tq, C, D)``
        — so RoPE applies the SAME rotation to every C slot at the same absolute
        time position, matching standard RoPE semantics.

        (Previous versions used ``.view(B, H, C, Tq, D)`` which mismatched the
        upstream layout and produced scrambled rotations at C > 1. That
        inconsistency was preserved during training as well; reverting to the
        correct layout may not match checkpoints trained under the buggy view.)
        """
        B, H, Lq, D = q.shape
        _, _, Lk, _ = k.shape

        assert Lq % C == 0, f"Lq={Lq} must be divisible by C={C}"
        assert Lk % C == 0, f"Lk={Lk} must be divisible by C={C}"

        Tq = Lq // C
        Tk = Lk // C

        # ============================================================
        # 1) Reshape flat -> (B, H, T, C, D) matching upstream layout.
        #    Position (t, c) in the original is at q_tc[b, h, t, c].
        # ============================================================
        q_tc = q.view(B, H, Tq, C, D)
        k_tc = k.view(B, H, Tk, C, D)
        v_tc = v.view(B, H, Tk, C, D)

        # ============================================================
        # 2) Permute to (B, H, C, T, D) so each (b, h, c) is a T-length
        #    sequence for flash_attn's RotaryEmbedding API, and merge
        #    (B, H, C) into a batch dim of BHC independent sequences.
        # ============================================================
        q_cthd = q_tc.permute(0, 1, 3, 2, 4).contiguous()   # (B, H, C, Tq, D)
        k_cthd = k_tc.permute(0, 1, 3, 2, 4).contiguous()   # (B, H, C, Tk, D)
        v_cthd = v_tc.permute(0, 1, 3, 2, 4).contiguous()

        BHC = B * H * C
        q_bc = q_cthd.reshape(BHC, Tq, D).unsqueeze(2)      # (BHC, Tq, 1, D)
        k_bc = k_cthd.reshape(BHC, Tk, D).unsqueeze(2)
        v_bc = v_cthd.reshape(BHC, Tk, D).unsqueeze(2)
        kv_bc = torch.stack([k_bc, v_bc], dim=2)            # (BHC, Tk, 2, 1, D)

        # ============================================================
        # 3) Apply RoPE. Each (b, h, c) sequence gets the SAME offset
        #    per position t, matching standard RoPE semantics.
        # ============================================================
        needed_seqlen = max(q_offset + Tq, kv_offset + Tk)
        if q_bc.is_cuda:
            q_rot_bc, kv_rot_bc = self.rotary_emb(
                q_bc, kv_bc,
                q_seqlen_offset=q_offset,
                kv_seqlen_offset=kv_offset,
                max_seqlen=needed_seqlen,
            )
        else:
            # CPU fallback: RotaryEmbedding.forward dispatches to a CUDA-only Triton
            # kernel (flash_attn.ops.triton.rotary.apply_rotary). Replicate it with
            # flash_attn's own pure-PyTorch RoPE, reusing the SAME cos/sin cache so the
            # numerics match the GPU path exactly. scale_base is None for this model
            # (cos_k/sin_k == cos/sin); q and k are rotated, v passes through.
            assert self.rotary_emb.scale is None, (
                "CPU RoPE fallback assumes scale_base=None (no XPos scaling)"
            )
            self.rotary_emb._update_cos_sin_cache(
                needed_seqlen, device=q_bc.device, dtype=q_bc.dtype
            )
            cos, sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
            interleaved = self.rotary_emb.interleaved
            q_rot_bc = apply_rotary_emb_torch(
                q_bc,
                cos[q_offset:q_offset + Tq],
                sin[q_offset:q_offset + Tq],
                interleaved=interleaved,
            )
            k_rot_bc_only = apply_rotary_emb_torch(
                kv_bc[:, :, 0],
                cos[kv_offset:kv_offset + Tk],
                sin[kv_offset:kv_offset + Tk],
                interleaved=interleaved,
            )
            kv_rot_bc = torch.stack([k_rot_bc_only, kv_bc[:, :, 1]], dim=2)

        k_rot_bc = kv_rot_bc[:, :, 0]                       # (BHC, Tk, 1, D)
        v_rot_bc = kv_rot_bc[:, :, 1]

        # ============================================================
        # 4) Restore (B, H, C, T, D), permute back to (B, H, T, C, D),
        #    then flatten to (B, H, T*C, D) with T-major C-minor layout.
        # ============================================================
        q_cthd_rot = q_rot_bc.squeeze(2).view(B, H, C, Tq, D)
        k_cthd_rot = k_rot_bc.squeeze(2).view(B, H, C, Tk, D)
        v_cthd_rot = v_rot_bc.squeeze(2).view(B, H, C, Tk, D)

        q_tc_rot = q_cthd_rot.permute(0, 1, 3, 2, 4).contiguous()   # (B, H, Tq, C, D)
        k_tc_rot = k_cthd_rot.permute(0, 1, 3, 2, 4).contiguous()
        v_tc_rot = v_cthd_rot.permute(0, 1, 3, 2, 4).contiguous()

        q_rot = q_tc_rot.reshape(B, H, Lq, D)
        k_rot = k_tc_rot.reshape(B, H, Lk, D)
        v_rot = v_tc_rot.reshape(B, H, Lk, D)

        return q_rot, k_rot, v_rot

    def forward_qkv(self, query, key, value, expand_kv=False):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            expand_kv (bool): Used only for partially autoregressive (PAR) decoding.

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)

        if expand_kv:
            k_shape = key.shape
            k = (
                self.linear_k(key[:1, :, :])
                .expand(n_batch, k_shape[1], k_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
            v_shape = value.shape
            v = (
                self.linear_v(value[:1, :, :])
                .expand(n_batch, v_shape[1], v_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
        else:
            _n_batch = key.size(0) # might be 1 for inference efficiency
            k = self.linear_k(key).view(_n_batch, -1, self.h, self.d_k)
            v = self.linear_v(value).view(_n_batch, -1, self.h, self.d_k)

        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        q = self.q_norm(q)
        k = self.k_norm(k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(
        self,
        query: torch.Tensor,   # (B, Tq, D) or (B, Tq, Cq, D)
        key: torch.Tensor,     # (B, Tk, D) or (B, Tk, Ck, D)
        value: torch.Tensor,   # (B, Tk, D) or (B, Tk, Ck, D)
        mask: Optional[torch.Tensor] = None,
        expand_kv: bool = False,
        mask_cfg: Optional[Union[ChunkedMaskConfig, ResolvedChunkedMaskConfig]] = None,
        position_offset: int = 0,
        nonpad_mask_1d: Optional[torch.Tensor] = None,
        flex_block_mask=None,
    ):
        """Compute scaled dot product attention with C-axis and streaming support.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size) or, with a
                choice axis, (#batch, time1, C, size). 4D inputs are flattened
                to (#batch, time1*C, size) internally and the output is
                restored to the input shape.
            key (torch.Tensor): Key tensor (#batch, time2, size) or
                (#batch, time2, C, size).
            value (torch.Tensor): Value tensor (#batch, time2, size) or
                (#batch, time2, C, size).
            mask (torch.Tensor): Boolean mask (True = allow) of shape
                (#batch, time1*C, time2*C), a broadcast-compatible shape, or
                (#batch, time1, Cq, time2, Ck) which is flattened first.
                During KV-cache inference the mask covers the full cache
                instead of the raw key.
            expand_kv (bool): Used only for partially autoregressive (PAR)
                decoding.
            mask_cfg (ChunkedMaskConfig or ResolvedChunkedMaskConfig, optional):
                Chunked streaming mask configuration; resolved per batch when
                given as an unresolved config.
            position_offset (int): Absolute position offset for RoPE, used by
                the streaming encoder (offset of this chunk's first frame).
            nonpad_mask_1d (torch.Tensor, optional): Per-frame non-padding
                mask required by the FlexAttention fast path.
            flex_block_mask (BlockMask, optional): Precomputed FlexAttention
                block mask shared across layers.

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model) or
                (#batch, time1, C, d_model), matching the query shape.

        Dispatch order: FlexAttention fast path (opt-in), SDPA (use_sdpa,
        including the decoder KV-cache branches), flash-attn (currently
        disabled at runtime with a warning), then the default matmul path.
        """
        if query.dim() == 4:
            c_axis_enabled = True
        else:
            c_axis_enabled = False
        
        B, D = query.size(0), query.size(-1)

        # Prepare mask configuration
        if mask_cfg is not None:
            if isinstance(mask_cfg, ChunkedMaskConfig):
                r = mask_cfg.resolve(B, query.device)
            else:
                r = mask_cfg
            chunk_size = r.chunk_size
            num_left_chunks = r.num_left_chunks
            num_global_tokens = r.num_global_tokens
        else:
            r = None

        # --- 0. Parse and flatten choice axes ---
        # If there are Cq/Ck axes, flatten them into batch dims.
        if query.dim() == 4:   # (B, Tq, Cq, D)
            Tq, Cq = query.size(1), query.size(2)
            query_flat = query.reshape(B, Tq * Cq, D)          # (B, Tq*Cq, D)
        else:
            Tq, Cq = query.size(1), 1
            query_flat = query                                 # (B, Tq, D)

        if key.dim() == 4:     # (B, Tk, Ck, D)
            Tk, Ck = key.size(1), key.size(2)
            key_flat = key.reshape(B, Tk * Ck, D)              # (B, Tk*Ck, D)
            value_flat = value.reshape(B, Tk * Ck, D)          # (B, Tk*Ck, D)
        else:
            Tk, Ck = key.size(1), 1
            key_flat = key                                     # (B, Tk, D)
            value_flat = value                                 # (B, Tk, D)

        # Flatten mask to match query/key shapes.
        if mask is not None and mask.dim() == 5: # (B, Tq, Cq, Tk, Ck)
            mask_flat = mask.reshape(B, Tq*Cq, Tk*Ck)
        else:
            mask_flat = mask

        # Verify mask shape matches expected dimensions.
        # Skip for cross-attention (allows broadcasting with shape (B, 1, Tk*Ck))
        # Skip for kvcache inference: mask covers the full cache but raw key is only the
        # latest token(s); the actual SDPA call uses self.k_cache, not the raw key.
        _in_kvcache_infer = getattr(self, "use_kvcache", False) and not self.training and not self.validation_mode
        if mask_flat is not None and not self.cross_attn and not _in_kvcache_infer:
            expected_shape = (B, Tq*Cq, Tk*Ck)
            # Allow broadcasting in batch dimension: mask can be (1, Tq*Cq, Tk*Ck) or (B, Tq*Cq, Tk*Ck)
            if mask_flat.shape != expected_shape:
                # Check if it's a valid broadcast-compatible shape
                # Allow broadcasting in batch dim (1 or B) and query dim (1 or Tq*Cq)
                if not (mask_flat.shape[0] in (1, B) and
                        mask_flat.shape[1] in (1, Tq*Cq) and
                        mask_flat.shape[2] == Tk*Ck):
                    expected_shape_str = f"{expected_shape} == ({B}, {Tq}*{Cq}, {Tk}*{Ck}) or broadcast-compatible"
                    raise ValueError(f"mask_flat must have shape {expected_shape_str}, got {tuple(mask_flat.shape)}")

        query, key, value, mask = query_flat, key_flat, value_flat, mask_flat

        # RoPE rotation initialization
        if self.use_rope and self.rotary_emb is None:
            self.rotary_emb = RotaryEmbedding(self.d_k, device=query.device)

        # FlexAttention fast path (opt-in; encoder self-attn only; skipped during
        # KV-cache inference and cross-attention).
        if (
            self.use_flex_attention
            and not self.cross_attn
            and not self.use_kvcache
            and mask_cfg is not None
            and nonpad_mask_1d is not None
        ):
            try:
                from torch.nn.attention.flex_attention import (
                    flex_attention as _flex_attn,
                )

                H, d_k = self.h, self.d_k
                T_flat = Tq * Cq

                q, k, v = self.forward_qkv(query, key, value, expand_kv)
                if self.use_rope:
                    q, k, v = self._RotaryPositionalEncoding(
                        q, k, v, C=Cq,
                        q_offset=position_offset,
                        kv_offset=position_offset,
                    )

                # Reuse BlockMask built once at the encoder level, or build
                # fresh if the caller didn't provide one.
                if flex_block_mask is not None:
                    block_mask = flex_block_mask
                else:
                    block_mask = build_flex_block_mask_for_encoder(
                        resolved_cfg=r,
                        T=Tq,
                        C=Cq,
                        nonpad_mask=nonpad_mask_1d,
                        H=H,
                        device=query.device,
                    )

                out = _flex_attn(q, k, v, block_mask=block_mask)
                out = out.transpose(1, 2).reshape(B, T_flat, H * d_k)
                out_flat = self.linear_out(out)

                if not self._logged_flex_success:
                    logging.info(
                        "FlexAttention active for encoder self-attention "
                        f"(B={B}, T={Tq}, C={Cq}, H={H}, d_k={d_k})"
                    )
                    self._logged_flex_success = True

                if c_axis_enabled:
                    return out_flat.view(B, Tq, Cq, D)
                else:
                    return out_flat.view(B, Tq, D)
            except Exception as e:
                logging.warning(
                    f"FlexAttention failed, falling back to default attention: {e}"
                )
                self.use_flex_attention = False

        # Use PyTorch's Scaled Dot Product Attention implementation
        if getattr(self, "use_sdpa", False):

            ### DECODER INFERENCE
            if self.use_kvcache and not self.training and not self.validation_mode:
                B, T_new, _ = query.shape
                H, D = self.h, self.d_k

                # decoder self-attn
                # if not self.cross_attn and self.causal and not self.training:
                if not self.cross_attn:
                    if self.k_cache is None:
                        # First call: materialize cache with prompt's KV
                        q, k, v = self.forward_qkv(query, key, value, expand_kv)
                        # RoPE
                        if self.use_rope:
                            q, k, v = self._RotaryPositionalEncoding(
                                q, k, v, C=Cq
                            )

                        self.k_cache = torch.empty(B, self.kvcache_maxlen, H, D, dtype=k.dtype, device=k.device)
                        self.v_cache = torch.empty_like(self.k_cache)
                        self.k_cache[:, :T_new] = k.transpose(1, 2) # store as (B, T, H, D)
                        self.v_cache[:, :T_new] = v.transpose(1, 2)
                        self._kvcache_curlen = T_new
                    else:
                        # Stepwise decoding
                        q, k, v = self.forward_qkv(query, key[:, -1:], value[:, -1:], expand_kv)
                        # RoPE: new query and new key both correspond to absolute
                        # position == self._kvcache_curlen (the index where the new
                        # token will be stored in the cache). Old cached keys were
                        # already rotated at their own positions.
                        if self.use_rope:
                            q, k, v = self._RotaryPositionalEncoding(
                                q, k, v, C=Cq,
                                q_offset=self._kvcache_curlen,
                                kv_offset=self._kvcache_curlen,
                            )
                        self.k_cache[:, self._kvcache_curlen] = k[:, :, 0] # store as (B, T, H, D)
                        self.v_cache[:, self._kvcache_curlen] = v[:, :, 0] # kv are (B, H, T=1, D)
                        self._kvcache_curlen += 1
                    _m = mask
                    if _m is not None:
                        _m = _m[..., :self._kvcache_curlen]
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q,
                        self.k_cache[:, :self._kvcache_curlen].transpose(1, 2),
                        self.v_cache[:, :self._kvcache_curlen].transpose(1, 2),
                        _m.unsqueeze(1) if _m is not None else None,
                        dropout_p=self.dropout_rate if self.training else 0.0,
                    )  # (B, H, T, D)
                    out = out.transpose(1, 2)  # (B, T, H, D)
                    out = out.reshape(out.shape[0], out.shape[1], -1)
                    return self.linear_out(out)  # (B, T, D)

                # decoder cross-attn
                if self.cross_attn:
                    if self.k_cache is None:
                        # First call: materialize cache with full encoder KV
                        # + Cross attention sequence is not batch-dependent, so [:1]
                        q, k, v = self.forward_qkv(query, key[:1], value[:1], expand_kv)
                        # RoPE
                        if self.use_rope:
                            q, k, v = self._RotaryPositionalEncoding(
                                q, k, v, C=Cq
                            )
                        self.k_cache = k.expand(B, -1, -1, -1).transpose(1, 2) # store as (B, T, H, D)
                        self.v_cache = v.expand(B, -1, -1, -1).transpose(1, 2)
                        self._kvcache_curlen = key.size(1)
                    else:
                        # Stepwise decoding
                        q = self.linear_q(query).view(B, -1, self.h, self.d_k)
                        q = self.q_norm(q.transpose(1, 2))
                        # RoPE
                        if self.use_rope:
                            # k/v are not defined in this branch (they live in
                            # the cache, rotated at materialization) — the old
                            # code here referenced them and would crash with
                            # UnboundLocalError. The decoder disables RoPE for
                            # cross-attention by design, so fail loudly if
                            # someone enables it.
                            raise NotImplementedError(
                                "RoPE for decoder cross-attention in the SDPA "
                                "stepwise KV-cache path is not implemented; "
                                "cross-attn RoPE is disabled by design "
                                "(see TransformerDecoder embed wiring)."
                            )
                    _k_for_attn = self.k_cache[:B, :self._kvcache_curlen].transpose(1, 2)
                    _v_for_attn = self.v_cache[:B, :self._kvcache_curlen].transpose(1, 2)
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q, _k_for_attn, _v_for_attn,
                        mask.unsqueeze(1) if mask is not None else None,
                        dropout_p=self.dropout_rate if self.training else 0.0,
                    )  # (B, H, T, D)

                    out = out.transpose(1, 2)  # (B, T, H, D)
                    out = out.reshape(out.shape[0], out.shape[1], -1)
                    return self.linear_out(out)  # (B, T, D)


            ### OTHER (i.e., encoder self-attn or decoder training)
            #
            q, k, v = self.forward_qkv(query, key, value, expand_kv)

            # Add rope info
            # For training, offset is 0
            # For self-attn, offset is same for both q and k
            # For streaming encoder, use position_offset from chunk index
            if self.use_rope:
                q, k, v = self._RotaryPositionalEncoding(
                    q, k, v, C=Cq,
                    q_offset=position_offset,
                    kv_offset=position_offset,
                )

            _full_attn_mask = mask.unsqueeze(1) if mask is not None else None

            # Fully-masked query rows (e.g. padded frames under the chunked
            # mask) make SDPA emit NaN (softmax over an all--inf row), and the
            # NaN would then spread into real frames through the depthwise
            # conv. The classic forward_attention path gives such rows a zero
            # attention CONTEXT (weights masked to 0); reproduce that here:
            # let dead rows attend to key 0 for the compute, then zero their
            # context afterwards (linear_out then maps it to its bias, exactly
            # as in the classic path).
            _dead_rows = None
            if _full_attn_mask is not None and _full_attn_mask.dtype == torch.bool:
                _row_ok = _full_attn_mask.any(dim=-1, keepdim=True)  # (..., Tq, 1)
                if not bool(_row_ok.all()):
                    _dead_rows = ~_row_ok
                    _full_attn_mask = _full_attn_mask.clone()
                    _full_attn_mask[..., 0:1] = _full_attn_mask[..., 0:1] | _dead_rows

            # The shape of mask must be broadcastable to the shape of attention weights
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                _full_attn_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
            )  # (batch, head, time1, d_k)
            if _dead_rows is not None:
                out = out.masked_fill(_dead_rows, 0.0)
            out = out.transpose(1, 2)  # (batch, time1, head, d_k)
            out = out.reshape(out.shape[0], out.shape[1], -1)  # (batch, time1, d_model)
            out_flat = self.linear_out(out)  # (batch, time1, d_model)
            # Restore original shape
            if c_axis_enabled:  # Was query input 4D?
                return out_flat.view(B, Tq, Cq, D)                  # (B, Tq, Cq, D) - preserves C=1 dimension
            else:
                return out_flat.view(B, Tq, D)                      # (B, Tq, D)

        # Use Flash Attention implementation
        if self.use_flash_attn:
            try:
                # Flash attention only support fp16 or bf16
                # Does Flex Attn support fp16/bf16 internally?
                # Flex Attention implementation
                # Mask
                # q, k, v
                self.use_kvcache = False  # TODO: Flex attention with KV cache not supported yet
                if self.use_kvcache:
                    # Decoder inference
                    pass
                else:
                    # Others (i.e., encoder self-attn or decoder training)
                    q = self.linear_q(query).view(B, -1, self.h, self.d_k)
                    k = self.linear_k(key).view(B, -1, self.h, self.d_k)
                    v = self.linear_v(value).view(B, -1, self.h, self.d_k)

                    q = self.q_norm(q)
                    k = self.k_norm(k)

                    # Apply RoPE
                    if self.use_rope:
                        q, k, v = self._RotaryPositionalEncoding(
                            q, k, v, C=Cq,
                            q_offset=position_offset,
                            kv_offset=position_offset,
                        )
                    
                    # Create block mask
                    if mask_cfg is not None:
                        block_mask = build_flex_blockmask_with_future_chunks(
                            C = mask_cfg.num_right_chunks,              # number of chunks
                            L = 128,           # frames per chunk
                            G = mask_cfg.num_global_tokens,             # global tokens per chunk
                            block_size = 16,   # FlexAttention block size
                            num_prev_blocks = 2, 
                            B = batch_size,    # your training batch size
                            H = num_heads,     # number of attention heads
                        )
                    else:
                        # Phrase the mask into block mask for better efficiency
                        pass
                    # Mask comes form both default mask and chunked mask config

                    out = flex_attention(q, k, v, block_mask=block_mask)  # (total, nheads, headdim)

            except Exception as e:
                logging.warning(
                    f"Flash Attention failed, falling back to default attention: {e}"
                )
                self.use_flash_attn = False

        #
        # Fall back to the default implementation
        #

        ### DECODER INFERENCE
        if self.use_kvcache:
            B, T_new, _ = query.shape
            H, D_k = self.h, self.d_k

            # decoder self-attn
            if not self.cross_attn and self.causal and not self.training and not self.validation_mode:
                if self.k_cache is None:
                    # First call: materialize cache with prompt's KV, and compute
                    # attention using freshly-computed k/v directly (same memory
                    # layout as the no-cache OTHER branch) so this call is
                    # bit-identical to the no-cache path.
                    q, k, v = self.forward_qkv(query, key, value, expand_kv)

                    # RoPE
                    if self.use_rope:
                        q, k, v = self._RotaryPositionalEncoding(q, k, v, C=Cq)

                    self.k_cache = torch.empty(B, self.kvcache_maxlen, H, D_k, dtype=k.dtype, device=k.device)
                    self.v_cache = torch.empty_like(self.k_cache)
                    self.k_cache[:, :T_new] = k.transpose(1, 2) # store as (B, T, H, D)
                    self.v_cache[:, :T_new] = v.transpose(1, 2)
                    self._kvcache_curlen = T_new

                    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
                    return self.forward_attention(v, scores, mask)

                # Stepwise decoding
                q, k, v = self.forward_qkv(query, key[:, -1:], value[:, -1:], expand_kv)

                # RoPE: new query and new key both correspond to absolute
                # position == self._kvcache_curlen.
                if self.use_rope:
                    q, k, v = self._RotaryPositionalEncoding(
                        q, k, v, C=Cq,
                        q_offset=self._kvcache_curlen,
                        kv_offset=self._kvcache_curlen,
                    )

                self.k_cache[:B, self._kvcache_curlen] = k[:, :, 0]
                self.v_cache[:B, self._kvcache_curlen] = v[:, :, 0]
                self._kvcache_curlen += 1
                # Stepwise: single new token attends to all past tokens in the cache.
                # Causality is enforced by the cache itself, so no mask is needed.

                scores = torch.matmul(
                    q, self.k_cache[:B, :self._kvcache_curlen].permute(0, 2, 3, 1)
                ) / math.sqrt(self.d_k)
                return self.forward_attention(
                    self.v_cache[:B, :self._kvcache_curlen].transpose(1, 2),
                    scores,
                    None,
                )

            # decoder cross-attn
            if self.cross_attn and not self.training and not self.validation_mode:
                if self.k_cache is None:
                    # First call: materialize cache with full encoder KV
                    # + Cross attention sequence is not batch-dependent, so [:1]
                    q, k, v = self.forward_qkv(query, key[:1], value[:1], expand_kv)

                    # RoPE
                    if self.use_rope:
                        # Cross-attn RoPE is disabled by design (see
                        # TransformerDecoder embed wiring); fail loudly if
                        # someone enables it.
                        raise NotImplementedError(
                            "RoPE for decoder cross-attention in the fallback "
                            "KV-cache path is not implemented; cross-attn "
                            "RoPE is disabled by design."
                        )

                    self.k_cache = k.expand(B, -1, -1, -1).transpose(1, 2) # store as (B, T, H, D)
                    self.v_cache = v.expand(B, -1, -1, -1).transpose(1, 2)
                    self._kvcache_curlen = key.size(1)
                else:
                    # Stepwise decoding
                    q = self.linear_q(query).view(B, 1, H, D_k)
                    q = self.q_norm(q.transpose(1, 2))

                    if self.use_rope:
                        # Cross-attn RoPE is disabled by design (see
                        # TransformerDecoder embed wiring); fail loudly if
                        # someone enables it.
                        raise NotImplementedError(
                            "RoPE for decoder cross-attention in the fallback "
                            "stepwise KV-cache path is not implemented; "
                            "cross-attn RoPE is disabled by design."
                        )

                # q: (B, H, Tq, D), k^T: (B, H, D, Tk)
                scores = torch.matmul(
                    q, self.k_cache[:B, :self._kvcache_curlen].permute(0, 2, 3, 1)
                ) / math.sqrt(self.d_k) # (B, H, Tq, Tk)
                return self.forward_attention(self.v_cache[:B, :self._kvcache_curlen].transpose(1, 2), scores, mask) # (B, Tq, D)

        ### OTHER (i.e., encoder self-attn or decoder training)
        q, k, v = self.forward_qkv(query, key, value, expand_kv)

        # RoPE for OTHER
        # For streaming encoder, use position_offset from chunk index
        if self.use_rope:
            q, k, v = self._RotaryPositionalEncoding(
                q, k, v, C=Cq,
                q_offset=position_offset,
                kv_offset=position_offset,
            )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        out = self.forward_attention(v, scores, mask)  # (batch, Tq*Cq, d_model)
        # Restore original shape to match input query shape
        if c_axis_enabled:
            return out.view(B, Tq, Cq, D)                       # (B, Tq, Cq, D)
        else:
            return out                           # (B, Tq, D)


class LegacyRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask, **kwargs):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            **kwargs: Additional keyword arguments (e.g. mask_cfg) accepted
                for compatibility but not used by the legacy implementation.

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        zero_triu=False,
    ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(
        self,
        query: torch.Tensor,         # (B, Tq, D) or (B, Tq, Cq, D)
        key: torch.Tensor,           # (B, Tk, D) or (B, Tk, Ck, D)
        value: torch.Tensor,         # (B, Tk, D) or (B, Tk, Ck, D)
        pos_emb: torch.Tensor,       # (B, 2*Tq-1, D)
        mask_bool: torch.BoolTensor, # (B, Tq, Tk) or (B, Tq, Cq, Tk, Ck)
        mask_cfg: Optional[Union[ChunkedMaskConfig, ResolvedChunkedMaskConfig]] = None,
    ):
        """
        Compute multi-head attention with optional streaming-aware relative-position masking.

        Args:
            query (torch.Tensor): (B, Tq, D) or (B, Tq, Cq, D)
            key (torch.Tensor): (B, Tk, D) or (B, Tk, Ck, D)
            value (torch.Tensor): (B, Tk, D) or (B, Tk, Ck, D)
            pos_emb (torch.Tensor): (B, 2*Tq-1, D) relative-position embeddings
            mask_bool (torch.BoolTensor):
                Attention mask of shape (B, Tq*Cq, Tk*Ck), where True = allow, False = mask.
                Must align with the flattened query/key choice axes and respect mask_cfg.
            mask_cfg (ChunkedMaskConfig, optional):
                If provided, fixes relative positional encoding calculation for streaming setup
                with global tokens.

        Returns:
            torch.Tensor: (B, Tq, D) or (B, Tq, Cq, D)
        """
        B, D = query.size(0), query.size(-1)

        # --- 0. Parse and flatten choice axes ---
        # If there are Cq/Ck axes, flatten them into batch dims.
        if query.dim() == 4:   # (B, Tq, Cq, D)
            Tq, Cq = query.size(1), query.size(2)
            query_flat = query.reshape(B, Tq * Cq, D)          # (B, Tq*Cq, D)
        else:
            Tq, Cq = query.size(1), 1
            query_flat = query                                 # (B, Tq, D)

        if key.dim() == 4:     # (B, Tk, Ck, D)
            Tk, Ck = key.size(1), key.size(2)
            key_flat = key.reshape(B, Tk * Ck, D)              # (B, Tk*Ck, D)
            value_flat = value.reshape(B, Tk * Ck, D)          # (B, Tk*Ck, D)
        else:
            Tk, Ck = key.size(1), 1
            key_flat = key                                     # (B, Tk, D)
            value_flat = value                                 # (B, Tk, D)

        # Flatten mask to match query/key shapes.
        if mask_bool.dim() == 5: # (B, Tq, Cq, Tk, Ck)
            mask_bool_flat = mask_bool.reshape(B, Tq*Cq, Tk*Ck)
        else:
            mask_bool_flat = mask_bool

        # Verify mask shape matches expected dimensions.
        expected_shape = (B, Tq*Cq, Tk*Ck)
        if mask_bool_flat.shape != expected_shape:
            expected_shape_str = f"{expected_shape} == ({B}, {Tq}*{Cq}, {Tk}*{Ck})"
            raise ValueError(f"mask_bool_flat must have shape {expected_shape_str}, got {tuple(mask_bool_flat.shape)}")

        # --- 1. Project to Q, K, V ---
        # Output shapes:
        #   q: (B, H, Tq*Cq, d_k)
        #   k: (B, H, Tk*Ck, d_k)
        #   v: (B, H, Tk*Ck, d_k)
        q, k, v = self.forward_qkv(query_flat, key_flat, value_flat)
        
        # Prepare query for content/position bias:
        #   q_t:   (B, Tq*Cq, H, d_k)
        #   q_u/v: (B, H, Tq*Cq, d_k) with biases added per head
        q_t = q.transpose(1, 2)                                # (B, Tq*Cq, H, d_k)
        q_u = (q_t + self.pos_bias_u).transpose(1, 2)          # (B, H, Tq*Cq, d_k)
        q_v = (q_t + self.pos_bias_v).transpose(1, 2)          # (B, H, Tq*Cq, d_k)
        q_v_unflat = q_v.view(B, self.h, Tq, Cq, self.d_k)     # (B, H, Tq, Cq, d_k)

        # --- 2. Content-based attention score ---
        # scores_ac: Standard dot product attention between Q (with bias_u) and K
        # Output: (B, H, Tq*Cq, Tk*Ck)
        # Note: only one out of every Ck scores will actually be used...
        scores_ac = torch.matmul(q_u, k.transpose(-2, -1))     # (B, H, Tq*Cq, Tk*Ck)

        # --- 3. Relative positional attention score ---
        # Compute position-augmented K for all positions:
        #   p:        (B, H, 2*Tq-1, d_k)
        #   matrix_bd: scores from Q (with bias_v) to all positions, shape (B, H, Tq*Cq, 2*Tq-1)
        p = self.linear_pos(pos_emb).view(B, -1, self.h, self.d_k).transpose(1, 2)   # (B, H, 2*Tq-1, d_k)
        matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))      # (B, H, Tq*Cq, 2*Tq-1)
        
        # Merge Cq into head dimension for rel_shift trick, then recover after
        matrix_bd = (
            matrix_bd
                .view(B, self.h, Tq, Cq, -1)        # (B, H, Tq, Cq, 2*Tq-1)
                .permute(0, 1, 3, 2, 4)             # (B, H, Cq, Tq, 2*Tq-1)
                .contiguous()
                .view(B, self.h * Cq, Tq, -1)       # (B, H*Cq, Tq, 2*Tq-1)
        )
        matrix_bd = self.rel_shift(matrix_bd)                   # (B, H*Cq, Tq, Tk)
        # Restore original order
        matrix_bd = (
            matrix_bd
                .view(B, self.h, Cq, Tq, Tk)        # (B, H, Cq, Tq, Tk)
                .permute(0, 1, 3, 2, 4)             # (B, H, Tq, Cq, Tk)
                .contiguous()
        )

        # --- 4. Streaming-aware relative global position update ---
        if mask_cfg is not None:
            if isinstance(mask_cfg, ChunkedMaskConfig):
                r = mask_cfg.resolve(B, q.device)
            else:
                r = mask_cfg

            # Fetch per-batch streaming parameters (all shape (B,))
            chunk_size = r.chunk_size
            left_chunks = r.num_left_chunks
            num_global_tokens = r.num_global_tokens

            # Compute max number of global tokens for any batch
            G_max = int(num_global_tokens.max().item())

            # Base index in relative embedding (distance zero)
            base = Tq - 1

            # Calculate maximal allowed distance from each query to each global key.
            # This distance becomes capped due to evicted chunks.
            # d_max(q, g) = chunk_size*num_left_chunks + q%chunk_size + (G-g)
            global_idx = torch.arange(G_max, device=q.device)           # (G_max,)
            chunk_idx = torch.remainder(
                torch.arange(Tq, device=q.device).unsqueeze(0),
                chunk_size.unsqueeze(1),
            )                                                           # (B, Tq)
            d_max = (
                chunk_idx.unsqueeze(-1)                                           # (B, Tq, 1)
                + (chunk_size * left_chunks + num_global_tokens)[:, None, None]   # (B, 1, 1)
                - global_idx[None, None, :]                                       # (1, 1, G_max)
            )                                                                     # (B, Tq, G_max)
            row_idx = base + d_max                                                # (B, Tq, G_max)

            # Gather capped positional embeddings for distant queries:
            #   p:       (B, H, 2*Tq-1, d_k)
            #   row_idx: (B, Tq, G_max)
            # Match tensor ranks, then gather along the position axis
            row_idx_expanded = (
                row_idx.unsqueeze(1)            # (B, 1, Tq, G_max)
                    .unsqueeze(-1)              # (B, 1, Tq, G_max, 1)
                    .expand(-1, self.h, -1, -1, self.d_k)
            )                                   # (B, H, Tq, G_max, d_k)
            p_dmax = torch.gather(               
                p.unsqueeze(3).expand(-1, -1, -1, G_max, -1), # (B, H, 2*Tq-1, G_max, d_k)
                dim=2,
                index=row_idx_expanded,                       # (B, H,   Tq,   G_max, d_k)
            ) # (B, H, Tq, G_max, d_k)

            # Compute replacement positional scores for (query, global_key) pairs where
            # the relative distance would otherwise exceed the modeled window. These
            # are computed using the capped distance (d_max), and will overwrite only
            # the necessary entries in the matrix_bd positional term.
            global_capped_bd = torch.matmul(
                q_v_unflat,                    # (B, H, Tq, Cq, d_k)
                p_dmax.transpose(-2, -1),      # (B, H, Tq, d_k, G_max)
            ) # (B, H, Tq, Cq, G_max)

            # Build mask for where to overwrite relative positions to global tokens with
            # capped values.
            q_idx = torch.arange(Tq, device=q.device)[None, :, None]     # (1, Tq, 1)
            g_idx = global_idx[None, None, :]                            # (1,  1, G_max)
            dist  = q_idx - g_idx                                        # (1, Tq, G_max)

            exists = (g_idx < num_global_tokens[:, None, None])          # (B,  1, G_max)
            too_far = (dist > d_max)                                     # (B, Tq, G_max)
            replace_mask = exists & too_far                              # (B, Tq, G_max)

            # Broadcast for all heads and Cq slices
            replace_mask = (
                replace_mask                                # (B, Tq, G_max)
                    .unsqueeze(1)                           # (B, 1, Tq, G_max)
                    .unsqueeze(3)                           # (B, 1, Tq, 1, G_max)
                    .expand(-1, self.h, -1, Cq, -1)         # (B, H, Tq, Cq, G_max)
            )

            # Replace the first G_max positions in matrix_bd with the capped values
            matrix_bd[..., :G_max] = torch.where(
                replace_mask,               # (B, H, Tq, Cq, G_max)
                global_capped_bd,           # (B, H, Tq, Cq, G_max)
                matrix_bd[..., :G_max],     # (B, H, Tq, Cq, G_max)
            )

        # --- 5. Final attention scores and masking ---
        # Collapse Cq/Ck axes as needed and scale for softmax
        matrix_bd = matrix_bd.view(B, self.h, Tq * Cq, Tk)     # (B,H,Tq*Cq,Tk)
        if Ck > 1:
            matrix_bd = matrix_bd.repeat_interleave(Ck, dim=-1)# (B,H,Tq*Cq,Tk*Ck)

        # Combine content and position terms, apply mask, and compute output
        full_scores = (scores_ac + matrix_bd) / math.sqrt(self.d_k) # (B, H, Tq*Cq, Tk*Ck)
        out_flat = self.forward_attention(
            v,              # (B, H, Tk*Ck, d_k)
            full_scores,    # (B, H, Tq*Cq, Tk*Ck)
            mask_bool_flat, # (B, Tq*Cq, Tk*Ck)
        )                   # → (B, Tq*Cq, D)

        # Restore original shape
        if query.dim() == 4:
            return out_flat.view(B, Tq, Cq, D)                  # (B, Tq, Cq, D)
        else:
            return out_flat.view(B, Tq, D)                      # (B, Tq, D)
