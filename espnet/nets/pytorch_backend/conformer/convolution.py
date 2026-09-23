#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionModule definition with dynamic chunked convolution (DCConv) support."""

import logging
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Union
from espnet.nets.pytorch_backend.nets_utils import (
    ChunkedMaskConfig,
    ResolvedChunkedMaskConfig,
)

class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model with optional dynamic chunked convolution.

    _warned_align_pad warns once per layer instance.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernel size of conv layers.
        activation (nn.Module): Activation function. Default: nn.ReLU()
        bias (bool): Whether to use bias. Default: True.
        dcconv_sync_lookahead (bool): At c_q >= 1 slots with nrc > 0, give
            DCConv Lc right-context taps synced with self-attention lookahead;
            c_q = 0 stays causal. Default: False.
        dcconv_cross_c_q (bool): Enable XCQ-DCConv (per-tap cross-c_q reads
            that bound the future info horizon per encoder depth). Mutually
            exclusive with dcconv_sync_lookahead. Default: False.

    The module applies pointwise conv + GLU, depthwise conv (with context masking if
    chunked_mask_config is provided), normalization, and pointwise conv again.

    If chunked_mask_config is passed to forward(), uses a chunked streaming-aware
    convolution with precise future masking.
    """

    _warned_align_pad = False  # warns once per layer instance
    _warned_xcq_fallback = False  # one-time XCQ-dispatch fallback notice

    def __init__(
        self,
        channels,
        kernel_size,
        activation=nn.ReLU(),
        bias=True,
        dcconv_sync_lookahead: bool = False,
        dcconv_cross_c_q: bool = False,
    ):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        self.channels = channels
        self.kernel_size = kernel_size
        self.Lc = (kernel_size - 1) // 2  # left/right context

        # Stage B: streaming conv state cache.
        # Holds the last Lc frames of this layer's conv INPUT at the chunk
        # being finalized in the previous call, to serve as left context
        # for the current call's oldest tail chunk. Shape (1, Lc, C, D).
        self.stream_state = None

        # Inference-only ablation flag: if True (and self.training is False),
        # forces right_context=Lc at every c_q slot, undoing the trained
        # "causal at every c_q" behaviour. Only meaningful when right-context
        # frames are actually present in the conv input window (i.e. C>=2 at
        # inference). Train/test mismatch — for diagnostics only.
        self.force_inference_right_context = False

        # Training/inference flag: when True, at c_q >= 1 slots with nrc > 0
        # the DCConv right_context is Lc (synced with self-attn's lookahead).
        # c_q = 0 stays causal. Default False preserves the prior behaviour.
        self.dcconv_sync_lookahead = dcconv_sync_lookahead

        # XCQ-DCConv: cross-c_q DCConv that extends the asymmetric attention
        # mask's "info bound = [i+k-R, i+k]" guarantee to the depthwise conv.
        # When True and the chunked path is active with nrc>=1:
        #   For the c_q=k slot's DCConv on chunk i, each kernel tap at position
        #   s in chunk j reads from c_q = clamp(k - (j-i), 0, R) of chunk j.
        #   Taps with j-i > k are zeroed (past the c_q=k allowance).
        # This stops the depth-stacked growing-RF problem that
        # `dcconv_sync_lookahead=True` introduces while still letting DCConv
        # mix in real future-chunk audio. Mutually exclusive with
        # dcconv_sync_lookahead (XCQ supersedes SYNC).
        self.dcconv_cross_c_q = dcconv_cross_c_q
        if dcconv_cross_c_q and dcconv_sync_lookahead:
            raise ValueError(
                "dcconv_cross_c_q and dcconv_sync_lookahead are mutually "
                "exclusive: XCQ is the principled fix for the SYNC growing-RF "
                "problem and replaces SYNC's per-layer Lc right-context."
            )

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=self.Lc,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward_simple(self, x: torch.Tensor):
        """Compute convolution module (without dynamic-chunk convolution).

        Args:
            x (torch.Tensor): Input tensor (#batch, time, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
        """
        x = x.transpose(1, 2)            # (B, D, T)

        # GLU mechanism
        x = self.pointwise_conv1(x)      # (B, 2*D, T)
        x = F.glu(x, dim=1)  # (B, D, T)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)       # (B, D, T)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)      # (B, D, T)

        return x.transpose(1, 2)  # (B, T, D)

    def _forward_dcconv(
        self,
        x: torch.Tensor,
        chunk_size: int,
        right_context: torch.Tensor,
    ):
        """Forward pass with dynamic-chunk convolution.

        Note: this internal function requires a constant chunk size.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            chunk_size (int): Chunk size used in chunked attention.
            right_context (torch.Tensor): Per-batch-item number of future
                frames legal for the conv window. The caller passes 0
                (causal, chunked path) or Lc (full-attention or
                unlimited-right); see forward(). Expected shape (#batch,)

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        B, T, D = x.size()

        # --
        # Example: chunk_size=4, Lc=2, right_context=0, input=aaaabbbbcc
        # |
        # |  Step 2: Pad input
        # |             0 0|a a a a|b b b b|c c 0 0|0 0
        # |
        # |  Step 3: Create overlapping windows
        # |    chunk 1: 0 0|a a a a|b b
        # |    chunk 2:         a a|b b b b|c c
        # |    chunk 3:                 b b|c c 0 0|0 0
        # |
        # |  Step 4: Mask illegal frames
        # |    chunk 1: 0 0|a a a a|0 0
        # |    chunk 2: a a|b b b b|0 0
        # |    chunk 3: b b|c c 0 0|0 0
        # |
        # |  Step 5: Convolution
        # |    chunk 1: A A A A
        # |    chunk 2: B B B B
        # |    chunk 3: C C 0 0
        # |
        # |  Step 6: Flatten [+ truncate]
        # |    A A A A B B B B C C[0 0]
        #
        # Example: chunk_size=4, Lc=2, right_context=4, input=aaaabbbbcc
        # |
        # |  ...
        # |
        # |  Step 4: Mask illegal frames
        # |    chunk 1: 0 0|a a a a|b b
        # |    chunk 2: a a|b b b b|c c
        # |    chunk 3: b b|c c 0 0|0 0
        # |
        # |  Step 5: Convolution
        # |    chunk 1: A A A A
        # |    chunk 2: B B B B
        # |    chunk 3: C C 0 0
        # |
        # |  Step 6: Flatten
        # |    A A A A B B B B C C 0 0
        #

        # 1. Pointwise conv + GLU, keeping (B, D, T) shape
        y = x.transpose(1, 2)  # (B, D, T)
        y = self.pointwise_conv1(y)
        y = F.glu(y, dim=1)    # (B, D, T)

        # Fast path: chunk_size=1 with zero right context (per-frame causal,
        # e.g. the FastEmit-style causal baselines). The future-tap masking
        # below then keeps only taps [t-Lc, t], which equals a causal conv
        # with the left Lc+1 kernel taps (bitwise-identical output, and the
        # masked taps receive exactly zero gradient in both formulations).
        # The unfold path materializes T windows of 2*Lc+1 frames and is
        # several times slower; skip it entirely.
        _rc_all_zero = (
            bool((right_context == 0).all())
            if torch.is_tensor(right_context)
            else right_context == 0
        )
        if chunk_size == 1 and _rc_all_zero:
            y = F.conv1d(
                F.pad(y, (self.Lc, 0)),
                weight=self.depthwise_conv.weight[:, :, : self.Lc + 1],
                bias=self.depthwise_conv.bias,
                stride=1,
                padding=0,
                groups=self.depthwise_conv.groups,
            )  # (B, D, T)
            y = self.activation(self.norm(y))
            y = self.pointwise_conv2(y)  # (B, D, T)
            return y.transpose(1, 2)  # (B, T, D)

        # 2. Padding
        # -- Calculate number of frames to complete the final chunk
        align_pad = (chunk_size - T % chunk_size) % chunk_size

        # -- Padded input: [Lc left pad] + [input] + [align + Lc right pad]
        y = F.pad(y, (self.Lc, align_pad + self.Lc)) # (B, D, T+2*Lc+align_pad)

        # 3. Unfold into overlapping windows of size=(Lc+chunk_size+Lc), step=chunk_size.
        window_size = chunk_size + 2*self.Lc
        y = y.unfold(
            dimension=2,
            size=window_size,
            step=chunk_size,
        ) # (B, D, n_chunks, (chunk_size+2*self.Lc))

        # 4. Mask out future beyond allowed right_context
        pos = torch.arange(window_size, device=x.device).view(1, 1, window_size)
        future_limit = (self.Lc + chunk_size + right_context).view(B, 1, 1)
        mask = (pos >= future_limit).unsqueeze(1)  # (B, 1, 1, window_size)
        y = y.masked_fill(mask, 0.0) # (B, D, n_chunks, (chunk_size+2*self.Lc))
        n_chunks = y.size(2)

        # 5. Convolution
        y = y.transpose(1, 2).flatten(0, 1) # (B * n_chunks, D, (chunk_size+2*self.Lc))
        y = F.conv1d(
            y,
            weight=self.depthwise_conv.weight,
            bias=self.depthwise_conv.bias,
            stride=1,
            padding=0,
            groups=self.depthwise_conv.groups,
        ) # (B * n_chunks, D, chunk_size)

        # 6. Flatten
        y = y.view(B, n_chunks, D, chunk_size)
        y = y.permute(0, 2, 1, 3).reshape(B, D, -1) # (B, D, T+align_pad)

        # ... rest of convolutions
        y = self.activation(self.norm(y))
        y = self.pointwise_conv2(y)  # (B, D, T+align_pad)

        # Remove align padding
        if align_pad > 0:
            if not self._warned_align_pad:
                logging.warning(
                    f"align_pad={align_pad} (> 0). The tail is being truncated; "
                    "check that your input length is a multiple of the DCConv chunk size."
                )
                self._warned_align_pad = True
            y = y[:, :, :-align_pad] # (B, D, T)

        return y.transpose(1, 2) # (B, T, D)

    def _forward_dcconv_cross_cq(
        self,
        x: torch.Tensor,    # (B, T, C, D)  full multi-c_q tensor
        chunk_size: int,
        num_right_chunks: int,  # R; same for all batch items
    ):
        """Cross-c_q depthwise convolution (XCQ-DCConv).

        For the c_q=k slot's DCConv on chunk i, each kernel tap at position s
        in chunk j reads from c_q = clamp(k - (j-i), 0, R) of chunk j. Taps
        with j-i > k are zeroed. This bounds the FUTURE info horizon to i+k
        at every encoder layer depth — no growing right RF:

        - Future taps (0 < j-i <= k) read slot k-(j-i), whose own horizon is
          exactly i+k — perfectly aligned. Taps with j-i > k are zeroed.
        - Past taps (j < i) ideally want slot k+(i-j) (horizon i+k), which
          only exists while k+(i-j) <= R. Beyond that the index CLAMPS to
          slot R — the freshest version of chunk j that exists (horizon
          j+R < i+k). This is intentional ("clamp" semantics): the tap reads
          real audio with a one-sided staleness of k+(i-j)-R chunks rather
          than zeros. No future leak is possible (every read slot's horizon
          is <= i+k); only the past-edge freshness degrades, which the model
          trains with. The past side is NOT bounded below (consistent with
          unlimited left context in the attention mask).

        Args:
            x: Input (B, T, C, D) with C = num_right_chunks + 1.
            chunk_size: Encoder-frame chunk size (constant per call).
            num_right_chunks: R; defines C and the per-c_q lookahead bound.

        Returns:
            (B, T, C, D) output tensor.
        """
        B, T, C, D = x.size()
        Lc = self.Lc
        R = num_right_chunks
        assert C == R + 1, (
            f"_forward_dcconv_cross_cq: C={C} != num_right_chunks+1={R+1}"
        )

        # 1) pointwise_conv1 + GLU per c_q slot (no cross-c_q here)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * C, T, D)   # (B*C, T, D)
        y = x_flat.transpose(1, 2)                            # (B*C, D, T)
        y = self.pointwise_conv1(y)
        y = F.glu(y, dim=1)                                   # (B*C, D, T)
        # Back to (B, T, C, D)
        y = y.transpose(1, 2).view(B, C, T, D).permute(0, 2, 1, 3).contiguous()

        # 2) Pad along T axis: (B, T, C, D) -> (B, T+2*Lc+align_pad, C, D)
        align_pad = (chunk_size - T % chunk_size) % chunk_size
        # F.pad with 4D input: pad order is (D_l, D_r, C_l, C_r, T_l, T_r) — last-axis-first
        y = F.pad(y, (0, 0, 0, 0, Lc, align_pad + Lc))

        # 3) Unfold along T axis (axis 1) into overlapping windows
        window_size = chunk_size + 2 * Lc
        # y.unfold returns (B, n_chunks, C, D, window_size)
        y_unfold = y.unfold(dimension=1, size=window_size, step=chunk_size)
        n_chunks = y_unfold.size(1)

        # 4) Per-window-position c_q index + illegal mask. Same for every
        #    chunk i because chunk_distance depends only on position p, not i:
        #       s_abs = i*cs + p - Lc
        #       j     = floor(s_abs / cs) = i + floor((p - Lc) / cs)
        #    so j - i = floor((p - Lc) / cs).
        pos = torch.arange(window_size, device=x.device)
        chunk_distance = (pos - Lc).div(chunk_size, rounding_mode='floor')  # (window_size,)
        cq_k = torch.arange(C, device=x.device).view(C, 1)            # (C, 1)
        chunk_dist_b = chunk_distance.view(1, window_size)            # (1, window_size)
        # clamp(max=R): past taps whose ideal slot k-(j-i) exceeds R read the
        # freshest existing slot R instead (stale-but-real; see docstring).
        read_cq = (cq_k - chunk_dist_b).clamp(min=0, max=R)           # (C, window_size) long
        # illegal = FUTURE taps beyond the c_q=k allowance (j-i > k); these
        # are the only taps that could leak future info, so only these are
        # zeroed. Past-edge taps are kept (clamped read above, not zeroed).
        illegal = chunk_dist_b > cq_k                                 # (C, window_size) bool

        # 5) Gather from y_unfold along the C axis (dim=2) per (k, position)
        #    y_unfold:     (B, n_chunks, C_in,    D, window_size)
        #    gathered:     (B, n_chunks, C_out=C, D, window_size)
        idx = read_cq.view(1, 1, C, 1, window_size).expand(B, n_chunks, C, D, window_size)
        gathered = y_unfold.gather(dim=2, index=idx)

        # 6) Zero out illegal FUTURE taps (j-i > k); past taps are never zeroed
        illegal_mask = illegal.view(1, 1, C, 1, window_size)
        gathered = gathered.masked_fill(illegal_mask, 0.0)

        # 7) Depthwise conv: flatten (B, n_chunks, C) into the batch dim
        conv_in = gathered.reshape(B * n_chunks * C, D, window_size)
        conv_out = F.conv1d(
            conv_in,
            weight=self.depthwise_conv.weight,
            bias=self.depthwise_conv.bias,
            stride=1,
            padding=0,
            groups=self.depthwise_conv.groups,
        )                                                              # (B*n_chunks*C, D, chunk_size)

        # 8) Restore (B, T+align_pad, C, D): reshape, permute axes
        y_dc = conv_out.view(B, n_chunks, C, D, chunk_size)
        # (B, n_chunks, C, D, chunk_size) -> (B, n_chunks, chunk_size, C, D)
        # -> (B, T+align_pad, C, D)
        y_dc = y_dc.permute(0, 1, 4, 2, 3).reshape(
            B, n_chunks * chunk_size, C, D
        )

        # 9) Norm + activation + pointwise_conv2 per c_q slot
        y_flat = y_dc.permute(0, 2, 1, 3).reshape(B * C, n_chunks * chunk_size, D)
        y_flat = y_flat.transpose(1, 2)                                # (B*C, D, Tp)
        y_flat = self.activation(self.norm(y_flat))
        y_flat = self.pointwise_conv2(y_flat)                          # (B*C, D, Tp)

        # 10) Remove align_pad (truncate from the right) then restore (B, T, C, D)
        if align_pad > 0:
            if not self._warned_align_pad:
                logging.warning(
                    f"[XCQ-DCConv] align_pad={align_pad} (> 0). The tail is being "
                    "truncated; check input length is a multiple of chunk_size."
                )
                self._warned_align_pad = True
            y_flat = y_flat[:, :, :-align_pad]                         # (B*C, D, T)

        y_out = y_flat.view(B, C, D, T).permute(0, 3, 1, 2).contiguous()
        return y_out  # (B, T, C, D)

    def forward(
        self,
        x: torch.Tensor,
        chunked_mask_config: Optional[Union[ChunkedMaskConfig, ResolvedChunkedMaskConfig]] = None,
    ):
        """Forward pass with optional dynamic-chunk convolution.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, size) or (#batch, time, c, size).
                The 4-D (#batch, time, c, size) shape is required when
                chunked_mask_config is provided (DCConv path).
            chunked_mask_config: DCConv configuration if provided

        Returns:
            torch.Tensor: Output tensor (#batch, time, size) or (#batch, time, c, size) .
        """
        if chunked_mask_config is not None:
            # Prepare Dynamic Chunk Convolution (DCConv)...
            assert x.dim() == 4, f"DCConv path but x.dim() == {x.dim()} instead of 4"
            B, T, Cq, D = x.size()

            if isinstance(chunked_mask_config, ChunkedMaskConfig):
                # Batchify fields in chunked_mask_config
                r = chunked_mask_config.resolve(B, device=x.device)
            else:
                r = chunked_mask_config

            # XCQ-DCConv dispatch: when enabled and chunked path with R>=1.
            # Requires (a) all batch items share the same chunk_size and
            # num_right_chunks (the C axis is fixed by the encoder caller),
            # and (b) full_attention is False on all batch items (full-attn
            # batches don't have the chunked structure XCQ depends on).
            if self.dcconv_cross_c_q:
                same_chunk = (r.chunk_size == r.chunk_size[0]).all().item()
                same_nrc = (r.num_right_chunks == r.num_right_chunks[0]).all().item()
                full_attn = r.full_attention.any().item()
                R = int(r.num_right_chunks[0].item())
                if same_chunk and same_nrc and not full_attn and R >= 1 and Cq == R + 1:
                    _chunk_size = int(r.chunk_size[0].item())
                    return self._forward_dcconv_cross_cq(x, _chunk_size, R)
                # If batch mixes full-attention and chunked batches, or R=0,
                # fall through to the legacy per-c_q-causal path below
                # (which is correct under those degenerate conditions).
                if not ConvolutionModule._warned_xcq_fallback:
                    ConvolutionModule._warned_xcq_fallback = True
                    logging.info(
                        "[XCQ-DCConv] dcconv_cross_c_q=True but the XCQ path "
                        "did not dispatch (R=0, full-attention, or mixed "
                        "batch) — using the legacy per-c_q-causal conv. "
                        "Logged once; expected for R=0 decode of XCQ models."
                    )

            # indices 0 … Cq-1
            cq_idx = torch.arange(Cq, device=x.device).view(1, Cq)       # (1, Cq)
            _num_right_chunks = cq_idx.expand(B, Cq)                      # (B, Cq)
            # DCConv is CAUSAL at every c_q slot: no right-chunk info leaks
            # across chunks via the conv window. This preserves the asymmetric
            # attention mask's guarantee that chunk N's c_q=F output depends
            # only on input chunks [N..N+F] regardless of encoder depth.
            # (Prior code used ``c_q * chunk_size`` right-context at c_q=F
            # which let chunk N+1's c=F leak into chunk N's c=F via Lc frames
            # per layer, making chunk 0's c=F output depth-dependent on many
            # future chunks — incompatible with Stage B streaming inference.)
            # Full-attention / unlimited-right still uses Lc (unchanged).
            right_context = torch.where(
                ((r.num_right_chunks < 0) | r.full_attention).unsqueeze(1),     # (B, 1)
                torch.full_like(_num_right_chunks, self.Lc, device=x.device),   # (B, Cq)
                torch.zeros_like(_num_right_chunks),                            # (B, Cq) causal
            ) # (B, Cq)

            # Synced-DCConv (training/inference): when nrc > 0, c_q >= 1 slots
            # get Lc right-context taps so DCConv lookahead matches self-attn's.
            # c_q = 0 stays causal to preserve the "no lookahead" preliminary view.
            if self.dcconv_sync_lookahead:
                sync_mask = (cq_idx > 0).expand(B, Cq) & (r.num_right_chunks > 0).unsqueeze(1)
                right_context = torch.where(
                    sync_mask,
                    torch.full_like(right_context, self.Lc, device=x.device),
                    right_context,
                )

            # Diagnostic override (inference only): drop the causal-at-every-c_q
            # constraint and let the conv kernel see Lc frames of right context.
            # Train/test mismatch — for ablations only. Requires C>=2 so that
            # right-context frames exist in the conv input window.
            if self.force_inference_right_context and not self.training:
                right_context = torch.full_like(
                    right_context, self.Lc, device=x.device
                )

            # Push everything into batch dimension.
            right_context_flat = right_context.view(-1)
            x_perm = x.permute(0, 2, 1, 3).contiguous()  # (B, Cq, T, D)
            x_flat = x_perm.view(B*Cq, T, D)             # (B*Cq, T, D)

            # We only need to compute convolutions on valid indices in Cq dimension.
            # Valid choices: c_q in [0, num_right_chunks[b]]
            nrc_exp = r.num_right_chunks.unsqueeze(1).expand(B, Cq) # (B, Cq)
            valid = cq_idx.expand(B, Cq) <= nrc_exp
            valid_flat = valid.view(-1)  # (B*Cq)
            valid_idx = torch.nonzero(valid_flat, as_tuple=True)[0] # (N_valid,)

            ### SHORTCUT: if right_context >= Lc, then no folding tricks are needed
            if (right_context_flat[valid_idx] >= self.Lc).all():
                y_valid = self.forward_simple(x_flat[valid_idx]) # (N_valid, T, D)
                # Under AMP autocast, y_valid may be bf16 while x_flat is fp32.
                # Cast the clone to match y_valid so scatter-assign dtypes agree.
                y_flat = x_flat.clone().to(y_valid.dtype)
                y_flat[valid_idx] = y_valid
                y = y_flat.view(B, Cq, T, D).permute(0, 2, 1, 3).contiguous()
                return y # (B, T, Cq, D)

            # Check if all chunk sizes match for vectorized DCConv path
            if (r.chunk_size == r.chunk_size[0]).all():
                _chunk_size = r.chunk_size[0].item()
                y_valid = self._forward_dcconv(
                    x_flat[valid_idx],
                    _chunk_size,
                    right_context_flat[valid_idx],
                ) # (N_valid, T, D)
                # Match dtype for AMP (see note above).
                y_flat = x_flat.clone().to(y_valid.dtype)
                y_flat[valid_idx] = y_valid
                y = y_flat.view(B, Cq, T, D).permute(0, 2, 1, 3).contiguous()
            else:
                ys = []
                for b in range(B):
                    _chunk_size = r.chunk_size[b].item()
                    _x = x_perm[b, :, :, :] # (Cq, T, D)
                    _right_context = right_context[b, :] # (Cq,)
                    _y_valid = self._forward_dcconv(_x[valid[b]], _chunk_size, _right_context[valid[b]]) # (n_valid, T, D)
                    # Match dtype for AMP (see note above).
                    _y = _x.clone().to(_y_valid.dtype)
                    _y[valid[b]] = _y_valid
                    ys.append(_y)
                y = torch.stack(ys, dim=0) # (B, Cq, T, D)
                y = y.permute(0, 2, 1, 3).contiguous()
            return y # (B, T, Cq, D)

        else:
            return self.forward_simple(x) # (B, T, D)

    def reset_stream_state(self):
        """Clear the Stage B conv streaming state between utterances."""
        self.stream_state = None

    def forward_streaming(
        self,
        x_tail: torch.Tensor,   # (1, T_tail, C, D)
        chunked_mask_config,
        chunk_size: int,
    ):
        """Stage B streaming-aware conv forward.

        Prepends a chunk-sized buffer whose last Lc frames are the cached
        state (from the chunk being finalized in the previous call), giving
        the tail's first chunk real left context. Runs the regular DCConv
        forward on the combined input, then slices out only the tail
        positions of the output. Updates ``self.stream_state`` with the
        last Lc frames of the chunk being finalized this call.

        The prepended region is made of ``chunk_size`` frames to keep the
        DCConv's chunk alignment intact: ``(chunk_size - Lc)`` zero frames
        followed by ``Lc`` state frames. The first output chunk_size frames
        (corresponding to the prepended region) are discarded.

        Args:
            x_tail: Layer-internal tail representation entering the conv
                module, shape ``(1, T_tail, C, D)``.
            chunked_mask_config: Same as ``forward``. Must be provided in
                streaming mode (asymmetric).
            chunk_size: Encoder-rate frames per chunk.

        Returns:
            Tail output, shape ``(1, T_tail, C, D)``.
        """
        assert x_tail.dim() == 4, (
            f"forward_streaming expects (B, T, C, D), got {tuple(x_tail.shape)}"
        )
        B, T_tail, C, D = x_tail.shape
        assert B == 1, "Stage B encoder streaming cache assumes batch=1 in encoder"

        Lc = self.Lc
        device = x_tail.device
        dtype = x_tail.dtype

        if self.stream_state is None:
            # First call: x_tail starts at the utterance boundary, so forward()'s
            # own internal post-GLU left F.pad supplies exactly the post-GLU-zero
            # left context the full path gives chunk 0. Skip the prepend and slice.
            # Prepending input-level zeros here would be WRONG: they pass through
            # pointwise_conv1 + GLU and become GLU(pointwise_conv1(0)) = GLU(bias)
            # != 0 at the depthwise taps, corrupting chunk 0's first Lc frames.
            y_tail = self.forward(x_tail, chunked_mask_config=chunked_mask_config)
        else:
            # Build the prepended chunk: (chunk_size - Lc) zeros + Lc state frames.
            state = self.stream_state
            if chunk_size - Lc > 0:
                zeros = torch.zeros(
                    1, chunk_size - Lc, C, D, device=device, dtype=dtype
                )
                prepend = torch.cat([zeros, state], dim=1)  # (1, chunk_size, C, D)
            else:
                # kernel_size == 2*chunk_size + 1 edge case — prepend is entirely state
                prepend = state

            x_combined = torch.cat([prepend, x_tail], dim=1)  # (1, chunk_size + T_tail, C, D)

            y = self.forward(x_combined, chunked_mask_config=chunked_mask_config)
            # y shape: (1, chunk_size + T_tail, C, D)

            y_tail = y[:, chunk_size:, :, :]

        # Cache state for next call: last Lc of the chunk being finalized at
        # this call. The chunk being finalized occupies positions [0, chunk_size)
        # of x_tail, so its last Lc frames are x_tail[:, chunk_size-Lc : chunk_size].
        self.stream_state = x_tail[:, chunk_size - Lc : chunk_size, :, :].detach().clone()

        return y_tail
