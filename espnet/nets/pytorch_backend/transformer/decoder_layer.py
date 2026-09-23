#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder self-attention layer definition."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        sequential_attn (bool): computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.


    """

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        sequential_attn=None,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sequential_attn = sequential_attn
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        if sequential_attn is not None:
            self.norm4 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)
            if sequential_attn is not None:
                self.concat_linear3 = nn.Linear(size + size, size)

        self.tgt_ids = None

    def _clear_kvcache(self):
        """Clear key-value cache used for inference-time attention.

        This method deletes the stored key and value tensors along with their
        associated length tracking variables. It is typically used to reset
        the internal state before starting a new sequence or utterance.
        """
        self.self_attn._clear_kvcache()
        if self.src_attn is not None:
            self.src_attn._clear_kvcache()

    def _extend_cross_kvcache(
        self, new_memory: "torch.Tensor", max_cache_frames: int = -1
    ):
        """Extend cross-attention KV cache with new encoder frames.

        Args:
            new_memory: New encoder frames (B, T_new, D).
            max_cache_frames: Max frames to keep (-1 = unlimited).
        """
        if self.src_attn is not None:
            self.src_attn._extend_cross_kvcache(
                new_memory, max_cache_frames=max_cache_frames
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
        self.self_attn._update_hyp_order(order)
        if self.src_attn is not None:
            self.src_attn._update_hyp_order(order)

    def forward(
        self,
        tgt,
        tgt_mask,
        memory,
        memory_mask,
        cache=None,
        pre_memory=None,
        pre_memory_mask=None,
        use_kvcache=False,
    ):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, 1, maxlen_in).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).
            pre_memory (torch.Tensor): Encoded memory (#batch, maxlen_in, size).
            pre_memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
            use_kvcache (bool): If True, run single-step decoding using the KV
                caches stored inside the attention modules; the legacy `cache`
                concatenation is skipped.

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        # set the validation mode for self attn and cross attn
        if hasattr(self, "validation_mode") and self.validation_mode:
            self.self_attn.validation_mode = True
            self.src_attn.validation_mode = True
            # only set True here; attention modules default to False and
            # inference never sets it
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        elif use_kvcache:
            # cache is empty (dummy "False" variable) since the KV cache
            # is stored in each attention module
            # tgt: (B, 1, D)
            # tgt_mask: (B, 1, Tk)
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        if self.sequential_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm4(x)
            if self.concat_after:
                x_concat = torch.cat(
                    (
                        x,
                        self.sequential_attn(
                            x, pre_memory, pre_memory, pre_memory_mask
                        ),
                    ),
                    dim=-1,
                )
                x = residual + self.concat_linear3(x_concat)
            else:
                x = residual + self.dropout(
                    self.sequential_attn(x, pre_memory, pre_memory, pre_memory_mask)
                )
            if not self.normalize_before:
                x = self.norm4(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None and not use_kvcache:
            x = torch.cat([cache, x], dim=1)

        if pre_memory is not None:
            return x, tgt_mask, memory, memory_mask, None, pre_memory, pre_memory_mask
        return x, tgt_mask, memory, memory_mask

    def forward_partially_AR(
        self, tgt, tgt_mask, tgt_lengths, memory, memory_mask, cache=None
    ):
        """Forward partially in autoregression fashion."""
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"

            if self.tgt_ids is None or self.tgt_ids.shape[0] < tgt.shape[0]:
                self.tgt_ids = torch.arange(tgt.size(0), device=tgt.device).view(-1, 1)

            tgt_q = tgt[
                self.tgt_ids[: tgt.size(0)], tgt_lengths.view(-1, 1) - 1
            ]  # (n_mask * n_beam, 1, D)
            residual = residual[
                self.tgt_ids[: tgt.size(0)], tgt_lengths.view(-1, 1) - 1
            ]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[
                    self.tgt_ids[: tgt.size(0)], tgt_lengths.view(-1, 1) - 1
                ]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask, expand_kv=True)),
                dim=-1,
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(
                self.src_attn(x, memory, memory, memory_mask, expand_kv=True)
            )
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            _tmp = torch.cat([cache, torch.zeros_like(x)], dim=1)
            _tmp[self.tgt_ids[: tgt.size(0)], tgt_lengths.view(-1, 1) - 1] = x
            return _tmp, tgt_mask, tgt_lengths, memory, memory_mask

        return x, tgt_mask, tgt_lengths, memory, memory_mask
