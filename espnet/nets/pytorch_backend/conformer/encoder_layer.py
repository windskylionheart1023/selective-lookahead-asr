#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch
from torch import nn
from typing import Optional, Union
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.nets_utils import (
    ChunkedMaskConfig,
    ResolvedChunkedMaskConfig,
)

class EncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.

    Streaming extensions: ``forward`` accepts chunked/asymmetric mask
    configuration, RoPE position offsets, and FlexAttention block masks;
    ``forward_streaming`` implements the Stage B chunked streaming path
    with self-attention KV and conv state caches (see ``reset_stream_state``).
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(
        self,
        x_input,
        mask,
        cache=None,
        chunked_mask_config: Optional[Union[ChunkedMaskConfig, ResolvedChunkedMaskConfig]] = None,
        position_offset: int = 0,
        nonpad_mask_1d: Optional[torch.Tensor] = None,
        flex_block_mask=None,
    ):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size)), (1, time, size)]
                              or [(#batch, time, c, size)), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).
            chunked_mask_config: Chunked/asymmetric mask configuration passed to
                self-attention and, for 4D (C-axis) input, to the conv module.
            position_offset (int): RoPE position offset for streaming chunks.
            nonpad_mask_1d (torch.Tensor): 1D non-padding mask for streaming
                padding handling.
            flex_block_mask: Precomputed FlexAttention block mask
                (optional fast path).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size) or (#batch, time, c, size).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        self._propagate_debug_layer_idx()
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape[:2] == (x.shape[0], x.shape[1] - 1)
            x_q = x[:, -1:, ...]
            residual = residual[:, -1:, ...]
            mask = None if mask is None else mask[:, -1:, ...]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask, mask_cfg=chunked_mask_config)
        else:
            x_att = self.self_attn(
                x_q, x, x, mask,
                mask_cfg=chunked_mask_config,
                position_offset=position_offset,
                nonpad_mask_1d=nonpad_mask_1d,
                flex_block_mask=flex_block_mask,
            )

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            # Only pass chunked_mask_config to DCConv when input is 4D (has C-axis).
            # rel_pos path keeps 3D input, so skip DCConv for it.
            args = (chunked_mask_config,) if chunked_mask_config is not None and x.dim() == 4 else tuple()
            x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x, *args))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
            self.feed_forward(x)
        )
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask

    def reset_stream_state(self):
        """Clear Stage B streaming state (conv + self-attn caches) between utterances."""
        if self.conv_module is not None and hasattr(self.conv_module, "reset_stream_state"):
            self.conv_module.reset_stream_state()
        if hasattr(self.self_attn, "reset_enc_stream_cache"):
            self.self_attn.reset_enc_stream_cache()

    def forward_streaming(
        self,
        x_tail,                 # (1, T_tail, C, D)
        mask_flat,              # (1, T_tail*C, (T_cached+T_tail)*C), bool True=attend
        n_finalized_chunks,
        chunk_size,
        num_left_chunks,
        chunked_mask_config,
    ):
        """Stage B streaming forward.

        Operates on the unfinalized tail only. Position-local components
        (LayerNorms, FFNs, macaron FFN) act on the tail as-is. Self-attn
        and conv module use their respective streaming paths that consume
        cached state and fresh tail input.

        Requires ``normalize_before=True`` and ``concat_after=False`` (the
        Conformer defaults used in our configs); other combinations are
        not supported in this path.

        Args:
            x_tail: Layer input for the tail, shape ``(1, T_tail, C, D)``.
            mask_flat: Precomputed flat self-attn mask.
            n_finalized_chunks: Counter for RoPE absolute offset.
            chunk_size: Encoder-rate frames per chunk.
            num_left_chunks: Sliding-window bound for the KV cache (<=0 = unlimited).
            chunked_mask_config: DCConv config.

        Returns:
            Tail output, shape ``(1, T_tail, C, D)``.
        """
        assert self.normalize_before and not self.concat_after, (
            "Stage B forward_streaming only supports normalize_before=True "
            "and concat_after=False"
        )

        self._propagate_debug_layer_idx()

        # Macaron FFN
        if self.feed_forward_macaron is not None:
            residual = x_tail
            x = self.norm_ff_macaron(x_tail)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
        else:
            x = x_tail

        # Self-attention (Stage B streaming)
        residual = x
        x = self.norm_mha(x)
        x_att = self.self_attn._forward_encoder_streaming(
            x,
            mask_flat,
            n_finalized_chunks=n_finalized_chunks,
            chunk_size=chunk_size,
            num_left_chunks=num_left_chunks,
        )
        x = residual + self.dropout(x_att)

        # Conv module (Stage B streaming via DCConv state cache)
        if self.conv_module is not None:
            residual = x
            x = self.norm_conv(x)
            x = residual + self.dropout(
                self.conv_module.forward_streaming(
                    x, chunked_mask_config, chunk_size
                )
            )

        # FFN
        residual = x
        x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x

    def _propagate_debug_layer_idx(self):
        """Forward self._debug_layer_idx onto self_attn so dump hooks can filter."""
        if hasattr(self, "_debug_layer_idx") and self.self_attn is not None:
            self.self_attn._debug_layer_idx = self._debug_layer_idx
