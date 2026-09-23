# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

import logging
import warnings
from typing import List, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import (
    get_activation,
    make_pad_mask,
    trim_by_ctc_posterior,
    ChunkedMaskConfig,
    build_chunked_mask_from_config,
    build_flex_block_mask_for_encoder,
    build_streaming_attn_mask_flat,
)
from espnet.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    ConvolutionalPositionalEmbedding,
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
    NoPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)

from espnet.nets.pytorch_backend.transformer.subsampling_without_posenc import (
    Conv2dSubsamplingWOPosEnc,
)


class ConformerEncoder(AbsEncoder):
    """Conformer encoder module.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.
        num_attention_sinks (int): Number of learned attention-sink tokens
            prepended to the sequence as global states. 0 disables sinks.
        use_kvcache (bool): Whether to enable per-layer self-attention K/V
            caching for streaming inference.
        use_rope (bool): Whether to apply rotary position embedding (RoPE)
            inside the self-attention modules instead of an input-side
            positional encoding.
        use_flex_attention (bool): Whether to route self-attention through
            torch FlexAttention with a shared per-batch BlockMask (training
            only; falls back to default attention if unavailable).
        use_sdpa (bool): Whether to route self-attention through
            torch.nn.functional.scaled_dot_product_attention.
        dcconv_sync_lookahead (bool): Whether the depthwise conv module uses
            the synchronized-lookahead (SYNCDCCONV) variant.
        dcconv_cross_c_q (bool): Whether the depthwise conv module uses the
            cross-C-query (XCQ) variant.

    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        ctc_trim: bool = False,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
        qk_norm: bool = False,
        use_flash_attn: bool = False,
        num_attention_sinks: int = 0,
        use_kvcache: bool = False,
        use_rope: bool = False,
        use_flex_attention: bool = False,
        # Route self-attention through torch.nn.functional.scaled_dot_product_
        # attention (fused, memory-efficient) instead of the naive matmul+
        # softmax path. Unlike FlexAttention this needs no per-batch kernel
        # rebuild, so it stays fast under dynamic chunk sampling. Default
        # False preserves existing behavior exactly.
        use_sdpa: bool = False,
        dcconv_sync_lookahead: bool = False,
        dcconv_cross_c_q: bool = False,
    ):
        super().__init__()
        self._output_size = output_size

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "conv":
            pos_enc_class = ConvolutionalPositionalEmbedding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        elif pos_enc_layer_type == "rope":
            selfattention_layer_type = "selfattn"
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d6_wo_posenc":
            self.embed = Conv2dSubsamplingWOPosEnc(
                input_size, output_size, dropout_rate, kernels=[3, 5], strides=[2, 3]
            )
            # For non-rope PE types (e.g. abs_pos), add a separate PE module after subsampling
            # since Conv2dSubsamplingWOPosEnc does not inject any positional encoding itself.
            if pos_enc_layer_type != "rope":
                self._posenc_after_embed = pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)
            else:
                self._posenc_after_embed = None
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # Gate FlexAttention on a compatible self-attention layer type.
        if use_flex_attention and selfattention_layer_type != "selfattn":
            logging.warning(
                f"use_flex_attention=True requires selfattention_layer_type='selfattn' "
                f"(got '{selfattention_layer_type}'). Disabling FlexAttention."
            )
            use_flex_attention = False
        if use_flex_attention and attention_dropout_rate > 0.0:
            logging.warning(
                f"use_flex_attention=True is incompatible with attention_dropout_rate="
                f"{attention_dropout_rate} (flex_attention has no dropout arg). "
                "Disabling FlexAttention."
            )
            use_flex_attention = False
        if use_flex_attention and use_kvcache:
            logging.warning(
                "use_flex_attention=True is skipped during KV-cache inference; "
                "flex path only activates for encoder self-attention training."
            )

        if selfattention_layer_type == "selfattn":
            # When requested, verify flash attention is actually available;
            # fall back otherwise.
            if use_flash_attn:
                try:
                    from espnet2.torch_utils.get_flash_attn_compatability import (
                        is_flash_attn_supported,
                    )

                    use_flash_attn = is_flash_attn_supported()
                    import flash_attn  # noqa
                except Exception:
                    use_flash_attn = False

            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                qk_norm,
                use_flash_attn,
                False,
                False,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks

        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        # Only MultiHeadedAttention (selfattn) accepts use_kvcache / use_rope / use_flex_attention
        _attn_extra_kwargs = (
            {
                "use_kvcache": use_kvcache,
                "use_rope": use_rope,
                "use_flex_attention": use_flex_attention,
                "use_sdpa": use_sdpa,
            }
            if selfattention_layer_type == "selfattn"
            else {}
        )
        self.use_flex_attention = use_flex_attention

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args, **_attn_extra_kwargs),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args,
                    dcconv_sync_lookahead=dcconv_sync_lookahead,
                    dcconv_cross_c_q=dcconv_cross_c_q,
                ) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate[lnum],
            ),
            layer_drop_rate,
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        self.ctc_trim = ctc_trim

        if num_attention_sinks > 0:
            self.attention_sink = torch.nn.Parameter(
                torch.randn(num_attention_sinks, output_size) * 0.02
            )
        else:
            self.register_parameter("attention_sink", None)

        # Stage B: encoder-level counter of finalized chunks across a streaming utterance.
        # Incremented by forward_streaming after a successful pass; reset between utterances.
        self.n_finalized_chunks = 0

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        masks: torch.Tensor = None,
        ctc: CTC = None,
        return_all_hs: bool = False,
        chunked_mask_config: ChunkedMaskConfig = None,
        global_states: torch.Tensor = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
            ctc (CTC): ctc module for intermediate CTC loss
            return_all_hs (bool): whether to return all hidden states
            chunked_mask_config (optional, ChunkedMaskConfig): Streaming configuration
            global_states (optional, torch.Tensor): global states to be prepended before
                the embedded speech sequence. Size (#batch, G, input_size) or (1, G, input_size)
                which will be expanded along the batch dimension.
            position_offset (int): RoPE position offset for streaming. When processing
                chunk N, set this to N * chunk_size to get correct absolute positions.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        if masks is None:
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        else:
            masks = ~masks[:, None, :]

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
            or isinstance(self.embed, Conv2dSubsamplingWOPosEnc)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
            if getattr(self, "_posenc_after_embed", None) is not None:
                xs_pad = self._posenc_after_embed(xs_pad)
            _is_conv2d_subsampling = True
        else:
            xs_pad = self.embed(xs_pad)
            _is_conv2d_subsampling = False

        # Unpack pos_emb if the embedding returned a tuple (e.g. RelPositionalEncoding).
        pos_emb = None
        if isinstance(xs_pad, tuple):
            xs_pad, pos_emb = xs_pad

        # Check valid global_states
        if global_states is not None:
            if global_states.dim() != 3:
                raise ValueError(f"Expected global_states to have dim 3, got {global_states.dim()}")
            if global_states.size(-1) != xs_pad.size(-1):
                d1 = f"global_states.size(-1) ({global_states.size(-1)})"
                d2 = f"xs_pad.size(-1) ({xs_pad.size(-1)})"
                raise ValueError(f"Mismatch in global_states size. {d1} != {d2}")

        if self.attention_sink is not None:
            if global_states is None:
                global_states = self.attention_sink.unsqueeze(0) # (1, num_sink, D)
            else:
                _B = global_states.size(0)
                global_states = torch.cat([
                    self.attention_sink.unsqueeze(0).expand(_B, -1, -1),
                    global_states
                ], dim=1) # (_B, num_sink+num_other_global, D)

        # Capture pre-global-prepend column non-pad mask for FlexAttention.
        if self.use_flex_attention:
            # masks is (B, 1, T_sub) here; column j non-pad iff masks[:, 0, j] == True.
            _nonpad_1d_pre_global = masks[:, 0, :]
        else:
            _nonpad_1d_pre_global = None

        # Prepend global states if provided.
        if global_states is not None:
            B = xs_pad.size(0)
            T0, T1 = global_states.size(1), xs_pad.size(1)
            T_new = T0 + T1
            if global_states.size(0) == 1:
                global_states = global_states.expand(B, -1, -1)
            xs_pad = torch.cat([global_states, xs_pad], dim=1) # (B, T_new, D)
            # Copy original mask. At first, allow attention from/to global tokens. Later, this
            # will be adjusted based on "chunked_mask_config".
            new_masks = torch.ones((B, T_new, T_new), dtype=torch.bool, device=masks.device)
            new_masks[:, T0:, T0:] = masks
            masks = new_masks

        # Build 1D non-pad mask over the final (post-global-prepend) sequence.
        if self.use_flex_attention and _nonpad_1d_pre_global is not None:
            if global_states is not None:
                _B = xs_pad.size(0)
                _G = global_states.size(1)
                nonpad_mask_1d = torch.cat(
                    [
                        torch.ones(
                            _B, _G,
                            dtype=torch.bool,
                            device=_nonpad_1d_pre_global.device,
                        ),
                        _nonpad_1d_pre_global,
                    ],
                    dim=1,
                )
            else:
                nonpad_mask_1d = _nonpad_1d_pre_global
        else:
            nonpad_mask_1d = None

        # The attribute `self.embed.do_padding` should be True if we are training for
        # streaming or low-latency purposes. Warn once if it's not set properly.
        if (
            chunked_mask_config is not None
            and _is_conv2d_subsampling
            and getattr(self.embed, "do_padding", True) is False
        ):
            if not hasattr(self, "_chunked_pad_warned"):
                _t = type(self.embed)
                warnings.warn(
                    f"chunked_mask_config is set while the {_t} module has do_padding = False."
                    "This combination may not be ideal for low-latency purposes.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._chunked_pad_warned = True

        # Modify mask for chunked attenton setup.
        if chunked_mask_config is not None:

            # Check number of global states
            if global_states is not None:
                expected_g = chunked_mask_config.num_global_tokens.max()
                actual_g = global_states.size(1)
                if expected_g != actual_g:
                    d1 = f"chunked_mask_config.num_global_tokens.max() ({expected_g})"
                    d2 = f"global_states.size(1) ({actual_g})"
                    raise ValueError(f"Expected {d1} == {d2}")

            # Chunked attention mask — applied identically at every layer.
            # When num_right_chunks=1 this is the asymmetric chunk attention mask:
            # each chunk Ct attends to L past chunks + itself + Ct+1, and Ct+1 itself
            # never attends beyond itself, bounding the future receptive field to 1
            # chunk regardless of network depth.
            B, T = xs_pad.size(0), xs_pad.size(1)
            r = chunked_mask_config.resolve(B, xs_pad.device)
            chunked_mask = build_chunked_mask_from_config(
                r,
                xs_pad.size(1),
                device=xs_pad.device,
            ) # (B, T, T)

            # masks (B, 1, T) or (B, T, T) after global-state prepend: 0 = mask, 1 = allow
            # chunked_mask (B, T, T): 0 = allow, 1 = mask
            masks = masks & (~chunked_mask) # (B, T, T)

            # Inject Cq=Ck dimension into the input
            if r.use_asymmetric_mask.any():
                C = int(r.num_right_chunks.max().item()) + 1
            else:
                C = 1
            if pos_emb is not None and C > 1:
                raise ValueError(
                    f"Legacy rel_pos positional encoding is not supported with "
                    f"use_asymmetric_mask=True (C={C}). "
                    "Use pos_enc_layer_type='rope' or 'abs_pos' instead."
                )
            if not hasattr(self, '_logged_asymmetric_mask'):
                self._logged_asymmetric_mask = True
                nrc = int(r.num_right_chunks.max().item())
                if C > 1:
                    logging.info(
                        f"[Encoder] Asymmetric C-axis mask ACTIVE: C={C}, "
                        f"right receptive field bounded to {nrc} chunk(s)"
                    )
                elif nrc > 0:
                    logging.warning(
                        f"[Encoder] Symmetric mask with num_right_chunks={nrc}: "
                        f"right receptive field grows with depth "
                        f"(~{len(self.encoders)} layers x {nrc} = "
                        f"~{len(self.encoders) * nrc} chunks). "
                        f"Set use_asymmetric_mask=True to prevent this."
                    )

            # Skip C-axis expansion for rel_pos (legacy attention doesn't handle 4D inputs).
            if pos_emb is None:
                xs_pad = xs_pad.unsqueeze(2).expand(-1, -1, C, -1) # (B, T, C, D)

                if C > 1:
                    # C-axis related mask
                    c_mask = r.build_age_mask(T) # (B, Tq, Cq, Tk, Ck) -- Tq=Tk=T, Cq=Ck=C

                    # masks (B, T, T): 0 = mask, 1 = allow
                    # c_mask (B, T, C, T, C): 0 = mask, 1 = allow
                    masks = masks.unsqueeze(2).unsqueeze(4) & c_mask # (B, Tq, Cq, Tk, Ck)
                else:
                    # C=1: no age mask needed, just add trivial C dimensions
                    masks = masks.unsqueeze(2).unsqueeze(4) # (B, Tq, 1, Tk, 1)

        # Re-pack pos_emb into xs_pad tuple for encoder layers that expect it.
        if pos_emb is not None:
            xs_pad = (xs_pad, pos_emb)

        # Build the FlexAttention BlockMask ONCE per batch and share across
        # every encoder layer. Building inside each layer's attention forward
        # would repeat O(B·H·T·C)² mask evaluations ``num_layers`` times.
        flex_block_mask = None
        if (
            self.use_flex_attention
            and chunked_mask_config is not None
            and nonpad_mask_1d is not None
            and pos_emb is None
        ):
            try:
                T_curr = nonpad_mask_1d.size(1)
                flex_block_mask = build_flex_block_mask_for_encoder(
                    resolved_cfg=r,
                    T=T_curr,
                    C=C,
                    nonpad_mask=nonpad_mask_1d,
                    H=self.encoders[0].self_attn.h,
                    device=nonpad_mask_1d.device,
                )
            except Exception as e:
                logging.warning(
                    f"FlexAttention BlockMask build failed, falling back to "
                    f"default attention: {e}"
                )
                flex_block_mask = None
                self.use_flex_attention = False

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            for encoder_layer in self.encoders:
                xs_pad, masks = encoder_layer(
                    xs_pad, masks,
                    chunked_mask_config=chunked_mask_config,
                    position_offset=position_offset,
                    nonpad_mask_1d=nonpad_mask_1d,
                    flex_block_mask=flex_block_mask,
                )
                if return_all_hs:
                    if isinstance(xs_pad, tuple):
                        intermediate_outs.append(xs_pad[0])
                    else:
                        intermediate_outs.append(xs_pad)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(
                    xs_pad, masks,
                    chunked_mask_config=chunked_mask_config,
                    position_offset=position_offset,
                    nonpad_mask_1d=nonpad_mask_1d,
                    flex_block_mask=flex_block_mask,
                )

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x = x + self.conditioning_layer(ctc_out)
                            xs_pad = (x, pos_emb)
                        else:
                            xs_pad = xs_pad + self.conditioning_layer(ctc_out)

                    if self.ctc_trim and ctc is not None:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x, masks, pos_emb = trim_by_ctc_posterior(
                                x, ctc_out, masks, pos_emb
                            )
                            xs_pad = (x, pos_emb)
                        else:
                            x, masks, _ = trim_by_ctc_posterior(x, ctc_out, masks)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        # olens under chunked masks: valid positions are the diagonal of the
        # (possibly C-expanded) self-attention mask.
        if chunked_mask_config is not None:
            if masks.dim() == 5:
                # (B, Tq, Cq, Tk, Ck) -> extract (B, T, T) diagonal
                sequence_mask_2d = masks[:, :, 0, :, 0] # (B, T, T)
            else:
                # (B, T, T) — rel_pos path skips C-axis expansion
                sequence_mask_2d = masks
            olens_mask = sequence_mask_2d.diagonal(dim1=1, dim2=2) # (B, T)
            # masks convention: 1 = allow, 0 = mask, so diagonal has 1 for valid positions
            olens = olens_mask.sum(1) # (B)
        else:
            olens = masks.squeeze(1).sum(1)

        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None

    def reset_streaming_state(self):
        """Stage B: clear the encoder-streaming state between utterances.

        Resets the finalized-chunk counter and clears each layer's
        self-attention KV cache and conv state cache.
        """
        self.n_finalized_chunks = 0
        for layer in self.encoders:
            if hasattr(layer, "reset_stream_state"):
                layer.reset_stream_state()

    def subsample_only(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> torch.Tensor:
        """Stage B helper: run the subsampling stem only and return subsampled features.

        For Stage B streaming the inference wrapper runs subsampling on the
        full accumulated raw features each call (keeps subsampling numerically
        identical to training), then slices the tail and calls
        :meth:`forward_streaming`. This method performs just the subsampling
        step. The returned tensor is the plain feature tensor (no positional
        embedding tuple) since RoPE is applied inside the self-attn modules.

        Args:
            xs_pad: Raw pre-subsampling features ``(1, T_raw, D)``.
            ilens: Input lengths ``(1,)``.

        Returns:
            Subsampled features, shape ``(1, T_subsampled, D)``.
        """
        # Build non-pad mask directly without the O(maxlen^2) triu allocation
        # in make_pad_mask. For long-form streaming, ilens can be tens of
        # thousands of frames; the triu mask in nets_utils.py:251 explodes
        # GPU memory. arange-based construction is O(B*maxlen).
        maxlen = int(xs_pad.size(1))
        idx = torch.arange(maxlen, device=xs_pad.device)
        masks = (idx.unsqueeze(0) < ilens.to(xs_pad.device).unsqueeze(-1)).unsqueeze(1)
        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
            or isinstance(self.embed, Conv2dSubsamplingWOPosEnc)
        ):
            xs_pad, _ = self.embed(xs_pad, masks)
            if getattr(self, "_posenc_after_embed", None) is not None:
                xs_pad = self._posenc_after_embed(xs_pad)
        else:
            xs_pad = self.embed(xs_pad)
        if isinstance(xs_pad, tuple):
            # strip pos_emb — RoPE path does not use it
            xs_pad = xs_pad[0]
        return xs_pad

    def forward_streaming(
        self,
        xs_pad_tail: torch.Tensor,              # (1, T_tail, D) subsampled tail
        chunk_size: int,
        num_right_chunks: int,
        num_left_chunks: int,
        chunked_mask_config: ChunkedMaskConfig,
        n_pad_tail: int = 0,                     # trailing zero-pad frames in the tail
    ) -> torch.Tensor:
        """Stage B streaming forward over the unfinalized tail.

        Given the subsampled tail (a slice of the full accumulated encoder
        input), runs every encoder layer via its streaming path. Each layer
        consumes cached K/V for the finalized prefix and fresh K/V for the
        tail. After the pass, every layer has appended the oldest-tail
        chunk's K/V to its permanent cache; this method advances the
        encoder-level ``n_finalized_chunks`` counter by 1.

        Args:
            xs_pad_tail: Subsampled tail (batch=1) of the unfinalized chunks.
            chunk_size: Encoder-rate frames per chunk.
            num_right_chunks: F; number of right-context chunks (asym bound).
            num_left_chunks: Sliding-window bound for the self-attn cache (<=0 = unlimited).
            chunked_mask_config: Chunked mask configuration; must have
                ``use_asymmetric_mask=True`` whenever ``num_right_chunks > 0``.
            n_pad_tail: Number of trailing zero-pad frames in the tail;
                forwarded as ``n_pad_keys`` so padded key positions are masked.

        Returns:
            Tail output, shape ``(1, T_tail, C, D)`` where ``C = num_right_chunks + 1``.
            Caller slices out the C=F (max right-context) view of the oldest chunk
            as the released finalized encoding.
        """
        assert xs_pad_tail.dim() == 3 and xs_pad_tail.size(0) == 1, (
            f"forward_streaming expects (1, T_tail, D), got {tuple(xs_pad_tail.shape)}"
        )
        T_tail = xs_pad_tail.size(1)
        C = num_right_chunks + 1
        device = xs_pad_tail.device

        # Resolve mask config
        if isinstance(chunked_mask_config, ChunkedMaskConfig):
            r = chunked_mask_config.resolve(1, device=device)
        else:
            r = chunked_mask_config
        if not r.use_asymmetric_mask.any() and num_right_chunks > 0:
            raise ValueError(
                "forward_streaming requires use_asymmetric_mask=True"
            )

        # Expand C-axis: (1, T_tail, D) -> (1, T_tail, C, D)
        # Match full-recompute path which uses stride-0 expand (no .contiguous()).
        # Materialising via .contiguous() causes CUDA kernels to dispatch
        # differently (different fp accumulation order), producing ~5e-3 diff
        # at layer 0 that compounds to ~1e-1 at deep layers.
        x_tail = xs_pad_tail.unsqueeze(2).expand(-1, -1, C, -1)

        # Build the flat streaming self-attn mask once and share across layers.
        # Cache is sliding-trimmed to ``num_left_chunks * chunk_size`` keys
        # per layer (see attention.py::_forward_encoder_streaming). The mask
        # must match this trimmed length, not the absolute history length.
        # Shift query_offset accordingly so chunk-distance still computes
        # correctly relative to the cache window. RoPE uses absolute
        # ``n_finalized_chunks * chunk_size`` and is unaffected.
        n_fin = self.n_finalized_chunks
        # L >= 0 is finite (L=0 -> own-chunk-only, no cached history); must stay
        # consistent with the cache trim in attention.py. L=-1 (unlimited) keeps
        # the full finalized prefix.
        n_in_cache = (
            min(n_fin, num_left_chunks)
            if num_left_chunks is not None and num_left_chunks >= 0
            else n_fin
        )
        mask_flat = build_streaming_attn_mask_flat(
            r,
            T_tail=T_tail,
            T_cached=n_in_cache * chunk_size,
            query_offset=n_in_cache * chunk_size,
            device=device,
            n_pad_keys=n_pad_tail,
        )

        # Each layer consumes its own cache and advances it; counter n_fin
        # is the same for every layer this call.
        for layer in self.encoders:
            x_tail = layer.forward_streaming(
                x_tail,
                mask_flat=mask_flat,
                n_finalized_chunks=n_fin,
                chunk_size=chunk_size,
                num_left_chunks=num_left_chunks,
                chunked_mask_config=r,
            )

        if self.normalize_before:
            # after_norm operates position-local, safe on (B, T, C, D)
            x_tail = self.after_norm(x_tail)

        # Advance counter AFTER all layers succeed — atomic wrt crashes mid-pass.
        self.n_finalized_chunks += 1

        return x_tail
