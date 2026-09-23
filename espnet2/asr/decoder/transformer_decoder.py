# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
import logging
from typing import Any, List, Optional, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import (
    BatchScorerInterface,
    MaskParallelScorerInterface,
)


class BaseTransformerDecoder(
    AbsDecoder, BatchScorerInterface, MaskParallelScorerInterface
):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        use_rope: whether to use RoPE in the decoder self-attention; when True,
            the absolute positional encoding on the embedding is removed.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        gradient_checkpoint_layers: List[int] = [],
        use_rope: bool = False, # RoPE is only applied for the self-attention
    ):
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            if use_rope:
                # remove the abs positional encoding when using RoPE
                self.embed = torch.nn.Embedding(vocab_size, attention_dim)
            else:
                self.embed = torch.nn.Sequential(
                    torch.nn.Embedding(vocab_size, attention_dim),
                    pos_enc_class(attention_dim, positional_dropout_rate),
                )
        elif input_layer == "linear":
            if use_rope:
                self.embed = torch.nn.Sequential(
                    torch.nn.Linear(vocab_size, attention_dim),
                    torch.nn.LayerNorm(attention_dim),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.ReLU(),
                )
            else:
                self.embed = torch.nn.Sequential(
                    torch.nn.Linear(vocab_size, attention_dim),
                    torch.nn.LayerNorm(attention_dim),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.ReLU(),
                    pos_enc_class(attention_dim, positional_dropout_rate),
                )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        self._output_size_bf_softmax = attention_dim
        # Must set by the inheritance
        self.decoders = None
        self.batch_ids = None

        # For gradient checkpointing, start from 1 (not 0)
        self.gradient_checkpoint_layers = gradient_checkpoint_layers
        logging.info(f"Gradient checkpoint layers: {self.gradient_checkpoint_layers}")

    def _clear_kvcache(self):
        """Clear key-value cache used for inference-time attention.

        This method deletes the stored key and value tensors along with their
        associated length tracking variables. It is typically used to reset
        the internal state before starting a new sequence or utterance.
        """
        for decoder in self.decoders:
            decoder._clear_kvcache()

    def _extend_cross_kvcache(self, new_memory, max_cache_frames: int = -1):
        """Extend cross-attention KV cache with new encoder frames.

        Args:
            new_memory: New encoder frames (B, T_new, D).
            max_cache_frames: Max frames to keep (-1 = unlimited).
        """
        for decoder in self.decoders:
            decoder._extend_cross_kvcache(new_memory, max_cache_frames=max_cache_frames)

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
        for decoder in self.decoders:
            decoder._update_hyp_order(order)

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        cross_attn_mask: Optional[torch.Tensor] = None,
        return_hs: bool = False,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
            cross_attn_mask: optional causal cross-attention mask, bool
                (batch, maxlen_out, maxlen_in) where True = valid, False = masked.
                If provided, each decoder position can only attend to encoder
                positions where the mask is True.
            return_hs: (bool) whether to return the last hidden output
                                  before output layer
            return_all_hs: (bool) whether to return all the hidden intermediates
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        # Create padding mask: (B, 1, src_len)
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )
        # Padding for Longformer
        if memory_mask.shape[-1] != memory.shape[1]:
            padlen = memory.shape[1] - memory_mask.shape[-1]
            memory_mask = torch.nn.functional.pad(
                memory_mask, (0, padlen), "constant", False
            )

        # Combine with causal cross-attention mask if provided
        if cross_attn_mask is not None:
            # cross_attn_mask: (B, tgt_len, src_len) - True = valid
            # memory_mask: (B, 1, src_len) - True = valid
            # Result: (B, tgt_len, src_len) - both conditions must be True
            memory_mask = memory_mask & cross_attn_mask

        x = self.embed(tgt)
        intermediate_outs = []
        for layer_idx, decoder_layer in enumerate(self.decoders):
            # set the validation mode for decoder layers
            decoder_layer.validation_mode = getattr(self, "validation_mode", False)
            if layer_idx + 1 in self.gradient_checkpoint_layers:
                x, tgt_mask, memory, memory_mask = torch.utils.checkpoint.checkpoint(
                    decoder_layer, x, tgt_mask, memory, memory_mask, use_reentrant=False
                )
            else:
                x, tgt_mask, memory, memory_mask = decoder_layer(
                    x, tgt_mask, memory, memory_mask
                )
            if return_all_hs:
                intermediate_outs.append(x)
        if self.normalize_before:
            x = self.after_norm(x)
        if return_hs:
            hidden = x
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        if return_hs:
            return (x, hidden), olens
        elif return_all_hs:
            return (x, intermediate_outs), olens
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor = None,
        *,
        cache: List[torch.Tensor] = None,
        return_hs: bool = False,
        new_encoder_frames: torch.Tensor = None,
        cross_attn_max_cache_frames: int = -1,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask (batch, 1, maxlen_in)
            cache: cached output list of (batch, max_time_out-1, size)
            return_hs: dec hidden state corresponding to ys,
                used for searchable hidden ints
            new_encoder_frames: optional new encoder frames (batch, T_new, feat)
                for incremental cross-attention KV cache extension in streaming.
            cross_attn_max_cache_frames: sliding window size for cross-attn
                KV cache. -1 means unlimited.
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        use_kvcache = getattr(self, "use_kvcache", False)
        _kv_kwargs = {"use_kvcache": use_kvcache}

        # Extend cross-attention KV cache with new encoder frames (streaming)
        if new_encoder_frames is not None and use_kvcache:
            self._extend_cross_kvcache(
                new_encoder_frames, max_cache_frames=cross_attn_max_cache_frames
            )

        if use_kvcache and cache is not None:
            # Note: The first iteration must skip this path (in order to process
            #       the entire prompt). This is done with the "cache is not None"
            #       condition above.
            if isinstance(self.embed, torch.nn.Embedding):
                # RoPE case: bare Embedding, no positional encoding layer.
                # Positions are handled by RoPE inside the attention modules.
                x = self.embed(tgt[:, -1:])  # (B, 1, D)
            else:
                assert (
                    len(self.embed) == 2
                ), "kvcache: currently only supporting Embedding + pos_encoding"
                assert isinstance(
                    self.embed[0], torch.nn.Embedding
                ), "kvcache: self.embed[0] must be Embedding"
                assert isinstance(
                    self.embed[1], PositionalEncoding
                ), "kvcache: self.embed[1] must be PositionalEncoding"
                x = self.embed[0](tgt[:, -1:])  # (B, 1, D)
                x = self.embed[1](x, t_offset=tgt.size(1) - 1)  # (B, 1, D)
            if tgt_mask is not None:
                tgt_mask = tgt_mask[:, -1:]  # (B, 1, Tk)
        else:
            x = self.embed(tgt)
            if use_kvcache:
                # Full reprocessing path: clear stale attention KV caches so they
                # get reinitialized from scratch (k_cache is None → init path).
                self._clear_kvcache()

        if cache is None:
            cache = [None] * len(self.decoders)

        # When using incremental decoding (cache-based or KV cache),
        # the decoder layer processes only the last query frame.
        # Slice memory_mask to match that query size so cross-attention
        # broadcasting does not inflate the output shape.
        if (
            cache[0] is not None
            and memory_mask is not None
            and memory_mask.dim() == 3
            and memory_mask.shape[1] > 1
        ):
            memory_mask = memory_mask[:, -1:, :]

        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c, **_kv_kwargs
            )
            if use_kvcache:
                dummy_state = torch.zeros(
                    (tgt.size(0),), dtype=torch.bool, device="cpu"
                )
                new_cache.append(dummy_state)  # dummy variable that's not None
            else:
                new_cache.append(x)  # x = cat(previous cache, new state vector)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if return_hs:
            hidden = y
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        if return_hs:
            return (y, hidden), new_cache
        return y, new_cache

    def score(self, ys, state, x, return_hs=False):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        if return_hs:
            (logp, hs), state = self.forward_one_step(
                ys.unsqueeze(0),
                ys_mask,
                x.unsqueeze(0),
                cache=state,
                return_hs=return_hs,
            )
            return logp.squeeze(0), hs, state
        else:
            logp, state = self.forward_one_step(
                ys.unsqueeze(0),
                ys_mask,
                x.unsqueeze(0),
                cache=state,
                return_hs=return_hs,
            )
            return logp.squeeze(0), state

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        return_hs: bool = False,
        new_encoder_frames: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        cross_attn_max_cache_frames: int = -1,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
            new_encoder_frames: optional new encoder frames (batch, T_new, feat)
                for incremental cross-attention KV cache extension in streaming.
            memory_mask: optional cross-attention mask (n_batch, ylen, xlen)
                where True=allow. Used for chunked streaming inference.
            cross_attn_max_cache_frames: sliding window size for cross-attn
                KV cache. -1 means unlimited.

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if any(s is None for s in states):
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        if return_hs:
            (logp, hs), states = self.forward_one_step(
                ys, ys_mask, xs, memory_mask=memory_mask,
                cache=batch_state, return_hs=return_hs,
                new_encoder_frames=new_encoder_frames,
                cross_attn_max_cache_frames=cross_attn_max_cache_frames,
            )
        else:
            logp, states = self.forward_one_step(
                ys, ys_mask, xs, memory_mask=memory_mask,
                cache=batch_state, return_hs=return_hs,
                new_encoder_frames=new_encoder_frames,
                cross_attn_max_cache_frames=cross_attn_max_cache_frames,
            )

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        if return_hs:
            return (logp, hs), state_list
        return logp, state_list

    def forward_partially_AR(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_lengths: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (n_mask * n_beam, maxlen_out)
            tgt_mask: input token mask,  (n_mask * n_beam, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            tgt_lengths: (n_mask * n_beam, )
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)  # (n_mask * n_beam, maxlen_out, D)
        new_cache = []
        if cache is None:
            cache = [None] * len(self.decoders)

        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, tgt_lengths, memory, memory_mask = (
                decoder.forward_partially_AR(
                    x, tgt_mask, tgt_lengths, memory, None, cache=c
                )
            )
            new_cache.append(x)

        if self.batch_ids is None or len(self.batch_ids) < x.size(0):
            self.batch_ids = torch.arange(x.size(0), device=x.device)

        if self.normalize_before:
            y = self.after_norm(
                x[self.batch_ids[: x.size(0)], tgt_lengths.unsqueeze(0) - 1].squeeze(0)
            )
        else:
            y = x[self.batch_ids, tgt_lengths.unsqueeze(0) - 1].squeeze(0)

        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, torch.stack(new_cache)

    def batch_score_partially_AR(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        yseq_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Any]]:
        # merge states
        if states[0] is None:
            batch_state = None
        else:
            # reshape state of [mask * batch, layer, 1, D]
            # into [layer, mask * batch, 1, D]
            batch_state = states.transpose(0, 1)

        # batch decoding
        tgt_mask = (~make_pad_mask(yseq_lengths)[:, None, :]).to(xs.device)
        m = subsequent_mask(tgt_mask.size(-1), device=xs.device).unsqueeze(0)
        tgt_mask = tgt_mask & m

        logp, states = self.forward_partially_AR(
            ys, tgt_mask, yseq_lengths, xs, cache=batch_state
        )

        # states is torch.Tensor, where shape is (layer, n_mask * n_beam, yseq_len, D)
        # reshape state to [n_mask * n_beam, layer, yseq_len, D]
        state_list = states.transpose(0, 1)
        return logp, state_list


class TransformerDecoder(BaseTransformerDecoder):
    """Transformer decoder with optional KV caching and RoPE.

    In addition to the standard arguments of BaseTransformerDecoder:

    Args:
        use_kvcache: whether to enable key-value caching in the attention
            modules for incremental (streaming) decoding.
        kvcache_maxlen: maximum number of cached key-value positions.
        use_rope: whether to use RoPE in the decoder self-attention
            (cross-attention never uses RoPE); when True, the absolute
            positional encoding on the embedding is removed.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        layer_drop_rate: float = 0.0,
        qk_norm: bool = False,
        use_flash_attn: bool = True,
        use_kvcache: bool = False,
        kvcache_maxlen: int = 8192,
        use_rope: bool = False, # RoPE is only applied for the self-attention
        gradient_checkpoint_layers: List[int] = [],
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            gradient_checkpoint_layers=gradient_checkpoint_layers,
            use_rope=use_rope,
        )

        if use_flash_attn:
            try:
                from espnet2.torch_utils.get_flash_attn_compatability import (
                    is_flash_attn_supported,
                )

                use_flash_attn = is_flash_attn_supported()
                import flash_attn  # noqa
            except Exception:
                use_flash_attn = False
        self.use_kvcache = use_kvcache
        self.kvcache_maxlen = kvcache_maxlen
        self.use_rope = use_rope

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                # Self attention
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    self_attention_dropout_rate,
                    qk_norm,
                    use_flash_attn,
                    True,
                    False,
                    use_kvcache=use_kvcache,
                    kvcache_maxlen=kvcache_maxlen,
                    use_rope=use_rope, # RoPE in self attention
                ),
                # Cross attention
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    src_attention_dropout_rate,
                    qk_norm,
                    use_flash_attn,
                    False,
                    True,
                    use_kvcache=use_kvcache,
                    kvcache_maxlen=kvcache_maxlen,
                    use_rope=False, # RoPE is not used for cross attention
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
            layer_drop_rate,
        )


class LightweightConvolutionTransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                LightweightConvolution(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class LightweightConvolution2DTransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                LightweightConvolution2D(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class DynamicConvolutionTransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )
        attention_dim = encoder_output_size

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                DynamicConvolution(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class DynamicConvolution2DTransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )
        attention_dim = encoder_output_size

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                DynamicConvolution2D(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class TransformerMDDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        use_speech_attn: bool = True,
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
                (
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    )
                    if use_speech_attn
                    else None
                ),
            ),
        )

        self.use_speech_attn = use_speech_attn

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        speech: torch.Tensor = None,
        speech_lens: torch.Tensor = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
            return_hs: dec hidden state corresponding to ys,
                used for searchable hidden ints
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )

        if speech is not None:
            speech_mask = (~make_pad_mask(speech_lens, maxlen=speech.size(1)))[
                :, None, :
            ].to(speech.device)
        else:
            speech_mask = None

        x = self.embed(tgt)
        if self.use_speech_attn:
            x, tgt_mask, memory, memory_mask, _, speech, speech_mask = self.decoders(
                x, tgt_mask, memory, memory_mask, None, speech, speech_mask
            )
        else:
            x, tgt_mask, memory, memory_mask = self.decoders(
                x, tgt_mask, memory, memory_mask
            )
        if self.normalize_before:
            x = self.after_norm(x)
            if return_hs:
                hs_asr = x
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)

        if return_hs:
            return x, olens, hs_asr

        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor = None,
        *,
        speech: torch.Tensor = None,
        speech_mask: torch.Tensor = None,
        cache: List[torch.Tensor] = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask (batch, 1, maxlen_in)
            speech: encoded speech, float32  (batch, maxlen_in, feat)
            speech_mask: encoded memory mask (batch, 1, maxlen_in)
            cache: cached output list of (batch, max_time_out-1, size)
            return_hs: dec hidden state corresponding to ys,
                used for searchable hidden ints
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            if self.use_speech_attn:
                x, tgt_mask, memory, memory_mask, _, speech, speech_mask = decoder(
                    x,
                    tgt_mask,
                    memory,
                    memory_mask,
                    cache=c,
                    pre_memory=speech,
                    pre_memory_mask=speech_mask,
                )
            else:
                x, tgt_mask, memory, memory_mask = decoder(
                    x, tgt_mask, memory, memory_mask, cache=c
                )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]

        if return_hs:
            h_asr = y

        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        if return_hs:
            return y, h_asr, new_cache
        return y, new_cache

    def score(self, ys, state, x, speech=None):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0),
            ys_mask,
            x.unsqueeze(0),
            speech=speech.unsqueeze(0) if speech is not None else None,
            cache=state,
        )
        return logp.squeeze(0), state

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        speech: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(
            ys, ys_mask, xs, speech=speech, cache=batch_state
        )

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
