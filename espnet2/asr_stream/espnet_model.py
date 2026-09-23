"""Streaming ASR task model: joint CTC/attention with dynamic chunked-mask training.

Wraps the standard ESPnet2 ASR model with the training-side pieces of the
bounded-lookahead chunk encoder (ChunkedMaskSampler: chunk size, left context
and number of future chunks sampled per batch, mixed with full-attention
batches) and the alignment-supervised soft early-emission losses (cross-attention
and CTC mass after each token's deadline chunk). Inference-time encode() applies
the same chunked mask (fixed_chunk_config) so training matches streaming decoding.
"""
import inspect
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.linear_decoder import LinearDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import (
    ChunkedMaskConfig,
    DynamicChunkConfigSampler,
    th_accuracy,
)
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

autocast_type = torch.float16
_use_new_autocast_api = False
if V(torch.__version__) >= V("1.6.0"):
    if V(torch.__version__) >= V("2.0.0"):
        from torch.amp import autocast as _autocast_impl
        _use_new_autocast_api = True
    else:
        from torch.cuda.amp import autocast as _autocast_impl

    if (
        V(torch.__version__) >= V("1.10.0")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        autocast_type = torch.bfloat16

    # Wrapper to handle both old and new API
    @contextmanager
    def autocast(enabled=True, **kwargs):
        if _use_new_autocast_api:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with _autocast_impl(device_type, enabled=enabled, **kwargs):
                yield
        else:
            with _autocast_impl(enabled=enabled, **kwargs):
                yield
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True, **kwargs):
        yield


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model for streaming ASR.

    Streaming fork of espnet2.asr.espnet_model.ESPnetASRModel. It adds
    dynamic chunked-mask training (with offline-mix batches), chunked
    decoder cross-attention masks driven by per-token alignments,
    attention- and CTC-based early-emission losses, 4D (C-axis) encoder
    output handling, and a FastEmit-regularized transducer loss.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: Optional[AbsEncoder],
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: Optional[dict] = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        transducer_multi_blank_durations: List = [],
        transducer_multi_blank_sigma: float = 0.05,
        # FastEmit regularization (https://arxiv.org/abs/2010.11148) for the
        # transducer loss; 0.0 disables it.
        fastemit_lambda: float = 0.0,
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        autocast_frontend: bool = False,
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        # New primary params
        dynamic_chunk_training: bool = True,
        dynamic_chunk_training_config: Optional[dict] = None,
        fixed_chunk_config: Optional[dict] = None,
        # Deprecated: kept for backward compat with saved config.yaml files
        use_chunked_training: bool = True,
        chunked_training_config: Optional[dict] = None,
        use_chunked_cross_attn: bool = True,
        cross_attn_sync_with_encoder_chunk: bool = False,
        use_alignment_loss: bool = True,
        early_emit_weight: float = 0.1,
        # If >0, decoder cross-attn mask at training widens to expose K[C+1..C+nrc]
        # to query positions in chunk C as context (paired with chunk-alignment
        # loss to teach "context vs emission target" distinction). Default 0
        # preserves the original training-time behavior of every saved config.
        cross_attn_num_right_train: int = 0,
        # If True, the decoder cross-attn future-chunk allowance follows the
        # encoder's per-batch sampled num_right_chunks instead of the static
        # cross_attn_num_right_train. Pairs with cross_attn_sync_with_encoder_chunk
        # (False default → left context already mirrors encoder) to make the
        # decoder cross-attn region fully track the encoder's chunk window —
        # the in-distribution diagonal of the (encR, xR) inference grid.
        # Default False preserves existing training-time behavior.
        cross_attn_num_right_sync_with_encoder: bool = False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # NOTE (Shih-Lun): else case is for OpenAI Whisper ASR model,
        #                  which doesn't use <blank> token
        if sym_blank in token_list:
            self.blank_id = token_list.index(sym_blank)
        else:
            self.blank_id = 0
        if sym_sos in token_list:
            self.sos = token_list.index(sym_sos)
        else:
            self.sos = vocab_size - 1
        if sym_eos in token_list:
            self.eos = token_list.index(sym_eos)
        else:
            self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.aux_ctc = aux_ctc
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        self._encoder_accepts_chunked_mask = (
            encoder is not None
            and "chunked_mask_config"
            in inspect.signature(encoder.forward).parameters
        )

        self.autocast_frontend = autocast_frontend

        if self.encoder and getattr(self.encoder, "interctc_use_conditioning", False):
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.use_transducer_decoder = joint_network is not None
        self.use_linear_decoder = isinstance(decoder, LinearDecoder)

        self.error_calculator = None

        if self.use_transducer_decoder:
            self.decoder = decoder
            self.joint_network = joint_network

            if not transducer_multi_blank_durations:
                from warprnnt_pytorch import RNNTLoss

                if fastemit_lambda > 0.0:
                    logging.info(f"Transducer loss with fastemit_lambda={fastemit_lambda}")
                self.criterion_transducer = RNNTLoss(
                    blank=self.blank_id,
                    fastemit_lambda=fastemit_lambda,
                )
            else:
                from espnet2.asr.transducer.rnnt_multi_blank.rnnt_multi_blank import (
                    MultiblankRNNTLossNumba,
                )

                self.criterion_transducer = MultiblankRNNTLossNumba(
                    blank=self.blank_id,
                    big_blank_durations=transducer_multi_blank_durations,
                    sigma=transducer_multi_blank_sigma,
                    reduction="mean",
                    fastemit_lambda=fastemit_lambda,
                )
                self.transducer_multi_blank_durations = transducer_multi_blank_durations

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        elif self.use_linear_decoder:
            assert ctc_weight == 0.0, "CTC is not supported with LinearDecoder."
            self.decoder = decoder
            self.criterion_classif = torch.nn.CrossEntropyLoss(
                ignore_index=ignore_id, label_smoothing=lsm_weight
            )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight < 1.0:
                assert (
                    decoder is not None
                ), "decoder should not be None when attention is used"
            else:
                decoder = None
                logging.warning("Set decoder to none as ctc_weight==1.0")

            self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        if self.encoder is not None:
            self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__
        else:
            self.is_encoder_whisper = False

        if self.is_encoder_whisper:
            assert (
                self.frontend is None
            ), "frontend should be None when using full Whisper model"

        if lang_token_id != -1:
            self.lang_token_id = torch.tensor([[lang_token_id]])
        else:
            self.lang_token_id = None

        # Ablation study flags
        # Backward compat: if old 'use_chunked_training=False' was set, honour it
        # unless new 'dynamic_chunk_training' was explicitly set to False too.
        if not use_chunked_training and dynamic_chunk_training:
            dynamic_chunk_training = False
        self.dynamic_chunk_training = dynamic_chunk_training

        # Backward compat: old 'chunked_training_config' maps to new key
        self.dynamic_chunk_training_config = dynamic_chunk_training_config or chunked_training_config or {}

        # Fixed chunk config: used only when dynamic_chunk_training=False
        self.fixed_chunk_config = fixed_chunk_config  # Optional[dict] of ChunkedMaskConfig fields

        # Materialize the fixed config now so that inference-time encode()
        # (called directly by asr_inference.py, without forward()) applies the
        # same mask as training. forward() overwrites this per batch.
        if fixed_chunk_config is not None and not dynamic_chunk_training:
            self.chunked_mask_config = ChunkedMaskConfig(**fixed_chunk_config)
        else:
            self.chunked_mask_config = None

        self.use_chunked_cross_attn = use_chunked_cross_attn
        self.cross_attn_sync_with_encoder_chunk = cross_attn_sync_with_encoder_chunk
        self.use_alignment_loss = use_alignment_loss
        self.early_emit_weight = early_emit_weight
        self.cross_attn_num_right_train = cross_attn_num_right_train
        self.cross_attn_num_right_sync_with_encoder = cross_attn_num_right_sync_with_encoder

        chunk_training_active = self.dynamic_chunk_training or self.fixed_chunk_config is not None
        if not chunk_training_active and self.use_alignment_loss:
            logging.warning(
                "use_alignment_loss=True has no effect when no chunk config is active "
                "(alignment losses require chunk_size from chunked_mask_config)"
            )
        if not chunk_training_active and self.use_chunked_cross_attn:
            logging.warning(
                "use_chunked_cross_attn=True has no effect when no chunk config is active "
                "(chunked cross-attention requires chunked_mask_config)"
            )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        alignment: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            alignment: (Batch, Length) per-token encoder-frame emission
                index, -1 for padding; drives chunk-alignment masking and
                early-emission losses. None disables alignment-based terms.
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]
        if alignment is not None:
            alignment = alignment[:, : text_lengths.max()]

        # 0. For streaming training: set chunked_mask_config for this batch
        if self.dynamic_chunk_training:
            sampler_kwargs = dict(training=self.training)
            if self.dynamic_chunk_training_config:
                sampler_kwargs.update(self.dynamic_chunk_training_config)
            sampler = DynamicChunkConfigSampler(**sampler_kwargs)
            sampled = sampler()
            if (not sampled.full_attention) and self.fixed_chunk_config is not None:
                # Online batch: replace randomly-sampled config with fixed config
                self.chunked_mask_config = ChunkedMaskConfig(**self.fixed_chunk_config)
            else:
                # Offline batch (full_attention=True), or pure dynamic (no fixed_chunk_config)
                self.chunked_mask_config = sampled
        elif self.fixed_chunk_config is not None:
            # Static mode: always use the fixed config (no offline mixing)
            self.chunked_mask_config = ChunkedMaskConfig(**self.fixed_chunk_config)
        else:
            # Offline / full-attention training
            self.chunked_mask_config = None

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        loss_classif, acc_classif = None, None  # noqa
        stats = dict()

        # 1a. Handle alignment offset for global tokens
        num_global_tokens = 0
        if self.chunked_mask_config is not None and hasattr(self.encoder, 'attention_sink'):
            if self.encoder.attention_sink is not None:
                num_global_tokens = self.encoder.attention_sink.size(0)

        # Adjust alignment for global token offset
        alignment_adjusted = alignment
        if alignment is not None and num_global_tokens > 0:
            # Alignment is in encoder output frame space, but global tokens are prepended
            # Need to shift alignment indices by num_global_tokens
            alignment_adjusted = alignment.clone()
            valid_mask = alignment_adjusted != -1
            alignment_adjusted[valid_mask] = alignment_adjusted[valid_mask] + num_global_tokens
            if not getattr(self, "_warned_global_token_offset", False):
                self._warned_global_token_offset = True
                logging.warning(
                    f"Alignment indices adjusted by +{num_global_tokens} to account for global tokens. "
                    f"Ensure your alignment was generated WITHOUT global tokens in the encoder."
                )

        # 1b. Validate alignment consistency with encoder output
        if alignment_adjusted is not None:
            # Check that alignment values are within encoder output range
            valid_alignments = alignment_adjusted[alignment_adjusted != -1]
            if valid_alignments.numel() > 0:
                max_alignment = valid_alignments.max().item()
                max_encoder_len = encoder_out_lens.max().item()

                # Account for potential 4D encoder_out (with C axis)
                if encoder_out.dim() == 4:
                    # encoder_out: (B, T, C, D), so max frame index is T-1
                    max_encoder_len = encoder_out.size(1) - 1 + num_global_tokens
                else:
                    # encoder_out: (B, T, D)
                    max_encoder_len = max_encoder_len - 1

                if max_alignment > max_encoder_len:
                    overshoot = max_alignment - max_encoder_len
                    if overshoot <= 8:
                        # Small overshoots (up to 8 frames) from time→frame rounding
                        # or slight encoder-config drift; clamp instead of erroring out.
                        # 8 frames at chunk_size=16 is half a chunk — still safe for
                        # chunk-alignment computation (chunk_alignment // chunk_size).
                        logging.warning(
                            f"Alignment frame {max_alignment} exceeds encoder max {max_encoder_len} "
                            f"by {overshoot} (likely rounding/encoder-config drift). "
                            f"Clamping to {max_encoder_len}."
                        )
                        valid_mask = alignment_adjusted != -1
                        alignment_adjusted[valid_mask] = alignment_adjusted[valid_mask].clamp(
                            max=max_encoder_len
                        )
                    else:
                        raise ValueError(
                            f"Alignment validation failed: max alignment frame {max_alignment} "
                            f"exceeds max encoder output frame {max_encoder_len}. "
                            f"This likely means the alignment was generated with a different encoder "
                            f"configuration (different subsampling rate, different conv layers, etc.). "
                            f"Please regenerate alignment using the SAME model configuration as training."
                        )

        # 1c. Convert frame alignment to chunk alignment once
        # Compute chunk_alignment whenever alignment data is available and chunking is active.
        # This is used for both cross-attention masking (use_chunked_cross_attn) and
        # early emission losses (use_alignment_loss), which are independently controlled.
        chunk_alignment = None
        if (alignment_adjusted is not None
                and self.chunked_mask_config is not None):
            chunk_size = self.chunked_mask_config.chunk_size
            chunk_alignment = alignment_adjusted // chunk_size  # 0-based chunk numbers

        # 1. CTC branch
        loss_ctc_early_emit = None
        if self.ctc_weight != 0.0:
            ctc_chunk_alignment = chunk_alignment if self.use_alignment_loss else None
            loss_ctc, cer_ctc, loss_ctc_early_emit = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths, ctc_chunk_alignment
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc
            stats["loss_ctc_early_emit"] = loss_ctc_early_emit.detach() if loss_ctc_early_emit is not None else None

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out

                # use auxillary ctc data if specified
                loss_ic = None
                if self.aux_ctc is not None:
                    idx_key = str(layer_idx)
                    if idx_key in self.aux_ctc:
                        aux_data_key = self.aux_ctc[idx_key]
                        aux_data_tensor = kwargs.get(aux_data_key, None)
                        aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                        if aux_data_tensor is not None and aux_data_lengths is not None:
                            loss_ic, cer_ic, _ = self._calc_ctc_loss(
                                intermediate_out,
                                encoder_out_lens,
                                aux_data_tensor,
                                aux_data_lengths,
                            )
                        else:
                            raise Exception(
                                "Aux. CTC tasks were specified but no data was found"
                            )
                if loss_ic is None:
                    loss_ic, cer_ic, _ = self._calc_ctc_loss(
                        intermediate_out, encoder_out_lens, text, text_lengths
                    )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        elif self.use_linear_decoder:
            # 2b. Linear decoder branch for classification tasks
            loss, acc = self._calc_classif_loss(encoder_out, encoder_out_lens, text)
            stats["loss"] = loss
            stats["acc"] = acc
        else:
            # 2c. Attention decoder branch
            loss_early_emit = None
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att, loss_early_emit = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths, chunk_alignment
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # 4. Add early emission losses if available
            early_emit_weight = self.early_emit_weight
            # Add attention-based early emission loss
            if loss_early_emit is not None:
                loss = loss + early_emit_weight * loss_early_emit
            # Add CTC-based early emission loss
            if loss_ctc_early_emit is not None:
                loss = loss + early_emit_weight * loss_ctc_early_emit

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["loss_early_emit"] = loss_early_emit.detach() if loss_early_emit is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(self.autocast_frontend, dtype=autocast_type):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder is None:
            encoder_out, encoder_out_lens = feats, feats_lengths
        else:
            # Pass the chunked mask config to encoders that accept it
            if getattr(self.encoder, "interctc_use_conditioning", False) or getattr(
                self.encoder, "ctc_trim", False
            ):
                encoder_out, encoder_out_lens, _ = self.encoder(
                    feats, feats_lengths, ctc=self.ctc
                )
            elif self._encoder_accepts_chunked_mask:
                encoder_out, encoder_out_lens, _ = self.encoder(
                    feats, feats_lengths,
                    chunked_mask_config=getattr(self, 'chunked_mask_config', None),
                )
            else:
                encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if self.encoder is not None and (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            # encoder_out shape: (B, T, D) or (B, T, C, D) with chunked_mask_config
            # encoder_out_lens represents T, so check dimension 1 (time dimension)
            assert encoder_out.size(1) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        chunk_alignment: Optional[torch.Tensor] = None,
    ):
        """Compute attention decoder loss with optional chunked cross-attention.

        Compared to the offline model, this version supports a chunked
        cross-attention mask built from chunk_alignment, an attention-based
        early-emission loss, and 4D encoder output (B, T, C, D) which is
        stacked along the batch dimension per C-axis config.

        Args:
            encoder_out: Encoder output, (B, T, D) or (B, T, C, D).
            encoder_out_lens: Encoder output lengths, (B,).
            ys_pad: Padded target token ids, (B, Lmax).
            ys_pad_lens: Target lengths, (B,).
            chunk_alignment: Per-token emission chunk index, (B, Lmax),
                -1 for padding. None disables chunked masking and the
                early-emission loss.

        Returns:
            Tuple of (loss_att, acc_att, cer_att, wer_att, loss_early_emit).
        """
        # Handle 4D encoder_out by stacking C into batch dimension
        if encoder_out.dim() == 4:
            B, T, C, D = encoder_out.size()
            # (B, T, C, D) -> (B, C, T, D) -> (B*C, T, D)
            encoder_out = encoder_out.permute(0, 2, 1, 3).reshape(B * C, T, D)
            encoder_out_lens = encoder_out_lens.repeat_interleave(C)
            ys_pad = ys_pad.repeat_interleave(C, dim=0)
            ys_pad_lens = ys_pad_lens.repeat_interleave(C)
            # Also repeat chunk_alignment if provided
            if chunk_alignment is not None:
                chunk_alignment = chunk_alignment.repeat_interleave(C, dim=0)
        else:
            B, C = encoder_out.size(0), None

        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            ys_pad = torch.cat(
                [
                    self.lang_token_id.repeat(ys_pad.size(0), 1).to(ys_pad.device),
                    ys_pad,
                ],
                dim=1,
            )
            ys_pad_lens += 1

        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # Create causal cross-attention mask if chunk_alignment is provided and in streaming mode
        cross_attn_mask = None
        if (self.use_chunked_cross_attn and chunk_alignment is not None
                and self.chunked_mask_config is not None):
            # --- Left (past) context source ---
            if self.cross_attn_sync_with_encoder_chunk:
                # Tight to current chunk: attend only to current chunk's frames
                cross_attn_num_left = 0
            else:
                # Default: mirror the encoder's left context for this batch.
                cross_attn_num_left = self.chunked_mask_config.num_left_chunks
            # --- Right (future) context source ---
            # When cross_attn_num_right_sync_with_encoder is True, the cross-
            # attn future allowance follows the encoder's per-batch sampled
            # num_right_chunks (in-distribution diagonal: enc R = xR per batch).
            # Otherwise, the static cross_attn_num_right_train is used.
            if self.cross_attn_num_right_sync_with_encoder:
                cross_attn_num_right = self.chunked_mask_config.num_right_chunks
            else:
                cross_attn_num_right = self.cross_attn_num_right_train
            cross_attn_mask = self._create_causal_cross_attn_mask(
                chunk_alignment=chunk_alignment,  # Note: chunk_alignment doesn't include SOS
                encoder_out_lens=encoder_out_lens,
                src_len=encoder_out.size(1),
                tgt_len_with_sos=ys_in_pad.size(1),
                chunk_size=self.chunked_mask_config.chunk_size,
                num_right_chunks=cross_attn_num_right,
                num_left_chunks=cross_attn_num_left,
            )

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens,
            cross_attn_mask=cross_attn_mask,
        )

        # Get the last decoder layer's cross-attention weight
        # The cross-attention module is src_attn in each decoder layer
        # After forward pass, attention weights are stored in src_attn.attn
        last_decoder_layer = self.decoder.decoders[-1]
        cross_attn_weight = last_decoder_layer.src_attn.attn  # (batch, n_head, tgt_len, src_len)

        # Compute early emission loss if alignment loss is enabled
        loss_early_emit = None
        if (self.use_alignment_loss and chunk_alignment is not None
                and self.chunked_mask_config is not None):
            # Remove the first position (SOS token) from attention weights
            # cross_attn_weight: (batch, n_head, tgt_len, src_len) where tgt_len includes SOS
            # chunk_alignment: (batch, tgt_len) where tgt_len does NOT include SOS
            cross_attn_weight_no_sos = cross_attn_weight[:, :, 1:, :]
            loss_early_emit = self._calc_early_emission_loss(
                cross_attn_weight_no_sos, chunk_alignment, ys_pad_lens, encoder_out_lens
            )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            if C is not None:
                # Reshape back to (B, C, L) and average over C
                L = ys_hat.size(-1)
                ys_hat_per_c = ys_hat.reshape(B, C, L)
                # ys_pad is (B*C, L), get original ys_pad (B, L)
                ys_pad_orig = ys_pad[::C]
                cer_att, wer_att = 0.0, 0.0
                for c in range(C):
                    cer_c, wer_c = self.error_calculator(ys_hat_per_c[:, c, :].cpu(), ys_pad_orig.cpu())
                    cer_att += cer_c
                    wer_att += wer_c
                cer_att, wer_att = cer_att / C, wer_att / C
            else:
                cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_early_emit

    def _calc_early_emission_loss(
        self,
        cross_attn_weight: torch.Tensor,  # (batch, n_head, tgt_len, src_len)
        chunk_alignment: torch.Tensor,    # (batch, tgt_len) - chunk indices for each token
        ys_pad_lens: torch.Tensor,        # (batch,)
        encoder_out_lens: torch.Tensor,   # (batch,)
    ) -> torch.Tensor:
        """Compute early emission loss to encourage tokens to be emitted at or before their aligned chunk.

        This loss penalizes attention to frames that come after the target chunk (based on chunk_alignment).
        The goal is to encourage the model to attend to earlier frames, enabling lower-latency streaming.

        Args:
            cross_attn_weight: Cross-attention weights from decoder, shape (batch, n_head, tgt_len, src_len)
            chunk_alignment: Chunk index where each token should be emitted, shape (batch, tgt_len)
                             Values of -1 indicate padding tokens.
            ys_pad_lens: Length of each target sequence (excluding padding), shape (batch,)
            encoder_out_lens: Length of each encoder output sequence, shape (batch,)

        Returns:
            loss_early_emit: Scalar loss encouraging early emission
        """
        cfg = self.chunked_mask_config
        device = cross_attn_weight.device

        # Edge case 1: full-attention batch (sampler returned full_attention=True
        # with chunk_size=1, num_left=-1, num_right=-1). This batch has no
        # streaming constraint by design — penalizing here would push the
        # offline-trained attention pattern toward a tight per-frame diagonal,
        # conflicting with what the full-attention batch is supposed to teach.
        # Skip and return zero.
        if bool(getattr(cfg, "full_attention", False)):
            return torch.tensor(0.0, device=device)

        chunk_size = cfg.chunk_size

        B, n_head, tgt_len, src_len = cross_attn_weight.shape

        # Align tgt_len dimensions: cross_attn_weight may be longer than
        # chunk_alignment if text includes <eos> but alignment does not.
        align_len = chunk_alignment.size(1)
        if tgt_len > align_len:
            cross_attn_weight = cross_attn_weight[:, :, :align_len, :]
            tgt_len = align_len

        # Average attention weights across heads: (batch, tgt_len, src_len)
        attn_avg = cross_attn_weight.mean(dim=1)

        # --- Cross-attn window edges (mirror the hard mask's logic) ---
        # Past edge: L chunks before c[t]. If cross_attn_sync_with_encoder_chunk
        # is True, the hard mask uses L=0 (tight to current chunk). Otherwise the
        # hard mask mirrors the encoder's num_left_chunks (which may be -1 for
        # unlimited past — in that case we DO NOT apply the past-side penalty).
        if self.cross_attn_sync_with_encoder_chunk:
            nLc = torch.zeros(B, 1, device=device, dtype=torch.long)
        else:
            nLc = self._resolve_num_left_chunks(
                cfg.num_left_chunks, B, device
            )  # (B, 1); may be -1 (unlimited)
        # Future edge: R chunks after c[t]. Either the per-batch encoder R
        # (sync flag) or the static cross_attn_num_right_train. In streaming
        # batches the future is always finite (sampler never returns R=-1
        # for streaming configs).
        if self.cross_attn_num_right_sync_with_encoder:
            nRc = self._resolve_num_right_chunks(
                cfg.num_right_chunks, B, device
            )  # (B, 1)
        else:
            nRc = torch.full(
                (B, 1), int(self.cross_attn_num_right_train),
                device=device, dtype=torch.long,
            )

        # --- CHUNK-LEVEL aggregation ---
        # Aggregate attention from per-frame to per-chunk mass. The streaming
        # constraint is at chunk granularity (whether mass is in chunk C or
        # beyond), so the natural unit of the penalty is also chunks. This
        # collapses the big (B, tgt_len, T_enc) tensor down to
        # (B, tgt_len, num_chunks) — typically a 16× reduction at cs=16.
        num_chunks = (src_len + chunk_size - 1) // chunk_size
        pad_amount = num_chunks * chunk_size - src_len
        if pad_amount > 0:
            attn_padded = torch.nn.functional.pad(
                attn_avg, (0, pad_amount), value=0.0
            )
        else:
            attn_padded = attn_avg
        # (B, tgt_len, num_chunks*cs) → (B, tgt_len, num_chunks, cs) → sum
        attn_per_chunk = attn_padded.view(
            B, tgt_len, num_chunks, chunk_size
        ).sum(dim=-1)  # (B, tgt_len, num_chunks)

        # Window edges in CHUNK units:
        # - future_end_chunk = c[t] + nRc (inclusive); first chunk past = +1
        # - past_start_chunk = c[t] - nLc (inclusive); chunk c is "before" if c < this
        future_end_chunk = (chunk_alignment + nRc).float().unsqueeze(-1)
        past_start_chunk = (
            (chunk_alignment - nLc).clamp(min=0).float().unsqueeze(-1)
        )

        # Per-batch gate: only penalize past side when the row has a finite
        # left context. nLc = -1 (unlimited) disables past penalty for that row.
        apply_past = (nLc >= 0).view(B, 1, 1).float()  # (B, 1, 1)
        # Same gate on the future side: nRc = -1 (unlimited / full-attention
        # sentinel) disables the late penalty for that row. Normally shielded
        # by the full_attention early-return above; this keeps per-row
        # semantics correct if mixed batches are ever enabled.
        apply_future = (nRc >= 0).view(B, 1, 1).float()  # (B, 1, 1)

        # Chunk indices: (1, 1, num_chunks)
        chunk_idx = torch.arange(
            num_chunks, device=device
        ).view(1, 1, num_chunks).float()

        # Distances OUTSIDE the window, in CHUNK units (zero inside the window).
        # A chunk c at distance d past the future edge gets weight 1 + d^2.
        late_distance = torch.clamp(
            chunk_idx - future_end_chunk, min=0
        )  # (B, tgt_len, num_chunks)
        early_distance = torch.clamp(
            past_start_chunk - chunk_idx, min=0
        )

        # Quadratic-curve weights (distance already in chunk units, no /cs
        # normalization needed). The "+1" base keeps a unit penalty for the
        # first chunk past the edge; squared term ramps up for further chunks.
        late_mask = (late_distance > 0).float()
        early_mask = (early_distance > 0).float()
        late_weight = late_mask * (1.0 + late_distance ** 2)
        early_weight = early_mask * (1.0 + early_distance ** 2)

        # Combine: penalize per-chunk attention mass outside the window. Past
        # side is gated by apply_past so rows with unlimited left context
        # (nLc = -1) only get the future-side penalty.
        weighted_outside_per_chunk = attn_per_chunk * (
            apply_future * late_weight + apply_past * early_weight
        )

        # Sum over chunks to get total outside-window mass per token,
        # weighted by squared chunk-distance from the nearest window edge.
        outside_attn_per_token = weighted_outside_per_chunk.sum(dim=-1)  # (B, tgt_len)

        # Mask out padding tokens (chunk_alignment == -1).
        valid_mask = (chunk_alignment != -1).float()
        outside_attn_per_token = outside_attn_per_token * valid_mask

        # Normalize by number of valid tokens.
        num_valid = valid_mask.sum()
        if num_valid > 0:
            loss_early_emit = outside_attn_per_token.sum() / num_valid
        else:
            loss_early_emit = torch.tensor(0.0, device=device)

        return loss_early_emit

    @staticmethod
    def _resolve_num_right_chunks(num_right_chunks, B, device):
        """Normalize num_right_chunks to a broadcastable (B, 1) tensor.

        Handles int, list, or tensor input. -1 (the full-attention sentinel)
        means UNLIMITED future and is passed through, mirroring
        _resolve_num_left_chunks. Consumers must gate on (nrc >= 0) and apply
        no right-edge restriction/penalty to -1 rows — never use -1
        arithmetically (the old clamp to 0 turned "unlimited" into "zero
        future", which at chunk_size=1 in full-attention batches collapsed
        each token's cross-attn window to its own onset frame and broke the
        HARD training runs).
        """
        if isinstance(num_right_chunks, (list, tuple)):
            nrc = torch.tensor(num_right_chunks, device=device, dtype=torch.long)
        elif torch.is_tensor(num_right_chunks):
            nrc = num_right_chunks.to(device=device, dtype=torch.long)
        else:
            nrc = torch.tensor([num_right_chunks], device=device, dtype=torch.long)
        if nrc.numel() == 1:
            nrc = nrc.expand(B)
        return nrc.view(B, 1)  # (B, 1) for broadcast

    @staticmethod
    def _resolve_num_left_chunks(num_left_chunks, B, device):
        """Normalize num_left_chunks to a broadcastable (B, 1) tensor.

        -1 means unlimited (no left restriction). Values >= 0 limit left window.
        """
        if isinstance(num_left_chunks, (list, tuple)):
            nlc = torch.tensor(num_left_chunks, device=device, dtype=torch.long)
        elif torch.is_tensor(num_left_chunks):
            nlc = num_left_chunks.to(device=device, dtype=torch.long)
        else:
            nlc = torch.tensor([num_left_chunks], device=device, dtype=torch.long)
        if nlc.numel() == 1:
            nlc = nlc.expand(B)
        return nlc.view(B, 1)  # (B, 1) for broadcast

    def _create_causal_cross_attn_mask(
        self,
        chunk_alignment: torch.Tensor,  # (B, tgt_len) - chunk index for each token (no SOS)
        encoder_out_lens: torch.Tensor, # (B,)
        src_len: int,                   # max encoder output length
        tgt_len_with_sos: int,          # target length including SOS
        chunk_size: int,                # chunk size for boundary calculation
        num_right_chunks=0,             # int, list, or tensor — future chunks visible
        num_left_chunks=-1,             # int, list, or tensor — past chunks visible (-1 = unlimited)
    ) -> torch.Tensor:
        """Create sliding-window cross-attention mask from chunk alignment.

        For each token at position t with chunk index c[t], the mask allows
        attention to encoder frames in the window [chunk_start, chunk_end]:
            chunk_start = max(0, c[t] - num_left_chunks) * chunk_size
            chunk_end   = (c[t] + 1 + num_right_chunks) * chunk_size - 1

        When num_left_chunks = -1 (unlimited), the left boundary is 0 (prefix mask).

        Args:
            chunk_alignment: Chunk index for each token, shape (B, tgt_len) where tgt_len excludes SOS.
                             Values of -1 indicate padding.
            encoder_out_lens: Length of each encoder output, shape (B,).
            src_len: Maximum encoder output length.
            tgt_len_with_sos: Target sequence length including SOS token.
            chunk_size: Chunk size for computing chunk boundaries.
            num_right_chunks: Number of future chunks visible (-1 = unlimited, no right restriction).
            num_left_chunks: Number of past chunks visible (-1 = unlimited, no left restriction).

        Returns:
            mask: (B, tgt_len_with_sos, src_len) where True = valid, False = masked
        """
        B = chunk_alignment.size(0)
        device = chunk_alignment.device

        # Normalize to (B, 1) tensors
        nrc = self._resolve_num_right_chunks(num_right_chunks, B, device)
        nlc = self._resolve_num_left_chunks(num_left_chunks, B, device)

        # Clip or pad chunk_alignment to match expected tgt_len (without SOS)
        expected_tgt_len = tgt_len_with_sos - 1
        if chunk_alignment.size(1) > expected_tgt_len:
            chunk_alignment = chunk_alignment[:, :expected_tgt_len]
        elif chunk_alignment.size(1) < expected_tgt_len:
            pad_len = expected_tgt_len - chunk_alignment.size(1)
            chunk_alignment = torch.nn.functional.pad(
                chunk_alignment, (0, pad_len), value=-1
            )

        # Create frame indices: (1, 1, src_len)
        frame_indices = torch.arange(src_len, device=device).view(1, 1, src_len)

        # --- Right boundary ---
        # Compute chunk end for each token: (chunk_num + 1 + nrc) * chunk_size - 1
        chunk_end = (chunk_alignment + 1 + nrc) * chunk_size - 1  # (B, tgt_len)

        # Handle SOS token: can attend to first (1 + nrc) chunks
        sos_chunk_end = (1 + nrc) * chunk_size - 1  # (B, 1)

        full_chunk_end = torch.cat([sos_chunk_end, chunk_end], dim=1)  # (B, tgt_len_with_sos)
        full_chunk_end = full_chunk_end.unsqueeze(-1)  # (B, tgt_len_with_sos, 1)

        # Right mask: frame_indices <= chunk_end. Mirroring apply_left below,
        # nrc = -1 means UNLIMITED future (full-attention sentinel): those
        # rows get no right-edge restriction at all. Using -1 arithmetically
        # would instead SHRINK the window by one chunk.
        apply_right = (nrc >= 0).view(B, 1, 1)
        mask = (frame_indices <= full_chunk_end) | ~apply_right  # (B, tgt_len_with_sos, src_len)

        # --- Left boundary (sliding window) ---
        # Only applies when num_left_chunks >= 0; -1 = unlimited (no left restriction).
        if (nlc >= 0).any():
            chunk_start = (chunk_alignment - nlc) * chunk_size  # (B, tgt_len)
            chunk_start = chunk_start.clamp(min=0)

            # SOS: left boundary naturally clamps to 0
            sos_chunk_start = torch.zeros(B, 1, device=device, dtype=torch.long)
            full_chunk_start = torch.cat([sos_chunk_start, chunk_start], dim=1).unsqueeze(-1)

            left_mask = frame_indices >= full_chunk_start  # (B, tgt_len_with_sos, src_len)

            # Per-batch: only restrict items where nlc >= 0
            apply_left = (nlc >= 0).view(B, 1, 1)
            mask = mask & (~apply_left | left_mask)

        # Apply encoder padding mask: frame_indices < encoder_out_lens
        padding_mask = frame_indices < encoder_out_lens.view(B, 1, 1)  # (B, 1, src_len)
        mask = mask & padding_mask

        return mask

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        chunk_alignment: Optional[torch.Tensor] = None,
    ):
        """Compute CTC loss with optional chunk-alignment plumbing.

        Compared to the offline model, this version passes chunk_alignment
        and chunk_size to the CTC module, computes a CTC early-emission
        loss, and supports 4D encoder output (B, T, C, D) by stacking the
        C axis into the batch dimension.

        Args:
            encoder_out: Encoder output, (B, T, D) or (B, T, C, D).
            encoder_out_lens: Encoder output lengths, (B,).
            ys_pad: Padded target token ids, (B, Lmax).
            ys_pad_lens: Target lengths, (B,).
            chunk_alignment: Per-token emission chunk index, (B, Lmax),
                -1 for padding. None disables the early-emission loss.

        Returns:
            Tuple of (loss_ctc, cer_ctc, loss_ctc_early_emit).
        """
        # Extract chunk_size for CTC soft masking
        ctc_chunk_size = self.chunked_mask_config.chunk_size if self.chunked_mask_config else None

        # Calc CTC loss
        if encoder_out.dim() == 3:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
                                chunk_alignment, chunk_size=ctc_chunk_size)
        else:
            # Parallelize over C axis by stacking into batch dimension
            B, T, C, D = encoder_out.size()
            # (B, T, C, D) -> (B, C, T, D) -> (B*C, T, D)
            encoder_out_stacked = encoder_out.permute(0, 2, 1, 3).reshape(B * C, T, D)
            # Repeat encoder_out_lens, ys_pad, ys_pad_lens C times
            encoder_out_lens_stacked = encoder_out_lens.repeat_interleave(C)
            ys_pad_stacked = ys_pad.repeat_interleave(C, dim=0)
            ys_pad_lens_stacked = ys_pad_lens.repeat_interleave(C)
            chunk_alignment_stacked = chunk_alignment.repeat_interleave(C, dim=0) if chunk_alignment is not None else None
            # Compute CTC loss in one forward pass
            loss_ctc = self.ctc(
                encoder_out_stacked, encoder_out_lens_stacked,
                ys_pad_stacked, ys_pad_lens_stacked, chunk_alignment_stacked,
                chunk_size=ctc_chunk_size
            )

        # Compute CTC early emission loss if alignment is provided
        loss_ctc_early_emit = None
        if chunk_alignment is not None and self.chunked_mask_config is not None:
            if encoder_out.dim() == 3:
                loss_ctc_early_emit = self._calc_ctc_early_emission_loss(
                    encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, chunk_alignment
                )
            else:
                # Use stacked version for 4D encoder_out
                loss_ctc_early_emit = self._calc_ctc_early_emission_loss(
                    encoder_out_stacked, encoder_out_lens_stacked,
                    ys_pad_stacked, ys_pad_lens_stacked, chunk_alignment_stacked
                )

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            if encoder_out.dim() == 3:
                ys_hat = self.ctc.argmax(encoder_out).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            else:
                # Parallelize CER computation over C axis
                B, T, C, D = encoder_out.size()
                # (B, T, C, D) -> (B*C, T, D)
                encoder_out_stacked = encoder_out.permute(0, 2, 1, 3).reshape(B * C, T, D)
                ys_hat_stacked = self.ctc.argmax(encoder_out_stacked).data  # (B*C, T)
                # Reshape back to (B, C, T) and average CER over C
                ys_hat_per_c = ys_hat_stacked.reshape(B, C, T)
                cer_ctc = 0.0
                for c in range(C):
                    cer_ctc += self.error_calculator(ys_hat_per_c[:, c, :].cpu(), ys_pad.cpu(), is_ctc=True)
                cer_ctc = cer_ctc / C
        return loss_ctc, cer_ctc, loss_ctc_early_emit

    def _calc_ctc_early_emission_loss(
        self,
        encoder_out: torch.Tensor,      # (batch, src_len, dim)
        encoder_out_lens: torch.Tensor, # (batch,)
        ys_pad: torch.Tensor,           # (batch, tgt_len) - token ids
        ys_pad_lens: torch.Tensor,      # (batch,)
        chunk_alignment: torch.Tensor,  # (batch, tgt_len) - chunk indices for each token
    ) -> torch.Tensor:
        """Compute CTC early emission loss to encourage tokens to be emitted at or before their aligned chunk.

        This loss uses CTC output probabilities to penalize late emissions. For each token in the target,
        we sum up the probability of that token at frames after its chunk deadline, weighted
        by how late the frame is.

        Args:
            encoder_out: Encoder output, shape (batch, src_len, dim)
            encoder_out_lens: Length of each encoder output sequence, shape (batch,)
            ys_pad: Padded target token ids, shape (batch, tgt_len)
            ys_pad_lens: Length of each target sequence, shape (batch,)
            chunk_alignment: Chunk index where each token should be emitted, shape (batch, tgt_len)
                             Values of -1 indicate padding tokens.

        Returns:
            loss_ctc_early_emit: Scalar loss encouraging early emission in CTC
        """
        cfg = self.chunked_mask_config

        # Full-attention batches carry no streaming constraint, so nothing is
        # "late" — penalizing here would fight normal CTC spike timing in
        # exactly the batches meant to preserve offline behavior (with the
        # sentinel values chunk_size=1 / nrc=-1 the deadline would collapse to
        # each token's onset frame). Mirror _calc_early_emission_loss's skip.
        if bool(getattr(cfg, "full_attention", False)):
            return torch.tensor(0.0, device=encoder_out.device)

        chunk_size = cfg.chunk_size

        B = encoder_out.size(0)
        device = encoder_out.device

        # Get CTC softmax probabilities: (batch, src_len, vocab_size)
        ctc_probs = self.ctc.softmax(encoder_out)  # (B, T, V)

        # Get target frame deadline for each token from chunk_alignment.
        # Deadline = end of the chunk containing this token + future context.
        # With num_right_chunks > 0, the encoder output incorporates future info,
        # so the deadline shifts forward by num_right_chunks chunks.
        nrc = self._resolve_num_right_chunks(
            self.chunked_mask_config.num_right_chunks, B, device
        )  # (B, 1)
        # Rows with nrc = -1 (unlimited future sentinel) have no deadline and
        # get no late penalty. Normally shielded by the full_attention skip
        # above; kept for per-row correctness under mixed batches.
        finite_future = (nrc >= 0).view(B)  # (B,)

        # ---- Vectorized late-probability computation ----
        # Same math as the previous per-token Python loop, in a handful of
        # batched ops. The loop forced ~3 GPU->CPU syncs per token (~2400 per
        # optimizer step at effective batch 64), making this loss latency-
        # bound: SOFT steps ran ~2.5x slower than HARD on every GPU class.
        U = min(ys_pad.size(1), chunk_alignment.size(1))
        tok = ys_pad[:, :U].clamp(min=0)              # (B, U); pad ids may be -1
        align = chunk_alignment[:, :U]
        deadline = ((align + 1 + nrc) * chunk_size).float()  # (B, U)

        # Valid tokens: within target length, aligned, finite future budget.
        # Tokens whose deadline lies beyond the encoder length stay valid
        # (counted in the normalizer) with a naturally-zero contribution,
        # matching the loop's accounting.
        u_idx = torch.arange(U, device=device).view(1, U)
        valid = (
            (u_idx < ys_pad_lens.view(B, 1).clamp(max=U))
            & (align != -1)
            & finite_future.view(B, 1)
        )                                              # (B, U)

        T = ctc_probs.size(1)
        t_idx = torch.arange(T, device=device).view(1, T, 1)
        # Per-token probability tracks: (B, T, U)
        probs_tok = ctc_probs.gather(2, tok.unsqueeze(1).expand(B, T, U))
        d = deadline.unsqueeze(1)                      # (B, 1, U)
        # Weight: prob * (1 + lateness/chunk_size) on frames past the deadline
        late_w = (t_idx > d).float() * (
            1.0 + (t_idx.float() - d).clamp(min=0) / chunk_size
        )
        frame_ok = (t_idx < encoder_out_lens.view(B, 1, 1)).float()
        contrib = (probs_tok * late_w * frame_ok).sum(dim=1)  # (B, U)

        num_valid_tokens = int(valid.sum())
        if num_valid_tokens > 0:
            loss_ctc_early_emit = (contrib * valid.float()).sum() / num_valid_tokens
        else:
            loss_ctc_early_emit = torch.tensor(0.0, device=device)

        return loss_ctc_early_emit

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc) or (B, T, C, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        # Handle 4D encoder_out (with C axis) by stacking C into the batch.
        if encoder_out.dim() == 4:
            B, T, C, D = encoder_out.size()
            encoder_out = encoder_out.permute(0, 2, 1, 3).reshape(B * C, T, D)
            encoder_out_lens = encoder_out_lens.repeat_interleave(C)
            labels = labels.repeat_interleave(C, dim=0)

        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        # warprnnt_pytorch's CUDA kernel only accepts float32 logits; cast and
        # disable autocast for the loss to avoid the "unsupported data type" error
        # when training with AMP.
        with torch.amp.autocast("cuda", enabled=False):
            loss_transducer = self.criterion_transducer(
                joint_out.float(),
                target,
                t_len,
                u_len,
            )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer

    def _calc_batch_ctc_loss(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        if self.ctc is None:
            return
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # Calc CTC loss
        do_reduce = self.ctc.reduce
        self.ctc.reduce = False
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        self.ctc.reduce = do_reduce
        return loss_ctc

    def _calc_classif_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute classification loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, 1)
        Return:
            loss_classif: Classification loss value.
            acc_classif: Classification accuracy.
        """
        # Calc classification loss
        assert labels.dim() == 2, labels.shape
        assert labels.shape[1] == 1, labels.shape
        logits = self.decoder(encoder_out, encoder_out_lens)  # (B, n_class)
        assert logits.shape[1] == self.vocab_size - 3, logits.shape
        # Shift up labels to remove blank and unk.
        labels = labels - 2
        loss_classif = self.criterion_classif(logits, labels.squeeze(-1))
        acc_classif = th_accuracy(logits, labels, ignore_label=self.ignore_id)
        return loss_classif, acc_classif
