#!/usr/bin/env python3
"""Chunked streaming ASR inference with dynamic future chunks.

Entry point for streaming decoding with chunked attention masks:
Speech2TextStreamingChunked runs the encoder chunk by chunk (optionally
with an asymmetric C-axis mask and incremental KV caches), delays
decoding until the configured future context is available, and supports
dynamic future-chunks deferral (geometric, confidence, SR-CEM, or
learned wait-policy triggers), commit-stable-prefix decoding, and a
final-chunk offline re-decode. The inference() driver simulates chunked
audio feeding and writes n-best and latency outputs.
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from typeguard import typechecked

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.batch_beam_search_online import BatchBeamSearchOnline
from espnet.nets.beam_search import Hypothesis
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.pytorch_backend.nets_utils import ChunkedMaskConfig
from espnet.utils.cli_utils import get_commandline_args
from espnet2.asr_stream.chunk_tracker import ChunkTracker


class Speech2TextStreamingChunked:
    """Streaming ASR with chunked attention and delayed decoding.

    This class implements true chunk-by-chunk processing with:
    1. Configurable past/future context (num_left_chunks, num_right_chunks)
    2. Delayed decoding: waits for future context before decoding
    3. Latency tracking for each emitted token

    Unlike Speech2TextStreaming (asr_inference_streaming.py), which uses
    ContextualBlockEncoder, this class works with standard ConformerEncoder
    using ChunkedMaskConfig.

    Examples:
        >>> speech2text = Speech2TextStreamingChunked(
        ...     "asr_config.yml", "asr.pth",
        ...     chunk_size=16, num_left_chunks=4, num_right_chunks=2
        ... )
        >>> for chunk in audio_chunks:
        ...     results = speech2text(chunk, is_final=False)
        >>> final_results = speech2text(last_chunk, is_final=True)
    """

    @typechecked
    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str, None] = None,
        lm_train_config: Union[Path, str, None] = None,
        lm_file: Union[Path, str, None] = None,
        token_type: Optional[str] = None,
        bpemodel: Optional[str] = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        penalty: float = 0.0,
        nbest: int = 1,
        normalize_length: bool = False,
        # Chunked streaming parameters
        chunk_size: Optional[int] = None,
        num_left_chunks: int = -1,
        num_right_chunks: int = 0,
        cross_attn_num_left_chunks: Optional[int] = None,
        cross_attn_num_right_chunks: int = 0,
        encoder_num_right_chunks_override: Optional[int] = None,
        num_global_tokens: int = 0,
        # KV cache options
        use_decoder_self_kvcache: bool = True,
        use_decoder_cross_kvcache: bool = True,
        # Stage B: proper incremental encoder self-attn KV cache (per-layer, chunk-wise)
        use_encoder_self_kvcache: bool = False,
        # Enable asymmetric C-axis mask at inference (C = num_right_chunks + 1).
        # When True and num_right_chunks > 0, encoder produces 4D output and we
        # extract the c_q = num_right_chunks slice (max right-context view).
        # When False, encoder uses symmetric mask with availability bound (C=1).
        use_asymmetric_mask_at_inference: bool = False,
        # Softmax-confidence-based hallucination suppression policies
        chunk_end_confidence_threshold: float = 0.0,
        chunk_start_confidence_threshold: float = 0.0,
        rollback_confidence_threshold: float = 0.0,
        # Alternative softmax-distribution stop signals (drop-in alternatives to P1).
        chunk_end_entropy_threshold: float = 0.0,
        chunk_end_margin_threshold: float = 0.0,
        chunk_end_topk_mass_threshold: float = 0.0,
        # Diagnostic: undo the trained "causal at every c_q" DCConv constraint
        # at inference. Train/test mismatch - for ablation only.
        force_dcconv_right_context_at_inference: bool = False,
        # Diagnostic: which c_q slot of the asymmetric encoder output to use
        # as the decoder-visible representation. None = c_q=num_right_chunks
        # (the default max-right-context view). Set to 0 to extract the
        # strictly-causal-self-attn slot while keeping nrc>=1 buffering - used
        # to decouple self-attn chunk-boundary from DCConv future access
        # when paired with force_dcconv_right_context_at_inference=True.
        extract_cq_slot: Optional[int] = None,
        # Reading-C strict-streaming slot taper. Requires
        # use_asymmetric_mask_at_inference=True. When True, the per-chunk c_q
        # slot is set so each chunk uses only the lookahead audio it can afford
        # under the current decoder budget:
        #   c_q(i) = min(num_right_chunks, num_chunks - 1 - i)
        # Most chunks (i <= c) get c_q = num_right_chunks; the rightmost
        # num_right_chunks chunks taper linearly down to c_q = 0 (causal).
        # Overrides extract_cq_slot when True. Default False.
        cross_attn_slot_taper: bool = False,
        # Monotonic cross-attention floor: prevent decoder cross-attention
        # from drifting backward into already-covered audio frames.
        # Targets AED word-level duplications. -1 = disabled.
        monotonic_attn_floor: int = -1,
        # Diagnostic: encoder runs chunk-by-chunk (streaming), but the beam
        # search is held until is_final=True and then runs once on the
        # concatenated encoder output with no per-token cross-attn mask.
        # Isolates beam-loop streaming behaviour from encoder representation.
        defer_decoding_until_last_chunk: bool = False,
        # Diagnostic: directory to dump per-utterance decoder cross-attention
        # distributions (last layer, beam-0, last query, mean over heads).
        # Saved as one .npz per utterance.
        dump_xattn_dir: Optional[str] = None,
        # Attention-edge stop rule: stop the chunk and wait for more encoder
        # context when the top beam's cross-attn argmax lands at the rightmost
        # frames of the available encoder. Value is the number of "danger"
        # frames at the right edge. 0 disables. Active only on non-final calls.
        attn_edge_stop_margin: int = 0,
        # F4 fix: at the final chunk, boost EOS log-probability by this amount
        # so the model is more willing to terminate cleanly instead of
        # emitting plausible-but-unsupported tail content.
        eos_bonus_at_final: float = 0.0,
        # F4 fix: at is_final, disable the per-token cross-attn mask so all
        # prefix rows get full encoder attention (like defer mode). Targets
        # streaming-specific tail substitutions caused by restricted prefix
        # representations.
        unmask_xattn_at_final: bool = False,
        # F3 fix: penalize subword-piece repetition at chunk boundaries
        # (e.g. ``▁CONTAIN`` + ``IN`` + ``ING`` → "CONTAININING"). When > 0,
        # at the first emission of each new chunk, candidate tokens whose
        # text is a >=2-char suffix of the prefix's last piece are penalized
        # by this many nats. 0 disables (default).
        subword_overlap_penalty: float = 0.0,
        # Resume-from-prefix: re-decode only the final chunk with the offline
        # beam, keeping the committed pre-final-chunk tokens as a fixed prefix.
        resume_offline_at_final: bool = False,
        # Branch-D commit-stable-prefix: on each deferred re-decode pass, freeze
        # the longest prefix agreeing with the previous pass and re-decode only
        # the divergent suffix deeper (lower latency; rare frozen-flip WER cost).
        commit_stable_prefix: bool = False,
        # Dynamic future-chunks: enable to dynamically defer per-chunk
        # release for more future context when a confidence signal trips.
        # See dynamic_future_chunks.DynamicFutureChunksController. When
        # disabled (default), behaviour is unchanged. Three branches:
        #   - pre_emptive: signal evaluated on CTC log-probs before beam search
        #   - chunk_rollback: full chunk decoded, then maybe rolled back
        #   - token_resume: per-token mid-decode stop and resume
        dynamic_future_chunks: bool = False,
        max_future_chunks: int = 1,
        dynamic_mode: str = "pre_emptive",
        dynamic_signal_type: str = "top1_prob",
        dynamic_top1_prob_threshold: float = 0.6,
        dynamic_entropy_threshold: float = 1.5,
        dynamic_margin_threshold: float = 0.3,
        dynamic_topk_mass_threshold: float = 0.8,
        dynamic_fake_random_prob: float = 0.5,
        dynamic_fake_random_seed: int = 1234,
        # SR-CEM (Score-Rank Confidence Estimation Module) calibrator
        # for the dynamic-future-chunks trigger. See
        # espnet2/asr_stream/sr_cem.py. The calibrator outputs a
        # p_correct in (0,1) that is used ONLY to drive the
        # defer/commit decision; it does NOT replace the raw softmax
        # confidence in the beam search.
        sr_cem_ckpt: Optional[str] = None,
        sr_cem_threshold: Optional[float] = None,
        sr_cem_variant: str = "A",  # 'A' (causal, Branch A) or 'B' (chunk, Branch B)
        sr_cem_chunk_agg: Optional[str] = None,  # 'mean'/'min'; None -> checkpoint's
        sr_cem_feat_dump_dir: Optional[str] = None,
        # Temperature scaling for DFC trigger signals (Guo et al. 2017).
        # T=1.0 is a no-op. T>1.0 flattens, T<1.0 sharpens. Applies only
        # to the trigger signal (top1_prob/entropy/margin/topk_mass);
        # does NOT modify the beam search ranking or final decoded
        # hypothesis. Argmax is preserved at any T.
        dynamic_trigger_temperature: float = 1.0,
        # Long-form virtual finals: when > 0, the sim loop flushes the
        # current segment with a real is_final pass whenever the CTC
        # blank run reaches this many frames (sentence-gap silence) and
        # resets all decode state. Decoder then only ever operates on
        # trained-regime sequence lengths; encoder/CTC are unaffected.
        virtual_final_blank_frames: int = 0,
        virtual_final_min_chunks: int = 8,
        mask_streaming_pad: bool = False,
    ):
        """Initialize Speech2TextStreamingChunked.

        Args:
            asr_train_config: Path to ASR training config.
            asr_model_file: Path to ASR model checkpoint.
            lm_train_config: Path to LM training config (optional).
            lm_file: Path to LM checkpoint (optional).
            token_type: Token type (char/bpe/None).
            bpemodel: BPE model path.
            device: Device (cpu/cuda).
            maxlenratio: Max output length ratio.
            minlenratio: Min output length ratio.
            batch_size: Batch size (must be 1 for streaming).
            dtype: Data type (float16/float32/float64).
            beam_size: Beam size for beam search.
            ctc_weight: CTC weight in joint decoding.
            lm_weight: LM weight.
            penalty: Length penalty.
            nbest: Number of hypotheses to return.
            normalize_length: Normalize scores by length.
            chunk_size: Encoder frames per chunk (None/0 for offline).
            num_left_chunks: Past chunks visible (-1 = unlimited).
            num_right_chunks: Future chunks visible (determines delay).
            num_global_tokens: Number of attention sink tokens.
            use_decoder_self_kvcache: Cache decoder KV for autoregressive decoding.
            use_decoder_cross_kvcache: Cache encoder projections for cross-attention.
            use_encoder_self_kvcache: Stage B incremental per-layer
                encoder self-attn KV cache (chunk-wise append + asymmetric
                C-axis + conv state cache).

        Note:
            The remaining streaming/trigger arguments (asymmetric mask,
            cross_attn_slot_taper, dynamic_future_chunks family, sr_cem_*,
            commit_stable_prefix, resume_offline_at_final,
            mask_streaming_pad, ...) are documented inline in the signature
            above and in the get_parser() help strings.
        """
        # 1. Build ASR model
        logging.info(f"Loading ASR model from {asr_train_config}")
        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()
        logging.info(f"Encoder type: {type(asr_model.encoder).__name__}")
        logging.info(f"Decoder type: {type(asr_model.decoder).__name__}")

        # 2. Setup decoder
        decoder = asr_model.decoder
        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 3. Build Language model
        if lm_train_config is not None:
            logging.info(f"Loading LM from {lm_train_config}")
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 4. Build BeamSearch object
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            length_bonus=penalty,
        )

        assert batch_size == 1, "Streaming inference only supports batch_size=1"

        beam_search = BatchBeamSearchOnline(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
            normalize_length=normalize_length,
            block_size=0,
            disable_repetition_detection=False,
            cross_attn_chunk_size=chunk_size if chunk_size is not None and chunk_size > 0 else 0,
            cross_attn_num_left_chunks=cross_attn_num_left_chunks if cross_attn_num_left_chunks is not None else num_left_chunks,
            cross_attn_num_right_chunks=cross_attn_num_right_chunks,
            chunk_end_confidence_threshold=chunk_end_confidence_threshold,
            chunk_start_confidence_threshold=chunk_start_confidence_threshold,
            rollback_confidence_threshold=rollback_confidence_threshold,
            chunk_end_entropy_threshold=chunk_end_entropy_threshold,
            chunk_end_margin_threshold=chunk_end_margin_threshold,
            chunk_end_topk_mass_threshold=chunk_end_topk_mass_threshold,
            monotonic_attn_floor=monotonic_attn_floor,
        )
        if monotonic_attn_floor >= 0:
            logging.info(
                f"Monotonic cross-attn floor enabled: {monotonic_attn_floor} frames"
            )
        if chunk_end_confidence_threshold > 0.0:
            logging.info(
                f"Policy 1 (end-of-chunk early stop) enabled: "
                f"chunk_end_confidence_threshold={chunk_end_confidence_threshold:.3f}"
            )
        if chunk_start_confidence_threshold > 0.0:
            logging.info(
                f"Policy 2 (start-of-chunk mask+retry) enabled: "
                f"chunk_start_confidence_threshold={chunk_start_confidence_threshold:.3f}"
            )
        if rollback_confidence_threshold > 0.0:
            logging.info(
                f"Policy 3 (rollback + agreement confirm) enabled: "
                f"rollback_confidence_threshold={rollback_confidence_threshold:.3f}"
            )
        if chunk_end_entropy_threshold > 0.0:
            logging.info(
                f"Entropy stop enabled: chunk_end_entropy_threshold="
                f"{chunk_end_entropy_threshold:.3f} (nats)"
            )
        if chunk_end_margin_threshold > 0.0:
            logging.info(
                f"Margin stop enabled: chunk_end_margin_threshold="
                f"{chunk_end_margin_threshold:.3f}"
            )
        if chunk_end_topk_mass_threshold > 0.0:
            logging.info(
                f"Top-k mass stop enabled: chunk_end_topk_mass_threshold="
                f"{chunk_end_topk_mass_threshold:.3f}"
            )
        if chunk_end_confidence_threshold > 0.0 and rollback_confidence_threshold > 0.0:
            logging.warning(
                "Policy 1 and Policy 3 are both enabled - these are redundant. "
                "Policy 1 would intercept low-confidence tokens before Policy 3 "
                "can roll them back. Consider disabling one."
            )

        non_batch = [
            k
            for k, v in beam_search.full_scorers.items()
            if not isinstance(v, BatchScorerInterface)
        ]
        assert len(non_batch) == 0, f"Non-batch scorers: {non_batch}"

        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()

        # 5. Build Text converter
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)

        # 6. Store attributes
        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        # Trigger-only temperature scaling (Guo et al. 2017). T=1.0 is a
        # no-op; with T != 1.0, the trigger signals built inside
        # batch_beam_search are computed from softmax(z/T) while beam
        # ranking + final hypothesis use the unmodified softmax(z).
        self.beam_search.trigger_temperature = float(
            dynamic_trigger_temperature
        )
        # Enable xattn dump on the beam_search if a dir was requested.
        # SDPA/flash_attn paths don't populate src_attn.attn, so force the
        # last decoder layer's cross-attn through the eager path.
        # The same eager-path requirement applies for attention-edge stop,
        # since it needs self.attn_center which is read from src_attn.attn.
        _need_eager_xattn = (
            dump_xattn_dir is not None or attn_edge_stop_margin > 0
        )
        if dump_xattn_dir is not None:
            self.beam_search.xattn_dump_enabled = True
        if attn_edge_stop_margin > 0:
            self.beam_search.attn_edge_stop_margin = int(attn_edge_stop_margin)
        if eos_bonus_at_final != 0.0:
            self.beam_search.eos_bonus_at_final = float(eos_bonus_at_final)
        if subword_overlap_penalty > 0.0:
            self.beam_search.subword_overlap_penalty = float(subword_overlap_penalty)
        if _need_eager_xattn:
            _last_xattn = asr_model.decoder.decoders[-1].src_attn
            _last_xattn.use_sdpa = False
            _last_xattn.use_flash_attn = False
            logging.info(
                "xattn capture: forcing eager attention on last decoder "
                "layer's src_attn so weights are stored in .attn."
            )
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

        # 7. Streaming configuration
        self.chunk_size = chunk_size
        self.num_left_chunks = num_left_chunks
        self.num_right_chunks = num_right_chunks
        # If set, encoder mask uses this nrc while the chunk tracker delay
        # continues to use self.num_right_chunks. Decouples encoder lookahead
        # from decoder-delay budget (e.g. causal encoder + cross-attn K>0).
        self.encoder_num_right_chunks_override = encoder_num_right_chunks_override
        self.num_global_tokens = num_global_tokens
        self.use_decoder_self_kvcache = use_decoder_self_kvcache
        self.use_decoder_cross_kvcache = use_decoder_cross_kvcache
        self.use_encoder_self_kvcache = use_encoder_self_kvcache
        self.use_asymmetric_mask_at_inference = use_asymmetric_mask_at_inference
        self.extract_cq_slot = extract_cq_slot
        self.cross_attn_slot_taper = cross_attn_slot_taper
        if cross_attn_slot_taper and not use_asymmetric_mask_at_inference:
            raise ValueError(
                "cross_attn_slot_taper=True requires "
                "use_asymmetric_mask_at_inference=True (the C-axis must be "
                "exposed so per-chunk c_q gathering can run)."
            )
        self.defer_decoding_until_last_chunk = defer_decoding_until_last_chunk
        self.dump_xattn_dir = dump_xattn_dir
        self.unmask_xattn_at_final = unmask_xattn_at_final
        self.resume_offline_at_final = resume_offline_at_final
        self.commit_stable_prefix = commit_stable_prefix

        # Dynamic future-chunks controller. None when disabled.
        if dynamic_future_chunks:
            from espnet2.asr_stream.dynamic_future_chunks import (
                DynamicFutureChunksConfig,
                DynamicFutureChunksController,
            )
            blank_id = (
                asr_model.blank_id
                if hasattr(asr_model, "blank_id")
                else 0
            )
            dyn_cfg = DynamicFutureChunksConfig(
                enabled=True,
                mode=dynamic_mode,
                max_future_chunks=max_future_chunks,
                signal_type=dynamic_signal_type,
                top1_prob_threshold=dynamic_top1_prob_threshold,
                entropy_threshold=dynamic_entropy_threshold,
                margin_threshold=dynamic_margin_threshold,
                topk_mass_threshold=dynamic_topk_mass_threshold,
                fake_random_prob=dynamic_fake_random_prob,
                fake_random_seed=dynamic_fake_random_seed,
                ctc_blank_id=int(blank_id),
                sr_cem_threshold=(
                    float(sr_cem_threshold)
                    if sr_cem_threshold is not None else 0.5
                ),
                # ctc_disagree is binary; reuse --sr_cem_threshold as the
                # operating point (any value in (0,1) defers all disagreements).
                ctc_disagree_threshold=(
                    float(sr_cem_threshold)
                    if sr_cem_threshold is not None else 0.5
                ),
            )
            self.dynamic_fc_controller = DynamicFutureChunksController(dyn_cfg)
            # Token-level mode (Branch A) consults the controller from
            # inside the beam-search step loop. Expose the controller on
            # the beam_search object so search() can reach it.
            self.beam_search.dynamic_fc_controller = self.dynamic_fc_controller
            logging.info(
                f"Dynamic future-chunks enabled: "
                f"mode={dynamic_mode}, signal={dynamic_signal_type}, "
                f"max_future_chunks={max_future_chunks}, "
                f"thresholds(top1={dynamic_top1_prob_threshold}, "
                f"entropy={dynamic_entropy_threshold}, "
                f"margin={dynamic_margin_threshold}, "
                f"topk_mass={dynamic_topk_mass_threshold}), "
                f"fake_random(prob={dynamic_fake_random_prob}, "
                f"seed={dynamic_fake_random_seed})"
            )
        else:
            self.dynamic_fc_controller = None
            self.beam_search.dynamic_fc_controller = None

        # ----- SR-CEM scorer (calibrator for the trigger) -----
        # Loaded if a checkpoint path is provided AND the signal type
        # is one of the SR-CEM family. The scorer reads beam-search
        # state and outputs p_correct; it does NOT modify any
        # beam-search internals (see espnet2/asr_stream/sr_cem.py
        # docstring's design invariant).
        self.sr_cem_scorer = None
        self.sr_cem_variant = sr_cem_variant.upper() if sr_cem_variant else "A"
        self.sr_cem_chunk_agg = sr_cem_chunk_agg
        self.sr_cem_feat_dump_dir = sr_cem_feat_dump_dir

        # Long-form virtual-final segmentation (consumed by the sim loop).
        self.virtual_final_blank_frames = int(virtual_final_blank_frames or 0)
        self.virtual_final_min_chunks = int(virtual_final_min_chunks or 0)
        self.trailing_blank_frames = 0
        # When True, the Stage-B is_final pad frames are excluded from the
        # encoder self-attention (real frames never attend to silence pad).
        self.mask_streaming_pad = bool(mask_streaming_pad)

        if sr_cem_ckpt and dynamic_future_chunks and dynamic_signal_type in (
            "sr_cem_causal", "sr_cem_chunk", "wait_policy", "wait_policy_chunk",
        ):
            from espnet2.asr_stream.sr_cem import load_sr_cem_checkpoint
            # Derive variant from signal_type. The learned wait-policy reuses the
            # SAME SRCemScorer / checkpoint format / --sr_cem_ckpt arg as SR-CEM;
            # only the controller comparison polarity differs (p_wait > thr).
            # 'A' family = causal token-level (sr_cem_causal / wait_policy):
            # both 'A' and the 9-feature 'A_MARGIN' are valid here. 'B' = chunk.
            inferred_family = (
                "A" if dynamic_signal_type in ("sr_cem_causal", "wait_policy")
                else "B"
            )
            _ok = (
                self.sr_cem_variant in ("A", "A_MARGIN") if inferred_family == "A"
                else self.sr_cem_variant == "B"
            )
            if not _ok:
                logging.warning(
                    "sr_cem_variant=%s incompatible with dynamic_signal_type=%s; "
                    "overriding to %s",
                    self.sr_cem_variant, dynamic_signal_type, inferred_family,
                )
                self.sr_cem_variant = inferred_family
            self.sr_cem_scorer = load_sr_cem_checkpoint(
                path=sr_cem_ckpt,
                variant=self.sr_cem_variant,
                threshold=sr_cem_threshold,
                device=device,
                chunk_agg=sr_cem_chunk_agg,
            )
            if self.sr_cem_scorer is not None:
                # Sync resolved values (CLI > checkpoint > default) into
                # the controller config and the Variant B aggregator.
                if self.dynamic_fc_controller is not None:
                    self.dynamic_fc_controller.config.sr_cem_threshold = (
                        self.sr_cem_scorer.threshold
                    )
                self.sr_cem_chunk_agg = (
                    self.sr_cem_scorer.chunk_agg or "mean"
                )
                logging.info(
                    "SR-CEM Variant %s loaded from %s; threshold=%.3f",
                    self.sr_cem_variant, sr_cem_ckpt,
                    self.sr_cem_scorer.threshold,
                )
            # Hand the scorer to beam_search for the Variant A per-step hook.
            self.beam_search.sr_cem_scorer = self.sr_cem_scorer
        else:
            self.beam_search.sr_cem_scorer = None
        # Per-utterance ID stamp for SR-CEM feature dumps. Written by
        # the main inference() loop before each utterance; the
        # beam_search per-step hook reads it for the JSONL row.
        self.beam_search._sr_cem_utt_id = ""
        self.beam_search._sr_cem_feat_dump_fp = None
        # Deploy-time learned wait-policy aux head on the DECODER HIDDEN STATE.
        # WAIT_AUX_CKPT=<fine_tune_waitaux .pth> loads the fine-tuned decoder
        # weights into the model AND the aux MLP, then sets return_hs so
        # batch_beam_search reads hs[top1_hyp_idx] -> p_wait (signal_type must
        # be wait_policy). Replaces the beam-feature SR-CEM scorer.
        import os as _os_wa
        _wait_aux_ckpt = _os_wa.environ.get("WAIT_AUX_CKPT")
        if _wait_aux_ckpt:
            from espnet2.asr_stream.wait_aux import load_wait_aux
            _st, _wa_scorer = load_wait_aux(_wait_aux_ckpt, device=device)
            self.asr_model.decoder.load_state_dict(_st["decoder"])
            self.beam_search.wait_aux_head = _wa_scorer
            self.beam_search.return_hs = True
            logging.info(
                "[wait_aux] loaded fine-tuned decoder + aux head from %s; "
                "return_hs=True; wait_policy signal from decoder hidden state",
                _wait_aux_ckpt,
            )
        else:
            self.beam_search.wait_aux_head = None
        # Hidden-state probe: when dumping features AND the env flag is set,
        # accumulate per-token decoder hidden states (hyp.hs) so the dump can
        # save them for a lookahead-sensitivity probe richer than the scalar
        # output features. Only enable on static (no-rollback) dump runs.
        import os as _os_hs
        self._enc_last = None
        if sr_cem_feat_dump_dir and _os_hs.environ.get("DUMP_HIDDEN_STATES") == "1":
            self.beam_search.return_hs = True
            logging.info("[hidden-probe] return_hs=True; per-token decoder "
                         "hidden states will be dumped as hs_<utt>.npy")
            # Encoder-context probe: forward hook on the encoder captures its
            # raw output (last call per utt = full utt); dumped to enc/<utt>.npy.
            def _enc_hook(_m, _i, _o):
                try:
                    self._enc_last = (
                        _o[0] if isinstance(_o, (tuple, list)) else _o).detach()
                except Exception:
                    pass
            try:
                self.asr_model.encoder.register_forward_hook(_enc_hook)
                logging.info("[enc-probe] encoder forward hook registered")
            except Exception as _e:
                logging.warning(f"[enc-probe] hook failed: {_e}")

        # Per-utterance CTC log-prob buffer (numpy (T, V) when populated).
        # Filled on every is_final=True call; consumed by the inference loop.
        self._last_ctc_logprobs = None

        # Apply KV cache inference flags to the decoder modules.
        # Encoder self-attn caching uses the Stage B streaming path
        # (see forward_streaming), not the module's use_kvcache flag.
        decoder = self.asr_model.decoder
        decoder.use_kvcache = use_decoder_self_kvcache
        for decoder_layer in decoder.decoders:
            decoder_layer.self_attn.use_kvcache = use_decoder_self_kvcache
            decoder_layer.src_attn.use_kvcache = use_decoder_cross_kvcache

        logging.info(
            f"KV cache: decoder_self={use_decoder_self_kvcache}, "
            f"decoder_cross={use_decoder_cross_kvcache}, "
            f"encoder_streaming={use_encoder_self_kvcache}"
        )

        # Diagnostic: flip the inference-time DCConv non-causal flag on every
        # ConvolutionModule in the encoder. Each conformer layer holds its
        # conv module on .conv_module (espnet conformer encoder convention).
        if force_dcconv_right_context_at_inference:
            n_flipped = 0
            from espnet.nets.pytorch_backend.conformer.convolution import (
                ConvolutionModule,
            )
            for m in self.asr_model.encoder.modules():
                if isinstance(m, ConvolutionModule):
                    m.force_inference_right_context = True
                    n_flipped += 1
            logging.info(
                f"DCConv inference-time non-causal override: enabled on "
                f"{n_flipped} ConvolutionModule(s) - train/test mismatch, "
                f"diagnostic only."
            )

        # Determine if streaming mode is enabled
        self.streaming_mode = chunk_size is not None and chunk_size > 0
        if self.streaming_mode:
            logging.info(
                f"Streaming mode: chunk_size={chunk_size}, "
                f"num_left_chunks={num_left_chunks}, "
                f"num_right_chunks={num_right_chunks}"
            )
            if num_right_chunks > 0:
                logging.info(
                    f"Encoder uses {num_right_chunks} right chunk(s); "
                    f"decoder cross-attention right chunks="
                    f"{cross_attn_num_right_chunks}"
                )
        else:
            logging.info("Offline mode (no chunking)")

        # 8. Frontend parameters
        if "n_fft" in asr_train_args.frontend_conf:
            self.n_fft = asr_train_args.frontend_conf["n_fft"]
        else:
            self.n_fft = 512
        if "hop_length" in asr_train_args.frontend_conf:
            self.hop_length = asr_train_args.frontend_conf["hop_length"]
        else:
            self.hop_length = 128
        if (
            "win_length" in asr_train_args.frontend_conf
            and asr_train_args.frontend_conf["win_length"] is not None
        ):
            self.win_length = asr_train_args.frontend_conf["win_length"]
        else:
            self.win_length = self.n_fft

        # 9. Calculate subsampling factor for chunk size conversion
        # This is used to convert chunk_size (encoder frames) to audio samples
        self.subsampling_factor = self._get_subsampling_factor()
        logging.info(f"Subsampling factor: {self.subsampling_factor}")

        # 10. Initialize state
        self.reset()

    def _get_subsampling_factor(self) -> int:
        """Determine subsampling factor from encoder config."""
        encoder_conf = getattr(self.asr_train_args, "encoder_conf", {})
        input_layer = encoder_conf.get("input_layer", "conv2d")

        # Common subsampling factors based on input layer type
        subsampling_map = {
            "conv2d": 4,
            "conv2d2": 2,
            "conv2d6": 6,
            "conv2d8": 8,
            "conv2d6_wo_posenc": 6,
            "linear": 1,
            "embed": 1,
        }
        return subsampling_map.get(input_layer, 4)

    def reset(self):
        """Reset all states for new utterance."""
        self.frontend_states = None
        self.accumulated_features = None
        # Stage B raw-feature trim: how many subsampled chunks have been
        # dropped from the head of accumulated_features. Used to translate
        # absolute chunk indices to local buffer indices in _streaming_b_call.
        self._b_sub_chunks_dropped = 0

        # Initialize chunk tracker if in streaming mode
        if self.streaming_mode:
            self.chunk_tracker = ChunkTracker(
                chunk_size=self.chunk_size,
                num_left_chunks=self.num_left_chunks,
                num_right_chunks=self.num_right_chunks,
                num_global_tokens=self.num_global_tokens,
            )
        else:
            self.chunk_tracker = None

        # Track the last encoder chunk index passed to beam search
        # (for including right-context chunks without duplication)
        self._last_enc_chunk_sent = -1

        # Long-form virtual finals: trailing CTC-blank run length (frames).
        self.trailing_blank_frames = 0

        # Stage B: clear encoder streaming state (per-layer KV cache, conv state, counter).
        if (
            getattr(self, "use_encoder_self_kvcache", False)
            and hasattr(self.asr_model.encoder, "reset_streaming_state")
        ):
            self.asr_model.encoder.reset_streaming_state()

        # Reset beam search
        self.beam_search.reset()

        # Reset dynamic future-chunks controller (per-utterance state).
        if getattr(self, "dynamic_fc_controller", None) is not None:
            self.dynamic_fc_controller.reset()
        # Chunks that have been encoded and added to the chunk tracker but
        # not yet released for beam search because the controller deferred
        # them. Sorted, contiguous range from chunk_tracker.next_decode_idx.
        self._dyn_pending: List[int] = []

        # commit-stable-prefix (Branch D): per-chunk frozen committed prefix
        # (chunk-local token ids) and the previous re-decode pass's chunk-local
        # token list, for the agreement rule (longest common prefix across
        # consecutive passes). Empty/unused unless commit_stable_prefix is on.
        self._csp_committed = {}
        self._csp_prev = {}

        # SR-CEM feat-dump utt-id is set EXTERNALLY by the main
        # inference() loop right before each utterance. We do NOT
        # increment it here because reset() is called multiple times
        # per utt (end-of-call, error paths) which would double-count.

    def apply_frontend(
        self, speech: torch.Tensor, prev_states=None, is_final: bool = False
    ):
        """Apply frontend processing (feature extraction) with buffering.

        Handles waveform buffering for STFT overlap and extracts acoustic features.

        Args:
            speech: Input waveform chunk (1D tensor).
            prev_states: Previous frontend states (waveform buffer).
            is_final: Whether this is the final chunk.

        Returns:
            feats: Extracted features (B, T, D) or None if not enough samples.
            feats_lengths: Feature lengths (B,).
            next_states: Next frontend states or None if final.
        """
        if prev_states is not None:
            buf = prev_states["waveform_buffer"]
            speech = torch.cat([buf, speech], dim=0)

        has_enough_samples = False if speech.size(0) <= self.win_length else True
        if not has_enough_samples:
            if is_final:
                pad = torch.zeros(self.win_length - speech.size(0), dtype=speech.dtype)
                speech = torch.cat([speech, pad], dim=0)
            else:
                feats = None
                feats_lengths = None
                next_states = {"waveform_buffer": speech.clone()}
                return feats, feats_lengths, next_states

        if is_final:
            speech_to_process = speech
            waveform_buffer = None
        else:
            n_frames = speech.size(0) // self.hop_length
            n_residual = speech.size(0) % self.hop_length
            speech_to_process = speech.narrow(0, 0, n_frames * self.hop_length)
            waveform_buffer = speech.narrow(
                0,
                speech.size(0)
                - (math.ceil(math.ceil(self.win_length / self.hop_length) / 2) * 2 - 1)
                * self.hop_length
                - n_residual,
                (math.ceil(math.ceil(self.win_length / self.hop_length) / 2) * 2 - 1)
                * self.hop_length
                + n_residual,
            ).clone()

        # data: (Nsamples,) -> (1, Nsamples)
        speech_to_process = speech_to_process.unsqueeze(0).to(
            getattr(torch, self.dtype)
        )
        lengths = speech_to_process.new_full(
            [1], dtype=torch.long, fill_value=speech_to_process.size(1)
        )
        batch = {"speech": speech_to_process, "speech_lengths": lengths}

        # To device
        batch = to_device(batch, device=self.device)

        feats, feats_lengths = self.asr_model._extract_feats(**batch)
        if self.asr_model.normalize is not None:
            feats, feats_lengths = self.asr_model.normalize(feats, feats_lengths)

        # Trimming for STFT overlap
        _trim = math.ceil(math.ceil(self.win_length / self.hop_length) / 2)
        if is_final:
            if prev_states is None:
                pass
            else:
                _length = feats.size(1) - _trim
                if _length > 0:
                    feats = feats.narrow(1, _trim, _length)
                # else: final chunk too short to trim; keep feats as-is
        else:
            if prev_states is None:
                feats = feats.narrow(
                    1,
                    0,
                    feats.size(1) - _trim,
                )
            else:
                feats = feats.narrow(
                    1,
                    _trim,
                    feats.size(1) - 2 * _trim,
                )

        feats_lengths = feats.new_full([1], dtype=torch.long, fill_value=feats.size(1))

        if is_final:
            next_states = None
        else:
            next_states = {"waveform_buffer": waveform_buffer}
        return feats, feats_lengths, next_states

    def _encode_features(
        self,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        position_offset: int = 0,
        num_right_chunks_override: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode features through the encoder.

        Args:
            feats: Input features (B, T, D).
            feats_lengths: Feature lengths (B,).
            position_offset: RoPE position offset for streaming.
            num_right_chunks_override: If set, overrides ``num_right_chunks`` in
                the chunked mask config for this call only.  Pass ``0`` to obtain
                a strictly causal (preliminary) encoding; pass ``None`` to use
                the tracker's configured value (the asymmetric / final encoding).

        Returns:
            Encoder output (B, T, D).
        """
        # Build chunked mask config if in streaming mode
        if self.streaming_mode:
            nrc = (
                self.num_right_chunks
                if num_right_chunks_override is None
                else num_right_chunks_override
            )
            chunked_mask_config = ChunkedMaskConfig(
                chunk_size=self.chunk_size,
                num_left_chunks=self.num_left_chunks,
                num_right_chunks=nrc,
                num_global_tokens=self.num_global_tokens,
                use_asymmetric_mask=self.use_asymmetric_mask_at_inference,
                full_attention=False,
            )
        else:
            chunked_mask_config = None

        # Encode through encoder with position offset for RoPE
        enc, enc_lengths, _ = self.asr_model.encoder(
            feats,
            feats_lengths,
            prev_states=None,
            chunked_mask_config=chunked_mask_config,
            position_offset=position_offset,
        )

        # When use_asymmetric_mask=True the encoder returns (B, T, C, D)
        # where C = num_right_chunks + 1.  Collapse the C dim back to (B, T, D).
        if enc.dim() == 4:
            C = enc.size(2)
            if C == 1:
                enc = enc[:, :, 0, :]
            elif C == self.num_right_chunks + 1:
                if self.cross_attn_slot_taper:
                    # Reading-C strict-streaming taper: per-chunk c_q so each
                    # chunk uses only the lookahead audio it can afford under
                    # the current encoder budget. With num_chunks = T/cs and
                    # decode chunk c = num_chunks - 1 - R:
                    #   chunks i in [0, c]:  c_q = R     (final view)
                    #   chunk i = c + k>=1:  c_q = R - k (audio for k future)
                    #   chunk i = c + R:     c_q = 0     (causal)
                    # In one formula: c_q(i) = min(R, num_chunks - 1 - i).
                    B, T, _, D = enc.shape
                    cs = self.chunk_size
                    R = self.num_right_chunks
                    num_chunks = (T + cs - 1) // cs
                    # Per-frame c_q indices. frame t belongs to chunk t // cs.
                    chunk_idx = torch.arange(T, device=enc.device) // cs
                    cq_idx = torch.clamp(num_chunks - 1 - chunk_idx, max=R)
                    # Gather: enc shape (B, T, C, D) -> (B, T, D)
                    enc = enc.gather(
                        2,
                        cq_idx.view(1, T, 1, 1).expand(B, T, 1, D),
                    ).squeeze(2)
                else:
                    # Default: extract c_q = num_right_chunks slot (max-right-
                    # context view, matches the training slice). Override via
                    # extract_cq_slot for diagnostic decoupling of self-attn
                    # boundary from DCConv future access.
                    cq_to_extract = (
                        self.extract_cq_slot
                        if self.extract_cq_slot is not None
                        else self.num_right_chunks
                    )
                    assert 0 <= cq_to_extract < C, (
                        f"extract_cq_slot={cq_to_extract} out of range [0, {C})"
                    )
                    enc = enc[:, :, cq_to_extract, :]
            else:
                # TODO: implement per-chunk slice selection for use_asymmetric_mask.
                # Each position t should use slice c_q based on its chunk age.
                raise NotImplementedError(
                    f"Chunked encoder returned C={C} (use_asymmetric_mask=True). "
                    "Per-chunk slice selection is not implemented yet."
                )

        return enc

    @torch.no_grad()
    @typechecked
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray], is_final: bool = True
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis, dict]]:
        """Streaming inference on audio chunk.

        This method implements chunk-by-chunk processing with delayed decoding:
        1. Extract features from audio chunk
        2. Accumulate features (handles STFT edge effects)
        3. Encode accumulated features with chunked attention mask
        4. Track chunks and implement delayed decoding per ChunkTracker

        Args:
            speech: Input audio chunk (1D array/tensor).
            is_final: Whether this is the final chunk of the utterance.

        Returns:
            List of (text, token, token_int, hyp, latency_info) tuples.
        """
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # 1. Extract features
        feats, feats_lengths, self.frontend_states = self.apply_frontend(
            speech, self.frontend_states, is_final=is_final
        )

        if feats is None:
            return []

        # 2. Accumulate features (handles STFT edge effects)
        if self.accumulated_features is None:
            self.accumulated_features = feats
        else:
            self.accumulated_features = torch.cat(
                [self.accumulated_features, feats], dim=1
            )
        accumulated_lengths = torch.tensor(
            [self.accumulated_features.shape[1]],
            dtype=torch.long,
            device=self.device,
        )

        # Stage B: dispatch to incremental streaming encoder KV cache path
        # when enabled. This path does subsampling once (on full accumulated
        # features, for training parity), then slices and runs only the
        # unfinalized tail through the encoder layers, with per-layer K/V
        # caching of finalized chunks.
        if self.streaming_mode and self.use_encoder_self_kvcache:
            assert self.dynamic_fc_controller is None, (
                "dynamic_future_chunks is incompatible with "
                "use_encoder_self_kvcache (Stage B). Stage B's incremental "
                "KV cache assumes a fixed num_right_chunks; dynamic deferral "
                "would require flushing/recomputing parts of the cache."
            )
            return self._streaming_b_call(is_final, accumulated_lengths)

        # Dynamic future-chunks dispatch: per-chunk dynamic deferral based
        # on CTC-confidence trigger. Default-off; behaviour identical to
        # static path when controller is absent.
        if self.streaming_mode and self.dynamic_fc_controller is not None:
            return self._dynamic_future_chunks_call(
                accumulated_lengths, is_final
            )

        # 3. Encode all accumulated features.
        #
        # Two-pass asymmetric chunk attention protocol (when num_right_chunks=1):
        #
        #   When chunk Ct arrives, this single encoder call on all accumulated
        #   features naturally implements both passes of the asymmetric protocol:
        #
        #   * FINAL pass for Ct-1: the encoder (with the same asymmetric mask at
        #     every layer) gives Ct-1 access to Ct as future context.  The chunk
        #     tracker's num_right_chunks=1 delay ensures Ct-1 is only released for
        #     decoding at this moment - i.e. enc[:, (t-1)*S:t*S, :] IS the final
        #     encoding of Ct-1 (with Ct as future), exactly as described in the
        #     design document.
        #
        #   * PRELIMINARY pass for Ct: enc[:, t*S:(t+1)*S, :] is a causal encoding
        #     of Ct (no Ct+1 is in the context yet), corresponding to the preliminary
        #     pass.  It is buffered in the tracker and used for decoding only after
        #     Ct+1 arrives (at which point it will be superseded by the final pass).
        #
        # Use num_right_chunks_override=0 for an explicit causal-only pass if you
        # need the preliminary encoding separately (e.g. for low-latency display).
        enc = self._encode_features(
            self.accumulated_features,
            accumulated_lengths,
            position_offset=0,
            num_right_chunks_override=self.encoder_num_right_chunks_override,
        )

        # Capture per-frame CTC log-probs immediately after the encoder runs,
        # so the dump survives any downstream failure (e.g. beam-search OOM).
        if is_final:
            self._capture_ctc_logprobs(enc[0])

        # 4. Handle streaming vs offline mode
        if self.streaming_mode and self.chunk_tracker is not None:
            num_frames = enc.shape[1]
            num_complete_chunks = num_frames // self.chunk_size

            # Track how many new chunks we have compared to last call
            prev_num_chunks = self.chunk_tracker.current_chunk_idx + 1
            new_chunks = num_complete_chunks - prev_num_chunks

            # Add new chunks to tracker and check for decode readiness.
            #
            # Two-pass semantics: enc_chunk at position i is the PRELIMINARY
            # encoding of chunk i (no future context), and it becomes the FINAL
            # encoding of chunk i-1 (with chunk i as future) simultaneously.
            # The tracker's 1-chunk delay releases chunk i-1 for decoding only
            # now, so the decoder always sees the final (asymmetric) encoding.
            decode_ready_chunks = []
            for i in range(new_chunks):
                chunk_idx = prev_num_chunks + i
                start_frame = chunk_idx * self.chunk_size
                end_frame = start_frame + self.chunk_size
                enc_chunk = enc[0, start_frame:end_frame, :]  # (chunk_size, D)

                # Long-form virtual finals: track the trailing CTC-blank run
                # so the sim loop can flush segments at sentence-gap silences.
                if self.virtual_final_blank_frames > 0:
                    _am = self.asr_model.ctc.ctc_lo(enc_chunk).argmax(-1)
                    _nb = (_am != 0).nonzero()
                    if _nb.numel() == 0:
                        self.trailing_blank_frames += _am.numel()
                    else:
                        self.trailing_blank_frames = (
                            _am.numel() - 1 - int(_nb[-1].item())
                        )

                # Store raw (pre-encoder) features for this chunk so ChunkTracker
                # can later re-encode with a different mask if needed (e.g. for a
                # future KV-cache-optimised 2-pass implementation).
                sf = self.subsampling_factor
                raw_start = chunk_idx * self.chunk_size * sf
                raw_end = raw_start + self.chunk_size * sf
                raw_chunk = self.accumulated_features[0, raw_start:raw_end, :]
                self.chunk_tracker.add_raw_chunk(raw_chunk)

                decode_idx = self.chunk_tracker.add_chunk(enc_chunk)
                if decode_idx is not None:
                    decode_ready_chunks.append(decode_idx)

            # Stage A fix: for the prior num_right_chunks chunk(s) already in the
            # tracker, overwrite their stored encoder output with the slice from
            # the CURRENT encoder call - which now includes right-context chunks
            # in the input tensor. Only the chunk being released this call needs
            # this update (older chunks were finalized on prior calls).
            if new_chunks > 0 and self.num_right_chunks > 0:
                for past_idx in range(
                    max(0, prev_num_chunks - self.num_right_chunks),
                    prev_num_chunks,
                ):
                    start = past_idx * self.chunk_size
                    end = start + self.chunk_size
                    self.chunk_tracker.encoder_chunks[past_idx] = enc[0, start:end, :]
                    logging.debug(
                        f"[Stage A] Overwrote encoder_chunks[{past_idx}] with "
                        f"right-context slice from call at chunk_idx="
                        f"{prev_num_chunks + new_chunks - 1}"
                    )

            # On final, add any partial chunk before flushing
            if is_final:
                partial_frames = num_frames % self.chunk_size
                if partial_frames > 0:
                    partial_start = num_complete_chunks * self.chunk_size
                    partial_chunk = enc[0, partial_start:, :]
                    # Store raw features for consistency with complete chunks
                    sf = self.subsampling_factor
                    raw_start = num_complete_chunks * self.chunk_size * sf
                    raw_chunk = self.accumulated_features[0, raw_start:, :]
                    self.chunk_tracker.add_raw_chunk(raw_chunk)
                    # Route through add_chunk for proper tracking
                    decode_idx = self.chunk_tracker.add_chunk(partial_chunk)
                    if decode_idx is not None:
                        decode_ready_chunks.append(decode_idx)
                remaining = self.chunk_tracker.flush_remaining()
                decode_ready_chunks.extend(remaining)

            # Defer-decode diagnostic: skip per-chunk beam search; on is_final
            # run the beam search once over the full concatenated encoder
            # output with no per-token cross-attn mask. Isolates the beam loop
            # from the encoder representation.
            if self.defer_decoding_until_last_chunk:
                if not is_final:
                    return []
                dec_enc = torch.cat(self.chunk_tracker.encoder_chunks, dim=0)
                _saved_cs = self.beam_search.cross_attn_chunk_size
                self.beam_search.cross_attn_chunk_size = 0
                try:
                    nbest_hyps = self.beam_search(
                        x=dec_enc,
                        maxlenratio=self.maxlenratio,
                        minlenratio=self.minlenratio,
                        is_final=True,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    logging.warning(
                        f"Beam search OOM (defer streaming branch): {e}. "
                        "CTC log-prob dump was already saved; returning empty hyps."
                    )
                    nbest_hyps = []
                    torch.cuda.empty_cache()
                finally:
                    self.beam_search.cross_attn_chunk_size = _saved_cs
            # Run beam search only on the newly decode-ready chunk(s)
            elif decode_ready_chunks or is_final:
                if decode_ready_chunks:
                    max_ctx_idx = min(
                        max(decode_ready_chunks),
                        self.chunk_tracker.current_chunk_idx,
                    )
                    # Expose extra "lookahead" encoded chunks as cross-attn K/V
                    # (context only). Each encoded chunk is sent EXACTLY ONCE:
                    # we advance _last_enc_chunk_sent to max_kv_idx so the C+1
                    # future chunk doesn't get re-sent on the next call as
                    # "current". Re-sending duplicates chunks in
                    # beam_search.encbuffer, doubling the effective per-slot
                    # frame count vs the mask's cs and causing the new-token
                    # right_bound=(C+1+R)*cs-1 to land mid-K-tensor at the
                    # final chunk - root cause of the K=R>0 trailing-tail.
                    extra_right = max(0, getattr(
                        self.beam_search, "cross_attn_num_right_chunks", 0
                    ))
                    max_kv_idx = min(
                        max_ctx_idx + extra_right,
                        len(self.chunk_tracker.encoder_chunks) - 1,
                    )
                    first_new = self._last_enc_chunk_sent + 1
                    ready_chunks = [
                        self.chunk_tracker.encoder_chunks[idx]
                        for idx in range(first_new, max_kv_idx + 1)
                    ]
                    if ready_chunks:
                        dec_enc = torch.cat(ready_chunks, dim=0)  # (T_ready, D)
                    else:
                        # No new chunks to send (final chunk in R>0 mode: C+1
                        # was already sent at the previous call). Pass a
                        # zero-row tensor; beam_search's cat-into-encbuffer is
                        # a no-op for this call.
                        ref = self.chunk_tracker.encoder_chunks[-1]
                        dec_enc = ref.new_zeros((0, ref.shape[-1]))
                    self._last_enc_chunk_sent = max_kv_idx
                else:
                    # is_final with no new ready chunks - pass a zero-row
                    # tensor (the last chunk was already sent in a previous
                    # call; re-sending it would duplicate it in encbuffer and
                    # misalign the final cross-attn mask). beam_search still
                    # runs its is_final pass on the existing buffer.
                    ref = self.chunk_tracker.encoder_chunks[-1]
                    dec_enc = ref.new_zeros((0, ref.shape[-1]))

                # F4 fix: at is_final, disable the per-token cross-attn mask
                # so ALL prefix rows get full encoder access (not just the new
                # token row). This makes the final beam call behave like
                # defer-style cross-attention for the trailing tokens,
                # eliminating the streaming-specific tail substitutions that
                # arise from restricted prefix representations.
                _saved_cs = None
                if is_final and getattr(self, "unmask_xattn_at_final", False):
                    _saved_cs = self.beam_search.cross_attn_chunk_size
                    self.beam_search.cross_attn_chunk_size = 0
                try:
                    nbest_hyps = self.beam_search(
                        x=dec_enc,
                        maxlenratio=self.maxlenratio,
                        minlenratio=self.minlenratio,
                        is_final=is_final,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    logging.warning(
                        f"Beam search OOM (streaming branch): {e}. "
                        "CTC log-prob dump was already saved; returning empty hyps."
                    )
                    nbest_hyps = []
                    torch.cuda.empty_cache()
                finally:
                    if _saved_cs is not None:
                        self.beam_search.cross_attn_chunk_size = _saved_cs
            else:
                # Not ready to decode yet (waiting for future context)
                return []
        else:
            # Offline mode - standard beam search
            try:
                nbest_hyps = self.beam_search(
                    x=enc[0],
                    maxlenratio=self.maxlenratio,
                    minlenratio=self.minlenratio,
                    is_final=is_final,
                )
            except torch.cuda.OutOfMemoryError as e:
                logging.warning(
                    f"Beam search OOM (offline branch): {e}. "
                    "CTC log-prob dump was already saved; returning empty hyps."
                )
                nbest_hyps = []
                torch.cuda.empty_cache()

        # 5. Only return results on final chunk (but beam search has already updated state)
        if is_final:
            # Populate chunk tracking info for latency analysis
            if self.chunk_tracker is not None and nbest_hyps:
                self._populate_chunk_tracking(nbest_hyps)

            # DEBUG: dump encoder_chunks if env var set (for Stage B vs full-recompute diff).
            self._maybe_dump_encoder_chunks(tag="full_recompute")

            # Assemble results
            ret = self.assemble_hyps(nbest_hyps)
            # Snapshot xattn buffer before reset clears it on the beam_search.
            self._last_xattn_buffer = list(
                getattr(self.beam_search, "xattn_dump_buffer", [])
            )
            self.reset()
            return ret
        else:
            # Intermediate chunk: beam search ran to update state, but don't return results
            return []

    @torch.no_grad()
    def _dynamic_future_chunks_call(
        self,
        accumulated_lengths: torch.Tensor,
        is_final: bool,
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis, dict]]:
        """Dispatch on ``self.dynamic_fc_controller.mode``.

        Three branches:

        * ``pre_emptive`` (cheapest baseline): signal evaluated on CTC
          log-probs *before* beam search. Defer simply means "don't run
          beam search on this chunk yet".
        * ``chunk_rollback`` (Branch B): signal evaluated *after* beam
          search runs on the chunk. Defer means restore beam state,
          unrelease the chunk, and wait for one more audio chunk before
          re-decoding.
        * ``token_resume`` (Branch A): signal evaluated *per token* inside
          beam search. Defer means drop the just-emitted token, return,
          and resume the same chunk from the same token step on the next
          call (with one more chunk of encoder context).
        """
        mode = self.dynamic_fc_controller.config.mode
        if mode == "pre_emptive":
            return self._branch_pre_emptive(accumulated_lengths, is_final)
        if mode == "chunk_rollback":
            return self._branch_chunk_rollback(accumulated_lengths, is_final)
        if mode == "token_resume":
            return self._branch_token_resume(accumulated_lengths, is_final)
        if mode == "token_chunk_resume":
            # Branch D = B's look-ahead chunk-rollback machinery (future chunk
            # in BOTH the encoder re-encode and the cross-attn window), but
            # triggered PER-TOKEN: the beam-search hook stops the chunk at the
            # FIRST low-conf token instead of B's post-chunk aggregate.
            return self._branch_chunk_rollback(accumulated_lengths, is_final)
        raise ValueError(f"unknown dynamic_fc mode {mode!r}")

    def _track_trailing_blank(self, enc_chunk):
        """Long-form virtual finals: track the trailing CTC-blank run on a new
        chunk's encoding so the sim loop flushes segments at sentence-gap
        silences. The dynamic branches bypass __call__'s copy of this tracking,
        so without it the decoder never resets and collapses on multi-minute
        audio. Mirrors the static path exactly. Inert when
        virtual_final_blank_frames == 0 (all non-long-form decodes)."""
        if self.virtual_final_blank_frames <= 0:
            return
        _am = self.asr_model.ctc.ctc_lo(enc_chunk).argmax(-1)
        _nb = (_am != 0).nonzero()
        if _nb.numel() == 0:
            self.trailing_blank_frames += _am.numel()
        else:
            self.trailing_blank_frames = _am.numel() - 1 - int(_nb[-1].item())

    @torch.no_grad()
    def _branch_pre_emptive(
        self,
        accumulated_lengths: torch.Tensor,
        is_final: bool,
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis, dict]]:
        """Pre-emptive baseline: evaluate signal on CTC log-probs before
        running beam search. See class docstring of
        :func:`_dynamic_future_chunks_call` for the three-mode taxonomy.

        State carried between calls: ``self._dyn_pending`` (list of pending
        chunk indices), and the controller's per-chunk ``defer_count``.
        """
        from espnet2.asr_stream.dynamic_future_chunks import (
            COMMIT,
            compute_chunk_confidence,
        )

        ctrl = self.dynamic_fc_controller
        ct = self.chunk_tracker
        chunk_size = self.chunk_size
        sf = self.subsampling_factor

        # -----------------------------------------------------------------
        # 1. Encoder pass - override = max defer-count among pending chunks
        # -----------------------------------------------------------------
        if self._dyn_pending:
            target_override = max(
                ctrl.defer_count.get(ch, 0) for ch in self._dyn_pending
            )
        else:
            target_override = 0
        # The override may not exceed the configured num_right_chunks; the
        # encoder/asymmetric-mask machinery is keyed off num_right_chunks at
        # build time. Cap to that. (Phase 1 limitation: bumping past the
        # static configured nrc would need an unbuilt mask config.)
        target_override = min(target_override, self.num_right_chunks)

        enc = self._encode_features(
            self.accumulated_features,
            accumulated_lengths,
            position_offset=0,
            num_right_chunks_override=target_override,
        )
        if is_final:
            self._capture_ctc_logprobs(enc[0])

        # -----------------------------------------------------------------
        # 2. CTC log-probs over the full encoder output, for chunk metrics.
        # -----------------------------------------------------------------
        ctc_logits = self.asr_model.ctc.ctc_lo(enc[0].to(self.device))
        full_logp_np = (
            torch.log_softmax(ctc_logits, dim=-1).detach().cpu().numpy()
        )  # (T, V)

        # -----------------------------------------------------------------
        # 3. Add newly arrived complete chunks to chunk_tracker, but undo
        #    its auto-release so the controller decides explicitly.
        # -----------------------------------------------------------------
        num_frames = enc.shape[1]
        num_complete_chunks = num_frames // chunk_size
        prev_num_chunks = ct.current_chunk_idx + 1
        new_chunks_n = num_complete_chunks - prev_num_chunks

        candidates = list(self._dyn_pending)  # carry-over pending
        for i in range(new_chunks_n):
            chunk_idx = prev_num_chunks + i
            start = chunk_idx * chunk_size
            end = start + chunk_size
            enc_chunk = enc[0, start:end, :]
            raw_start = chunk_idx * chunk_size * sf
            raw_end = raw_start + chunk_size * sf
            raw_chunk = self.accumulated_features[0, raw_start:raw_end, :]
            ct.add_raw_chunk(raw_chunk)
            decode_idx = ct.add_chunk(enc_chunk)
            # Undo the auto-release; controller decides below.
            if decode_idx is not None:
                ct.next_decode_idx -= 1
            candidates.append(chunk_idx)

        # Refresh stored encoder slices for pending chunks: this encoder
        # call may have produced a different representation under the
        # updated future-context window (analogous to the Stage A fix in
        # the static path).
        for ch in self._dyn_pending:
            start = ch * chunk_size
            end = start + chunk_size
            if end <= num_frames:
                ct.encoder_chunks[ch] = enc[0, start:end, :]

        # Final-call: also stage a partial-frames last chunk.
        if is_final:
            partial_frames = num_frames % chunk_size
            if partial_frames > 0:
                partial_start = num_complete_chunks * chunk_size
                partial_chunk = enc[0, partial_start:, :]
                raw_start = num_complete_chunks * chunk_size * sf
                raw_partial = self.accumulated_features[0, raw_start:, :]
                ct.add_raw_chunk(raw_partial)
                decode_idx = ct.add_chunk(partial_chunk)
                if decode_idx is not None:
                    ct.next_decode_idx -= 1
                candidates.append(ct.current_chunk_idx)

        candidates = sorted(set(candidates))

        # -----------------------------------------------------------------
        # 4. Per-candidate confidence eval + release in contiguous order.
        # -----------------------------------------------------------------
        decode_ready_chunks: List[int] = []
        new_pending: List[int] = []
        release_blocked = False
        for ch in candidates:
            start = ch * chunk_size
            end = min(start + chunk_size, full_logp_np.shape[0])
            chunk_logp = full_logp_np[start:end, :]
            metrics = compute_chunk_confidence(
                chunk_logp,
                blank_id=ctrl.config.ctc_blank_id,
            )
            future_avail = ct.current_chunk_idx - ch
            decision = ctrl.decide_chunk(
                chunk_idx=ch,
                metrics=metrics,
                future_available=future_avail,
                is_final=is_final,
            )
            can_release = (
                not release_blocked
                and decision == COMMIT
                and ch == ct.next_decode_idx
            )
            if can_release:
                ct.next_decode_idx += 1
                decode_ready_chunks.append(ch)
            else:
                new_pending.append(ch)
                release_blocked = True
            logging.debug(
                f"[dyn_fc] chunk={ch} future_avail={future_avail} "
                f"defer_count={ctrl.defer_count.get(ch, 0)} "
                f"top1={metrics.top1_prob_mean:.3f} "
                f"entropy={metrics.entropy_mean:.3f} "
                f"margin={metrics.margin_mean:.3f} "
                f"topk_mass={metrics.topk_mass_mean:.3f} "
                f"decision={decision} released={can_release}"
            )

        self._dyn_pending = new_pending

        # is_final must release everything: controller force-commits, but
        # the release-blocked ordering check might still have held chunks
        # behind a (now-removed) earlier blocker. Walk again.
        if is_final and self._dyn_pending:
            still_pending = []
            for ch in self._dyn_pending:
                if ch == ct.next_decode_idx:
                    ct.next_decode_idx += 1
                    decode_ready_chunks.append(ch)
                else:
                    still_pending.append(ch)
            self._dyn_pending = still_pending
            # Anything in chunk_tracker beyond next_decode_idx must also flush.
            decode_ready_chunks.extend(ct.flush_remaining())

        # -----------------------------------------------------------------
        # 5. Beam search over the contiguous newly-released chunks.
        #    Same logic as the static streaming branch of __call__.
        # -----------------------------------------------------------------
        if not decode_ready_chunks and not is_final:
            return []

        if decode_ready_chunks:
            max_ctx_idx = min(
                max(decode_ready_chunks), ct.current_chunk_idx
            )
            extra_right = max(
                0,
                getattr(self.beam_search, "cross_attn_num_right_chunks", 0),
            )
            max_kv_idx = min(
                max_ctx_idx + extra_right,
                len(ct.encoder_chunks) - 1,
            )
            first_new = self._last_enc_chunk_sent + 1
            ready_chunks = [
                ct.encoder_chunks[idx]
                for idx in range(first_new, max_kv_idx + 1)
            ]
            if ready_chunks:
                dec_enc = torch.cat(ready_chunks, dim=0)
            else:
                # Already sent in a previous call - zero-row tensor avoids
                # duplicating the last chunk in encbuffer at is_final.
                _ref = ct.encoder_chunks[-1]
                dec_enc = _ref.new_zeros((0, _ref.shape[-1]))
            self._last_enc_chunk_sent = max_kv_idx
        else:
            _ref = ct.encoder_chunks[-1]
            dec_enc = _ref.new_zeros((0, _ref.shape[-1]))

        _saved_cs = None
        if is_final and getattr(self, "unmask_xattn_at_final", False):
            _saved_cs = self.beam_search.cross_attn_chunk_size
            self.beam_search.cross_attn_chunk_size = 0
        try:
            nbest_hyps = self.beam_search(
                x=dec_enc,
                maxlenratio=self.maxlenratio,
                minlenratio=self.minlenratio,
                is_final=is_final,
            )
        except torch.cuda.OutOfMemoryError as e:
            logging.warning(
                f"Beam search OOM (dynamic future-chunks branch): {e}. "
                "CTC log-prob dump was already saved; returning empty hyps."
            )
            nbest_hyps = []
            torch.cuda.empty_cache()
        finally:
            if _saved_cs is not None:
                self.beam_search.cross_attn_chunk_size = _saved_cs

        if is_final:
            if ct is not None and nbest_hyps:
                self._populate_chunk_tracking(nbest_hyps)
            self._maybe_dump_encoder_chunks(tag="dynamic_future_chunks")
            ret = self.assemble_hyps(nbest_hyps)
            self._last_xattn_buffer = list(
                getattr(self.beam_search, "xattn_dump_buffer", [])
            )
            logging.info(
                f"[dyn_fc] utterance summary: {ctrl.summary()}"
            )
            self.reset()
            return ret
        return []

    # ------------------------------------------------------------------
    # Branch B: chunk-level rollback
    # ------------------------------------------------------------------
    def _csp_forced_prefix(self, ch):
        """commit-stable-prefix: build the ``_forced_prefix`` token list for a
        deeper re-decode pass - the top hypothesis's prior committed tokens plus
        chunk ``ch``'s frozen prefix - so ``search()`` teacher-forces them
        (advancing decoder + CTC state) before decoding the divergent suffix.
        Returns None when nothing is frozen yet (the first one/two passes)."""
        _csp_pref = self._csp_committed.get(ch, [])
        if not _csp_pref:
            return None
        _rh = self.beam_search.running_hyps
        if _rh is None or _rh.yseq.numel() == 0:
            return None
        _prior = _rh.yseq[0, 1:int(_rh.length[0].item())].tolist()
        return [int(t) for t in _prior] + [int(t) for t in _csp_pref]

    def _oracle_chunk_wrong(self, src_hyp) -> bool:
        """Oracle trigger signal (env ORACLE_REF_FILE; SLT rebuttal).

        True iff any word of the JUST-DECODED chunk (the tokens carrying the
        highest ChunkEmissionIndex, per the coordinate note in the sr_cem_chunk
        block) is not Levenshtein-matched to ``self._oracle_ref_words``.
        A word = consecutive BPE pieces from a word-initial token; its chunk =
        its LAST token's emission index. Substitutions/insertions trigger;
        deletions emit nothing and cannot. A still-growing word at the chunk
        edge counts as wrong until completed, so the oracle defers exactly
        where waiting can still change the output.
        """
        ref = getattr(self, "_oracle_ref_words", None)
        if not ref or src_hyp is None:
            return False
        ys = getattr(src_hyp, "yseq", None)
        cei = getattr(src_hyp, "ChunkEmissionIndex", None)
        if ys is None or cei is None or cei.numel() == 0:
            return False
        ids = ys.tolist()[1:]
        n = min(len(ids), cei.numel())
        toks = self.converter.ids2tokens(ids[:n])
        cks = [int(cei[i].item()) for i in range(n)]
        words, wchunks, cur, ch_last = [], [], None, None
        for t, c in zip(toks, cks):
            if t.startswith("<"):
                continue  # sos/eos/blank specials
            if t.startswith("▁"):
                if cur:
                    words.append(cur.upper())
                    wchunks.append(ch_last)
                cur, ch_last = t[1:], c
            else:
                cur = (cur or "") + t
                ch_last = c
        if cur:
            words.append(cur.upper())
            wchunks.append(ch_last)
        if not words:
            return False
        # Levenshtein backtrace: matched hyp-word indices vs full reference.
        h, r = words, ref
        N, M = len(h), len(r)
        D = [[0] * (M + 1) for _ in range(N + 1)]
        for i in range(N + 1):
            D[i][0] = i
        for j in range(M + 1):
            D[0][j] = j
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                c = 0 if h[i - 1] == r[j - 1] else 1
                D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                              D[i - 1][j - 1] + c)
        i, j, matched = N, M, set()
        while i > 0 and j > 0:
            if h[i - 1] == r[j - 1] and D[i][j] == D[i - 1][j - 1]:
                matched.add(i - 1)
                i, j = i - 1, j - 1
            elif D[i][j] == D[i - 1][j - 1] + 1:
                i, j = i - 1, j - 1
            elif D[i][j] == D[i - 1][j] + 1:
                i -= 1
            else:
                j -= 1
        tgt = max(wchunks)
        wrong = any(
            wchunks[k] == tgt and k not in matched for k in range(len(words))
        )
        if wrong:
            logging.info("[oracle_ref] chunk-local error detected -> DEFER")
        return wrong

    @torch.no_grad()
    def _branch_chunk_rollback(
        self,
        accumulated_lengths: torch.Tensor,
        is_final: bool,
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis, dict]]:
        """Branch B: decode each released chunk, then evaluate the signal
        on the just-decoded output. On DEFER, restore the beam_search
        state to its pre-chunk snapshot, unrelease the chunk from the
        chunk tracker, and return [] - the next audio chunk arrival
        will re-encode with one more future-context chunk and re-decode
        from the same snapshot.

        For the fake_random signal type the post-decode evaluation is
        a Bernoulli sample (used to exercise the rollback plumbing);
        for the confidence signal types the metrics are computed on the
        chunk's CTC frames under the *current* encoding, which is the
        cheapest implementation. A follow-up could use per-token
        decoder-output confidences instead.
        """
        from espnet2.asr_stream.dynamic_future_chunks import (
            COMMIT,
            DEFER,
            compute_chunk_confidence,
        )

        ctrl = self.dynamic_fc_controller
        ct = self.chunk_tracker
        chunk_size = self.chunk_size
        sf = self.subsampling_factor

        # 1. Encoder pass - override = max defer-count among pending chunks.
        if self._dyn_pending:
            target_override = max(
                ctrl.defer_count.get(ch, 0) for ch in self._dyn_pending
            )
        else:
            target_override = 0
        target_override = min(target_override, self.num_right_chunks)

        enc = self._encode_features(
            self.accumulated_features,
            accumulated_lengths,
            position_offset=0,
            num_right_chunks_override=target_override,
        )
        if is_final:
            self._capture_ctc_logprobs(enc[0])

        ctc_logits = self.asr_model.ctc.ctc_lo(enc[0].to(self.device))
        full_logp_np = (
            torch.log_softmax(ctc_logits, dim=-1).detach().cpu().numpy()
        )

        # 2. Add new chunks to chunk_tracker, undo auto-release.
        num_frames = enc.shape[1]
        num_complete_chunks = num_frames // chunk_size
        prev_num_chunks = ct.current_chunk_idx + 1
        new_chunks_n = num_complete_chunks - prev_num_chunks
        candidates = list(self._dyn_pending)
        for i in range(new_chunks_n):
            chunk_idx = prev_num_chunks + i
            start = chunk_idx * chunk_size
            end = start + chunk_size
            enc_chunk = enc[0, start:end, :]
            self._track_trailing_blank(enc_chunk)  # long-form silence reset
            raw_start = chunk_idx * chunk_size * sf
            raw_end = raw_start + chunk_size * sf
            raw_chunk = self.accumulated_features[0, raw_start:raw_end, :]
            ct.add_raw_chunk(raw_chunk)
            decode_idx = ct.add_chunk(enc_chunk)
            if decode_idx is not None:
                ct.next_decode_idx -= 1
            candidates.append(chunk_idx)
        for ch in self._dyn_pending:
            start = ch * chunk_size
            end = start + chunk_size
            if end <= num_frames:
                ct.encoder_chunks[ch] = enc[0, start:end, :]

        if is_final:
            partial_frames = num_frames % chunk_size
            if partial_frames > 0:
                partial_start = num_complete_chunks * chunk_size
                partial_chunk = enc[0, partial_start:, :]
                raw_start = num_complete_chunks * chunk_size * sf
                raw_partial = self.accumulated_features[0, raw_start:, :]
                ct.add_raw_chunk(raw_partial)
                decode_idx = ct.add_chunk(partial_chunk)
                if decode_idx is not None:
                    ct.next_decode_idx -= 1
                candidates.append(ct.current_chunk_idx)

        candidates = sorted(set(candidates))

        # 3. Per-candidate snapshot/decode/evaluate/maybe-rollback loop.
        # Process in order to preserve chunk_tracker invariant.
        nbest_hyps_last: List[Hypothesis] = []
        new_pending: List[int] = []
        release_blocked = False
        # Work queue (not a plain for-loop): a deferred head chunk is RE-INSERTED
        # to be re-processed within THIS call, one defer level deeper, whenever
        # its next future chunk is already buffered (in-call deep deferral; see
        # the DEFER handling below). Without it, deferral advances only one level
        # per audio arrival, so late chunks get starved to nrc0 at is_final and
        # always-defer mfc=k fails to reproduce static nrc=k.
        _queue = list(candidates)
        _qi = 0
        while _qi < len(_queue):
            ch = _queue[_qi]
            _qi += 1
            if release_blocked or ch != ct.next_decode_idx:
                # Cannot release out of order; defer this and everything after.
                new_pending.append(ch)
                release_blocked = True
                continue

            # Snapshot before running beam_search on this chunk.
            snap = self.beam_search.snapshot_state()
            saved_last_enc_chunk_sent = self._last_enc_chunk_sent

            # Build encoder slice for this chunk plus cross-attn future context.
            # Deep-deferral (max_future_chunks > 1): the future window must match
            # how many times THIS chunk has already been deferred
            # (defer_count[ch]), capped by the configured num_right_chunks. The
            # encoder override at the top of this method uses the same k, so a
            # chunk deferred k times decodes exactly like static nrc=k: its
            # encoder repr sees k future chunks, k future chunks' K/V are
            # exposed here, and the cross-attn mask right-bound R is widened to
            # k just before beam_search below. k=0 on the first pass = causal
            # (== static nrc0). A fixed extra_right/R would clamp deep deferral
            # to a single future chunk regardless of defer_count.
            _defer_k = min(ctrl.defer_count.get(ch, 0), self.num_right_chunks)
            if ctrl.config.mode == "token_chunk_resume":
                # Branch D: tell the per-token hook whether this chunk's
                # deep-deferral budget is already spent (then it must commit,
                # no further deferring), and clear the per-chunk defer flag
                # before this (re-)decode so a stale True can't carry over.
                self.beam_search._dyn_d_force_commit = (
                    ctrl.defer_count.get(ch, 0) >= ctrl.config.max_future_chunks
                )
                self.beam_search._dyn_d_token_deferred = False
            extra_right = _defer_k
            max_kv_idx = min(
                ch + extra_right, len(ct.encoder_chunks) - 1
            )
            first_new = self._last_enc_chunk_sent + 1
            # REPLACE (default): overwrite each already-in-buffer chunk in this
            # chunk's window [ch .. last_sent] with its OWN current rep
            # (ct.encoder_chunks at this chunk's level), instead of keeping the
            # rep a deep neighbor wrote. New chunks [first_new .. max_kv_idx] are
            # still appended below. Gives a per-chunk-consistent encbuffer (each
            # chunk at its committed level) rather than inheriting the deep
            # neighbor's level - verified >= the skip variant (29-utt deep cells
            # 9.7 -> 9.4), so it is the default (no env flag).
            if self.beam_search.encbuffer is not None:
                _eb = self.beam_search.encbuffer
                for _idx in range(ch, min(self._last_enc_chunk_sent, max_kv_idx) + 1):
                    _s = _idx * chunk_size
                    _e = _s + ct.encoder_chunks[_idx].shape[0]
                    if _e <= _eb.shape[0]:
                        _eb[_s:_e] = ct.encoder_chunks[_idx].to(_eb.device)
            ready_chunks = [
                ct.encoder_chunks[idx]
                for idx in range(first_new, max_kv_idx + 1)
            ]
            if ready_chunks:
                dec_enc = torch.cat(ready_chunks, dim=0)
            else:
                # Everything up to ch+extra_right is ALREADY in the beam's
                # encbuffer (e.g. ch went out earlier as a previous chunk's
                # lookahead). Send a zero-row tensor so beam_search re-runs the
                # decode over the EXISTING encbuffer instead of APPENDING a
                # duplicate copy of these frames - the duplicate shifts all
                # downstream encoder positions and is what breaks the
                # always-defer == static nrc invariant (over-emission). This
                # matches the pre_emptive / token_resume branches.
                _ref = ct.encoder_chunks[-1]
                dec_enc = _ref.new_zeros((0, _ref.shape[-1]))
            chunk_is_final_call = is_final and (ch == candidates[-1])

            # Tentatively release this chunk for beam search.
            ct.next_decode_idx += 1
            # The K/V dedup pointer must only ADVANCE. After a DEEP commit
            # (e.g. ch deferred to k=4 sent chunks ch..ch+4, so last_sent=ch+4),
            # the next chunk's zero-rows decode has max_kv_idx=ch+1 < last_sent;
            # assigning it unconditionally would REGRESS the pointer and make
            # later chunks re-send already-sent frames, duplicating them in the
            # beam's encbuffer -> backward cross-attn -> span re-emission. Only
            # mfc=1 was immune (deep send never overshoots the next chunk).
            self._last_enc_chunk_sent = max(self._last_enc_chunk_sent, max_kv_idx)

            _saved_cs = None
            # Widen the cross-attn mask's right bound R to this chunk's defer
            # level so the decoder may actually attend the k future chunks we
            # just exposed (mask right-bound = (c+1+R)*cs-1). Without this, R
            # stays at the configured value and clamps deep deferral to a
            # single future chunk regardless of defer_count. Restored below.
            _saved_R = self.beam_search.cross_attn_num_right_chunks
            self.beam_search.cross_attn_num_right_chunks = _defer_k
            if chunk_is_final_call and getattr(self, "unmask_xattn_at_final", False):
                _saved_cs = self.beam_search.cross_attn_chunk_size
                self.beam_search.cross_attn_chunk_size = 0
            if getattr(self, "commit_stable_prefix", False):
                self.beam_search._forced_prefix = self._csp_forced_prefix(ch)
            try:
                nbest_hyps = self.beam_search(
                    x=dec_enc,
                    maxlenratio=self.maxlenratio,
                    minlenratio=self.minlenratio,
                    is_final=chunk_is_final_call,
                )
            except torch.cuda.OutOfMemoryError as e:
                logging.warning(
                    f"Beam search OOM (chunk_rollback branch ch={ch}): {e}"
                )
                nbest_hyps = []
                torch.cuda.empty_cache()
            finally:
                self.beam_search.cross_attn_num_right_chunks = _saved_R
                if _saved_cs is not None:
                    self.beam_search.cross_attn_chunk_size = _saved_cs
                if getattr(self, "commit_stable_prefix", False):
                    self.beam_search._forced_prefix = None

            # Evaluate signal on this chunk's CTC frames (also works for
            # fake_random; metrics are unused in that mode).
            start = ch * chunk_size
            end = min(start + chunk_size, full_logp_np.shape[0])
            chunk_logp = full_logp_np[start:end, :]
            metrics = compute_chunk_confidence(
                chunk_logp, blank_id=ctrl.config.ctc_blank_id
            )

            # --- SR-CEM Variant B (chunk-local) ----------------------
            # Build per-token 8-dim features for tokens emitted in
            # THIS just-decoded chunk, compute chunk-local S>t per
            # token, run the SR-CEM scorer, aggregate to a single
            # chunk-level p_correct, attach to metrics. READS hyp
            # state; does NOT modify any beam-search internals.
            # BUGFIX: the online beam_search returns [] until is_final, so
            # nbest_hyps is empty mid-stream and the Variant-B signal was only
            # ever computed at the final chunk (~once/utterance) -> every other
            # chunk committed by default. Read the current top hypothesis from
            # the search's running state when nbest_hyps is empty.
            _src_hyp = None
            if nbest_hyps:
                _src_hyp = nbest_hyps[0]
            else:
                _rh = getattr(self.beam_search, "running_hyps", None)
                if (_rh is not None and getattr(_rh, "yseq", None) is not None
                        and _rh.yseq.shape[0] > 0):
                    _src_hyp = self.beam_search._select(_rh, 0)

            # --- per-depth re-decode trace (measured user-perceived latency) ----
            # Every pass of the rollback loop re-decodes chunk `ch` at depth
            # `_defer_k`; log this chunk's 1-best tokens at this depth BEFORE the
            # DEFER rollback discards them. One (chunk, depth) record per pass →
            # the real depth-0..D_c ladder. Parsed post-hoc to compute the
            # commit-stable-prefix latency from REAL traces (not the static-nrc
            # proxy). Records delimited per-utt by the later `best hypo:` line.
            try:
                if _src_hyp is not None:
                    _ys = getattr(_src_hyp, "yseq", None)
                    _ce = getattr(_src_hyp, "ChunkEmissionIndex", None)
                    if _ys is not None and _ce is not None and _ce.numel() > 0:
                        _ids = _ys.tolist()[1:]  # drop SOS
                        _n = min(len(_ids), _ce.numel())
                        _ck_ids = [_ids[i] for i in range(_n)
                                   if int(_ce[i].item()) == ch]
                        _ck_toks = self.converter.ids2tokens(_ck_ids)
                        logging.info(
                            f"[redecode_trace] chunk={ch} depth={_defer_k} "
                            f"toks={' '.join(_ck_toks)}"
                        )
            except Exception:
                pass

            # --- commit-stable-prefix bookkeeping (agreement rule) ------------
            # Extend chunk `ch`'s FROZEN prefix to the longest common prefix
            # between this pass's chunk-local tokens and the previous pass's.
            # Those tokens are re-forced (teacher-forced) on every deeper pass so
            # they can no longer change - committing at their stable depth (lower
            # latency) at the cost of the rare token that flips after freezing.
            # Same chunk-local selection (ChunkEmissionIndex == ch) as the trace.
            if getattr(self, "commit_stable_prefix", False) and _src_hyp is not None:
                _ys_c = getattr(_src_hyp, "yseq", None)
                _ce_c = getattr(_src_hyp, "ChunkEmissionIndex", None)
                if _ys_c is not None and _ce_c is not None and _ce_c.numel() > 0:
                    _ids_c = _ys_c.tolist()[1:]
                    _n_c = min(len(_ids_c), _ce_c.numel())
                    _tk = [_ids_c[i] for i in range(_n_c)
                           if int(_ce_c[i].item()) == ch]
                    # Skip empty passes (chunk emitted nothing at this depth):
                    # setting _csp_prev=[] would stall the agreement (every later
                    # lcp against [] is 0). Forcing guarantees a non-empty _tk
                    # once a prefix is frozen, so empty only occurs pre-freeze.
                    if _tk:
                        _prev_tk = self._csp_prev.get(ch)
                        if _prev_tk is not None:
                            _L = 0
                            for _a, _b in zip(_prev_tk, _tk):
                                if _a == _b:
                                    _L += 1
                                else:
                                    break
                            # The frozen prefix only grows: forcing re-emits the
                            # already-committed tokens, so lcp >= current
                            # committed length. Guard against an unexpected
                            # shrink (would un-freeze a committed token) by
                            # keeping the longer one.
                            _old_len = len(self._csp_committed.get(ch, []))
                            if _L >= _old_len:
                                self._csp_committed[ch] = _tk[:_L]
                                if _L > _old_len:
                                    logging.info(
                                        f"[csp_freeze] chunk={ch} "
                                        f"depth={_defer_k} committed_len={_L}"
                                    )
                        self._csp_prev[ch] = _tk

            if (
                ctrl.config.signal_type == "sr_cem_chunk"
                and self.sr_cem_scorer is not None
                and _src_hyp is not None
            ):
                _hyp0 = _src_hyp
                _scores_list = getattr(_hyp0, "scores_list", None) or []
                _yseq = getattr(_hyp0, "yseq", None)
                # ChunkEmissionIndex aligns 1:1 with yseq[1:] (excludes SOS).
                _cei = getattr(_hyp0, "ChunkEmissionIndex", None)
                if not (_scores_list and _yseq is not None
                        and _cei is not None and _cei.numel() > 0):
                    # Without chunk tracking, Variant B has no features and
                    # the controller silently behaves as "always confident" -
                    # make that visible (once) instead of looking like a
                    # deliberate no-defer decision.
                    if not getattr(self, "_warned_sr_cem_b_notrack", False):
                        self._warned_sr_cem_b_notrack = True
                        logging.warning(
                            "[sr_cem_chunk] scores_list/ChunkEmissionIndex "
                            "unavailable on the top hypothesis - Variant B "
                            "features cannot be built; defer/commit falls "
                            "back to COMMIT. (Logged once.)"
                        )
                if _scores_list and _yseq is not None and _cei is not None and _cei.numel() > 0:
                    # Identify tokens emitted in chunk index `ch`.
                    _per_token_p: List[float] = []
                    try:
                        _N = min(len(_scores_list), _yseq.shape[0] - 1, _cei.numel())
                        # Collect kept tokens with their CUMULATIVE selected
                        # scores (stored scores_list values are prefix-
                        # inclusive), then derive every feature through the
                        # canonical paper-recipe helper. The old code here
                        # treated stored cumulatives as per-step scores,
                        # corrupting score/S<t/S>t_chunk - and indexed
                        # compacted lists with uncompacted chunk indices.
                        from espnet2.asr_stream.sr_cem import (
                            build_features_chunk,
                            step_features_from_cumulative,
                        )
                        _selected_ids = _yseq[1:1 + _N].tolist()
                        _chunk_all = [int(_cei[i].item()) for i in range(_N)]
                        _kept = []  # (orig_i, sel_cum, vals, chunk)
                        for i in range(_N):
                            sd = _scores_list[i]
                            if not isinstance(sd, dict):
                                continue
                            sel_cum = sd.get(str(_selected_ids[i]))
                            if sel_cum is None:
                                continue
                            _kept.append(
                                (i, float(sel_cum), list(sd.values()),
                                 _chunk_all[i])
                            )
                        # Chunk-local S>t telescopes to (last kept cumulative
                        # in the chunk) - (this token's cumulative).
                        _last_cum_in_chunk = {}
                        for (_i, _cum, _vals, _ck) in _kept:
                            _last_cum_in_chunk[_ck] = _cum
                        # BUGFIX: ChunkEmissionIndex is tagged with the beam
                        # search's own counter (processed_block), a different
                        # coordinate system than the chunk-tracker's `ch`, so
                        # `_ck == ch` never matched -> _per_token_p stayed empty
                        # -> chunk p defaulted to 1.0 -> Variant B never
                        # deferred. The tokens emitted in the just-decoded chunk
                        # carry the HIGHEST emitted-chunk index; select those.
                        _target_ck = max((c for (_, _, _, c) in _kept), default=None)
                        for k, (_i, _cum, _vals, _ck) in enumerate(_kept):
                            if _target_ck is None or _ck != _target_ck:
                                continue
                            _prev = _kept[k - 1][1] if k > 0 else 0.0
                            _sc, _rk, _slt, _t4 = step_features_from_cumulative(
                                selected_cum=_cum,
                                prev_selected_cum=_prev,
                                candidate_cums=_vals,
                            )
                            feat = build_features_chunk(
                                score=_sc,
                                rank=_rk,
                                S_lt=_slt,
                                S_gt_chunk=_last_cum_in_chunk[_ck] - _cum,
                                top4=_t4,
                            )
                            try:
                                p = float(self.sr_cem_scorer.predict(feat))
                            except Exception as _e:
                                logging.warning(
                                    "[sr_cem_chunk] predict failed: %s", _e
                                )
                                p = 1.0
                            _per_token_p.append(p)
                        # Aggregate over the chunk's tokens.
                        if _per_token_p:
                            if self.sr_cem_chunk_agg == "min":
                                _agg = min(_per_token_p)
                            else:  # default mean
                                _agg = sum(_per_token_p) / len(_per_token_p)
                        else:
                            _agg = 1.0  # no tokens in chunk -> trivially confident
                        # Attach to metrics for the controller dispatch.
                        try:
                            setattr(metrics, "sr_cem_chunk_p", float(_agg))
                        except Exception:
                            # ConfidenceMetrics is a frozen dataclass in
                            # some configs; fall back to a side-channel
                            # attribute. Note: nothing currently reads it,
                            # so the Variant-B signal is dropped for this
                            # chunk on that path.
                            self._last_sr_cem_chunk_p = float(_agg)
                    except Exception as _e:
                        logging.warning(
                            "[sr_cem_chunk] feature-build failed: %s", _e
                        )

            # --- Temperature-scaled top1_prob (chunk-level, Variant B/C) ----
            # Parallel calibration method to SR-CEM: use the CALIBRATED DECODER
            # softmax confidence (per-token top1 persisted in confidence_list by
            # the beam search, temperature applied at commit) instead of the
            # CTC-frame confidence. Aggregate over THIS chunk's tokens
            # (min = Variant B / mean = Variant C) and overwrite
            # metrics.top1_prob_mean so the controller's signal_type=top1_prob
            # dispatch reads the decoder-confidence aggregate.
            if (
                ctrl.config.signal_type == "top1_prob"
                and ctrl.config.mode == "chunk_rollback"
                and _src_hyp is not None
            ):
                _cl = getattr(_src_hyp, "confidence_list", None)
                _cei2 = getattr(_src_hyp, "ChunkEmissionIndex", None)
                if _cl and _cei2 is not None and _cei2.numel() > 0:
                    _cl = list(_cl)
                    _idx = [int(v) for v in _cei2.tolist()]
                    _M = min(len(_cl), len(_idx))
                    if _M > 0:
                        _tgt = max(_idx[:_M])
                        _chunk_confs = [_cl[i] for i in range(_M) if _idx[i] == _tgt]
                        if _chunk_confs:
                            if self.sr_cem_chunk_agg == "min":
                                _agg_t1 = min(_chunk_confs)
                            else:  # default mean
                                _agg_t1 = sum(_chunk_confs) / len(_chunk_confs)
                            try:
                                metrics.top1_prob_mean = float(_agg_t1)
                            except Exception:
                                pass

            future_avail = ct.current_chunk_idx - ch
            if ctrl.config.mode == "token_chunk_resume":
                # Branch D: the per-token hook already decided by stopping the
                # chunk at the first low-conf token. DEFER iff it fired and the
                # deep-deferral budget isn't spent; the look-ahead deepening
                # below (restore + re-encode at +1 nrc + re-decode) is shared
                # with B. Increment defer_count here (B does it in decide_chunk).
                _spent = ctrl.defer_count.get(ch, 0)
                if (getattr(self.beam_search, "_dyn_d_token_deferred", False)
                        and _spent < ctrl.config.max_future_chunks):
                    decision = DEFER
                    ctrl.defer_count[ch] = _spent + 1
                else:
                    decision = COMMIT
            elif getattr(self, "_oracle_ref_words", None) is not None:
                # ORACLE trigger (env ORACLE_REF_FILE): DEFER iff the chunk's
                # just-decoded words contain an error against the reference and
                # budget remains. Mirrors the token_chunk_resume bookkeeping
                # (decide_chunk is bypassed, so increment defer_count here).
                _spent = ctrl.defer_count.get(ch, 0)
                if (self._oracle_chunk_wrong(_src_hyp)
                        and _spent < ctrl.config.max_future_chunks):
                    decision = DEFER
                    ctrl.defer_count[ch] = _spent + 1
                else:
                    decision = COMMIT
            else:
                # Pass is_final=False so the controller never force-commits on the
                # final call by itself. At is_final all remaining audio is buffered,
                # so a deferring tail chunk should still DEEPEN (up to
                # min(mfc, remaining) == static's end taper) before committing; the
                # in-call loop below does that deepening and the final commit.
                decision = ctrl.decide_chunk(
                    chunk_idx=ch,
                    metrics=metrics,
                    future_available=future_avail,
                    is_final=False,
                )
            if decision == DEFER:
                _k_next = min(ctrl.defer_count.get(ch, 0), self.num_right_chunks)
                # Can we deepen RIGHT NOW? Only if the chunk needed for the next
                # defer level is already buffered (has arrived). At is_final all
                # remaining audio is buffered, so the tail keeps deepening here
                # too, up to min(mfc, remaining) == static nrc's end taper.
                _can_deepen = (
                    _k_next > _defer_k
                    and (ch + _k_next) <= ct.current_chunk_idx
                )
                if is_final and not _can_deepen:
                    # Final call, no further audio to deepen with: ACCEPT the
                    # decode just made (this chunk reached min(mfc, remaining)).
                    # Do NOT roll back - fall through to COMMIT below. This is
                    # what makes always-defer mfc=k reproduce static nrc=k's end
                    # taper instead of force-committing the tail at nrc0.
                    if (ctrl.config.mode == "token_chunk_resume"
                            and getattr(self.beam_search, "_dyn_d_token_deferred", False)):
                        # Branch D ONLY: B's decode above ran to completion, but
                        # D's per-token hook EARLY-STOPPED this tail chunk
                        # (truncated - dropped its tail tokens). Accepting that
                        # truncated decode deletes the tail. Re-decode the chunk
                        # FULLY with the hook disabled (force_commit) at its
                        # current level, then commit the complete decode.
                        self.beam_search.restore_state(snap)
                        self.beam_search._dyn_d_force_commit = True
                        self.beam_search._dyn_d_token_deferred = False
                        _sR = self.beam_search.cross_attn_num_right_chunks
                        self.beam_search.cross_attn_num_right_chunks = _defer_k
                        if getattr(self, "commit_stable_prefix", False):
                            # Forced re-decode must rebuild decoder KV from the
                            # restored (pre-chunk) prefix - mirror the DEFER
                            # path's clear; the un-cleared module cache still
                            # reflects the just-truncated decode (stale curlen).
                            _dec = self.beam_search.full_scorers.get("decoder")
                            if _dec is not None and getattr(_dec, "use_kvcache", False) \
                                    and hasattr(_dec, "_clear_kvcache"):
                                _dec._clear_kvcache()
                                _rh = self.beam_search.running_hyps
                                if _rh is not None and "decoder" in _rh.states:
                                    _rh.states["decoder"] = (
                                        [None] * len(_rh.states["decoder"])
                                    )
                            self.beam_search._forced_prefix = (
                                self._csp_forced_prefix(ch)
                            )
                        try:
                            nbest_hyps = self.beam_search(
                                x=dec_enc,
                                maxlenratio=self.maxlenratio,
                                minlenratio=self.minlenratio,
                                is_final=chunk_is_final_call,
                            )
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                        finally:
                            self.beam_search.cross_attn_num_right_chunks = _sR
                            if getattr(self, "commit_stable_prefix", False):
                                self.beam_search._forced_prefix = None
                    pass
                else:
                    # Roll back this chunk's tentative decode + release.
                    self.beam_search.restore_state(snap)
                    ct.next_decode_idx -= 1
                    self._last_enc_chunk_sent = saved_last_enc_chunk_sent
                    # Decoder KV cache lives ON the decoder module; clear stale
                    # entries and force a full rebuild on the next batch_score
                    # (else stepwise path vs empty k_cache → shape mismatch).
                    _dec = self.beam_search.full_scorers.get("decoder")
                    if _dec is not None and getattr(_dec, "use_kvcache", False) \
                            and hasattr(_dec, "_clear_kvcache"):
                        _dec._clear_kvcache()
                        _rh = self.beam_search.running_hyps
                        if _rh is not None and "decoder" in _rh.states:
                            _n = len(_rh.states["decoder"])
                            _rh.states["decoder"] = [None] * _n
                    if _can_deepen:
                        # In-call deep deferral: re-encode at the deeper level and
                        # re-process THIS chunk now (re-insert into the queue)
                        # instead of waiting for the next audio arrival, so a
                        # chunk reaches min(mfc, buffered_future) in ONE call.
                        # Without it, deferral advances 1 level per arrival and
                        # late chunks starve to nrc0 at is_final.
                        _enc_d = self._encode_features(
                            self.accumulated_features, accumulated_lengths,
                            position_offset=0, num_right_chunks_override=_k_next,
                        )
                        _ncc = _enc_d.shape[1] // chunk_size
                        for _j in range(ch, min(ch + _k_next + 1, _ncc)):
                            ct.encoder_chunks[_j] = _enc_d[
                                0, _j * chunk_size:(_j + 1) * chunk_size, :
                            ]
                        _ctc_d = self.asr_model.ctc.ctc_lo(
                            _enc_d[0].to(self.device)
                        )
                        full_logp_np = torch.log_softmax(
                            _ctc_d, dim=-1
                        ).detach().cpu().numpy()
                        _queue.insert(_qi, ch)
                        logging.info(
                            f"[dyn_fc Branch B] chunk={ch} DEFER->deepen in-call "
                            f"k={_defer_k}->{_k_next} (buffered); re-decode now"
                        )
                        continue
                    # Non-final and cannot deepen yet: wait for next arrival.
                    new_pending.append(ch)
                    release_blocked = True
                    logging.info(
                        f"[dyn_fc Branch B] chunk={ch} DEFER "
                        f"(margin={metrics.margin_mean:.3f}) -> rollback, "
                        f"defer_count={ctrl.defer_count.get(ch, 0)}"
                    )
                    continue

            # COMMIT - keep beam state changes.
            nbest_hyps_last = nbest_hyps
            ctrl.note_chunk_committed(ch)
            logging.debug(
                f"[dyn_fc Branch B] chunk={ch} COMMIT "
                f"(top1={metrics.top1_prob_mean:.3f} signal={ctrl.config.signal_type})"
            )

        self._dyn_pending = new_pending

        if is_final:
            if not nbest_hyps_last:
                # Force flush whatever is still pending: run beam search on
                # each remaining chunk without the defer gate.
                still_pending = []
                for ch in self._dyn_pending:
                    if ch == ct.next_decode_idx:
                        # Route through the dedup protocol: send only chunks
                        # not yet in encbuffer (ch may already have gone out
                        # as a KV-extra), fill any gap, and keep
                        # _last_enc_chunk_sent consistent for later sends.
                        first_new = self._last_enc_chunk_sent + 1
                        if ch >= first_new:
                            dec_enc = torch.cat(
                                [ct.encoder_chunks[i] for i in range(first_new, ch + 1)],
                                dim=0,
                            )
                            self._last_enc_chunk_sent = ch
                        else:
                            _ref = ct.encoder_chunks[ch]
                            dec_enc = _ref.new_zeros((0, _ref.shape[-1]))
                        ct.next_decode_idx += 1
                        try:
                            nbest_hyps_last = self.beam_search(
                                x=dec_enc,
                                maxlenratio=self.maxlenratio,
                                minlenratio=self.minlenratio,
                                is_final=True,
                            )
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            nbest_hyps_last = []
                    else:
                        still_pending.append(ch)
                self._dyn_pending = still_pending
            if not nbest_hyps_last and ct.encoder_chunks:
                # FIX (long-form empty-flush): no candidate carried a final
                # beam call — e.g. a virtual-final flush (speech[0:0]) with
                # nothing pending yields candidates == [], so beam_search is
                # never called with is_final and the segment's ACCUMULATED
                # hypothesis would be silently discarded by reset() below
                # (observed: whole ~13 s segments deleted on LibriSpeech-Long,
                # 9/723 flushes). End the utterance explicitly with a zero-row
                # final call over the existing encbuffer, mirroring the static
                # path's empty-tensor final.
                _ref = ct.encoder_chunks[-1]
                _saved_cs_fb = None
                if getattr(self, "unmask_xattn_at_final", False):
                    _saved_cs_fb = self.beam_search.cross_attn_chunk_size
                    self.beam_search.cross_attn_chunk_size = 0
                try:
                    nbest_hyps_last = self.beam_search(
                        x=_ref.new_zeros((0, _ref.shape[-1])),
                        maxlenratio=self.maxlenratio,
                        minlenratio=self.minlenratio,
                        is_final=True,
                    )
                    logging.info(
                        "[dyn_fc Branch B] FALLBACK final beam call "
                        f"(no pending at is_final): {len(nbest_hyps_last)} hyps"
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    nbest_hyps_last = []
                finally:
                    if _saved_cs_fb is not None:
                        self.beam_search.cross_attn_chunk_size = _saved_cs_fb
            if ct is not None and nbest_hyps_last:
                self._populate_chunk_tracking(nbest_hyps_last)
            self._maybe_dump_encoder_chunks(tag="dynamic_chunk_rollback")
            ret = self.assemble_hyps(nbest_hyps_last)
            self._last_xattn_buffer = list(
                getattr(self.beam_search, "xattn_dump_buffer", [])
            )
            logging.info(f"[dyn_fc Branch B] summary: {ctrl.summary()}")
            self.reset()
            return ret
        return []

    # ------------------------------------------------------------------
    # Branch A: token-level stop and resume
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _branch_token_resume(
        self,
        accumulated_lengths: torch.Tensor,
        is_final: bool,
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis, dict]]:
        """Branch A: signal evaluated *per beam-search step* inside
        :class:`BatchBeamSearchOnline`. On DEFER, the existing
        ``stop_reason`` mechanism in :meth:`process_one_block` breaks the
        per-token loop while preserving ``running_hyps`` and
        ``process_idx`` - so the next call (with one more chunk of
        encoder context) resumes from the same token step.

        The outer flow mirrors the static streaming path; the only
        addition is publishing the current ``future_available`` count on
        the beam_search so the per-step controller hook can read it.
        """
        ctrl = self.dynamic_fc_controller
        ct = self.chunk_tracker
        chunk_size = self.chunk_size
        sf = self.subsampling_factor

        enc = self._encode_features(
            self.accumulated_features,
            accumulated_lengths,
            position_offset=0,
        )
        if is_final:
            self._capture_ctc_logprobs(enc[0])

        num_frames = enc.shape[1]
        num_complete_chunks = num_frames // chunk_size
        prev_num_chunks = ct.current_chunk_idx + 1
        new_chunks = num_complete_chunks - prev_num_chunks

        decode_ready_chunks: List[int] = []
        for i in range(new_chunks):
            chunk_idx = prev_num_chunks + i
            start_frame = chunk_idx * chunk_size
            end_frame = start_frame + chunk_size
            enc_chunk = enc[0, start_frame:end_frame, :]
            raw_start = chunk_idx * chunk_size * sf
            raw_end = raw_start + chunk_size * sf
            raw_chunk = self.accumulated_features[0, raw_start:raw_end, :]
            ct.add_raw_chunk(raw_chunk)
            decode_idx = ct.add_chunk(enc_chunk)
            if decode_idx is not None:
                decode_ready_chunks.append(decode_idx)
        # Stage A overwrite for prior chunks (matches static path)
        if new_chunks > 0 and self.num_right_chunks > 0:
            for past_idx in range(
                max(0, prev_num_chunks - self.num_right_chunks),
                prev_num_chunks,
            ):
                start = past_idx * chunk_size
                end = start + chunk_size
                ct.encoder_chunks[past_idx] = enc[0, start:end, :]
        if is_final:
            partial_frames = num_frames % chunk_size
            if partial_frames > 0:
                partial_start = num_complete_chunks * chunk_size
                partial_chunk = enc[0, partial_start:, :]
                raw_start = num_complete_chunks * chunk_size * sf
                raw_partial = self.accumulated_features[0, raw_start:, :]
                ct.add_raw_chunk(raw_partial)
                decode_idx = ct.add_chunk(partial_chunk)
                if decode_idx is not None:
                    decode_ready_chunks.append(decode_idx)
            decode_ready_chunks.extend(ct.flush_remaining())

        if not decode_ready_chunks and not is_final:
            return []

        # Build encoder slice (same as static path).
        if decode_ready_chunks:
            max_ctx_idx = min(
                max(decode_ready_chunks), ct.current_chunk_idx
            )
            extra_right = max(
                0, getattr(self.beam_search, "cross_attn_num_right_chunks", 0)
            )
            max_kv_idx = min(
                max_ctx_idx + extra_right, len(ct.encoder_chunks) - 1
            )
            first_new = self._last_enc_chunk_sent + 1
            ready_chunks = [
                ct.encoder_chunks[idx]
                for idx in range(first_new, max_kv_idx + 1)
            ]
            dec_enc = (
                torch.cat(ready_chunks, dim=0)
                if ready_chunks
                # Already sent previously - zero-row avoids encbuffer dup.
                else ct.encoder_chunks[-1].new_zeros(
                    (0, ct.encoder_chunks[-1].shape[-1])
                )
            )
            self._last_enc_chunk_sent = max_kv_idx
        else:
            dec_enc = ct.encoder_chunks[-1].new_zeros(
                (0, ct.encoder_chunks[-1].shape[-1])
            )

        # Publish future_available so the per-step hook can read it.
        # For Branch A we use processed_block as a proxy for "how much
        # encoder context the model has accumulated" - the per-step
        # controller logs this for diagnostics.
        self.beam_search._dyn_fc_future_available = int(
            getattr(self.beam_search, "processed_block", 0)
        )

        _saved_cs = None
        if is_final and getattr(self, "unmask_xattn_at_final", False):
            _saved_cs = self.beam_search.cross_attn_chunk_size
            self.beam_search.cross_attn_chunk_size = 0
        try:
            nbest_hyps = self.beam_search(
                x=dec_enc,
                maxlenratio=self.maxlenratio,
                minlenratio=self.minlenratio,
                is_final=is_final,
            )
        except torch.cuda.OutOfMemoryError as e:
            logging.warning(
                f"Beam search OOM (token_resume branch): {e}"
            )
            nbest_hyps = []
            torch.cuda.empty_cache()
        finally:
            if _saved_cs is not None:
                self.beam_search.cross_attn_chunk_size = _saved_cs

        if is_final:
            if ct is not None and nbest_hyps:
                self._populate_chunk_tracking(nbest_hyps)
            self._maybe_dump_encoder_chunks(tag="dynamic_token_resume")
            ret = self.assemble_hyps(nbest_hyps)
            self._last_xattn_buffer = list(
                getattr(self.beam_search, "xattn_dump_buffer", [])
            )
            logging.info(f"[dyn_fc Branch A] summary: {ctrl.summary()}")
            self.reset()
            return ret
        return []

    def _capture_ctc_logprobs(self, enc: torch.Tensor):
        """Compute per-frame CTC log-probs and buffer them for the inference loop.

        Args:
            enc: Final encoder output for the utterance, shape (T, D).
        """
        if enc is None or enc.numel() == 0:
            self._last_ctc_logprobs = None
            return
        with torch.no_grad():
            # ctc.ctc_lo: linear (D -> V); log_softmax over vocab.
            logits = self.asr_model.ctc.ctc_lo(enc.to(self.device))
            logp = torch.log_softmax(logits, dim=-1)
        self._last_ctc_logprobs = logp.detach().cpu().numpy()

    def _maybe_dump_encoder_chunks(self, tag: str):
        """Save chunk_tracker.encoder_chunks to a pickle when ESPNET_DUMP_ENCODER_CHUNKS
        env var is set. Used to diff Stage B vs full-recompute numerically.
        """
        import os
        # Encoder-context probe (runs BEFORE the ESPNET_DUMP_ENCODER_CHUNKS
        # early-return): save per-utt per-chunk mean-pooled RAW encoder output to
        # <feat_dump_dir>/enc/<utt>.npy when DUMP_HIDDEN_STATES=1. Last call per
        # utt wins (= full utt's encoder chunks).
        if (os.environ.get("DUMP_HIDDEN_STATES") == "1"
                and getattr(self, "sr_cem_feat_dump_dir", None)
                and self.chunk_tracker is not None):
            try:
                import numpy as _npe
                _ec = self.chunk_tracker.encoder_chunks
                _uid = getattr(self.beam_search, "_sr_cem_utt_id", "")
                if _ec and _uid:
                    _ed = os.path.join(self.sr_cem_feat_dump_dir, "enc")
                    os.makedirs(_ed, exist_ok=True)
                    _npe.save(
                        os.path.join(_ed, f"{_uid}.npy"),
                        _npe.stack([t.detach().cpu().numpy().mean(0) for t in _ec]))
            except Exception:
                pass
        dump_path = os.environ.get("ESPNET_DUMP_ENCODER_CHUNKS", "")
        if not dump_path or self.chunk_tracker is None:
            return
        try:
            chunks = [t.detach().cpu() for t in self.chunk_tracker.encoder_chunks]
        except Exception:
            return
        suffix = f".{tag}.pt"
        if not dump_path.endswith(".pt"):
            dump_path = dump_path + suffix
        else:
            dump_path = dump_path[:-3] + suffix
        torch.save({"encoder_chunks": chunks, "tag": tag}, dump_path)
        logging.info(f"[DEBUG] Saved {len(chunks)} encoder chunks to {dump_path}")

    def _ensure_offline_beam(self):
        """Build (once) an offline BatchBeamSearch reusing the streaming
        scorers/weights. Subclassed so that when ``hyp_primer`` is set, the CTC
        prefix-score state is advanced over the primer in init_hyp (cheap DP).
        The decoder needs no special handling: its first ``batch_score``
        processes the WHOLE primer in ONE call (with use_kvcache the prefix is
        cached in that single pass), so the beam only does the tail steps -
        avoiding the O(L)-step forced-prefix re-decode."""
        if getattr(self, "_offline_beam", None) is None:
            from espnet.nets.batch_beam_search import BatchBeamSearch

            class _PrimedBatchBeamSearch(BatchBeamSearch):
                def init_hyp(self, x):
                    import os as _os_d
                    # SEED-FULL-BEAM path: start from the whole committed-prefix
                    # beam (all hyps + scores) captured before the last chunk,
                    # so the beam re-ranks the prefixes while decoding the last
                    # chunk. Set up the scorer states over the full memory x,
                    # extend each hyp's CTC state to T_total, and reset decoder
                    # states (None → rebuilt on the first batch_score).
                    seed = getattr(self, "_beam_seed", None)
                    if seed is not None:
                        import copy as _cp
                        sh = _cp.deepcopy(seed)
                        for _k, _d in self.scorers.items():
                            _d.batch_init_state(x)
                        _ctc = self.part_scorers.get("ctc")
                        if (_ctc is not None and "ctc" in sh.states
                                and sh.states["ctc"]):
                            try:
                                sh.states["ctc"] = [
                                    _ctc.impl.extend_state(s)
                                    for s in sh.states["ctc"]
                                ]
                            except Exception as e:
                                logging.warning(
                                    f"[seed] ctc extend failed: {e!r}"
                                )
                        if "decoder" in sh.states:
                            sh.states["decoder"] = [
                                None for _ in range(int(sh.yseq.shape[0]))
                            ]
                        return sh
                    hyps = BatchBeamSearch.init_hyp(self, x)
                    primer = self.hyp_primer
                    if not primer or len(primer) <= 1:
                        return hyps
                    ctc = self.part_scorers.get("ctc")
                    if ctc is None:
                        return hyps
                    state = None
                    # REUSE path: inject the streaming committed-prefix CTC
                    # prefix-score state (captured before the final beam pass)
                    # instead of re-advancing it token-by-token. The captured r
                    # is sized to the streaming buffer (T_prefix); extend_state
                    # pads it to the offline memory length (T_total) using the
                    # final chunk's frames. The streaming state for yseq=
                    # [sos]+prefix already corresponds to fwd-probs of
                    # [sos]+prefix[:-1] (the DP state lags yseq by one), which is
                    # exactly r_prev for the first beam step over the primer.
                    reuse = getattr(self, "_reuse_ctc_state", None)
                    if reuse is not None:
                        try:
                            state = ctc.impl.extend_state(reuse)
                        except Exception as e:
                            logging.warning(
                                f"[resume_reuse] ctc extend failed: {e!r}; "
                                "falling back to re-advance"
                            )
                            state = None
                    if state is None:
                        # FALLBACK: advance CTC state over primer[:-1]
                        # (= sos + prefix[:-1]) token-by-token; the first beam
                        # step then scores extending the full primer.
                        ys = [int(primer[0])]
                        for tok in [int(t) for t in primer[1:-1]]:
                            y_t = torch.tensor(
                                [ys], dtype=torch.long, device=x.device
                            )
                            # Score ONLY the prefix token (1 candidate) so the
                            # CTC DP is cheap; the full vocab here is wasteful.
                            ids_t = torch.tensor(
                                [[tok]], dtype=torch.long, device=x.device
                            )
                            _sc, new_state = ctc.batch_score_partial(
                                y_t, ids_t, [state], x
                            )
                            state = ctc.select_state(new_state, 0, tok)
                            ys.append(tok)
                    for i in range(len(hyps.states["ctc"])):
                        hyps.states["ctc"][i] = state
                    # DEDUP (env RESUME_DEDUP_PRIMER, default OFF): correctness-
                    # neutral but measured to give no speedup (the beam_size
                    # primer copies are batched, not compute-bound), so off.
                    if _os_d.environ.get("RESUME_DEDUP_PRIMER") == "1":
                        hyps = self._batch_select(hyps, [0])
                    return hyps

            bs = self.beam_search
            scorers = {**bs.full_scorers, **bs.part_scorers}
            offline = _PrimedBatchBeamSearch(
                scorers=scorers,
                weights=bs.weights,
                beam_size=bs.beam_size,
                vocab_size=bs.n_vocab,
                sos=bs.sos,
                eos=bs.eos,
                token_list=bs.token_list,
                pre_beam_score_key=getattr(bs, "pre_beam_score_key", None),
                normalize_length=getattr(bs, "normalize_length", False),
            )
            offline.to(device=self.device).eval()
            self._offline_beam = offline
        return self._offline_beam

    def _resume_offline_at_final(self, nbest_hyps):
        """Resume-from-prefix fix: keep the committed pre-final-chunk tokens as
        a fixed prefix and re-decode only the final chunk with the offline beam
        over the (good) streaming encoder. Genuinely streaming (no re-encode,
        prefix kept). Fixes final-chunk drift/loops/truncation; interior errors
        stay frozen in the prefix.
        """
        if (not nbest_hyps or self.chunk_tracker is None
                or not self.chunk_tracker.encoder_chunks):
            return nbest_hyps
        top = nbest_hyps[0]
        ys = top.yseq
        cei = getattr(top, "ChunkEmissionIndex", None)
        if cei is None or cei.numel() == 0 or ys.numel() < 2:
            return nbest_hyps
        final = int(cei.max().item())
        n = min(ys.numel() - 1, cei.numel())
        # prefix = committed tokens (after sos) emitted before the final chunk
        prefix = [int(ys[i + 1].item()) for i in range(n)
                  if int(cei[i].item()) < final]
        memory = torch.cat(
            [c for c in self.chunk_tracker.encoder_chunks], dim=0
        )  # (T, D)
        self._ensure_offline_beam()
        _dec = self.beam_search.full_scorers.get("decoder")
        _saved_kv = getattr(_dec, "use_kvcache", None) if _dec is not None else None
        # Mode (env RESUME_MODE): "primed" (default, efficient) processes the
        # prefix in ONE decoder call via hyp_primer + KV cache and beam-decodes
        # only the tail; "forced" is the O(L)-step forced-prefix baseline.
        import os as _os_r
        _mode = _os_r.environ.get("RESUME_MODE", "primed")
        if _dec is not None and _saved_kv:
            _dec._clear_kvcache()
            if _mode == "forced":
                _dec.use_kvcache = False  # baseline: recompute fresh each step
            # primed: keep use_kvcache=True so the primer is cached in one pass
        try:
            if _mode == "forced":
                self._offline_beam._forced_prefix = prefix
                new = self._offline_beam(
                    x=memory, maxlenratio=self.maxlenratio,
                    minlenratio=self.minlenratio,
                )
                self._offline_beam._forced_prefix = None
            else:
                # Primed: decoder processes [sos]+prefix in one batch_score
                # (cache populated); CTC state injected/advanced in init_hyp;
                # beam runs only the final-chunk tail.
                # REUSE: if the streaming committed-prefix CTC state was
                # snapshotted and its tokens match this prefix, inject it
                # (skips the O(L) per-token CTC re-advance). Disable via
                # RESUME_REUSE_CTC=0.
                if _os_r.environ.get("RESUME_REUSE_CTC", "1") != "0":
                    self._offline_beam._reuse_ctc_state = \
                        self._match_ctc_snapshot(prefix)
                else:
                    self._offline_beam._reuse_ctc_state = None
                self._offline_beam.set_hyp_primer(
                    [self.asr_model.sos] + prefix
                )
                new = self._offline_beam(
                    x=memory, maxlenratio=self.maxlenratio,
                    minlenratio=self.minlenratio,
                )
                self._offline_beam.set_hyp_primer(None)
                self._offline_beam._reuse_ctc_state = None
            if new:
                nbest_hyps = new
        except Exception as e:
            logging.warning(f"[resume_offline] failed: {e!r}")
            self._offline_beam._forced_prefix = None
            self._offline_beam.set_hyp_primer(None)
            self._offline_beam._reuse_ctc_state = None
        finally:
            if _dec is not None and _saved_kv:
                _dec.use_kvcache = _saved_kv
        return nbest_hyps

    def _offline_decode_last_chunk(self):
        """Decode the LAST chunk ONCE via the offline beam - NO streaming
        last-chunk decode. The committed prefix is taken from the snapshot
        captured BEFORE the last chunk (the pre-last-chunk top hyp); all its
        tokens are frozen and the offline beam re-decodes only the final chunk
        over the full encoder. Reuses the snapshot's own CTC prefix state.
        Returns nbest_hyps, or [] to signal the caller to fall back to the
        streaming decode (e.g. no committed prefix yet)."""
        snap = getattr(self, "_ctc_prefix_snapshot", None)
        if (not snap or not snap.get("yseqs")
                or self.chunk_tracker is None
                or not self.chunk_tracker.encoder_chunks):
            return []
        yseq = snap["yseqs"][0]
        L = snap["lengths"][0]
        prefix = [int(yseq[t].item()) for t in range(1, L)]  # drop sos
        if not prefix:
            return []
        memory = torch.cat(
            [c for c in self.chunk_tracker.encoder_chunks], dim=0
        )
        self._ensure_offline_beam()
        _dec = self.beam_search.full_scorers.get("decoder")
        _saved_kv = getattr(_dec, "use_kvcache", None) if _dec is not None else None
        import os as _os_o
        if _dec is not None and _saved_kv:
            _dec._clear_kvcache()
        new = []
        _full = getattr(self, "_full_beam_snapshot", None)
        _seed_full = (
            _os_o.environ.get("RESUME_SEED_FULL_BEAM", "1") != "0"
            and _full is not None
        )
        try:
            if _seed_full:
                # Seed the offline beam with the WHOLE committed-prefix beam
                # (all hyps + scores). It decodes the last chunk AND re-ranks
                # the prefixes in one clean pass - recovering the re-ranking the
                # double-decode gets, without decoding the last chunk twice.
                self._offline_beam._beam_seed = _full
                new = self._offline_beam(
                    x=memory, maxlenratio=self.maxlenratio,
                    minlenratio=self.minlenratio,
                )
                self._offline_beam._beam_seed = None
            else:
                # top-1 primer (old): freeze only the snapshot's top hyp.
                if _os_o.environ.get("RESUME_REUSE_CTC", "1") != "0":
                    self._offline_beam._reuse_ctc_state = snap["ctc"][0]
                else:
                    self._offline_beam._reuse_ctc_state = None
                self._offline_beam.set_hyp_primer(
                    [self.asr_model.sos] + prefix
                )
                new = self._offline_beam(
                    x=memory, maxlenratio=self.maxlenratio,
                    minlenratio=self.minlenratio,
                )
                self._offline_beam.set_hyp_primer(None)
                self._offline_beam._reuse_ctc_state = None
            if new:
                # yseq = sos + prefix + tail + eos → tail = numel - 2 - prefix
                logging.info(
                    f"[lc_stats] prefix={len(prefix)} "
                    f"yseq={int(new[0].yseq.numel())} "
                    f"tail={int(new[0].yseq.numel()) - 2 - len(prefix)}"
                )
        except Exception as e:
            logging.warning(f"[offline_last_chunk] failed: {e!r}")
            self._offline_beam.set_hyp_primer(None)
            self._offline_beam._reuse_ctc_state = None
            self._offline_beam._beam_seed = None
            new = []
        finally:
            if _dec is not None and _saved_kv:
                _dec.use_kvcache = _saved_kv
        return new if new else []

    def _match_ctc_snapshot(self, prefix):
        """Find the committed-prefix CTC state captured before the final beam
        pass whose token sequence matches ``prefix``. Returns the per-hyp CTC
        state (r sized to the streaming buffer; extended to T_total later in
        init_hyp) or None if no exact match (then init_hyp re-advances)."""
        snap = getattr(self, "_ctc_prefix_snapshot", None)
        if not snap:
            return None
        target = [int(t) for t in prefix]
        for k, yseq in enumerate(snap["yseqs"]):
            L = snap["lengths"][k]
            toks = [int(yseq[t].item()) for t in range(1, L)]  # skip sos
            if toks == target:
                logging.info(
                    f"[resume_reuse] CTC snapshot HIT (beam {k}, "
                    f"prefix_len={len(target)}) - skipping re-advance"
                )
                return snap["ctc"][k]
        logging.info(
            f"[resume_reuse] CTC snapshot MISS (prefix_len={len(target)}, "
            f"{len(snap['yseqs'])} snapshot hyps) - re-advancing"
        )
        return None

    def _snapshot_ctc_prefix(self):
        """Snapshot the online beam's committed-prefix state (yseq + CTC
        prefix-score states) BEFORE the final beam pass, so the resume can
        reuse the CTC state instead of re-advancing it. Captured per-hyp; the
        resume matches by token sequence (beam reordering safe)."""
        self._ctc_prefix_snapshot = None
        rh = getattr(self.beam_search, "running_hyps", None)
        if rh is None or not hasattr(rh, "yseq") or rh.yseq.numel() == 0:
            return
        ctc_states = rh.states.get("ctc") if hasattr(rh, "states") else None
        if not ctc_states:
            return
        import copy as _copy
        nb = rh.yseq.shape[0]
        self._ctc_prefix_snapshot = {
            "yseqs": [rh.yseq[i].detach().clone() for i in range(nb)],
            "lengths": [int(rh.length[i].item()) for i in range(nb)],
            "ctc": [_copy.deepcopy(ctc_states[i]) for i in range(nb)],
        }
        # Also snapshot the FULL committed-prefix beam (all hyps + scores +
        # states) so the offline last-chunk decode can seed from the whole beam
        # and re-rank the prefixes (not just freeze the top-1). deepcopy so the
        # subsequent streaming pass can't mutate it.
        self._full_beam_snapshot = _copy.deepcopy(rh)

    def _streaming_b_call(
        self,
        is_final: bool,
        accumulated_lengths: torch.Tensor,
    ):
        """Stage B streaming inference: incremental encoder KV cache.

        Per-call flow:
          1. Run encoder subsampling on full accumulated raw features (cheap,
             keeps subsampled output numerically identical to training).
          2. Determine how many complete subsampled chunks we now have.
          3. While we can finalize (have F+1 tail chunks OR is_final with any
             remaining): slice the tail, call ``encoder.forward_streaming`` to
             produce the tail output, extract the oldest-tail chunk's C=F slice
             as the released finalized encoding, append to ``encoder_chunks``.
          4. Release newly finalized chunks to beam search; on is_final assemble
             and return.
        """
        encoder = self.asr_model.encoder
        F = self.num_right_chunks
        S = self.chunk_size
        sf = self.subsampling_factor

        # Long-form memory bound: trim accumulated_features so subsampling
        # never sees more than ``F + 1 + LOOKBACK_CHUNKS`` chunks of raw
        # log-mel audio. K/V cache already holds past info; the only reason
        # to keep raw audio around is conv-subsample lookback for the tail.
        # ``LOOKBACK_CHUNKS=2`` gives plenty of conv RF (a few raw frames).
        LOOKBACK_CHUNKS = 2
        n_fin_abs = encoder.n_finalized_chunks
        target_start_chunk = max(0, n_fin_abs - LOOKBACK_CHUNKS)
        drop_chunks = target_start_chunk - self._b_sub_chunks_dropped
        if drop_chunks > 0:
            drop_raw = drop_chunks * S * sf
            if drop_raw < self.accumulated_features.size(1):
                self.accumulated_features = self.accumulated_features[:, drop_raw:, :]
                self._b_sub_chunks_dropped += drop_chunks
                accumulated_lengths = torch.tensor(
                    [self.accumulated_features.size(1)],
                    dtype=torch.long,
                    device=self.accumulated_features.device,
                )

        # Step 1: Subsample only the kept raw window (now bounded ~F+3 chunks).
        xs_pad_full = encoder.subsample_only(
            self.accumulated_features, accumulated_lengths
        )  # (1, T_sub_kept, D)

        # Step 2: Determine complete chunks. On is_final, the tail's final chunk
        # may be partial. Do NOT zero-pad it at the encoder input: input-level
        # zeros pass through pointwise_conv1 + GLU and become GLU(bias) != 0 at
        # the DCConv depthwise taps, poisoning the terminal chunk's last real
        # frames (whose within-chunk conv window reads those tail positions).
        # This is the right-boundary mirror of the first-call prepend bug fixed
        # in ConvolutionModule.forward_streaming. Feed the true partial tail
        # instead: the conv's internal post-GLU F.pad (align_pad) then supplies
        # exactly the post-GLU zeros the full-recompute (Stage A) path uses, and
        # the self-attn mask naturally has no pad keys (also matching Stage A).
        # ``partial_len`` is the terminal chunk's real-frame count (0 when the
        # tail is chunk-aligned); that chunk releases ``partial_len`` frames
        # instead of a full ``S`` slice.
        T_sub = xs_pad_full.size(1)
        partial_len = (T_sub % S) if is_final else 0
        if is_final:
            # Round up so the partial terminal chunk is counted and finalized.
            num_sub_chunks_local = (T_sub + S - 1) // S
        else:
            num_sub_chunks_local = T_sub // S
        sub_chunks_dropped = self._b_sub_chunks_dropped
        num_sub_chunks = num_sub_chunks_local + sub_chunks_dropped  # absolute count
        n_fin = encoder.n_finalized_chunks

        # Step 3: Finalization loop.
        mask_config = ChunkedMaskConfig(
            chunk_size=S,
            num_left_chunks=self.num_left_chunks,
            num_right_chunks=F,
            num_global_tokens=self.num_global_tokens,
            use_asymmetric_mask=True,
            full_attention=False,
        )

        decode_ready_chunks = []
        while True:
            remaining = num_sub_chunks - n_fin
            if remaining <= 0:
                break
            if remaining < F + 1 and not is_final:
                break

            # Tail chunk count: full F+1 except at is_final end where we flush
            # whatever chunks are left (1..F available).
            tail_n_chunks = min(F + 1, remaining)
            # Map absolute chunk indices to local indices into the trimmed
            # xs_pad_full (which starts at chunk ``sub_chunks_dropped``).
            local_n_fin = n_fin - sub_chunks_dropped
            tail_start = local_n_fin * S
            tail_end = (local_n_fin + tail_n_chunks) * S
            xs_pad_tail = xs_pad_full[:, tail_start:tail_end, :]

            # No is_final zero-pad is applied at the encoder input any more
            # (see Step 2), so the tail carries only real frames and there are
            # no trailing pad keys to mask. (``mask_streaming_pad`` is therefore
            # inert for Stage B; kept as a CLI flag for the full-recompute sim.)
            n_pad_tail = 0

            # Run streaming encoder over the tail.
            enc_tail = encoder.forward_streaming(
                xs_pad_tail,
                chunk_size=S,
                num_right_chunks=F,
                num_left_chunks=self.num_left_chunks,
                chunked_mask_config=mask_config,
                n_pad_tail=n_pad_tail,
            )  # (1, T_tail, C, D)

            # Released chunk = oldest tail chunk (positions [0, S)). The
            # asymmetric age mask defines c_q = num_right_chunks as the slot
            # that sees the FULL right context (matches training's
            # right-context-aware encoding consumed by the decoder). c_q = 0
            # is the causal-only slot; picking C=0 would give the decoder an
            # encoding with no right context, undoing the point of Stage B.
            # Every finalized chunk is full except the terminal chunk on
            # is_final, which has ``partial_len`` real frames. The conv/self-attn
            # already produced exactly those frames (matching Stage A), so no
            # pad-frame drop is needed; release the real-frame count directly.
            released_len = (
                partial_len
                if (partial_len > 0 and n_fin == num_sub_chunks - 1)
                else S
            )
            finalized_chunk_enc = enc_tail[0, :released_len, F, :]  # (released_len, D)

            # Long-form virtual finals: track the trailing CTC-blank run on
            # the finalized chunk encoding (the C=F slice the decoder
            # consumes). Mirrors the full-recompute path; this is the branch
            # taken when use_encoder_self_kvcache=true (Stage B).
            if self.virtual_final_blank_frames > 0:
                _am = self.asr_model.ctc.ctc_lo(finalized_chunk_enc).argmax(-1)
                _nb = (_am != 0).nonzero()
                if _nb.numel() == 0:
                    self.trailing_blank_frames += _am.numel()
                else:
                    self.trailing_blank_frames = (
                        _am.numel() - 1 - int(_nb[-1].item())
                    )

            # Bookkeeping: push into chunk_tracker, bypass its delay logic (we
            # already waited for right context inside the encoder).
            # Raw-frame slicing is also LOCAL to the trimmed accumulated_features:
            # absolute raw start = n_fin * S * sf  →  local start = local_n_fin * S * sf
            local_raw_start = local_n_fin * S * sf
            local_raw_end = min(
                (local_n_fin + 1) * S * sf, self.accumulated_features.shape[1]
            )
            raw_chunk = self.accumulated_features[0, local_raw_start:local_raw_end, :]
            self.chunk_tracker.add_raw_chunk(raw_chunk)
            self.chunk_tracker.encoder_chunks.append(finalized_chunk_enc)
            self.chunk_tracker.current_chunk_idx = (
                len(self.chunk_tracker.encoder_chunks) - 1
            )
            self.chunk_tracker.next_decode_idx = len(self.chunk_tracker.encoder_chunks)

            decode_ready_chunks.append(n_fin)
            n_fin = encoder.n_finalized_chunks  # was incremented inside forward_streaming

        # Snapshot the committed-prefix beam state (CTC prefix-score + yseq)
        # BEFORE the final beam pass drifts it, so _resume_offline_at_final can
        # reuse the CTC state instead of re-advancing it token-by-token. At
        # this point self.beam_search.running_hyps is the state carried over
        # from the previous (non-final) chunk = the committed prefix.
        if (is_final and not self.defer_decoding_until_last_chunk
                and getattr(self, "resume_offline_at_final", False)):
            self._snapshot_ctc_prefix()

        # DECODE-ONCE (env RESUME_DECODE_LAST_ONCE, default on): at the final
        # chunk, REPLACE the streaming last-chunk decode with the offline decode
        # - do NOT decode the last chunk twice. Run the offline pass straight
        # from the pre-last-chunk snapshot prefix. Falls back to the streaming
        # path if there's no committed prefix yet or the offline pass fails.
        import os as _os_dc
        _decode_once = (
            is_final
            and not self.defer_decoding_until_last_chunk
            and getattr(self, "resume_offline_at_final", False)
            and self._last_enc_chunk_sent >= 0
            and _os_dc.environ.get("RESUME_DECODE_LAST_ONCE", "1") != "0"
        )
        _once_hyps = None
        if _decode_once:
            _once_hyps = self._offline_decode_last_chunk()
            if not _once_hyps:
                _decode_once = False  # no prefix / failed → use streaming

        # Defer-decode: skip per-chunk beam search; on is_final run once over
        # the full concatenated encoder output with no per-token cross-attn
        # mask. Same logic as the full-recompute branch.
        if self.defer_decoding_until_last_chunk:
            if not is_final:
                return []
            dec_enc = torch.cat(self.chunk_tracker.encoder_chunks, dim=0)
            _saved_cs = self.beam_search.cross_attn_chunk_size
            self.beam_search.cross_attn_chunk_size = 0
            try:
                nbest_hyps = self.beam_search(
                    x=dec_enc,
                    maxlenratio=self.maxlenratio,
                    minlenratio=self.minlenratio,
                    is_final=True,
                )
            finally:
                self.beam_search.cross_attn_chunk_size = _saved_cs
        # Decode-once: last chunk via OFFLINE only (no streaming last-chunk
        # decode). nbest_hyps already produced from the snapshot prefix.
        elif _decode_once:
            nbest_hyps = _once_hyps
        # Step 4: Release to beam search.
        elif decode_ready_chunks:
            max_ctx_idx = max(decode_ready_chunks)
            first_new = self._last_enc_chunk_sent + 1
            ready_chunks = [
                self.chunk_tracker.encoder_chunks[idx]
                for idx in range(first_new, max_ctx_idx + 1)
            ]
            dec_enc = torch.cat(ready_chunks, dim=0)
            self._last_enc_chunk_sent = max_ctx_idx

            # PROFILE (env RESUME_PROFILE): time the per-chunk streaming beam
            # pass so a non-last chunk (is_final=False) can be compared against
            # the last-chunk fix. cuda.synchronize for accurate GPU timing.
            import os as _os_sp, time as _time_sp
            _sp = _os_sp.environ.get("RESUME_PROFILE")
            if _sp and torch.cuda.is_available():
                torch.cuda.synchronize()
            _sp_t = _time_sp.perf_counter()
            # F4-fix parity with the static path (:1241-1243): at is_final,
            # disable the per-token cross-attn mask so all prefix rows get
            # full encoder attention during the final pass. Without this,
            # Stage B's final block keeps the restricted mask and reproduces
            # the F4 tail pathology (repetition loops / final-word drops)
            # that unmask_xattn_at_final exists to suppress.
            _saved_cs_s4 = None
            if is_final and getattr(self, "unmask_xattn_at_final", False):
                _saved_cs_s4 = self.beam_search.cross_attn_chunk_size
                self.beam_search.cross_attn_chunk_size = 0
            try:
                nbest_hyps = self.beam_search(
                    x=dec_enc,
                    maxlenratio=self.maxlenratio,
                    minlenratio=self.minlenratio,
                    is_final=is_final,
                )
            finally:
                if _saved_cs_s4 is not None:
                    self.beam_search.cross_attn_chunk_size = _saved_cs_s4
            if _sp:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                logging.info(
                    f"[STREAM_PROF] chunk_decode="
                    f"{_time_sp.perf_counter()-_sp_t:.3f}s is_final={is_final} "
                    f"nchunks_sent={len(ready_chunks)}"
                )
        elif is_final and self._last_enc_chunk_sent >= 0:
            # is_final with nothing new to finalize - run the final beam pass
            # on a zero-row tensor so EOS can be emitted without duplicating
            # the already-sent last chunk in encbuffer.
            _ref = self.chunk_tracker.encoder_chunks[-1]
            dec_enc = _ref.new_zeros((0, _ref.shape[-1]))
            # Same F4-fix parity as the Step 4 branch above.
            _saved_cs_zr = None
            if getattr(self, "unmask_xattn_at_final", False):
                _saved_cs_zr = self.beam_search.cross_attn_chunk_size
                self.beam_search.cross_attn_chunk_size = 0
            try:
                nbest_hyps = self.beam_search(
                    x=dec_enc,
                    maxlenratio=self.maxlenratio,
                    minlenratio=self.minlenratio,
                    is_final=is_final,
                )
            finally:
                if _saved_cs_zr is not None:
                    self.beam_search.cross_attn_chunk_size = _saved_cs_zr
        else:
            # Nothing to decode this step (waiting for more audio).
            return []

        # Step 5: On final, assemble results and reset.
        if is_final:
            # Capture per-frame CTC log-probs (full utterance) before reset.
            if self.chunk_tracker is not None and \
                    len(self.chunk_tracker.encoder_chunks) > 0:
                full_enc = torch.cat(
                    [c for c in self.chunk_tracker.encoder_chunks], dim=0
                )  # (T_total, D)
                self._capture_ctc_logprobs(full_enc)
            if self.chunk_tracker is not None and nbest_hyps:
                self._populate_chunk_tracking(nbest_hyps)
            # DEBUG: dump encoder_chunks if env var set (Stage B vs full-recompute diff).
            self._maybe_dump_encoder_chunks(tag="stage_b")
            # Resume only if we did NOT already decode the last chunk offline
            # (decode-once path already produced the offline nbest_hyps).
            if getattr(self, "resume_offline_at_final", False) \
                    and not _decode_once:
                nbest_hyps = self._resume_offline_at_final(nbest_hyps)
            ret = self.assemble_hyps(nbest_hyps)
            # Snapshot xattn buffer before reset clears it on the beam_search.
            self._last_xattn_buffer = list(
                getattr(self.beam_search, "xattn_dump_buffer", [])
            )
            self.reset()
            return ret
        else:
            return []

    def _populate_chunk_tracking(self, hyps: List[Hypothesis]):
        """Populate chunk tracking info for token-level latency analysis.

        Uses the actual ChunkEmissionIndex recorded by the beam search when
        available, falling back to a linear approximation otherwise.

        Args:
            hyps: List of hypotheses from beam search
        """
        if not hyps or not self.chunk_tracker:
            return

        # Get the best hypothesis for reference
        best_hyp = hyps[0]
        num_tokens = len(best_hyp.yseq) - 2  # Exclude SOS and EOS

        if num_tokens <= 0:
            return

        # Use actual ChunkEmissionIndex from beam search if available
        has_chunk_idx = (
            hasattr(best_hyp, 'ChunkEmissionIndex')
            and best_hyp.ChunkEmissionIndex.numel() > 0
        )

        current_chunk = self.chunk_tracker.current_chunk_idx
        if current_chunk < 0:
            current_chunk = 0

        for token_idx, token_id in enumerate(best_hyp.yseq[1:-1]):  # Skip SOS/EOS
            if has_chunk_idx and token_idx < best_hyp.ChunkEmissionIndex.numel():
                chunk_idx = int(best_hyp.ChunkEmissionIndex[token_idx].item())
            else:
                # Fallback: linear approximation
                tokens_per_chunk = max(1, num_tokens / (current_chunk + 1))
                chunk_idx = min(current_chunk, int(token_idx / tokens_per_chunk))
            self.chunk_tracker.record_token_emission(
                token_id=token_id.item() if hasattr(token_id, 'item') else int(token_id),
                decode_idx=chunk_idx
            )

    def assemble_hyps(
        self, hyps: List[Hypothesis]
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis, dict]]:
        """Assemble hypotheses into output format.

        Args:
            hyps: List of hypotheses from beam search.

        Returns:
            List of (text, token, token_int, hyp, latency_info) tuples.
        """
        nbest_hyps = hyps[: self.nbest]
        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # Remove sos/eos and get results
            token_int = hyp.yseq[1:-1].tolist()

            # Remove blank symbol id (assumed to be 0)
            token_int = list(filter(lambda x: x != 0, token_int))

            # remove last scores_list entry (corresponds to EOS)
            if hyp.scores_list:
                hyp.scores_list.pop()

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None

            # Get latency info if available
            if self.chunk_tracker is not None:
                latency_info = self.chunk_tracker.get_latency_summary()
                # Add per-token chunk indices for debugging.
                # Prefer the actual ChunkEmissionIndex from the hypothesis
                # over the chunk_tracker approximation.
                has_chunk_idx = (
                    hasattr(hyp, 'ChunkEmissionIndex')
                    and hyp.ChunkEmissionIndex.numel() > 0
                )
                if has_chunk_idx:
                    # yseq = [SOS, t1, t2, ..., tN, EOS]
                    # ChunkEmissionIndex = [c1, ..., cN, c_eos] (no SOS entry)
                    # token_int filters out blanks from yseq[1:-1], so
                    # we must apply the same filter to ChunkEmissionIndex.
                    raw_ids = hyp.yseq[1:-1].tolist()  # before blank removal
                    cei = hyp.ChunkEmissionIndex
                    # Build filtered chunk indices matching token_int/token
                    filtered_chunks = [
                        int(cei[i].item()) if i < cei.numel() else -1
                        for i, tid in enumerate(raw_ids)
                        if tid != 0  # same blank filter as token_int
                    ]
                    latency_info['token_chunk_map'] = {
                        i: (tok, filtered_chunks[i] if i < len(filtered_chunks) else -1)
                        for i, tok in enumerate(token)
                    }
                elif hasattr(self.chunk_tracker, 'token_emission_info'):
                    latency_info['token_chunk_map'] = {
                        i: (tok, info['decode_chunk_idx'])
                        for i, (tok, info) in enumerate(
                            zip(token, self.chunk_tracker.token_emission_info)
                        )
                    }
                # Per-token future-chunk depth: how many future chunks each
                # chunk deferred for. Join with token_chunk_map via emission
                # chunk -> future_chunks[token] = chunk_future_map[emission_chunk].
                # This is the actual per-token lookahead the latency analysis
                # needs (token_chunk_map alone records only the belonging chunk,
                # not the deferral). Exact for chunk_rollback (B/C); for
                # token_resume (A) defer_count is keyed on the acoustic focus
                # chunk, so it is a per-acoustic-chunk approximation.
                _ctrl = getattr(self, 'dynamic_fc_controller', None)
                if _ctrl is not None and getattr(_ctrl, 'defer_count', None):
                    latency_info['chunk_future_map'] = {
                        int(k): int(v) for k, v in _ctrl.defer_count.items()
                    }
            else:
                latency_info = {}

            results.append((text, token, token_int, hyp, latency_info))

        return results

    def get_audio_chunk_size(self) -> int:
        """Get recommended audio chunk size in samples.

        Returns:
            Number of audio samples per chunk, computed from:
            chunk_size * subsampling_factor * hop_length
        """
        if not self.streaming_mode:
            return 0
        return self.chunk_size * self.subsampling_factor * self.hop_length


@typechecked
def inference(
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    ctc_weight: float,
    lm_weight: float,
    penalty: float,
    nbest: int,
    normalize_length: bool,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: str,
    asr_model_file: str,
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    disable_repetition_detection: bool,
    encoded_feat_length_limit: int,
    decoder_text_length_limit: int,
    # New streaming parameters
    use_chunked_streaming: bool = False,
    chunk_size: Optional[int] = None,
    num_left_chunks: int = -1,
    num_right_chunks: int = 0,
    cross_attn_num_left_chunks: Optional[int] = None,
    cross_attn_num_right_chunks: int = 0,
    encoder_num_right_chunks_override: Optional[int] = None,
    num_global_tokens: int = 0,
    use_decoder_self_kvcache: bool = True,
    use_decoder_cross_kvcache: bool = True,
    use_encoder_self_kvcache: bool = False,
    use_asymmetric_mask_at_inference: bool = False,
    chunk_end_confidence_threshold: float = 0.0,
    chunk_start_confidence_threshold: float = 0.0,
    rollback_confidence_threshold: float = 0.0,
    chunk_end_entropy_threshold: float = 0.0,
    chunk_end_margin_threshold: float = 0.0,
    chunk_end_topk_mass_threshold: float = 0.0,
    force_dcconv_right_context_at_inference: bool = False,
    extract_cq_slot: Optional[int] = None,
    cross_attn_slot_taper: bool = False,
    monotonic_attn_floor: int = -1,
    defer_decoding_until_last_chunk: bool = False,
    dump_ctc_logprobs_dir: Optional[str] = None,
    dump_xattn_dir: Optional[str] = None,
    attn_edge_stop_margin: int = 0,
    eos_bonus_at_final: float = 0.0,
    unmask_xattn_at_final: bool = False,
    subword_overlap_penalty: float = 0.0,
    resume_offline_at_final: bool = False,
    commit_stable_prefix: bool = False,
    # Dynamic future-chunks (inference-only latency/WER trade-off)
    dynamic_future_chunks: bool = False,
    max_future_chunks: int = 1,
    dynamic_mode: str = "pre_emptive",
    dynamic_signal_type: str = "top1_prob",
    dynamic_top1_prob_threshold: float = 0.6,
    dynamic_entropy_threshold: float = 1.5,
    dynamic_margin_threshold: float = 0.3,
    dynamic_topk_mass_threshold: float = 0.8,
    dynamic_fake_random_prob: float = 0.5,
    dynamic_fake_random_seed: int = 1234,
    # SR-CEM (calibrator for the trigger; see espnet2/asr_stream/sr_cem.py).
    sr_cem_ckpt: Optional[str] = None,
    sr_cem_threshold: Optional[float] = None,
    sr_cem_variant: str = "A",
    sr_cem_chunk_agg: Optional[str] = None,
    sr_cem_feat_dump_dir: Optional[str] = None,
    # Trigger-only temperature scaling.
    dynamic_trigger_temperature: float = 1.0,
    virtual_final_blank_frames: int = 0,
    virtual_final_min_chunks: int = 8,
    pad_final_chunk: bool = True,
    mask_streaming_pad: bool = False,
):
    """Run chunked streaming ASR inference over a dataset.

    Builds a Speech2TextStreamingChunked instance, feeds each utterance
    to it in simulated audio chunks of sim_chunk_length samples (or in
    one offline call when sim_chunk_length is 0), and writes the n-best
    hypotheses, per-token latency information, and optional diagnostic
    dumps (CTC log-probs, cross-attention, SR-CEM features) under
    output_dir. Arguments mirror the get_parser() command-line flags.
    """
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    # When use_chunked_streaming=false, we still use Speech2TextStreamingChunked
    # but with chunk_size=None/0 so it runs in offline mode (full attention,
    # no chunking), analogous to training with chunked_mask_config=None.
    if not use_chunked_streaming:
        chunk_size = None  # Force offline mode
    logging.info("Using Speech2TextStreamingChunked")
    speech2text = Speech2TextStreamingChunked(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        penalty=penalty,
        nbest=nbest,
        normalize_length=normalize_length,
        chunk_size=chunk_size,
        num_left_chunks=num_left_chunks,
        num_right_chunks=num_right_chunks,
        cross_attn_num_left_chunks=cross_attn_num_left_chunks,
        cross_attn_num_right_chunks=cross_attn_num_right_chunks,
        encoder_num_right_chunks_override=encoder_num_right_chunks_override,
        num_global_tokens=num_global_tokens,
        use_decoder_self_kvcache=use_decoder_self_kvcache,
        use_decoder_cross_kvcache=use_decoder_cross_kvcache,
        use_encoder_self_kvcache=use_encoder_self_kvcache,
        use_asymmetric_mask_at_inference=use_asymmetric_mask_at_inference,
        chunk_end_confidence_threshold=chunk_end_confidence_threshold,
        chunk_start_confidence_threshold=chunk_start_confidence_threshold,
        rollback_confidence_threshold=rollback_confidence_threshold,
        chunk_end_entropy_threshold=chunk_end_entropy_threshold,
        chunk_end_margin_threshold=chunk_end_margin_threshold,
        chunk_end_topk_mass_threshold=chunk_end_topk_mass_threshold,
        force_dcconv_right_context_at_inference=force_dcconv_right_context_at_inference,
        extract_cq_slot=extract_cq_slot,
        cross_attn_slot_taper=cross_attn_slot_taper,
        monotonic_attn_floor=monotonic_attn_floor,
        defer_decoding_until_last_chunk=defer_decoding_until_last_chunk,
        dump_xattn_dir=dump_xattn_dir,
        attn_edge_stop_margin=attn_edge_stop_margin,
        eos_bonus_at_final=eos_bonus_at_final,
        unmask_xattn_at_final=unmask_xattn_at_final,
        subword_overlap_penalty=subword_overlap_penalty,
        resume_offline_at_final=resume_offline_at_final,
        commit_stable_prefix=commit_stable_prefix,
        dynamic_future_chunks=dynamic_future_chunks,
        max_future_chunks=max_future_chunks,
        dynamic_mode=dynamic_mode,
        dynamic_signal_type=dynamic_signal_type,
        dynamic_top1_prob_threshold=dynamic_top1_prob_threshold,
        dynamic_entropy_threshold=dynamic_entropy_threshold,
        dynamic_margin_threshold=dynamic_margin_threshold,
        dynamic_topk_mass_threshold=dynamic_topk_mass_threshold,
        dynamic_fake_random_prob=dynamic_fake_random_prob,
        dynamic_fake_random_seed=dynamic_fake_random_seed,
        sr_cem_ckpt=sr_cem_ckpt,
        sr_cem_threshold=sr_cem_threshold,
        sr_cem_variant=sr_cem_variant,
        sr_cem_chunk_agg=sr_cem_chunk_agg,
        sr_cem_feat_dump_dir=sr_cem_feat_dump_dir,
        dynamic_trigger_temperature=dynamic_trigger_temperature,
        virtual_final_blank_frames=virtual_final_blank_frames,
        virtual_final_min_chunks=virtual_final_min_chunks,
        mask_streaming_pad=mask_streaming_pad,
    )

    # Auto-compute audio chunk size based on streaming mode
    if use_chunked_streaming:
        sim_chunk_length = speech2text.get_audio_chunk_size()
        logging.info(f"Streaming mode: sim_chunk_length auto-computed as {sim_chunk_length} samples")
    else:
        sim_chunk_length = 0
        logging.info("Offline mode: processing entire utterance at once")

    # 3. Build data-iterator
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # Prepare CTC log-prob dump directory if requested.
    if dump_ctc_logprobs_dir is not None:
        Path(dump_ctc_logprobs_dir).mkdir(parents=True, exist_ok=True)
        logging.info(
            f"Per-frame CTC log-probs will be saved to {dump_ctc_logprobs_dir}/"
            f"<utt_key>.npy"
        )

    # Prepare cross-attention dump directory if requested.
    if dump_xattn_dir is not None:
        Path(dump_xattn_dir).mkdir(parents=True, exist_ok=True)
        logging.info(
            f"Per-utterance cross-attention will be saved to {dump_xattn_dir}/"
            f"<utt_key>.npz"
        )

    # ----- SR-CEM feature dump -----
    # Open the JSONL file once. Each token-step the beam_search hook
    # writes one line. Closed at function exit.
    _sr_cem_dump_fp = None
    if sr_cem_feat_dump_dir:
        Path(sr_cem_feat_dump_dir).mkdir(parents=True, exist_ok=True)
        # Suffix the dump file with the per-shard output_dir leaf (e.g.
        # "output.3") so that inference_nj>1 - N parallel processes sharing one
        # sr_cem_feat_dump_dir - do NOT open the same file in "w" mode and
        # truncate each other (silent data loss; only the last shard survived).
        # The dump wrapper concatenates sr_cem_features.*.jsonl afterwards.
        _shard = Path(output_dir).name or "output"
        _dump_path = f"{sr_cem_feat_dump_dir}/sr_cem_features.{_shard}.jsonl"
        _sr_cem_dump_fp = open(_dump_path, "w")
        speech2text.beam_search._sr_cem_feat_dump_fp = _sr_cem_dump_fp
        logging.info(
            f"SR-CEM feature dump: per-token features will be written to {_dump_path}"
        )

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    scores_list_data = {n: [] for n in range(1, nbest + 1)}
    # ORACLE per-utt deferral DIAGNOSTIC (env-driven, no CLI plumbing): when
    # ORACLE_PERUTT_BUDGET points at a {utt_id: k_U} JSON, override the
    # controller's max_future_chunks per utt so always-defer reproduces static
    # nrc=k_U per utt (= the per-utt oracle). Confirms whether the DFC mechanism
    # can reach the oracle WER with a perfect router.
    _oracle_budget = None
    import os as _os
    _ob_path = _os.environ.get("ORACLE_PERUTT_BUDGET", "")
    if _ob_path and _os.path.exists(_ob_path):
        import json as _json
        _oracle_budget = _json.load(open(_ob_path))
        logging.info(f"[oracle_perutt] loaded per-utt budget for "
                     f"{len(_oracle_budget)} utts from {_ob_path}")
    # ORACLE per-CHUNK trigger (env-driven, same pattern as above): when
    # ORACLE_REF_FILE points at a kaldi-style text file (utt_id WORD WORD ...),
    # chunk_rollback DEFERs exactly the chunks whose just-decoded words contain
    # an error against this reference (see _oracle_chunk_wrong). Upper bound
    # for any commit-time signal (SLT rebuttal, reviewer unDe).
    _oracle_refs = None
    _or_path = _os.environ.get("ORACLE_REF_FILE", "")
    if _or_path and _os.path.exists(_or_path):
        _oracle_refs = {}
        for _ln in open(_or_path):
            _parts = _ln.split()
            if _parts:
                _oracle_refs[_parts[0]] = [w.upper() for w in _parts[1:]]
        logging.info(f"[oracle_ref] loaded references for "
                     f"{len(_oracle_refs)} utts from {_or_path}")
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            assert len(batch.keys()) == 1

            # RTF start marker parsed by pyscripts/utils/calculate_rtf.py
            # (--start-times-marker "speech length"); it pairs 1:1 with the
            # beam search's final "best hypo" end marker. Length in samples.
            logging.info("speech length: " + str(batch["speech"].size(0)))

            # Reset ALL state before processing each utterance
            sys.stderr.write("\n")  # empty line before each utterance
            sys.stderr.flush()
            speech2text.reset()              # This internally resets beam_search, chunk_tracker, etc.
            # Stamp the utt_id for the SR-CEM feat-dump (one row per
            # per-step beam search call gets this id). Set AFTER reset
            # so subsequent end-of-call resets don't clobber it.
            speech2text.beam_search._sr_cem_utt_id = str(keys[0]) if keys else ""
            if _oracle_budget is not None and getattr(
                speech2text, "dynamic_fc_controller", None
            ) is not None:
                _kU = int(_oracle_budget.get(str(keys[0]), 0))
                speech2text.dynamic_fc_controller.config.max_future_chunks = _kU
            if _oracle_refs is not None:
                # Set AFTER reset so end-of-utterance resets can't clobber it.
                speech2text._oracle_ref_words = _oracle_refs.get(str(keys[0]))

            try:
                if sim_chunk_length == 0:
                    # N-best list of (text, token, token_int, hyp_object)
                    results = speech2text(**batch)
                else:
                    speech = batch["speech"]
                    n_full_chunks = len(speech) // sim_chunk_length
                    # Long-form virtual finals: when the trailing CTC-blank
                    # run reaches the threshold (sentence-gap silence), flush
                    # the current segment with a real is_final pass (the
                    # object fully resets itself afterwards) and bank its
                    # text. The decoder then never accumulates more target
                    # context than it saw in training; the healthy
                    # encoder/CTC stream is unaffected. An empty-tensor
                    # final call is the same code path exact-multiple-length
                    # utterances already take.
                    _vf_frames = getattr(
                        speech2text, "virtual_final_blank_frames", 0
                    )
                    _vf_min = getattr(speech2text, "virtual_final_min_chunks", 0)
                    _banked = []  # (text, token, token_int) per flushed segment
                    for i in range(n_full_chunks):
                        speech2text(
                            speech=speech[
                                i * sim_chunk_length : (i + 1) * sim_chunk_length
                            ],
                            is_final=False,
                        )
                        if (
                            _vf_frames > 0
                            and i < n_full_chunks - 1
                            and speech2text.trailing_blank_frames >= _vf_frames
                            and speech2text.chunk_tracker is not None
                            and speech2text.chunk_tracker.current_chunk_idx + 1
                            >= _vf_min
                        ):
                            seg = speech2text(speech[0:0], is_final=True)
                            if seg and seg[0][0]:
                                _banked.append((seg[0][0], seg[0][1], seg[0][2]))
                            else:
                                # A speech segment flushed with NO text: either
                                # a genuinely silent segment or a lost segment
                                # (the long-form empty-flush bug). Loud so it
                                # can never again hide inside a WER number.
                                logging.warning(
                                    f"[VIRTUAL_FINAL] EMPTY FLUSH at sim chunk "
                                    f"{i} - no text banked for this segment"
                                )
                            logging.info(
                                f"[VIRTUAL_FINAL] fired at sim chunk {i} "
                                f"(segments banked: {len(_banked)})"
                            )
                    final_speech = speech[n_full_chunks * sim_chunk_length : len(speech)]
                    # By default, pad the final partial chunk with silence to a
                    # full chunk so the encoder gets enough frames - prevents end
                    # deletion when num_right_chunks=0. With --pad_final_chunk
                    # false the ragged remainder is fed as-is, matching training
                    # (training uses a smaller final chunk, never a silence pad).
                    if pad_final_chunk:
                        remainder = len(final_speech) % sim_chunk_length
                        if remainder > 0:
                            pad_len = sim_chunk_length - remainder
                            final_speech = torch.nn.functional.pad(final_speech, (0, pad_len))
                    results = speech2text(
                        final_speech, is_final=True
                    )
                    if _banked:
                        # Prepend banked segments to the 1-best result so the
                        # writer/scorer see one full-utterance transcript.
                        _texts = [t for t, _, _ in _banked]
                        _toks = sum((tk for _, tk, _ in _banked), [])
                        _tints = sum((ti for _, _, ti in _banked), [])
                        if results:
                            _t0 = results[0]
                            _texts += [_t0[0]] if _t0[0] else []
                            results = [(
                                " ".join(_texts).strip(),
                                _toks + _t0[1],
                                _tints + _t0[2],
                                _t0[3],
                                _t0[4],
                            )] + list(results[1:])
                        else:
                            _hyp = Hypothesis(
                                score=0.0, scores={}, states={}, yseq=[]
                            )
                            results = [(
                                " ".join(_texts).strip(), _toks, _tints, _hyp, {}
                            )]
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["<space>"], [2], hyp, {}]] * nbest

            # Only supporting batch_size==1
            key = keys[0]

            # SR-CEM per-utterance feature dump (per COMMITTED token of
            # the winning hypothesis). Writes one JSON line per token,
            # aligned 1:1 with the sclite TER hypothesis used to derive
            # labels. Features built from hyp.scores_list +
            # hyp.ChunkEmissionIndex. Dump fires whenever the dump fp
            # is open, independent of any trigger being active.
            if _sr_cem_dump_fp is not None and results:
                _hyp0 = results[0][3]  # Hypothesis object
                _scores_list = getattr(_hyp0, "scores_list", None) or []
                _yseq = getattr(_hyp0, "yseq", None)
                _cei = getattr(_hyp0, "ChunkEmissionIndex", None)
                _conf_list = getattr(_hyp0, "confidence_list", None) or []
                if _yseq is not None and _scores_list:
                    # yseq = [SOS, t1, ..., tN, EOS]; scores_list aligns
                    # with yseq[1:N+1] (EOS already popped by assemble_hyps).
                    _selected_ids = _yseq[1:1 + len(_scores_list)].tolist()
                    _cei_list = (_cei[:len(_scores_list)].tolist()
                                 if _cei is not None and _cei.numel() > 0
                                 else [0] * len(_scores_list))
                    # Pass 1: collect kept tokens with their CUMULATIVE
                    # selected scores (stored scores_list values are
                    # prefix-inclusive). Feature derivation then goes
                    # through the canonical paper-recipe helper -
                    # the same one the decision-time hook uses - so the
                    # dump can never drift from inference again.
                    from espnet2.asr_stream.sr_cem import (
                        build_features_causal as _bfc,
                        build_features_chunk as _bfk,
                        step_features_from_cumulative as _sffc,
                    )
                    _kept = []  # (step_i, tid, sel_cum, vals, chunk)
                    for i, (_sd, _tid) in enumerate(zip(_scores_list, _selected_ids)):
                        if not isinstance(_sd, dict):
                            continue
                        # Skip blank token id (sclite hypothesis strips
                        # blanks; keeping them here would cause an
                        # off-by-one against the alignment-derived
                        # training labels).
                        if int(_tid) == 0:
                            continue
                        # Selected token's score (handles ESPnet's "tid-1"
                        # key convention for transducer; for AED the key
                        # equals the token id).
                        _sel = _sd.get(str(_tid))
                        if _sel is None:
                            _sel = _sd.get(str(_tid - 1))
                        if _sel is None:
                            continue
                        _ck = int(_cei_list[i]) if i < len(_cei_list) else 0
                        _kept.append(
                            (i, int(_tid), float(_sel), list(_sd.values()), _ck)
                        )
                    # Pass 2: derive + write. Variant B's chunk-local S>t
                    # is the suffix sum of diffs within the token's chunk,
                    # which telescopes to (last kept cumulative in the
                    # chunk) - (this token's cumulative).
                    _last_cum_in_chunk = {}
                    for (_i, _tid, _cum, _vals, _ck) in _kept:
                        _last_cum_in_chunk[_ck] = _cum  # kept order = ascending
                    for k, (_i, _tid, _cum, _vals, _ck) in enumerate(_kept):
                        _prev_cum = _kept[k - 1][2] if k > 0 else 0.0
                        _score, _rk, _s_lt, _top4 = _sffc(
                            selected_cum=_cum,
                            prev_selected_cum=_prev_cum,
                            candidate_cums=_vals,
                        )
                        _feat = _bfc(
                            score=_score, rank=_rk, S_lt=_s_lt, top4=_top4
                        )
                        _s_gt_chunk = _last_cum_in_chunk[_ck] - _cum
                        _feat_b = _bfk(
                            score=_score, rank=_rk, S_lt=_s_lt,
                            S_gt_chunk=_s_gt_chunk, top4=_top4,
                        )
                        _top1_prob_raw = (
                            float(_conf_list[_i]) if _i < len(_conf_list) else None
                        )
                        _sr_cem_dump_fp.write(json.dumps({
                            "utt_id": str(key),
                            "step": int(_i),
                            "current_chunk": int(_ck),
                            "top1_token": int(_tid),
                            "feat": _feat,
                            "feat_b": _feat_b,
                            "p_correct": None,
                            "top1_prob_raw": _top1_prob_raw,
                        }) + "\n")
                    _sr_cem_dump_fp.flush()
                    # Hidden-state probe: save per-kept-token decoder hidden
                    # states, row-aligned to the JSONL rows above (_kept order),
                    # as hs/<utt>.npy shape (N_kept, hidden_dim). Only present
                    # when return_hs was enabled (DUMP_HIDDEN_STATES=1).
                    _hs = getattr(_hyp0, "hs", None)
                    if _hs:
                        import os as _os2
                        import numpy as _np2
                        _hs_dir = _os2.path.join(
                            _os2.path.dirname(_sr_cem_dump_fp.name), "hs")
                        _os2.makedirs(_hs_dir, exist_ok=True)
                        _hrows = [
                            _hs[_i].detach().cpu().numpy().reshape(-1)
                            for (_i, _tid, _cum, _vals, _ck) in _kept
                            if _i < len(_hs)
                        ]
                        if _hrows:
                            _np2.save(
                                _os2.path.join(_hs_dir, f"{key}.npy"),
                                _np2.stack(_hrows),
                            )
                        # Encoder-context probe: per-chunk mean-pooled RAW encoder
                        # output (from the encoder forward hook, full-utt last call)
                        # to enc/<utt>.npy shape (n_chunks, enc_dim). Token context
                        # = enc[its current_chunk]. Tests whether the raw acoustic
                        # rep carries routing signal the decoder discards.
                        _el = getattr(speech2text, "_enc_last", None)
                        if _el is not None:
                            try:
                                _enc_dir = _os2.path.join(
                                    _os2.path.dirname(_sr_cem_dump_fp.name), "enc")
                                _os2.makedirs(_enc_dir, exist_ok=True)
                                _eo = _el[0].cpu().numpy()  # (T, enc_dim)
                                # Probe assumes chunk_size=16 (cs16 configs);
                                # wrong chunking for other chunk sizes. Kept
                                # fixed so probe outputs stay reproducible.
                                _csz = 16  # subsampled frames per chunk (cs16)
                                _nch = _eo.shape[0] // _csz
                                if _nch > 0:
                                    _np2.save(
                                        _os2.path.join(_enc_dir, f"{key}.npy"),
                                        _np2.stack([
                                            _eo[c * _csz:(c + 1) * _csz].mean(0)
                                            for c in range(_nch)]))
                            except Exception:
                                pass
                        # Frontend-acoustic probe: save the full accumulated fbank
                        # (raw frontend output, upstream of the encoder) to
                        # fbank/<utt>.npy shape (T_feat, feat_dim). Probe chunks it
                        # offline by the known #chunks. Direct attribute (no hook).
                        _af = getattr(speech2text, "accumulated_features", None)
                        if _af is not None:
                            try:
                                _fb_dir = _os2.path.join(
                                    _os2.path.dirname(_sr_cem_dump_fp.name), "fbank")
                                _os2.makedirs(_fb_dir, exist_ok=True)
                                _np2.save(_os2.path.join(_fb_dir, f"{key}.npy"),
                                          _af[0].detach().cpu().numpy())
                            except Exception:
                                pass

            # Per-frame CTC log-prob dump (one .npy per utterance, shape (T, V)).
            if dump_ctc_logprobs_dir is not None:
                logp = getattr(speech2text, "_last_ctc_logprobs", None)
                if logp is not None:
                    np.save(
                        str(Path(dump_ctc_logprobs_dir) / f"{key}.npy"),
                        logp,
                    )
                else:
                    logging.warning(
                        f"CTC log-probs unavailable for {key}; skipping dump."
                    )

            # Per-utterance cross-attention dump (one .npz per utterance).
            # Contains arrays: chunk_idx[N], enc_len[N], attn (object array of
            # length N, each entry a 1D array of attention weights over encoder
            # frames at that step), token_ids[M] (final 1-best tokens).
            if dump_xattn_dir is not None:
                buf = getattr(speech2text, "_last_xattn_buffer", None)
                if buf is None:
                    buf = getattr(
                        speech2text.beam_search, "xattn_dump_buffer", []
                    )
                if buf:
                    chunk_idx = np.array([b[0] for b in buf], dtype=np.int32)
                    enc_len = np.array([b[1] for b in buf], dtype=np.int32)
                    attn = np.empty(len(buf), dtype=object)
                    for i, b in enumerate(buf):
                        attn[i] = b[2]
                    top_hyp = results[0][3] if results else None
                    if top_hyp is not None and hasattr(top_hyp, "yseq"):
                        ys = top_hyp.yseq
                        token_ids = np.array(
                            ys.tolist() if hasattr(ys, "tolist") else list(ys),
                            dtype=np.int64,
                        )
                    else:
                        token_ids = np.array([], dtype=np.int64)
                    np.savez(
                        str(Path(dump_xattn_dir) / f"{key}.npz"),
                        chunk_idx=chunk_idx,
                        enc_len=enc_len,
                        attn=attn,
                        token_ids=token_ids,
                    )
                else:
                    logging.warning(
                        f"xattn buffer empty for {key}; skipping dump."
                    )

            for n, result in zip(range(1, nbest + 1), results):
                text, token, token_int, hyp, latency_info = result

                # Log chunk indices for debugging (1-best only, grouped by
                # chunk so long utterances stay readable: one line per chunk
                # showing the words emitted in it).
                if n == 1 and 'token_chunk_map' in latency_info:
                    total = latency_info.get('total_chunks', '?')
                    items = list(latency_info['token_chunk_map'].values())
                    by_chunk = {}
                    for tok, chunk in items:
                        by_chunk.setdefault(chunk, []).append(tok)
                    logging.info(f"[CHUNKS] {key} total_chunks={total}")
                    for chunk in sorted(by_chunk):
                        words = "".join(by_chunk[chunk]).replace("▁", " ").strip()
                        logging.info(f"    chunk {chunk:>2}: {words}")

                # Create a directory: outdir/{n}best_recog
                ibest_writer = writer[f"{n}best_recog"]

                # Write the result to each file
                ibest_writer["token"][key] = " ".join(token)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["score"][key] = str(hyp.score)

                if text is not None:
                    ibest_writer["text"][key] = text

                scores_list_data[n].append((key, hyp.scores_list))

                # Write latency info if available
                if latency_info:
                    ibest_writer["latency"][key] = json.dumps(latency_info)

                    # Write chunk indices in readable format: token,chunk token,chunk ...
                    if 'token_chunk_map' in latency_info:
                        chunk_str = " ".join([
                            f"{tok},{chunk}"
                            for tok, chunk in latency_info['token_chunk_map'].values()
                        ])
                        ibest_writer["chunk_indices"][key] = chunk_str

    # Write scores_list.json for each n-best
    for n in range(1, nbest + 1):
        score_file_path = f"{output_dir}/{n}best_recog/scores_list.json"
        with open(score_file_path, "w") as f:
            json.dump(scores_list_data[n], f)

    # Close SR-CEM feature dump file if opened.
    if _sr_cem_dump_fp is not None:
        _sr_cem_dump_fp.close()
        speech2text.beam_search._sr_cem_feat_dump_fp = None


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--asr_train_config", type=str, required=True)
    group.add_argument("--asr_model_file", type=str, required=True)
    group.add_argument("--lm_train_config", type=str)
    group.add_argument("--lm_file", type=str)
    group.add_argument("--word_lm_train_config", type=str)
    group.add_argument("--word_lm_file", type=str)

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain max output length. "
        "If maxlenratio=0.0 (default), it uses a end-detect "
        "function "
        "to automatically find maximum hypothesis lengths",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    group.add_argument(
        "--ctc_weight",
        type=float,
        default=0.5,
        help="CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--disable_repetition_detection", type=str2bool, default=False)

    group.add_argument(
        "--encoded_feat_length_limit",
        type=int,
        default=0,
        help="Limit the lengths of the encoded feature" "to input to the decoder.",
    )
    group.add_argument(
        "--decoder_text_length_limit",
        type=int,
        default=0,
        help="Limit the lengths of the text" "to input to the decoder.",
    )

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--normalize_length",
        type=str2bool,
        default=False,
        help="If true, best hypothesis is selected by length-normalized scores",
    )

    # Chunked streaming configuration
    group = parser.add_argument_group("Chunked streaming related")
    group.add_argument(
        "--use_chunked_streaming",
        type=str2bool,
        default=False,
        help="Use chunked streaming with delayed decoding",
    )
    group.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        help="Number of encoder frames per chunk",
    )
    group.add_argument(
        "--num_left_chunks",
        type=int,
        default=-1,
        help="Number of past chunks visible (-1 = unlimited)",
    )
    group.add_argument(
        "--num_right_chunks",
        type=int,
        default=0,
        help="Number of future chunks visible (determines decoding delay)",
    )
    group.add_argument(
        "--cross_attn_num_left_chunks",
        type=int,
        default=None,
        help="Decoder cross-attention left chunks (default: same as num_left_chunks)",
    )
    group.add_argument(
        "--cross_attn_num_right_chunks",
        type=int,
        default=0,
        help="Decoder cross-attention right chunks (default 0 = no future chunks visible "
        "in cross-attn, matches training). >0 opens chunk(s) of future encoder frames "
        "to the decoder; combine with --attn_edge_stop_margin to control commit-delay.",
    )
    group.add_argument(
        "--encoder_num_right_chunks_override",
        type=int,
        default=None,
        help="If set, overrides the encoder's chunked self-attention right-chunk count "
        "while leaving the chunk-tracker delay tied to --num_right_chunks. Use 0 for a "
        "fully causal encoder paired with a cross-attn-only future-chunk view "
        "(eliminates the redundancy between K[C]'s self-attn-mixed future and K[C+1]'s "
        "preliminary encoding).",
    )
    group.add_argument(
        "--num_global_tokens",
        type=int,
        default=0,
        help="Number of global attention sink tokens",
    )
    group.add_argument(
        "--use_decoder_self_kvcache",
        type=str2bool,
        default=True,
        help="Use KV cache for decoder self-attention",
    )
    group.add_argument(
        "--use_decoder_cross_kvcache",
        type=str2bool,
        default=True,
        help="Use KV cache for decoder cross-attention",
    )
    group.add_argument(
        "--use_encoder_self_kvcache",
        type=str2bool,
        default=False,
        help="Stage B: enable the incremental per-layer encoder self-attn "
        "KV cache (chunk-wise append, asymmetric C-axis, conv state cache). "
        "Activates a streaming forward path that only recomputes the "
        "unfinalized tail each call.",
    )
    group.add_argument(
        "--use_asymmetric_mask_at_inference",
        type=str2bool,
        default=False,
        help="Enable the asymmetric C-axis encoder mask at inference "
        "(matches one of the two training modes, C = num_right_chunks + 1). "
        "When True with num_right_chunks > 0, the encoder produces a 4D "
        "output; the decoder-visible slice is taken at c_q = num_right_chunks "
        "(max right-context view). When False, the encoder uses a symmetric "
        "chunked mask with availability-bounded right context.",
    )
    group.add_argument(
        "--chunk_end_confidence_threshold",
        type=float,
        default=0.0,
        help="Softmax confidence threshold for end-of-chunk early stop. "
        "If top-1 softmax probability drops below this value on a non-final "
        "chunk, decoding of the current chunk stops and resumes when more "
        "encoder context arrives. 0.0 disables the policy.",
    )
    group.add_argument(
        "--chunk_start_confidence_threshold",
        type=float,
        default=0.0,
        help="Softmax confidence threshold for start-of-chunk low-confidence "
        "filter. If the first token emitted in a new chunk has top-1 softmax "
        "probability below this value, the token is masked and beam selection "
        "retries. 0.0 disables the policy.",
    )
    group.add_argument(
        "--rollback_confidence_threshold",
        type=float,
        default=0.0,
        help="Softmax confidence threshold for Policy 3 (confidence-gated "
        "rollback + agreement confirmation). At each chunk boundary, tokens "
        "emitted in the previous chunk with confidence below this value are "
        "re-scored under the grown encoder. Agreement → confirm (keep token, "
        "update confidence); disagreement → replace the token and drop all "
        "subsequent tokens. 0.0 disables the policy.",
    )
    group.add_argument(
        "--chunk_end_entropy_threshold",
        type=float,
        default=0.0,
        help="Alternative softmax-distribution stop signal: at each beam step on "
        "a non-final chunk, fire the chunk-stop rule when the full softmax "
        "Shannon entropy (in nats) exceeds this value. High entropy = uncertain "
        "choice. 0.0 disables. Typical: 1.0-2.0 nats for BPE5000.",
    )
    group.add_argument(
        "--chunk_end_margin_threshold",
        type=float,
        default=0.0,
        help="Alternative softmax-distribution stop signal: fire when "
        "(top1_prob - top2_prob) < threshold. Catches the 'two competing "
        "candidates' case where the model wavers between two tokens. 0.0 "
        "disables. Typical: 0.3-0.5.",
    )
    group.add_argument(
        "--chunk_end_topk_mass_threshold",
        type=float,
        default=0.0,
        help="Alternative softmax-distribution stop signal: fire when the "
        "cumulative softmax mass of the top-3 tokens is below threshold "
        "(low concentration = long-tail uncertainty). 0.0 disables. "
        "Typical: 0.7-0.9.",
    )
    group.add_argument(
        "--force_dcconv_right_context_at_inference",
        type=str2bool,
        default=False,
        help="Diagnostic flag: drop the trained 'causal at every c_q' DCConv "
        "behaviour at inference and let the conv kernel see Lc frames of right "
        "context at every c_q slot. Train/test mismatch (the conv weights were "
        "trained with right_context=0 everywhere) - use only for ablation. "
        "Requires C >= 2 so right-context frames are present.",
    )
    group.add_argument(
        "--extract_cq_slot",
        type=int,
        default=None,
        help="Diagnostic: which c_q slot of the asymmetric encoder output "
        "to extract as the decoder-visible representation. None (default) "
        "uses c_q = num_right_chunks (the max-right-context training slice). "
        "Set to 0 to extract the strictly-causal-self-attn slot while keeping "
        "num_right_chunks >= 1 buffering - pair with "
        "--force_dcconv_right_context_at_inference to decouple self-attn "
        "chunk boundary from DCConv future access.",
    )
    group.add_argument(
        "--cross_attn_slot_taper",
        type=str2bool,
        default=False,
        help="Reading-C strict-streaming taper. When True, per-chunk c_q is "
        "set so each chunk uses only the lookahead audio it can afford under "
        "the current encoder budget: chunk i in [0, c] gets c_q=R, then chunk "
        "c+k gets c_q=R-k (R=num_right_chunks). Requires "
        "--use_asymmetric_mask_at_inference=true. Overrides --extract_cq_slot.",
    )
    group.add_argument(
        "--monotonic_attn_floor",
        type=int,
        default=-1,
        help="Frames of forgiveness for monotonic cross-attention floor. "
        "When >= 0, after each beam step records the argmax frame of the "
        "decoder's cross-attention; the next step's mask restricts the new "
        "token's attention to frames at or after (last_argmax - floor). "
        "Targets AED word-level duplications. -1 disables. Typical: 4-8.",
    )
    group.add_argument(
        "--defer_decoding_until_last_chunk",
        type=str2bool,
        default=False,
        help="Diagnostic: encoder runs chunk-by-chunk (streaming mask), but "
        "the beam search is held until is_final=True, then runs once on the "
        "concatenated encoder output with no per-token cross-attn mask. "
        "Isolates the beam-loop streaming behaviour from the encoder.",
    )
    group.add_argument(
        "--dump_ctc_logprobs_dir",
        type=str,
        default=None,
        help="If set, write per-frame CTC log-probabilities for every "
        "utterance to <dir>/<utt_key>.npy (shape (T, V), float32). "
        "Used for distribution-divergence analysis across decoding "
        "conditions (CTC-only diagnostic).",
    )
    group.add_argument(
        "--dump_xattn_dir",
        type=str,
        default=None,
        help="If set, write per-utterance decoder cross-attention "
        "(last layer, beam-0, last query, mean over heads) to "
        "<dir>/<utt_key>.npz. Each .npz contains chunk_idx[N], "
        "enc_len[N], attn (object array of N 1D distributions), and "
        "token_ids[M] (final 1-best yseq).",
    )
    group.add_argument(
        "--attn_edge_stop_margin",
        type=int,
        default=0,
        help="Attention-edge stop rule. If > 0, on a non-final beam call, "
        "stop the chunk (do not commit the new token) when the top beam's "
        "cross-attention argmax lands within this many frames of the "
        "rightmost encoder frame. Targets the cross-chunk word-duplication "
        "failure mode by holding the beam until more encoder context "
        "arrives. 0 disables. Typical: 1-2.",
    )
    group.add_argument(
        "--eos_bonus_at_final",
        type=float,
        default=0.0,
        help="F4 fix: at the final chunk (is_final=True), add this bonus "
        "to the EOS log-probability so the model is more willing to "
        "terminate cleanly instead of emitting plausible-but-unsupported "
        "tail content. Targets type-(a) hallucinated tails. 0 disables. "
        "Typical: 1.0-3.0.",
    )
    group.add_argument(
        "--unmask_xattn_at_final",
        type=str2bool,
        default=False,
        help="F4 fix: at is_final, disable the per-token cross-attn mask so "
        "all prefix rows get full encoder attention (like defer mode). "
        "Targets streaming-specific tail substitutions caused by "
        "restricted prefix representations.",
    )
    group.add_argument(
        "--resume_offline_at_final",
        type=str2bool,
        default=False,
        help="Resume-from-prefix fix: at is_final, keep the committed "
        "pre-final-chunk tokens as a fixed prefix and re-decode ONLY the final "
        "chunk with the offline beam over the streaming encoder. Genuinely "
        "streaming (no re-encode); fixes final-chunk drift, leaves interior "
        "errors frozen in the prefix.",
    )
    group.add_argument(
        "--commit_stable_prefix",
        type=str2bool,
        default=False,
        help="Branch-D (token_chunk_resume) commit-stable-prefix: on each "
        "deferred re-decode pass, freeze the longest prefix of the chunk that "
        "agrees with the previous pass and re-decode only the divergent suffix "
        "deeper. Frozen tokens commit at their stable depth (lower latency); a "
        "token that flips after being frozen is unrecoverable (small WER cost). "
        "Off (default) = whole-chunk rollback.",
    )
    group.add_argument(
        "--subword_overlap_penalty",
        type=float,
        default=0.0,
        help="F3 fix: penalize subword-piece repetition at chunk boundaries "
        "(e.g. ``▁CONTAIN`` + ``IN`` + ``ING`` → \"CONTAININING\"). When > 0, "
        "at the first token emission of each new chunk, candidate tokens "
        "whose text is a >=2-char suffix of the prefix's last word piece are "
        "penalized by this many nats. 0 disables. Typical: 5.0-15.0.",
    )

    # Dynamic future-chunks (latency/WER trade-off at inference time)
    group.add_argument(
        "--dynamic_future_chunks",
        type=str2bool,
        default=False,
        help="Enable dynamic future-chunks. When on, the per-chunk number "
        "of future-context chunks is decided dynamically based on a "
        "configurable signal, up to --max_future_chunks. When off (default), "
        "behaviour is identical to the static num_right_chunks setting.",
    )
    group.add_argument(
        "--max_future_chunks",
        type=int,
        default=1,
        help="Maximum future-chunks budget for --dynamic_future_chunks. "
        "Hard upper bound on how many chunks the controller may wait for "
        "on any single chunk before force-committing. Ignored when "
        "--dynamic_future_chunks=false.",
    )
    group.add_argument(
        "--dynamic_mode",
        type=str,
        default="pre_emptive",
        choices=["pre_emptive", "chunk_rollback", "token_resume", "token_chunk_resume"],
        help="Which branch to run. pre_emptive: evaluate signal on CTC "
        "log-probs before beam search runs (cheapest baseline). "
        "chunk_rollback (Branch B): run beam search on the chunk, then "
        "evaluate signal; on trigger, restore beam_search state and "
        "wait for one more audio chunk before re-decoding. token_resume "
        "(Branch A): per-step signal check inside beam search; on trigger, "
        "drop the just-emitted token and resume from that token step on "
        "the next call.",
    )
    group.add_argument(
        "--dynamic_signal_type",
        type=str,
        default="top1_prob",
        choices=["top1_prob", "entropy", "margin", "topk_mass", "fake_random",
                 "sr_cem_causal", "sr_cem_chunk", "wait_policy", "wait_policy_chunk",
                 "ctc_disagree"],
        help="Confidence signal that drives the defer/commit decision. "
        "top1_prob = mean softmax top-1; entropy = mean Shannon entropy "
        "(nats); margin = mean (top1 - top2); topk_mass = mean cumulative "
        "top-3 mass; fake_random = Bernoulli with prob --dynamic_fake_random_prob "
        "(for testing the plumbing only); sr_cem_causal = SR-CEM Variant A "
        "(7-dim causal features) on Branch A token_resume; sr_cem_chunk = "
        "SR-CEM Variant B (8-dim chunk-local features incl chunk-local S>t) "
        "on Branch B chunk_rollback. SR-CEM signals require --sr_cem_ckpt.",
    )
    group.add_argument(
        "--dynamic_top1_prob_threshold",
        type=float,
        default=0.6,
        help="Defer when mean top-1 softmax probability falls below this "
        "value. Used only when --dynamic_signal_type=top1_prob. "
        "Typical: 0.5-0.8.",
    )
    group.add_argument(
        "--dynamic_entropy_threshold",
        type=float,
        default=1.5,
        help="Defer when mean softmax entropy (nats) exceeds this value. "
        "Used only when --dynamic_signal_type=entropy. "
        "Typical: 1.0-2.0 for BPE5000.",
    )
    group.add_argument(
        "--dynamic_margin_threshold",
        type=float,
        default=0.3,
        help="Defer when mean (top1 - top2) margin falls below this value. "
        "Used only when --dynamic_signal_type=margin. Typical: 0.2-0.5.",
    )
    group.add_argument(
        "--dynamic_topk_mass_threshold",
        type=float,
        default=0.8,
        help="Defer when mean cumulative top-3 softmax mass falls below "
        "this value. Used only when --dynamic_signal_type=topk_mass. "
        "Typical: 0.7-0.9.",
    )
    group.add_argument(
        "--dynamic_fake_random_prob",
        type=float,
        default=0.5,
        help="Defer probability for the fake_random signal type. Used "
        "only when --dynamic_signal_type=fake_random; for testing the "
        "infrastructure without depending on the confidence logic.",
    )
    group.add_argument(
        "--dynamic_fake_random_seed",
        type=int,
        default=1234,
        help="RNG seed for the fake_random signal type. Reset per "
        "utterance for reproducibility.",
    )

    # ---- SR-CEM (calibration-based signal) ---------------------------
    group.add_argument(
        "--sr_cem_ckpt",
        type=str,
        default=None,
        help="Path to a trained SR-CEM .pt checkpoint (state_dict of a "
        "Score_CEM MLP). Required when --dynamic_signal_type is "
        "sr_cem_causal or sr_cem_chunk. If the file is missing the "
        "signal no-ops to COMMIT (warning logged).",
    )
    group.add_argument(
        "--sr_cem_threshold",
        type=float,
        default=None,
        help="Defer when SR-CEM p(correct) < threshold. Default None = "
        "use the checkpoint's calibrated threshold (falling back to 0.5). "
        "Used by both Variant A (sr_cem_causal) and Variant B "
        "(sr_cem_chunk). Variant B compares against the per-chunk "
        "aggregate (mean or min per --sr_cem_chunk_agg).",
    )
    group.add_argument(
        "--sr_cem_variant",
        type=str,
        default="A",
        choices=["A", "B", "A_margin", "A_MARGIN"],
        help="SR-CEM variant: A (7-dim causal features, Branch A "
        "token_resume) or B (8-dim chunk-local features, Branch B "
        "chunk_rollback). Overridden automatically by --dynamic_signal_type.",
    )
    group.add_argument(
        "--sr_cem_chunk_agg",
        type=str,
        default=None,
        choices=["mean", "min"],
        help="Aggregation over per-token p(correct) in Variant B to a "
        "single chunk-level signal. 'mean' (default) is more stable; "
        "'min' is more conservative (defer if ANY token in the chunk is "
        "uncertain).",
    )
    group.add_argument(
        "--sr_cem_feat_dump_dir",
        type=str,
        default=None,
        help="If set, dump per-token SR-CEM features + raw top1_prob to "
        "<dir>/sr_cem_features.jsonl during decoding. Used to build the "
        "SR-CEM training dataset. Activates the per-step JSON-line dump "
        "in batch_beam_search.search().",
    )
    group.add_argument(
        "--dynamic_trigger_temperature",
        type=float,
        default=1.0,
        help="Temperature scaling (Guo et al. 2017) applied to the per-step "
        "log-softmax before computing DFC trigger signals (top1_prob, "
        "entropy, margin, topk_mass). T=1.0 is no-op. T>1.0 flattens, "
        "T<1.0 sharpens. Does NOT modify beam ranking or final decoded "
        "hypothesis - argmax is preserved at any T. Only the trigger's "
        "defer/commit decisions change. Fit T on a held-out dev set with "
        "research_docs/calibration/fit_temperature_scaling.py.",
    )
    group.add_argument(
        "--virtual_final_blank_frames",
        type=int,
        default=0,
        help="Long-form decoding: if > 0, flush the current segment with a "
        "real is_final pass (and reset all decode state) whenever the "
        "trailing CTC-blank run reaches this many encoder frames "
        "(sentence-gap silence). Keeps the decoder in its trained "
        "short-sequence regime on long audio. 0 disables.",
    )
    group.add_argument(
        "--virtual_final_min_chunks",
        type=int,
        default=8,
        help="Minimum encoder chunks in a segment before a virtual final "
        "may fire (guards against rapid-fire resets in long silences).",
    )
    group.add_argument(
        "--pad_final_chunk",
        type=str2bool,
        default=True,
        help="If true (default), the final partial chunk is silence-padded to "
        "a full chunk before the is_final encoder pass. If false, the ragged "
        "remainder is fed as-is, matching training (which uses a smaller final "
        "chunk, never a silence pad). Set false to remove the train/inference "
        "trailing-pad mismatch.",
    )
    group.add_argument(
        "--mask_streaming_pad",
        type=str2bool,
        default=False,
        help="If true, exclude the Stage-B is_final pad frames from the encoder "
        "self-attention so real frames never attend to silence-pad keys. "
        "Default false preserves prior behaviour. Matters most when the chunk "
        "is large relative to the utterance (one-chunk regime), where the pad "
        "is a large fraction of the sequence.",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
