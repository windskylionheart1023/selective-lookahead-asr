"""Parallel beam search module for online simulation."""

import logging
from typing import Any  # noqa: H301
from typing import Dict  # noqa: H301
from typing import List  # noqa: H301
from typing import Tuple  # noqa: H301

import torch

from espnet2.asr.transducer.beam_search_transducer_streaming import (
    BeamSearchTransducerStreaming,
)
from espnet.nets.batch_beam_search import BatchBeamSearch  # noqa: H301
from espnet.nets.batch_beam_search import BatchHypothesis  # noqa: H301
from espnet.nets.beam_search import Hypothesis
from espnet.nets.beam_search_timesync_streaming import BeamSearchTimeSyncStreaming
from espnet.nets.e2e_asr_common import end_detect


class BatchBeamSearchOnline(BatchBeamSearch):
    """Online beam search implementation.

    This simulates streaming decoding.
    It requires encoded features of entire utterance and
    extracts block by block from it as it shoud be done
    in streaming processing.
    This is based on Tsunoo et al, "STREAMING TRANSFORMER ASR
    WITH BLOCKWISE SYNCHRONOUS BEAM SEARCH"
    (https://arxiv.org/abs/2006.14941).
    """

    def __init__(
        self,
        *args,
        block_size=40,
        hop_size=16,
        look_ahead=16,
        disable_repetition_detection=False,
        encoded_feat_length_limit=0,
        decoder_text_length_limit=0,
        incremental_decode=False,
        time_sync=False,
        ctc=None,
        hold_n=0,
        transducer_conf=None,
        joint_network=None,
        cross_attn_chunk_size: int = 0,
        cross_attn_num_left_chunks: int = -1,
        cross_attn_num_right_chunks: int = 0,
        chunk_end_confidence_threshold: float = 0.0,
        chunk_start_confidence_threshold: float = 0.0,
        rollback_confidence_threshold: float = 0.0,
        chunk_end_entropy_threshold: float = 0.0,
        chunk_end_margin_threshold: float = 0.0,
        chunk_end_topk_mass_threshold: float = 0.0,
        monotonic_attn_floor: int = -1,
        **kwargs,
    ):
        """Initialize beam search.

        Args:
            monotonic_attn_floor: When >= 0, enforces monotonic cross-attention
                across decoder steps. After each beam step, records the argmax
                frame of the decoder's cross-attn for the top beam; the next
                step's cross-attn mask restricts the new token to attend only
                to frames at or after (last_argmax - monotonic_attn_floor).
                Targets the AED word-level duplication failure mode.
                Set to -1 to disable. Typical values: 4-8 (frames of forgiveness).
        """
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.hop_size = hop_size
        self.look_ahead = look_ahead
        self.disable_repetition_detection = disable_repetition_detection
        self.encoded_feat_length_limit = encoded_feat_length_limit
        self.decoder_text_length_limit = decoder_text_length_limit
        self.incremental_decode = incremental_decode
        self.cross_attn_chunk_size = cross_attn_chunk_size
        self.cross_attn_num_left_chunks = cross_attn_num_left_chunks
        self.cross_attn_num_right_chunks = cross_attn_num_right_chunks
        self.chunk_end_confidence_threshold = chunk_end_confidence_threshold
        self.chunk_start_confidence_threshold = chunk_start_confidence_threshold
        self.rollback_confidence_threshold = rollback_confidence_threshold
        self.chunk_end_entropy_threshold = chunk_end_entropy_threshold
        self.chunk_end_margin_threshold = chunk_end_margin_threshold
        self.chunk_end_topk_mass_threshold = chunk_end_topk_mass_threshold
        self.monotonic_attn_floor = monotonic_attn_floor
        self.attn_center = None  # Last beam-0's cross-attn argmax frame
        self.time_sync = time_sync
        self.ctc = ctc
        self.hold_n = hold_n

        if time_sync:
            if transducer_conf is not None:
                self.time_sync_search = BeamSearchTransducerStreaming(
                    decoder=self.scorers["decoder"],
                    joint_network=joint_network,
                    beam_size=self.beam_size,
                    token_list=self.token_list,
                    hold_n=hold_n,
                    **transducer_conf,
                )
                del self.scorers["decoder"]
                self.t = 0
            else:
                scorers = self.scorers.copy()
                scorers["ctc"] = ctc
                self.time_sync_search = BeamSearchTimeSyncStreaming(
                    beam_size=self.beam_size,
                    weights=self.weights,
                    scorers=scorers,
                    sos=self.sos,
                    token_list=self.token_list,
                    hold_n=hold_n,
                )
                self.t = 0

        self._block_is_final = True
        self._new_encoder_frames = None
        # Per-utterance cross-attn dump buffer:
        # list of (processed_block, enc_len, attn_dist_np_1d_T_k) per scoring step.
        self.xattn_dump_enabled = False
        self.xattn_dump_buffer = []
        # Attention-edge stop: stop the chunk (don't commit the new token) when
        # the top beam's cross-attn argmax is within `attn_edge_stop_margin`
        # frames of the rightmost encoder frame on a non-final call. 0 disables.
        self.attn_edge_stop_margin = 0
        self.reset()

    def reset(self):
        """Reset parameters."""
        self.encbuffer = None
        self.running_hyps = None
        self.prev_hyps = []
        self.ended_hyps = []
        self.processed_block = 0
        self.process_idx = 0
        self.prev_output = None
        self.prev_incremental = None
        self.prev_enc_len = 0
        self.xattn_dump_buffer = []
        self._new_encoder_frames = None
        self.attn_center = None  # reset monotonic-attn tracker between utts

    def snapshot_state(self) -> dict:
        """Capture mutable per-utterance state for chunk-level rollback (Branch B).

        Returns a dict that can later be restored via :meth:`restore_state`.
        Used by the dynamic future-chunks chunk_rollback mode: snapshot
        before running beam search on a candidate chunk; if the post-decode
        signal trips, restore so the chunk can be re-decoded under more
        future context on the next call.
        """
        import copy
        return {
            "encbuffer": (
                None if self.encbuffer is None else self.encbuffer.detach().clone()
            ),
            "running_hyps": copy.deepcopy(self.running_hyps),
            "prev_hyps": copy.deepcopy(self.prev_hyps),
            "ended_hyps": copy.deepcopy(self.ended_hyps),
            "processed_block": int(self.processed_block),
            "process_idx": int(self.process_idx),
            "prev_output": copy.deepcopy(self.prev_output),
            "prev_incremental": copy.deepcopy(self.prev_incremental),
            "prev_enc_len": int(self.prev_enc_len),
            "_new_encoder_frames": (
                None
                if self._new_encoder_frames is None
                else self._new_encoder_frames.detach().clone()
            ),
            "attn_center": self.attn_center,
            # xattn_dump_buffer is append-only and used for diagnostics;
            # snapshot length so we can truncate back on restore.
            "_xattn_dump_buffer_len": len(self.xattn_dump_buffer),
        }

    def restore_state(self, snap: dict) -> None:
        """Restore per-utterance state from a :meth:`snapshot_state` snapshot."""
        self.encbuffer = snap["encbuffer"]
        self.running_hyps = snap["running_hyps"]
        self.prev_hyps = snap["prev_hyps"]
        self.ended_hyps = snap["ended_hyps"]
        self.processed_block = snap["processed_block"]
        self.process_idx = snap["process_idx"]
        self.prev_output = snap["prev_output"]
        self.prev_incremental = snap["prev_incremental"]
        self.prev_enc_len = snap["prev_enc_len"]
        self._new_encoder_frames = snap["_new_encoder_frames"]
        self.attn_center = snap["attn_center"]
        # Drop any xattn_dump_buffer entries appended after the snapshot.
        n_keep = snap["_xattn_dump_buffer_len"]
        self.xattn_dump_buffer = self.xattn_dump_buffer[:n_keep]

    def _build_cross_attn_mask(self, hyp, x):
        """Build per-token sliding-window cross-attention mask for decoder.

        Uses ChunkEmissionIndex (which chunk each token was emitted in) to
        build the same sliding-window mask used during training.

        Args:
            hyp: BatchHypothesis with ChunkEmissionIndex.
            x: encoder output (n_batch, T_enc, D).

        Returns:
            Bool mask (B, tgt_len, T_enc) where True=allow, or None if
            full attention (both L=-1 and R=-1, or offline mode).
        """
        L = self.cross_attn_num_left_chunks
        R = self.cross_attn_num_right_chunks
        cs = self.cross_attn_chunk_size

        # Only skip mask when both unlimited (full attention / offline)
        if (L < 0 and R < 0) or cs <= 0:
            return None

        B, T_enc = x.shape[0], x.shape[1]
        tgt_len = hyp.yseq.shape[1]
        device = x.device

        # Token chunk indices: SOS → chunk 0, others → ChunkEmissionIndex.
        # The LAST row of the mask is the query that scores the NEW token.
        # Its chunk should be the CURRENT processed_block (the chunk being
        # decoded now), not the previous token's emission chunk — otherwise
        # the new encoder frames that arrive while the beam is held remain
        # masked out and the decoder's prediction is frozen.
        sos_chunk = torch.zeros(B, 1, device=device, dtype=torch.long)
        if hyp.ChunkEmissionIndex.numel() > 0:
            token_chunks = hyp.ChunkEmissionIndex[:, :tgt_len - 1].to(
                device=device, dtype=torch.long
            )
            chunk_indices = torch.cat([sos_chunk, token_chunks], dim=1).clone()
        else:
            chunk_indices = sos_chunk.expand(B, tgt_len).clone()
        # Override the last position with the current chunk being processed.
        current_chunk = getattr(self, "processed_block", 0)
        chunk_indices[:, -1] = current_chunk

        frame_idx = torch.arange(T_enc, device=device).view(1, 1, T_enc)

        # Start with all-True mask
        mask = torch.ones(B, tgt_len, T_enc, dtype=torch.bool, device=device)

        # Right boundary: (c + 1 + R) * cs - 1 (only when R >= 0)
        if R >= 0:
            right_bound = ((chunk_indices + 1 + R) * cs - 1).unsqueeze(-1)
            mask = mask & (frame_idx <= right_bound)

        # Left boundary: max(0, c - L) * cs (only when L >= 0)
        if L >= 0:
            left_bound = ((chunk_indices - L).clamp(min=0) * cs).unsqueeze(-1)
            mask = mask & (frame_idx >= left_bound)

        # Monotonic attention floor: restrict the LAST row (the new token
        # being scored) to frames at or after (attn_center - floor). This
        # prevents cross-attention from drifting backward into already-
        # covered audio at chunk boundaries — the AED duplication failure.
        if (
            self.monotonic_attn_floor is not None
            and self.monotonic_attn_floor >= 0
            and self.attn_center is not None
        ):
            mono_floor_frame = max(0, self.attn_center - self.monotonic_attn_floor)
            if mono_floor_frame > 0 and tgt_len > 0:
                last_row_mask = (frame_idx >= mono_floor_frame)  # (1, 1, T_enc)
                mask[:, -1:, :] = mask[:, -1:, :] & last_row_mask

        return mask

    def score_full(
        self,
        hyp: BatchHypothesis,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature
            pre_x (torch.Tensor): Encoded speech feature for
                sequential attention (T, D)

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()

        # Build cross-attention mask for decoder (None if full attention).
        # When cross-attn KV cache sliding window is active, the cache itself
        # limits attention scope, so skip the mask to avoid shape mismatch.
        _dec_scorer = self.full_scorers.get("decoder")
        _use_sliding = (
            _dec_scorer is not None
            and getattr(_dec_scorer, "use_kvcache", False)
            and self.cross_attn_num_left_chunks >= 0
            and self.cross_attn_chunk_size > 0
        )
        cross_attn_mask = None if _use_sliding else self._build_cross_attn_mask(hyp, x)

        for k, d in self.full_scorers.items():
            if (
                self.decoder_text_length_limit > 0
                and len(hyp.yseq) > 0
                and len(hyp.yseq[0]) > self.decoder_text_length_limit
            ):
                temp_yseq = hyp.yseq.narrow(
                    1, -self.decoder_text_length_limit, self.decoder_text_length_limit
                ).clone()
                temp_yseq[:, 0] = self.sos
                self.running_hyps.states["decoder"] = [
                    None for _ in self.running_hyps.states["decoder"]
                ]
            else:
                temp_yseq = hyp.yseq

            # For decoder scorer with KV cache, pass new_encoder_frames for
            # incremental cross-attention extension (only on first call per chunk).
            _new_enc = None
            _max_cache = -1
            if "decoder" in k and getattr(d, "use_kvcache", False):
                _new_enc = self._new_encoder_frames
                self._new_encoder_frames = None  # extend only once per chunk
                # Sliding window: keep (num_left_chunks + 1) * chunk_size frames
                L = self.cross_attn_num_left_chunks
                cs = self.cross_attn_chunk_size
                if L >= 0 and cs > 0:
                    _max_cache = (L + 1) * cs

            if "decoder" in k and self.return_hs:
                (scores[k], hs), states[k] = d.batch_score(
                    temp_yseq, hyp.states[k], x, return_hs=self.return_hs,
                    new_encoder_frames=_new_enc,
                    memory_mask=cross_attn_mask,
                    cross_attn_max_cache_frames=_max_cache,
                )
            elif "decoder" in k and pre_x is not None:
                scores[k], states[k] = d.batch_score(temp_yseq, hyp.states[k], x, pre_x)
            elif "decoder" in k:
                scores[k], states[k] = d.batch_score(
                    temp_yseq, hyp.states[k], x,
                    new_encoder_frames=_new_enc,
                    memory_mask=cross_attn_mask,
                    cross_attn_max_cache_frames=_max_cache,
                )
            else:
                scores[k], states[k] = d.batch_score(temp_yseq, hyp.states[k], x)

        # Update self.attn_center from the decoder's last-layer cross-attn
        # for the top beam (beam 0). Used by _build_cross_attn_mask on the
        # NEXT call to enforce monotonic attention floor. Also push full
        # distribution into xattn_dump_buffer when dump is enabled.
        if (
            self.xattn_dump_enabled
            or self.attn_edge_stop_margin > 0
            or (self.monotonic_attn_floor is not None
                and self.monotonic_attn_floor >= 0)
        ):
            try:
                _dec = self.full_scorers.get("decoder")
                if _dec is None:
                    logging.warning("[xattn dump] decoder not in full_scorers")
                elif not hasattr(_dec, "decoders"):
                    logging.warning(
                        f"[xattn dump] _dec has no .decoders attr; type={type(_dec).__name__}"
                    )
                else:
                    _last = _dec.decoders[-1]
                    if not hasattr(_last, "src_attn") or _last.src_attn is None:
                        logging.warning("[xattn dump] last decoder has no src_attn")
                    else:
                        _attn = getattr(_last.src_attn, "attn", None)
                        if _attn is None:
                            logging.warning("[xattn dump] src_attn.attn is None")
                        elif _attn.numel() == 0:
                            logging.warning("[xattn dump] src_attn.attn is empty")
                        else:
                            # _attn shape: (B, H, T_q, T_k); read last query, beam 0.
                            _attn_last_q = _attn[0, :, -1, :]  # (H, T_k)
                            _weights = _attn_last_q.mean(dim=0)  # (T_k,)
                            self.attn_center = int(_weights.argmax().item())
                            if self.xattn_dump_enabled:
                                self.xattn_dump_buffer.append((
                                    int(self.processed_block),
                                    int(_weights.shape[0]),
                                    _weights.detach().cpu().numpy(),
                                ))
            except Exception as e:
                logging.warning(f"[xattn dump] capture failed: {e!r}")

        if self.return_hs:
            return hs, scores, states
        return scores, states

    def _is_within_chunk_repeat(self, hyp_idx: int, new_token: int) -> bool:
        """Return True if token was already emitted in the CURRENT chunk.

        Strategy 1: within-chunk repetition -> caller should stop the chunk.
        """
        running_hyps = getattr(self, '_search_running_hyps', None)
        current_chunk = getattr(self, '_search_current_chunk', 0)
        block_is_final = getattr(self, '_block_is_final', True)

        if (block_is_final
                or self.disable_repetition_detection
                or running_hyps is None):
            return False

        seq_len = int(running_hyps.length[hyp_idx].item())
        if seq_len <= 1:
            return False

        avail = (
            running_hyps.ChunkEmissionIndex.shape[1]
            if running_hyps.ChunkEmissionIndex.dim() >= 2
            else 0
        )
        check_len = min(seq_len - 1, avail)
        if check_len <= 0:
            return False

        chunk_idx = running_hyps.ChunkEmissionIndex[hyp_idx, :check_len]
        tokens = running_hyps.yseq[hyp_idx, 1:check_len + 1]
        in_current = tokens[chunk_idx == current_chunk]
        if len(in_current) > 0 and (in_current == new_token).any():
            logging.debug(
                f"[within_chunk_repeat] hyp {hyp_idx}: token {new_token} "
                f"already in chunk {current_chunk}"
            )
            return True
        return False

    def _is_cross_chunk_repeat(self, hyp_idx: int, new_token: int) -> bool:
        """Return True if new_token equals the LAST token of the previous
        chunk AND no token has been emitted in the current chunk yet.

        Strategy 2: cross-chunk repetition -> caller should mask out
        this token and re-run beam selection.

        Rationale: the streaming decoder sometimes "stutters" across chunk
        boundaries, re-emitting the prev chunk's last token as the first
        token of the new chunk. Blocking this specific pattern is safe
        (legitimate within-chunk repeats are not affected).
        """
        running_hyps = getattr(self, '_search_running_hyps', None)
        current_chunk = getattr(self, '_search_current_chunk', 0)
        block_is_final = getattr(self, '_block_is_final', True)

        if (block_is_final
                or self.disable_repetition_detection
                or running_hyps is None
                or current_chunk <= 0):
            return False

        seq_len = int(running_hyps.length[hyp_idx].item())
        if seq_len <= 1:
            return False

        avail = (
            running_hyps.ChunkEmissionIndex.shape[1]
            if running_hyps.ChunkEmissionIndex.dim() >= 2
            else 0
        )
        check_len = min(seq_len - 1, avail)
        if check_len <= 0:
            return False

        chunk_idx = running_hyps.ChunkEmissionIndex[hyp_idx, :check_len]
        tokens = running_hyps.yseq[hyp_idx, 1:check_len + 1]

        # Only fire on the FIRST emission of the current chunk: if any
        # token has already been committed in current_chunk, don't block.
        if (chunk_idx == current_chunk).any():
            return False

        # Find the LAST token emitted in the immediately previous chunk.
        prev_chunk = current_chunk - 1
        prev_mask = (chunk_idx == prev_chunk)
        if not prev_mask.any():
            return False  # prev chunk was silent, nothing to compare
        # last position where chunk_idx == prev_chunk
        last_idx = int(prev_mask.nonzero(as_tuple=True)[0].max().item())
        last_tok_of_prev = int(tokens[last_idx].item())

        if int(new_token) == last_tok_of_prev:
            logging.info(
                f"[cross_chunk_repeat] chunk {current_chunk}: top-1 "
                f"token {new_token} matches last token of chunk "
                f"{prev_chunk} — masking for retry"
            )
            return True
        return False

    def _is_low_confidence_chunk_start(
        self, hyp_idx: int, new_token: int, step_probs: torch.Tensor
    ) -> bool:
        """Return True if this is the first token of a new chunk AND
        the top-1 softmax confidence is below chunk_start_confidence_threshold.

        Used by search() to signal cross_chunk_repeat-style mask+retry.
        """
        if self.chunk_start_confidence_threshold <= 0.0:
            return False

        running_hyps = getattr(self, '_search_running_hyps', None)
        current_chunk = getattr(self, '_search_current_chunk', 0)
        block_is_final = getattr(self, '_block_is_final', True)

        if block_is_final or running_hyps is None or current_chunk == 0:
            return False

        seq_len = int(running_hyps.length[hyp_idx].item())
        if seq_len <= 1:
            return False

        avail = (
            running_hyps.ChunkEmissionIndex.shape[1]
            if running_hyps.ChunkEmissionIndex.dim() >= 2
            else 0
        )
        check_len = min(seq_len - 1, avail)
        if check_len <= 0:
            return False

        chunk_indices = running_hyps.ChunkEmissionIndex[hyp_idx, :check_len]
        if (chunk_indices == current_chunk).any():
            return False

        top1_prob = step_probs[new_token].item()
        if top1_prob < self.chunk_start_confidence_threshold:
            logging.info(
                f"[chunk_start_lowconf] P2 hyp {hyp_idx}: token {new_token} "
                f"at start of chunk {current_chunk} has conf {top1_prob * 100:.1f}% "
                f"< {self.chunk_start_confidence_threshold * 100:.1f}%"
            )
            return True
        return False

    def _rebuild_ctc_state(self, yseq_row: torch.Tensor):
        """Replay a (truncated) prefix through the CTC prefix scorer.

        Sequential decoding pairs hyp.states['ctc'] with its prefix via
        per-step select_state; after a rollback truncation the stored state
        still spans the old, longer prefix and would corrupt all subsequent
        CTC prefix scores. There is no one-shot API for "state of an
        arbitrary prefix", so rebuild it the way decoding would have:
        score each kept token in order and select its state.
        """
        ctc = self.scorers.get("ctc")
        if ctc is None or getattr(ctc, "impl", None) is None:
            return None
        state = None  # re-batched form, feeds the next impl call
        sel = None    # per-hyp form, what hyp.states["ctc"] stores
        for t in range(yseq_row.shape[0] - 1):
            prefix = yseq_row[: t + 1].unsqueeze(0)
            _, st = ctc.impl(prefix, state)
            sel = ctc.select_state(st, 0, new_id=int(yseq_row[t + 1]))
            # Re-batch the per-hyp selection for the next TH call, mirroring
            # CTCPrefixScorer.batch_score_partial's stacking.
            state = (
                torch.stack([sel[0]], dim=2),
                torch.stack([sel[1]]),
                sel[2],
                sel[3],
            )
        return sel

    def _rollback_and_confirm_boundary(
        self, h: torch.Tensor, running_hyps: BatchHypothesis
    ) -> BatchHypothesis:
        """Re-score low-confidence tokens from the previous chunk.

        Policy 3. At the start of a new chunk, walk the tokens emitted in the
        previous chunk whose emission-time softmax was below
        rollback_confidence_threshold (earliest first). For each such position:

          * Truncate the hypothesis to that position (drop that token and all
            subsequent tokens from the prefix).
          * Re-score the token at that position under the grown encoder.
          * If new top-1 == stored token → CONFIRM: update the stored
            confidence, keep walking.
          * If new top-1 != stored token → REPLACE: set that position to the
            new top-1, drop everything after it, stop walking. The next
            process_one_block will continue decoding from this shorter prefix.

        Scope: operates on the top-1 beam only.
        """
        if self.rollback_confidence_threshold <= 0.0:
            return running_hyps
        if running_hyps is None or running_hyps.yseq.shape[0] == 0:
            return running_hyps
        prev_chunk = self.processed_block - 1
        if prev_chunk < 0:
            return running_hyps

        # Scope to top-1 beam
        if running_hyps.yseq.shape[0] > 1:
            hyps = self._batch_select(running_hyps, torch.tensor([0]))
        else:
            hyps = running_hyps

        if not hyps.confidence_list or len(hyps.confidence_list[0]) == 0:
            return running_hyps

        conf_list = list(hyps.confidence_list[0])
        seq_len = int(hyps.length[0].item())
        chunk_idx_full = hyps.ChunkEmissionIndex[0]
        yseq = hyps.yseq[0].clone()

        # Word-tail unconditional rollback for streaming F3 cases.
        #
        # Cause: at chunk boundaries the chunked beam decoder picks SHORTER
        # BPE pieces because the encoder context is impoverished. Defer
        # (full-context single-call beam) picks longer in-vocab pieces
        # (``▁IRISH`` instead of ``▁I RI S``; ``ING`` instead of ``IN ING``
        # after ``▁CONTAIN``). Once the short pieces are committed, no
        # subsequent extension can recover the longer single-piece option.
        #
        # Strategy: at each chunk transition, find the tokens emitted in
        # the previous chunk that belong to the CURRENT in-progress word
        # (everything in prev_chunk AFTER the most recent ``▁``-prefixed
        # token), and unconditionally re-decode them under the grown
        # encoder. The continuation-only gate means we never touch the
        # ``▁``-prefixed word-start itself — that's still committed.
        # This is safe (does not cross word boundaries → cannot cause
        # word-substitution regressions like ``▁COLOR`` → ``▁COLLAR``)
        # and addresses the BPE over-segmentation mechanism directly.
        prev_chunk_positions = []
        for i in range(min(len(conf_list), seq_len - 1)):
            if int(chunk_idx_full[i].item()) == prev_chunk:
                prev_chunk_positions.append(i)
        if not prev_chunk_positions:
            return running_hyps
        # Find the most recent ``▁``-prefixed token within prev chunk
        # positions. Everything AFTER that is the current word's tail
        # emitted in prev_chunk → these are the rollback candidates.
        word_start_idx_in_list = -1
        for j in range(len(prev_chunk_positions) - 1, -1, -1):
            pos = prev_chunk_positions[j]
            tid = int(yseq[pos + 1].item())
            tok_str = (
                self.token_list[tid]
                if (self.token_list is not None and 0 <= tid < len(self.token_list))
                else ""
            )
            if tok_str.startswith("▁"):
                word_start_idx_in_list = j
                break
        # Positions to re-decode: the word-start (if it's in prev_chunk)
        # plus everything in prev_chunk AFTER it. Including the word-start
        # itself lets us upgrade ``▁I RI`` to ``▁IRISH`` when the grown
        # encoder prefers the longer single piece. The prefix-extension
        # safety rule in the REPLACE branch keeps this safe by allowing
        # word-start replacement ONLY when the OLD token's text is a
        # prefix of the NEW token's text (so ``▁I`` → ``▁IRISH`` works
        # but ``▁COLOR`` → ``▁COLLAR`` is blocked).
        if word_start_idx_in_list >= 0:
            tail_positions = prev_chunk_positions[word_start_idx_in_list:]
        else:
            # Word started in an earlier chunk; re-decode prev_chunk's
            # continuation pieces only.
            tail_positions = list(prev_chunk_positions)
        if not tail_positions:
            return running_hyps
        # Cap depth: rolling back too many tokens risks dropping content
        # that won't be re-emitted under the new encoder.
        K = getattr(self, "rollback_max_depth", 3)
        if len(tail_positions) > K:
            tail_positions = tail_positions[-K:]
        # Unconditional walk: walk EARLIEST-first (so re-decode from the
        # leftmost continuation piece in the current word's tail and
        # cascade forward). Confidence gate is NOT applied.
        low_conf_positions = list(tail_positions)
        logging.info(
            f"[rollback] boundary {prev_chunk}→{self.processed_block}: "
            f"ENTER word-tail rollback positions={low_conf_positions}"
        )

        _dec = self.full_scorers.get("decoder")
        if _dec is None:
            return running_hyps
        n_layers = len(_dec.decoders)
        _use_kvcache = getattr(_dec, "use_kvcache", False)

        n_confirmed = 0
        n_replaced = 0
        # Each rescore pass uses a clean state (states=[None]); the batch_score
        # path clears internal caches and rebuilds from the truncated prefix.
        for pos in low_conf_positions:
            target_len = pos + 1  # [SOS, t_0, ..., t_{pos-1}]
            stored_token = int(yseq[target_len].item())

            trunc_yseq = yseq[:target_len].unsqueeze(0)
            # Build cross-attn mask for the truncated prefix
            if pos > 0:
                trunc_chunk_idx = chunk_idx_full[:pos].unsqueeze(0)
            else:
                trunc_chunk_idx = torch.empty(
                    (1, 0), dtype=chunk_idx_full.dtype, device=chunk_idx_full.device
                )
            mask_hyp = BatchHypothesis(
                yseq=trunc_yseq, ChunkEmissionIndex=trunc_chunk_idx
            )
            memory_mask = self._build_cross_attn_mask(mask_hyp, h.unsqueeze(0))

            fresh_states = [None] * n_layers
            logp, _ = _dec.batch_score(
                trunc_yseq,
                fresh_states,
                h.unsqueeze(0),
                new_encoder_frames=None,
                memory_mask=memory_mask,
            )
            new_top1 = int(logp[0].argmax().item())
            new_conf = float(torch.softmax(logp[0], dim=-1)[new_top1].item())

            if new_top1 == stored_token:
                conf_list[pos] = new_conf
                n_confirmed += 1
                logging.info(
                    f"[rollback] chunk {prev_chunk} pos {pos}: CONFIRM token "
                    f"{stored_token}, conf {conf_list[pos] * 100:.1f}% → "
                    f"{new_conf * 100:.1f}%"
                )
                continue
            elif new_top1 == self.eos:
                # EOS veto: the re-decoder's top-1 is EOS, but we're mid-utterance
                # (Policy 3 only fires on non-final chunk boundaries via the
                # processed_block > 0 gate in forward()). REPLACE here would
                # write EOS mid-sequence and drop the rest of the hypothesis,
                # causing mass deletions. Skip this position — keep the stored
                # token, update its stored confidence, continue walking.
                conf_list[pos] = new_conf
                logging.info(
                    f"[rollback] chunk {prev_chunk} pos {pos}: EOS_VETO "
                    f"(new top1=EOS, keeping stored token {stored_token}, "
                    f"conf updated to {new_conf * 100:.1f}%)"
                )
                continue
            else:
                # F3 safety with prefix-extension allowance:
                # - If stored is a continuation piece (no ``▁``) and new is
                #   ``▁``-prefixed: refuse (would re-segment into different
                #   words under partial encoder).
                # - If stored is ``▁``-prefixed (word-start) and new is also
                #   ``▁``-prefixed: allow ONLY if stored_text is a strict
                #   prefix of new_text (extension: ``▁I`` → ``▁IRISH``).
                #   Block substitution (``▁COLOR`` → ``▁COLLAR``).
                # - If new is a continuation piece (no ``▁``): always allow.
                stored_tok_str = (
                    self.token_list[stored_token]
                    if (self.token_list is not None and 0 <= stored_token < len(self.token_list))
                    else ""
                )
                _new_tok_str = (
                    self.token_list[new_top1]
                    if (self.token_list is not None and 0 <= new_top1 < len(self.token_list))
                    else ""
                )
                if _new_tok_str.startswith("▁"):
                    stored_is_wordstart = stored_tok_str.startswith("▁")
                    if not stored_is_wordstart:
                        # Continuation piece being replaced by a word-start —
                        # would silently re-segment. Refuse.
                        conf_list[pos] = new_conf
                        logging.info(
                            f"[rollback] chunk {prev_chunk} pos {pos}: KEEP "
                            f"(new top1 {_new_tok_str!r} is word-start, stored is continuation; "
                            f"refusing to re-segment), conf updated to {new_conf * 100:.1f}%"
                        )
                        continue
                    # Both stored and new are word-starts. Apply prefix-extension
                    # safety: new must extend stored (stored_text is a strict
                    # prefix of new_text). Allows ``▁I`` → ``▁IRISH`` and blocks
                    # ``▁COLOR`` → ``▁COLLAR``.
                    stored_text = stored_tok_str.lstrip("▁")
                    new_text = _new_tok_str.lstrip("▁")
                    if not (
                        len(new_text) > len(stored_text)
                        and new_text.startswith(stored_text)
                    ):
                        conf_list[pos] = new_conf
                        logging.info(
                            f"[rollback] chunk {prev_chunk} pos {pos}: KEEP "
                            f"(new top1 {_new_tok_str!r} is not a prefix-extension of "
                            f"stored {stored_tok_str!r}; refusing), conf updated to {new_conf * 100:.1f}%"
                        )
                        continue
                yseq = yseq[: target_len + 1].clone()
                yseq[target_len] = new_top1
                conf_list = conf_list[: pos + 1]
                conf_list[pos] = new_conf
                n_replaced += 1
                logging.info(
                    f"[rollback] chunk {prev_chunk} pos {pos}: REPLACE token "
                    f"{stored_token} → {new_top1} (conf {new_conf * 100:.1f}%), "
                    f"dropped {seq_len - 1 - target_len} trailing token(s)"
                )
                break

        logging.info(
            f"[rollback] boundary {prev_chunk}→{self.processed_block}: "
            f"{len(low_conf_positions)} low-conf, {n_confirmed} confirmed, "
            f"{n_replaced} replaced"
        )

        # Rebuild BatchHypothesis with modified yseq / confidence_list.
        # Reset decoder states so the next process_one_block rebuilds caches
        # for the final (possibly truncated) prefix.
        if _use_kvcache:
            _dec._clear_kvcache()
        new_len = yseq.shape[0]
        device = hyps.yseq.device
        token_age = hyps.TokenAgeIndex[0]
        chunk_idx = chunk_idx_full
        new_token_age = (
            token_age[: new_len - 1]
            if new_len > 1
            else torch.empty((0,), dtype=token_age.dtype, device=device)
        )
        new_chunk_idx = (
            chunk_idx[: new_len - 1]
            if new_len > 1
            else torch.empty((0,), dtype=chunk_idx.dtype, device=device)
        )
        scores_list = (
            hyps.scores_list[0][: new_len - 1]
            if hyps.scores_list and hyps.scores_list[0]
            else []
        )
        new_states = {k: v for k, v in hyps.states.items()}
        new_states["decoder"] = [None]
        # On truncation (REPLACE), the shallow-copied CTC state still spans
        # the old, longer prefix — rebuild it for the kept prefix (mirrors
        # the decoder-state reset above). Confirm-only walks keep the full
        # prefix, for which the existing state remains correct.
        if n_replaced > 0 and "ctc" in new_states:
            new_states["ctc"] = [self._rebuild_ctc_state(yseq)]

        new_hyps = BatchHypothesis(
            yseq=yseq.unsqueeze(0),
            score=hyps.score,
            length=torch.tensor(
                [new_len], dtype=hyps.length.dtype, device=hyps.length.device
            ),
            scores=hyps.scores,
            states=new_states,
            hs=[],
            scores_list=[scores_list],
            confidence_list=[conf_list],
            TokenAgeIndex=new_token_age.unsqueeze(0),
            ChunkEmissionIndex=new_chunk_idx.unsqueeze(0),
        )
        # Invalidate _new_encoder_frames: score_full will use full h and the
        # decoder's "first call" path will materialize the cross-attn cache.
        self._new_encoder_frames = None
        return new_hyps

    def forward(
        self,
        x: torch.Tensor,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        is_final: bool = True,
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        if self.encbuffer is None:
            self.encbuffer = x
        else:
            self.encbuffer = torch.cat([self.encbuffer, x], axis=0)

        x = self.encbuffer

        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))

        if minlenratio < 0:
            minlen = -1 * int(minlenratio)
        else:
            minlen = int(minlenratio * x.size(0))

        # set block_size == 0 for recomputing
        if self.block_size == 0:
            block_is_final = is_final
            h = x
            logging.debug(
                "  Feature length: {}, current position: {}".format(
                    h.shape[0], self.process_idx
                )
            )

            if self.running_hyps is None:  # init hyps
                self.running_hyps = self.init_hyp(h)
                self.prev_incremental = self.running_hyps
                self.prev_enc_len = 0
            elif self.running_hyps.yseq.shape[0] == 0:
                # All beams ended on a previous non-final chunk.
                # Re-init so the decoder can re-decode with more encoder frames.
                logging.info("Re-initializing hypotheses (all ended on previous chunk).")
                self.running_hyps = self.init_hyp(h)
                self.process_idx = 0
                self.prev_enc_len = 0
                # Clear decoder KV caches for fresh start
                _dec = self.full_scorers.get("decoder")
                if _dec is not None and getattr(_dec, "use_kvcache", False):
                    _dec._clear_kvcache()

            # Handle decoder state for new chunk
            _dec = self.full_scorers.get("decoder")
            _use_kvcache = _dec is not None and getattr(_dec, "use_kvcache", False)

            # When decoder KV cache is enabled, keep decoder states across
            # chunk boundaries so the decoder continues incrementally:
            # - Self-attention: cached KV from past positions is preserved,
            #   only the new token position is computed.
            # - Cross-attention: recomputed for the new token using the full
            #   (grown) encoder output; past positions are frozen in cache.
            # When KV cache is disabled, reset states to force full
            # recomputation (original behavior).
            if not _use_kvcache:
                if (
                    self.running_hyps is not None
                    and "decoder" in self.running_hyps.states
                ):
                    self.running_hyps.states["decoder"] = [
                        None for _ in self.running_hyps.states["decoder"]
                    ]

            # Compute new encoder frames for cross-attention KV cache extension
            if _use_kvcache and self.prev_enc_len > 0 and h.shape[0] > self.prev_enc_len:
                self._new_encoder_frames = h[self.prev_enc_len:].unsqueeze(0)
            else:
                self._new_encoder_frames = None
            self.prev_enc_len = h.shape[0]

            # Policy 3: confidence-gated rollback + agreement confirmation.
            # Re-score low-confidence tokens from the previous chunk under the
            # grown encoder. Called before process_one_block so downstream
            # decoding uses the post-rollback prefix.
            if (
                self.rollback_confidence_threshold > 0.0
                and self.running_hyps is not None
                and self.running_hyps.yseq.shape[0] > 0
                and self.processed_block > 0
            ):
                self.running_hyps = self._rollback_and_confirm_boundary(
                    h, self.running_hyps
                )

            if self.time_sync:
                ret = self.process_one_block_time_sync(
                    h, block_is_final, maxlen, maxlenratio
                )
            else:
                ret = self.process_one_block(
                    h, block_is_final, maxlen, minlen, maxlenratio
                )
            logging.debug("Finished processing chunk: %d", self.processed_block)
            self.processed_block += 1

            # prune running_hyps, taking top as an incremental decoding
            if self.incremental_decode and not is_final:
                if (
                    self.running_hyps.yseq.shape[0] == 0
                ):  # running_hyps will be empty if maxlen is reached
                    logging.info(
                        "search stopped by maxlen in a non final chunk. \
                        reverting to prev running hyp"
                    )
                    self.running_hyps = self.prev_incremental
                logging.info(
                    "Hyps before incremental pruning: %d",
                    self.running_hyps.yseq.shape[0],
                )
                if self.running_hyps.yseq.shape[0] > 0:
                    self.running_hyps = self._batch_select(self.running_hyps, [0])
                    self.prev_incremental = self.running_hyps
                logging.info(
                    "Hyps after incremental pruning: %d",
                    self.running_hyps.yseq.shape[0],
                )

                if self.token_list is not None:
                    logging.info(
                        "best running hypo: "
                        + "".join(
                            [self.token_list[x] for x in self.running_hyps.yseq[0, 1:]]
                        )
                    )

                # hold_n
                if self.hold_n > 0 and self.running_hyps.length[0] > 2:
                    self.running_hyps = BatchHypothesis(
                        score=self.running_hyps.score,
                        scores=self.running_hyps.scores,
                        states=self.running_hyps.states,
                        length=self.running_hyps.length - self.hold_n,
                        yseq=self.running_hyps.yseq[:, : -self.hold_n],
                        hs=[],
                        TokenAgeIndex=self.running_hyps.TokenAgeIndex[:, : -self.hold_n],
                        ChunkEmissionIndex=self.running_hyps.ChunkEmissionIndex[:, : -self.hold_n],
                    )
                    if self.token_list is not None:
                        logging.info(
                            "best hypo after hold: "
                            + "".join(
                                [
                                    self.token_list[x]
                                    for x in self.running_hyps.yseq[0, 1:]
                                ]
                            )
                        )

            if is_final:
                if len(ret) == 0:
                    if self.prev_output is None:
                        return []
                    else:
                        return self.prev_output
                else:
                    return ret
            else:
                # dont return incremental hyps,
                # check them by grabbing top running_hyp
                if len(ret) > 0:
                    self.prev_output = ret
                return []

        # blockwise processing w/ rewinding
        else:
            ret = None
            while True:
                cur_end_frame = (
                    self.block_size
                    - self.look_ahead
                    + self.hop_size * self.processed_block
                )
                if cur_end_frame < x.shape[0]:
                    h = x.narrow(0, 0, cur_end_frame)
                    block_is_final = False
                else:
                    if is_final:
                        h = x
                        block_is_final = True
                    else:
                        break

                logging.debug("Start processing block: %d", self.processed_block)
                logging.debug(
                    "  Feature length: {}, current position: {}".format(
                        h.shape[0], self.process_idx
                    )
                )
                if (
                    self.encoded_feat_length_limit > 0
                    and h.shape[0] > self.encoded_feat_length_limit
                ):
                    h = h.narrow(
                        0,
                        h.shape[0] - self.encoded_feat_length_limit,
                        self.encoded_feat_length_limit,
                    )

                if self.running_hyps is None:
                    self.running_hyps = self.init_hyp(h)
                if self.time_sync:
                    ret = self.process_one_block_time_sync(
                        h, block_is_final, maxlen, maxlenratio
                    )
                else:
                    ret = self.process_one_block(
                        h, block_is_final, maxlen, minlen, maxlenratio
                    )
                logging.debug("Finished processing block: %d", self.processed_block)
                self.processed_block += 1

                # prune running_hyps, taking top as an incremental decoding
                if self.incremental_decode:
                    logging.debug(
                        "Hyps before incremental pruning: %d",
                        self.running_hyps.yseq.shape[0],
                    )
                    if self.running_hyps.yseq.shape[0] > 0:
                        self.running_hyps = self._batch_select(self.running_hyps, [0])
                    logging.debug(
                        "Hyps after incremental pruning: %d",
                        self.running_hyps.yseq.shape[0],
                    )

                if block_is_final:
                    return ret
            if ret is None:
                if self.prev_output is None:
                    return []
                else:
                    return self.prev_output
            else:
                self.prev_output = ret
                # N-best results
                return ret

    def process_one_block_time_sync(self, h, is_final, maxlen, maxlenratio):
        """Recognize one block w/ time sync."""
        hyps = self.time_sync_search(
            h,
            start_idx=self.t,
            is_final=is_final,
            incremental_decode=self.incremental_decode,
        )
        logging.debug("time:" + str(self.t))
        logging.debug("best_hyp:" + "".join([self.token_list[x] for x in hyps[0].yseq]))
        if is_final:
            self.t = 0
        else:
            self.t = len(h)
        return hyps

    def process_one_block(self, h, is_final, maxlen, minlen, maxlenratio):
        """Recognize one block.

        Two-strategy repetition handling:
        1. Within-chunk repeat: token already decoded in this chunk -> stop chunk.
        2. Cross-chunk repeat: token repeats from previous chunk -> mask it out
           in the score distribution and re-run beam selection.
        """
        self._block_is_final = is_final
        # extend states for ctc
        # Skip on the first block of an utterance (running_hyps is None):
        # init_hyp below (re)builds every scorer's state from the full
        # buffer anyway, so extending here would only touch the PREVIOUS
        # utterance's stale CTC impl — wasted work on an object about to be
        # replaced, with a shape-mismatch risk across utterances.
        if self.running_hyps is not None:
            self.extend(h, self.running_hyps)
        # Track silent-chunk diagnostics: how many tokens the inner loop
        # commits during this chunk's process_one_block call.
        _silent_entry_process_idx = self.process_idx
        _silent_chunk_idx = self.processed_block
        _silent_stop_reason = None
        _silent_top1_token = None
        while self.process_idx < maxlen:
            logging.debug("position " + str(self.process_idx))

            # Inner retry loop: if a cross-chunk repeat is detected,
            # mask the offending token and re-run beam selection.
            score_mask = None
            stop_reason = None
            _max_retries = self.n_vocab  # safety bound
            for _retry in range(_max_retries):
                best, stop_reason = self.search(
                    self.running_hyps, h,
                    current_chunk=self.processed_block,
                    score_mask=score_mask,
                )

                if stop_reason == "cross_chunk_repeat":
                    # Get the offending token and add it to the mask
                    hyp_idx, bad_token = self._cross_chunk_repeat_info
                    if score_mask is None:
                        n_batch = self.running_hyps.yseq.shape[0]
                        score_mask = torch.zeros(
                            n_batch, self.n_vocab, dtype=torch.bool,
                            device=self.running_hyps.yseq.device,
                        )
                    score_mask[hyp_idx, bad_token] = True
                    logging.debug(
                        f"[process_one_block] Masking cross-chunk repeat: "
                        f"token {bad_token} for hyp {hyp_idx}, retry {_retry + 1}"
                    )
                    # The failed attempt wrote one phantom position into each
                    # layer's SELF-attn KV cache (the stepwise path always
                    # appends at _kvcache_curlen). REWIND the write cursor so
                    # the retry overwrites that slot with identical content
                    # (same token, same RoPE offset == cursor) instead of
                    # accumulating phantoms. Do NOT _clear_kvcache() here:
                    # mid-utterance prompt rebuild at full beam width takes
                    # the first-call branch whose expand_kv shapes don't hold.
                    # Cross-attn caches are consume-once per block
                    # (new_encoder_frames is nulled after first use) and are
                    # unaffected by retries.
                    _dec = self.full_scorers.get("decoder")
                    if _dec is not None and getattr(_dec, "use_kvcache", False):
                        for _lyr in _dec.decoders:
                            _att = _lyr.self_attn
                            _cur = getattr(_att, "_kvcache_curlen", None)
                            if _cur:
                                _att._kvcache_curlen = max(0, _cur - 1)
                    continue  # retry with updated mask
                else:
                    break  # good token, EOS, or within-chunk repeat

            # EOS or within-chunk repeat -> stop this chunk
            if stop_reason is not None:
                _silent_stop_reason = stop_reason
                break

            # Attention-edge stop: if the top beam's cross-attn argmax is at
            # the rightmost frames of the available encoder on a non-final
            # call, the model is trying to look past its audio. Don't commit;
            # wait for more encoder context on the next call. The previous
            # iteration's running_hyps and process_idx are preserved.
            #
            # Anchor: end of chunk C (= processed_block) when chunk size is
            # known, so under R>0 the rule fires when argmax reaches/crosses
            # C's right edge, treating C+1+ as context-only.
            # Bypass while teacher-forcing a committed prefix token
            # (commit-stable-prefix): attn_edge_stop lives outside search() so
            # the _force_tok guards there don't cover it; without this it could
            # break the chunk mid-forced-prefix and truncate the frozen prefix.
            _fp_es = getattr(self, "_forced_prefix", None)
            _in_forced_es = (
                _fp_es is not None
                and self.running_hyps is not None
                and (self.running_hyps.yseq.shape[1] - 1) < len(_fp_es)
            )
            if (
                self.attn_edge_stop_margin > 0
                and not is_final
                and self.attn_center is not None
                and not _in_forced_es
            ):
                _enc_T = h.shape[0]
                if self.cross_attn_chunk_size > 0:
                    _anchor = min(
                        (self.processed_block + 1) * self.cross_attn_chunk_size,
                        _enc_T,
                    )
                else:
                    _anchor = _enc_T
                if self.attn_center >= _anchor - self.attn_edge_stop_margin:
                    # Capture the candidate token the trigger is blocking
                    # (best is the new beam selection from search() — its top-0
                    # row's last position holds the would-be-committed token).
                    try:
                        _cand_tok = int(best.yseq[0, self.process_idx + 1].item())
                    except Exception:
                        _cand_tok = -1
                    logging.info(
                        f"[attn_edge_stop] argmax={self.attn_center} "
                        f"anchor={_anchor} enc_T={_enc_T} "
                        f"margin={self.attn_edge_stop_margin} "
                        f"candidate_token={_cand_tok} "
                        f"-> stop chunk {self.processed_block} "
                        f"at process_idx={self.process_idx}"
                    )
                    _silent_stop_reason = "attn_at_right_edge"
                    break

            if self.process_idx == maxlen - 1 and (is_final or self.block_size > 0):
                # end decoding (skip for block_size=0 non-final: maxlen
                # will grow on the next chunk so don't force EOS yet)
                self.running_hyps = self.post_process(
                    self.process_idx, maxlen, minlen, maxlenratio, best, self.ended_hyps
                )

            if (
                is_final
                and maxlenratio == 0.0
                and end_detect(
                    [lh.asdict() for lh in self.ended_hyps], self.process_idx
                )
            ):
                logging.info(f"end detected at {self.process_idx}")
                return self.assemble_hyps(self.ended_hyps)

            self.prev_hyps = self.running_hyps
            pp_maxlen = maxlen + 1 if (self.block_size == 0 and not is_final) else maxlen
            self.running_hyps = self.post_process(
                self.process_idx, pp_maxlen, minlen, maxlenratio, best, self.ended_hyps
            )

            if len(self.running_hyps) == 0:
                if not is_final and self.block_size == 0:
                    logging.info(
                        "All hypotheses ended on non-final chunk. "
                        "Will re-init on next chunk."
                    )
                    break
                logging.info("no hypothesis. Finish decoding.")
                return self.assemble_hyps(self.ended_hyps, is_final=is_final)
            else:
                logging.debug(f"remained hypotheses: {len(self.running_hyps)}")
            # increment number
            self.process_idx += 1

        # BEAM-DUMP DIAGNOSTIC: at end of each chunk, dump top-K beams'
        # last tokens + cumul score. Gated by ESPNET_BEAM_DUMP=1.
        import os as _os
        if _os.environ.get("ESPNET_BEAM_DUMP", "") == "1":
            try:
                n_beams = self.running_hyps.yseq.shape[0]
                dump_k = min(5, n_beams)
                scores = self.running_hyps.score[:dump_k].detach().cpu().tolist()
                lengths = self.running_hyps.length[:dump_k].detach().cpu().tolist()
                for k in range(dump_k):
                    L = int(lengths[k])
                    # last up to 8 tokens of this beam
                    start = max(1, L - 8)
                    tail_ids = self.running_hyps.yseq[k, start:L].detach().cpu().tolist()
                    tok_list = getattr(self, "token_list", None)
                    if tok_list is not None:
                        tail_toks = [tok_list[t] if 0 <= t < len(tok_list) else f"<{t}>" for t in tail_ids]
                    else:
                        tail_toks = tail_ids
                    logging.info(
                        f"[BEAM_DUMP] chunk {_silent_chunk_idx} end: "
                        f"beam {k} len={L} cumul={scores[k]:.3f} tail={tail_toks}"
                    )
            except Exception as _e:
                logging.info(f"[BEAM_DUMP] error: {_e}")

        # SILENT-CHUNK DIAGNOSTIC: log non-final chunks that committed 0 tokens.
        _committed_this_chunk = self.process_idx - _silent_entry_process_idx
        if not is_final and _committed_this_chunk == 0:
            logging.info(
                f"[SILENT CHUNK] chunk {_silent_chunk_idx}: 0 tokens committed. "
                f"stop_reason={_silent_stop_reason}, "
                f"process_idx={self.process_idx}/{maxlen}"
            )

        if is_final:
            return self.assemble_hyps(self.ended_hyps, is_final=True)
        else:
            # For non-final chunks with block_size=0, return partial results from ended_hyps
            # (if any hypotheses truly ended and were moved there by post_process)
            rets = self.assemble_hyps(self.ended_hyps, is_final=False)

            if self.block_size > 0 and self.process_idx > 1 and len(self.prev_hyps) > 0:
                self.running_hyps = self.prev_hyps
                self.process_idx -= 1
                self.prev_hyps = []
                # The rewind reverts yseq to a shorter checkpoint, but
                # the decoder attention KV caches still hold entries from
                # the discarded steps. Reset states to None so the next
                # block's batch_score does a full recomputation (cache=None
                # in forward_one_step) which clears and rebuilds the caches.
                if "decoder" in self.running_hyps.states:
                    self.running_hyps.states["decoder"] = [
                        None for _ in self.running_hyps.states["decoder"]
                    ]

            # N-best results
            return rets

    def assemble_hyps(self, ended_hyps, is_final=True):
        """Assemble the hypotheses."""
        if self.normalize_length:
            # Note (Jinchuan): -1 since hyp starts with <sos> and
            # initially has score of 0.0
            nbest_hyps = sorted(
                ended_hyps, key=lambda x: x.score / (len(x.yseq) - 1), reverse=True
            )
        else:
            nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            return []

        # report the best result (only on final chunks to avoid log pollution)
        if is_final:
            best = nbest_hyps[0]
            for k, v in best.scores.items():
                logging.info(
                    f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
                )
            logging.info(f"total log probability: {best.score:.2f}")
            logging.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
            logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
            if self.token_list is not None:
                logging.info(
                    "best hypo: "
                    + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                )
        return nbest_hyps

    def extend(self, x: torch.Tensor, hyps: Hypothesis) -> List[Hypothesis]:
        """Extend probabilities and states with more encoded chunks.

        Args:
            x (torch.Tensor): The extended encoder output feature
            hyps (Hypothesis): Current list of hypothesis

        Returns:
            Hypothesis: The extended hypothesis

        """
        for k, d in self.scorers.items():
            if hasattr(d, "extend_prob"):
                d.extend_prob(x)
            if hasattr(d, "extend_state"):
                hyps.states[k] = d.extend_state(hyps.states[k])
