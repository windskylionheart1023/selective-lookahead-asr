"""Parallel beam search module."""

import logging
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Tuple

import torch
from packaging.version import parse as V
from torch.nn.utils.rnn import pad_sequence

from espnet.nets.beam_search import BeamSearch, Hypothesis

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

logger = logging.getLogger(__name__)


class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type.

    In addition to the upstream fields, carries streaming bookkeeping:
    per-step scorer breakdowns (``scores_list``), per-token top-1
    confidences (``confidence_list``), and per-token age / emission-chunk
    indices (``TokenAgeIndex``, ``ChunkEmissionIndex``) used by the
    chunked streaming beam search.
    """

    yseq: torch.Tensor = torch.tensor([])  # (batch, maxlen)
    score: torch.Tensor = torch.tensor([])  # (batch,)
    length: torch.Tensor = torch.tensor([])  # (batch,)
    scores: Dict[str, torch.Tensor] = dict()  # values: (batch,)
    states: Dict[str, Dict] = dict()
    hs: List[torch.Tensor] = []  # (batch, maxlen, adim)
    scores_list: List[Any] = []
    # Per-beam list of softmax top-1 confidences aligned with emitted tokens
    confidence_list: List[List[float]] = []
    # ESPnet inference runs one utterance per batch, so a scalar per beam suffices.
    TokenAgeIndex: torch.Tensor = torch.tensor([], dtype=torch.int32)  # (batch,)
    # Track which chunk each token was emitted in (batch, maxlen)
    ChunkEmissionIndex: torch.Tensor = torch.tensor([], dtype=torch.int32)  # (batch, maxlen)

    def __len__(self) -> int:
        """Return a batch size."""
        return len(self.length)


class BatchBeamSearch(BeamSearch):
    """Batch beam search implementation."""

    def batchfy(self, hyps: List[Hypothesis]) -> BatchHypothesis:
        """Convert list to batch."""
        if len(hyps) == 0:
            return BatchHypothesis()

        if self.return_hs:
            hs = [h.hs for h in hyps]
        else:
            hs = []

        yseq_batch = pad_sequence(
            [h.yseq for h in hyps], batch_first=True, padding_value=self.eos
        )
        # Pad TokenAgeIndex to handle hypotheses of different lengths
        # (happens when keeping EOS hypotheses in beam on non-final chunks)
        token_age_batch = pad_sequence(
            [h.TokenAgeIndex for h in hyps], batch_first=True, padding_value=0
        )
        # Pad ChunkEmissionIndex to handle hypotheses of different lengths
        chunk_emission_batch = pad_sequence(
            [h.ChunkEmissionIndex for h in hyps], batch_first=True, padding_value=0
        )
        return BatchHypothesis(
            yseq=yseq_batch,
            length=torch.tensor([len(h.yseq) for h in hyps], dtype=torch.int64),
            score=torch.tensor([h.score for h in hyps]),
            scores={k: torch.tensor([h.scores[k] for h in hyps]) for k in self.scorers},
            states={k: [h.states[k] for h in hyps] for k in self.scorers},
            hs=hs,
            scores_list=[h.scores_list for h in hyps],
            confidence_list=[list(h.confidence_list) for h in hyps],
            TokenAgeIndex=token_age_batch,
            ChunkEmissionIndex=chunk_emission_batch,
        )

    def _batch_select(self, hyps: BatchHypothesis, ids: List[int]) -> BatchHypothesis:
        if self.return_hs:
            hs = [hyps.hs[i] for i in ids]
        else:
            hs = []

        ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return BatchHypothesis(
            yseq=hyps.yseq[ids],
            score=hyps.score[ids],
            length=hyps.length[ids],
            scores={k: v[ids] for k, v in hyps.scores.items()},
            states={
                k: [self.scorers[k].select_state(v, i) for i in ids]
                for k, v in hyps.states.items()
            },
            hs=hs,
            scores_list=[hyps.scores_list[i] for i in ids_list] if hyps.scores_list else [],
            confidence_list=[hyps.confidence_list[i] for i in ids_list] if hyps.confidence_list else [],
            TokenAgeIndex=hyps.TokenAgeIndex[ids],
            ChunkEmissionIndex=hyps.ChunkEmissionIndex[ids],
        )

    def _select(self, hyps: BatchHypothesis, i: int) -> Hypothesis:
        return Hypothesis(
            yseq=hyps.yseq[i, : hyps.length[i]],
            score=hyps.score[i],
            scores={k: v[i] for k, v in hyps.scores.items()},
            states={
                k: self.scorers[k].select_state(v, i) for k, v in hyps.states.items()
            },
            hs=hyps.hs[i] if self.return_hs else [],
            scores_list=hyps.scores_list[i],
            confidence_list=hyps.confidence_list[i] if hyps.confidence_list else [],
            TokenAgeIndex=hyps.TokenAgeIndex[i, : hyps.length[i]],
            ChunkEmissionIndex=hyps.ChunkEmissionIndex[i, : hyps.length[i]],
        )

    def unbatchfy(self, batch_hyps: BatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        return [
            Hypothesis(
                yseq=batch_hyps.yseq[i][: batch_hyps.length[i]],
                score=batch_hyps.score[i],
                scores={k: batch_hyps.scores[k][i] for k in self.scorers},
                states={
                    k: v.select_state(batch_hyps.states[k], i)
                    for k, v in self.scorers.items()
                },
                hs=batch_hyps.hs[i] if self.return_hs else [],
                scores_list=batch_hyps.scores_list[i] if batch_hyps.scores_list else [],
                confidence_list=batch_hyps.confidence_list[i] if batch_hyps.confidence_list else [],
                TokenAgeIndex=batch_hyps.TokenAgeIndex[i][: batch_hyps.length[i]],
                ChunkEmissionIndex=batch_hyps.ChunkEmissionIndex[i][: batch_hyps.length[i]],
            )
            for i in range(len(batch_hyps.length))
        ]

    def batch_beam(
        self, weighted_scores: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-compute topk full token ids and partial token ids.

        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens.
                Its shape is `(n_beam, self.vocab_size)`.
            ids (torch.Tensor): The partial token ids to compute topk.
                Its shape is `(n_beam, self.pre_beam_size)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The topk full (prev_hyp, new_token) ids
                and partial (prev_hyp, new_token) ids.
                Their shapes are all `(self.beam_size,)`

        """
        top_ids = weighted_scores.view(-1).topk(self.beam_size)[1]
        # Because of the flatten above, `top_ids` is organized as:
        # [hyp1 * V + token1, hyp2 * V + token2, ..., hypK * V + tokenK],
        # where V is `self.n_vocab` and K is `self.beam_size`
        if is_torch_1_9_plus:
            prev_hyp_ids = torch.div(top_ids, self.n_vocab, rounding_mode="trunc")
        else:
            prev_hyp_ids = top_ids // self.n_vocab
        new_token_ids = top_ids % self.n_vocab
        return prev_hyp_ids, new_token_ids, prev_hyp_ids, new_token_ids

    def init_hyp(self, x: torch.Tensor) -> BatchHypothesis:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature

        Returns:
            Hypothesis: The initial hypothesis.

        """
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.batch_init_state(x)
            init_scores[k] = 0.0

        # NOTE (Shih-Lun): added for OpenAI Whisper ASR
        primer = [self.sos] if self.hyp_primer is None else self.hyp_primer

        return self.batchfy(
            [
                Hypothesis(
                    yseq=torch.tensor(primer, device=x.device),
                    score=0.0 if i == 0 else -1e9, # duplicates start with -1 billion score
                    scores=init_scores,
                    states=init_states,
                    hs=[],
                    TokenAgeIndex=torch.tensor([], device=x.device, dtype=torch.int32),
                    ChunkEmissionIndex=torch.tensor([], device=x.device, dtype=torch.int32),
                )
                for i in range(self.beam_size)
            ]
        )

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
            pre_x (torch.Tensor): Encoded speech feature for sequential attn (T, D)
                Sequential attn computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            if "decoder" in k and self.return_hs:
                (scores[k], hs), states[k] = d.batch_score(
                    hyp.yseq, hyp.states[k], x, return_hs=self.return_hs
                )
            elif "decoder" in k and pre_x is not None:
                scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x, pre_x)
            else:
                scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)

        if self.return_hs:
            return hs, scores, states
        return scores, states

    def score_partial(
        self,
        hyp: BatchHypothesis,
        ids: torch.Tensor,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            ids (torch.Tensor): 2D tensor of new partial tokens to score
            x (torch.Tensor): Corresponding input feature
            pre_x (torch.Tensor): Encoded speech feature for sequential attn (T, D)
                Sequential attn computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            if "ctc" in k and pre_x is not None:
                scores[k], states[k] = d.batch_score_partial(
                    hyp.yseq, ids, hyp.states[k], pre_x
                )
            else:
                scores[k], states[k] = d.batch_score_partial(
                    hyp.yseq, ids, hyp.states[k], x
                )
        return scores, states

    def merge_states(self, states: Any, part_states: Any, part_idx: int) -> Any:
        """Merge states for new hypothesis.

        Args:
            states: states of `self.full_scorers`
            part_states: states of `self.part_scorers`
            part_idx (int): The new token id for `part_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.

        """
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, v in part_states.items():
            new_states[k] = v
        return new_states

    def _is_within_chunk_repeat(self, _hyp_idx: int, _new_token: int) -> bool:
        """Return True if token repeats within the current chunk.

        Base class always returns False. Override in BatchBeamSearchOnline.
        """
        return False

    def _is_cross_chunk_repeat(self, _hyp_idx: int, _new_token: int) -> bool:
        """Return True if token repeats from the previous chunk.

        Base class always returns False. Override in BatchBeamSearchOnline.
        """
        return False

    def search(
        self,
        running_hyps: BatchHypothesis,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
        current_chunk: int = 0,
        score_mask: torch.Tensor = None,
    ) -> "Tuple[BatchHypothesis, str]":
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)
            pre_x (torch.Tensor): Encoded speech feature for sequential attention (T, D)
            current_chunk (int): Current chunk index for tracking token emission
            score_mask (torch.Tensor): Optional bool mask of shape (n_batch, n_vocab).
                True entries are set to -inf before beam selection. Used to mask
                out cross-chunk repeated tokens on retry.

        Returns:
            Tuple[BatchHypothesis, str]: Best sorted hypotheses and a stop reason.
                None: normal, continue decoding.
                "eos_stop_chunk": EOS reached the candidate set on a non-final
                    chunk, stop this chunk.
                "within_chunk_repeat": token repeated within current chunk, stop chunk.
                "cross_chunk_repeat": token repeated from previous chunk.
                    The offending (hyp_idx, token_id) is stored in
                    self._cross_chunk_repeat_info for the caller to build a mask.
                "low_confidence_stop" / "high_entropy_stop" / "low_margin_stop" /
                    "low_topk_mass_stop": confidence-policy chunk stops (Policy 1/2).
                "dyn_fc_token_defer": dynamic-future-chunks trigger deferred this
                    token; the caller re-decodes with one more future chunk.

        """
        n_batch = len(running_hyps)
        part_ids = None  # no pre-beam
        # batch scoring
        weighted_scores = torch.zeros(
            n_batch, self.n_vocab, dtype=x.dtype, device=x.device
        )
        if self.return_hs:
            hs, scores, states = self.score_full(
                running_hyps,
                x.expand(n_batch, *x.shape),
                pre_x=(
                    pre_x.expand(n_batch, *pre_x.shape) if pre_x is not None else None
                ),
            )
        else:
            scores, states = self.score_full(
                running_hyps,
                x.expand(n_batch, *x.shape),
                pre_x=(
                    pre_x.expand(n_batch, *pre_x.shape) if pre_x is not None else None
                ),
            )

        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        # Forced-prefix decoding (resume-from-prefix): force the first
        # len(_forced_prefix) emitted tokens so decoder AND CTC states advance
        # correctly over the kept prefix. No-op when _forced_prefix is unset.
        _force_tok = None
        _fp = getattr(self, "_forced_prefix", None)
        if _fp is not None:
            _step = running_hyps.yseq.shape[1] - 1
            if 0 <= _step < len(_fp):
                _force_tok = int(_fp[_step])
        # partial scoring
        if self.do_pre_beam:
            pre_beam_scores = (
                weighted_scores
                if self.pre_beam_score_key == "full"
                else scores[self.pre_beam_score_key]
            )
            part_ids = torch.topk(pre_beam_scores, self.pre_beam_size, dim=-1)[1]
        # Ensure the forced token is in the candidate set so CTC scores it
        # (advancing its prefix-score state correctly).
        if _force_tok is not None and part_ids is not None:
            _ins = (part_ids == _force_tok).any(dim=-1)
            for _b in range(part_ids.shape[0]):
                if not bool(_ins[_b]):
                    part_ids[_b, -1] = _force_tok
        # NOTE(takaaki-hori): Unlike BeamSearch, we assume that score_partial returns
        # full-size score matrices, which has non-zero scores for part_ids and zeros
        # for others.
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x, pre_x)
        for k in self.part_scorers:
            weighted_scores += self.weights[k] * part_scores[k]
        # Restrict selection to the forced token (states already advanced above).
        if _force_tok is not None:
            _fm = torch.full_like(weighted_scores, float("-inf"))
            _fm[:, _force_tok] = weighted_scores[:, _force_tok]
            weighted_scores = _fm
        # add previous hyp scores
        weighted_scores += running_hyps.score.to(
            dtype=x.dtype, device=x.device
        ).unsqueeze(1)

        # EOS is allowed on all chunks. On non-final chunks, if any beam
        # hits EOS it signals "stop this chunk" without appending EOS to hyp.
        block_is_final = getattr(self, '_block_is_final', True)

        # Mask cross-chunk repeated tokens (Strategy 2: mask and re-run)
        if score_mask is not None:
            weighted_scores[score_mask.to(device=weighted_scores.device)] = float('-inf')

        # TODO(karita): do not use list. use batch instead
        # see also https://github.com/espnet/espnet/pull/1402#discussion_r354561029
        # update hyps
        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)

        # Store context so repetition checks can access running_hyps and current_chunk
        self._search_running_hyps = running_hyps
        self._search_current_chunk = current_chunk

        # F4 fix: at the final chunk, add a bonus to EOS log-prob so that
        # the model is more willing to terminate cleanly instead of emitting
        # plausible-but-unsupported tail content. Targets type (a)
        # hallucination cases where 2-3 extra content tokens are emitted
        # past the actual audio end.
        eos_bonus = getattr(self, "eos_bonus_at_final", 0.0)
        if block_is_final and eos_bonus > 0.0:
            weighted_scores[:, self.eos] = weighted_scores[:, self.eos] + eos_bonus

        # F3 fix: penalize subword-piece repetition at chunk boundaries.
        # Example: ``▁CONTAIN`` (chunk N, last piece) + ``IN`` (chunk N+1)
        # → "CONTAININING". The candidate ``IN`` is a proper suffix of the
        # previous piece ``CONTAIN``; emitting it as a separate continuation
        # piece duplicates letters that are already part of the current word.
        # We mask such candidates only on the FIRST emission of a new chunk
        # (chunk transition) so within-chunk decoding is unaffected.
        sw_pen = getattr(self, "subword_overlap_penalty", 0.0)
        if sw_pen > 0.0 and self.token_list is not None and hasattr(
            self, "processed_block"
        ):
            if not hasattr(self, "_text_to_id_subword"):
                self._text_to_id_subword = {}
                for tid, tok in enumerate(self.token_list):
                    if tok is None or tok.startswith("▁"):
                        continue
                    self._text_to_id_subword.setdefault(tok, tid)
            cur_chunk = int(self.processed_block)
            for b in range(n_batch):
                seq_len = int(running_hyps.length[b])
                if seq_len < 2:
                    continue
                last_tid = int(running_hyps.yseq[b, seq_len - 1])
                last_tok = (
                    self.token_list[last_tid]
                    if 0 <= last_tid < len(self.token_list)
                    else ""
                )
                last_text = last_tok.lstrip("▁") if last_tok else ""
                if len(last_text) < 4:
                    continue
                # Chunk transition test: the prefix's last emitted token
                # was emitted in a chunk earlier than the current one.
                cei = running_hyps.ChunkEmissionIndex
                if cei.numel() == 0 or cei.ndim < 2 or cei.size(1) < 1:
                    continue
                # ChunkEmissionIndex indexes emitted tokens (excluding SOS at
                # yseq[0]); the last entry is the most recently emitted token.
                last_emit_chunk = int(cei[b, seq_len - 2])
                if last_emit_chunk >= cur_chunk:
                    continue
                # Penalize every continuation piece whose text equals a
                # non-trivial (>=2 char) suffix of the prev piece text.
                for L in range(2, len(last_text) + 1):
                    sfx = last_text[-L:]
                    cand_tid = self._text_to_id_subword.get(sfx)
                    if cand_tid is not None:
                        weighted_scores[b, cand_tid] = (
                            weighted_scores[b, cand_tid] - sw_pen
                        )

        r = self.batch_beam(weighted_scores, part_ids)
        new_hyp_order_with_eos, _next_tokens = r[:2]

        # Check top-1 candidate for repetition
        top1_hyp_idx = int(r[0][0].item())
        top1_token = int(r[1][0].item())

        # If any beam hits EOS on a non-final chunk, stop this chunk
        # without appending EOS to the hypothesis.
        if not block_is_final and (_next_tokens == self.eos).any():
            n_eos = int((_next_tokens == self.eos).sum().item())
            top1_is_eos = bool(_next_tokens[0].item() == self.eos)
            logger.info(
                f"[silent-chunk cause] chunk {current_chunk}: EOS_STOP "
                f"(top1_token={top1_token}, top1_is_eos={top1_is_eos}, "
                f"eos_in_topbeam={n_eos}/{len(_next_tokens)})"
            )
            return running_hyps, "eos_stop_chunk"

        # Check within-chunk repeat (Strategy 1: stop chunk).
        # Bypass while teacher-forcing a committed prefix token (_force_tok set):
        # it was decoded cleanly in a prior pass, so re-emitting it must not stop
        # the chunk (which would truncate the frozen prefix).
        if _force_tok is None and self._is_within_chunk_repeat(top1_hyp_idx, top1_token):
            logger.info(
                f"[silent-chunk cause] chunk {current_chunk}: WITHIN_CHUNK_REPEAT "
                f"(token {top1_token} already emitted in this chunk)"
            )
            return running_hyps, "within_chunk_repeat"

        # Check cross-chunk repeat (Strategy 2: mask and retry). Bypass while
        # forcing: the mask-retry would mask the forced token, leaving an
        # all -inf score row (deadlock against the force mask).
        if _force_tok is None and self._is_cross_chunk_repeat(top1_hyp_idx, top1_token):
            logger.debug(
                f"[search] Cross-chunk repeat: token {top1_token} "
                f"from previous chunk, signaling retry"
            )
            self._cross_chunk_repeat_info = (top1_hyp_idx, top1_token)
            return running_hyps, "cross_chunk_repeat"

        # Softmax-confidence-based hallucination suppression.
        # Policy 1: end-of-chunk early stop when top-1 softmax is below threshold.
        # Policy 2: mask+retry the first token of a new chunk when softmax is below threshold.
        # Policy 3 (rollback): record per-token confidence for later re-scoring at chunk boundary.
        end_thresh = getattr(self, 'chunk_end_confidence_threshold', 0.0)
        start_thresh = getattr(self, 'chunk_start_confidence_threshold', 0.0)
        rollback_thresh = getattr(self, 'rollback_confidence_threshold', 0.0)
        entropy_thresh = getattr(self, 'chunk_end_entropy_threshold', 0.0)
        margin_thresh = getattr(self, 'chunk_end_margin_threshold', 0.0)
        topk_mass_thresh = getattr(self, 'chunk_end_topk_mass_threshold', 0.0)
        # Compute per-beam softmax once, reused for Policy 1/2 checks and
        # per-candidate confidence recording. Shape: (n_batch, n_vocab).
        all_step_probs = None
        # Dynamic-future-chunks needs per-token confidence in confidence_list
        # even when the Policy 1/2 thresholds are all 0 (the DFC sweep config),
        # so force the per-step softmax when a DFC controller is active. This
        # is the SAME softmax the token hook computes, so token-mode signals are
        # unchanged; it only makes confidence_list informative for the chunk
        # variants (which aggregate it). Harmless for non-DFC runs.
        _dyn_active = (
            getattr(self, "dynamic_fc_controller", None) is not None
            and getattr(self.dynamic_fc_controller, "enabled", False)
        )
        if (end_thresh > 0.0 or start_thresh > 0.0 or rollback_thresh > 0.0
                or entropy_thresh > 0.0 or margin_thresh > 0.0
                or topk_mass_thresh > 0.0 or _dyn_active):
            prev_scores = running_hyps.score.to(
                dtype=weighted_scores.dtype, device=weighted_scores.device
            ).unsqueeze(1)
            all_step_probs = torch.softmax(weighted_scores - prev_scores, dim=-1)

        if all_step_probs is not None and not block_is_final and end_thresh > 0.0:
            top1_prob = all_step_probs[top1_hyp_idx, top1_token].item()
            if top1_prob < end_thresh:
                logger.info(
                    f"[low_confidence_stop] P1 top1 conf {top1_prob * 100:.1f}% < "
                    f"{end_thresh * 100:.1f}% for token {top1_token} in chunk "
                    f"{current_chunk}, stopping chunk"
                )
                return running_hyps, "low_confidence_stop"

        # Entropy-based stop: fires when the softmax distribution is spread out
        # (high entropy = uncertain choice). Computed on the top beam's full
        # vocab distribution in nats.
        if all_step_probs is not None and not block_is_final and entropy_thresh > 0.0:
            p = all_step_probs[top1_hyp_idx].clamp_min(1e-12)
            ent = float(-(p * p.log()).sum().item())
            if ent > entropy_thresh:
                logger.info(
                    f"[entropy_stop] top-beam entropy {ent:.3f} nats > "
                    f"{entropy_thresh:.3f} at token {top1_token} in chunk "
                    f"{current_chunk}, stopping chunk"
                )
                return running_hyps, "high_entropy_stop"

        # Margin-based stop: fires when (top1 - top2) < threshold. Catches the
        # "two competing candidates" case where the model is wavering between
        # two tokens but committed to one prematurely.
        if all_step_probs is not None and not block_is_final and margin_thresh > 0.0:
            top_probs, _ = all_step_probs[top1_hyp_idx].topk(2)
            margin = float((top_probs[0] - top_probs[1]).item())
            if margin < margin_thresh:
                logger.info(
                    f"[margin_stop] top1-top2 margin {margin:.3f} < "
                    f"{margin_thresh:.3f} at token {top1_token} in chunk "
                    f"{current_chunk}, stopping chunk"
                )
                return running_hyps, "low_margin_stop"

        # Top-k mass stop: fires when the top-3 cumulative softmax mass is
        # below threshold. Catches "long-tail uncertainty" where mass is
        # spread across many candidates rather than concentrated in the head.
        if all_step_probs is not None and not block_is_final and topk_mass_thresh > 0.0:
            top_probs, _ = all_step_probs[top1_hyp_idx].topk(3)
            mass = float(top_probs.sum().item())
            if mass < topk_mass_thresh:
                logger.info(
                    f"[topk_mass_stop] top-3 mass {mass:.3f} < "
                    f"{topk_mass_thresh:.3f} at token {top1_token} in chunk "
                    f"{current_chunk}, stopping chunk"
                )
                return running_hyps, "low_topk_mass_stop"

        if (all_step_probs is not None
                and not block_is_final
                and hasattr(self, '_is_low_confidence_chunk_start')):
            if self._is_low_confidence_chunk_start(
                top1_hyp_idx, top1_token, all_step_probs[top1_hyp_idx]
            ):
                self._cross_chunk_repeat_info = (top1_hyp_idx, top1_token)
                return running_hyps, "cross_chunk_repeat"

        # ----- SR-CEM Variant A: per-step inference prediction -----
        # Build features and run the scorer ONLY when the trigger is
        # active (signal_type=sr_cem_causal AND scorer loaded). The
        # TRAINING-data dump path lives elsewhere: at end-of-utterance
        # in inference() we extract features from the final committed
        # hypothesis's scores_list and dump those — that's what aligns
        # 1:1 with sclite labels. Per-step dumping would record
        # uncommitted EOS_STOP / cross-chunk-repeat / defer steps that
        # don't appear in the final hypothesis, causing label/feature
        # length mismatch at training-set build time.
        _sr_cem_scorer = getattr(self, "sr_cem_scorer", None)
        _dyn_ctrl_for_sr_cem = getattr(self, "dynamic_fc_controller", None)
        _sig_type = (
            _dyn_ctrl_for_sr_cem.config.signal_type
            if _dyn_ctrl_for_sr_cem is not None else None
        )
        # The same Variant-A scorer + feature hook serves BOTH SR-CEM (p_correct)
        # and the learned wait-policy (p_wait); they differ only in the controller
        # comparison polarity, so the producer is shared.
        _sr_cem_active = (
            _sr_cem_scorer is not None
            and _dyn_ctrl_for_sr_cem is not None
            and _sig_type in ("sr_cem_causal", "wait_policy")
        )
        _sr_cem_p_correct = None
        # Learned wait-policy via an aux head on the DECODER HIDDEN STATE
        # (preferred when loaded): p_wait = aux_head(hs[top1_hyp_idx]). This
        # replaces the beam-feature SR-CEM scorer for signal_type=wait_policy.
        # Requires return_hs=True so `hs` (n_batch, 256) is populated above.
        _wait_aux = getattr(self, "wait_aux_head", None)
        _aux_active = (
            _wait_aux is not None
            and _dyn_ctrl_for_sr_cem is not None
            and _sig_type == "wait_policy"
            and self.return_hs
        )
        if _aux_active:
            try:
                _sr_cem_p_correct = _wait_aux.predict_from_hidden(hs[top1_hyp_idx])
            except Exception as _e:
                logger.warning(
                    "[wait_aux] predict_from_hidden failed: %s; "
                    "signal will not populate.", _e,
                )
        if _sr_cem_active and not _aux_active:
            from espnet2.asr_stream.sr_cem import (
                build_features_causal,
                step_features_from_cumulative,
            )
            _prev_score_row = running_hyps.score.to(
                dtype=weighted_scores.dtype,
                device=weighted_scores.device,
            )
            # Canonical paper-recipe features from CUMULATIVE scores
            # (weighted_scores rows are prefix-inclusive). Notably top4 is
            # the RAW cumulative top-4 per the published training pipeline
            # (dataset_token_ours.py) — the old code here computed top4
            # over the incremental row, mismatching the trained features.
            _cum_row = weighted_scores[top1_hyp_idx].detach()
            _score, _rank, _s_lt, _top4 = step_features_from_cumulative(
                selected_cum=float(_cum_row[top1_token].item()),
                prev_selected_cum=float(_prev_score_row[top1_hyp_idx].item()),
                candidate_cums=_cum_row.cpu(),
            )
            _sr_cem_feat = build_features_causal(
                score=_score, rank=_rank, S_lt=_s_lt, top4=_top4,
            )
            try:
                _sr_cem_p_correct = _sr_cem_scorer.predict(_sr_cem_feat)
            except Exception as _e:
                logger.warning(
                    "[sr_cem_causal] scorer.predict failed: %s; "
                    "calibrated signal will not populate.", _e,
                )

        # Dynamic future-chunks Branch A (token_resume): consult the
        # controller mid-decode. On DEFER, signal "dyn_fc_token_defer";
        # process_one_block will break, preserving running_hyps and
        # process_idx so the next call resumes from this exact step with
        # (one more chunk of) richer encoder context.
        dyn_ctrl = getattr(self, "dynamic_fc_controller", None)
        if (
            dyn_ctrl is not None
            and dyn_ctrl.enabled
            and dyn_ctrl.mode in ("token_resume", "token_chunk_resume")
            and not block_is_final
        ):
            # Compute per-token softmax on the top beam's vocab row.
            if all_step_probs is None:
                prev_scores_row = running_hyps.score.to(
                    dtype=weighted_scores.dtype,
                    device=weighted_scores.device,
                ).unsqueeze(1)
                _step_probs_top = torch.softmax(
                    (weighted_scores - prev_scores_row)[top1_hyp_idx], dim=-1
                )
            else:
                _step_probs_top = all_step_probs[top1_hyp_idx]
            # Optional temperature scaling for the trigger signal only
            # (Guo et al. 2017). T=1.0 is no-op. Applies to the full step
            # softmax before extracting top1/top2/topk/entropy so all four
            # signals are calibrated consistently. Does NOT affect beam
            # ranking or final decoding — argmax is preserved at any T.
            _trigger_T = float(getattr(self, "trigger_temperature", 1.0))
            if _trigger_T != 1.0:
                _logp_step = _step_probs_top.clamp_min(1e-12).log()
                _logp_step = _logp_step / _trigger_T
                _logp_step = _logp_step - _logp_step.logsumexp(dim=-1)
                _step_probs_top = _logp_step.exp()
            _top_probs_k, _ = _step_probs_top.topk(3)
            _top1_p = float(_top_probs_k[0].item())
            _top2_p = float(_top_probs_k[1].item()) if _top_probs_k.numel() >= 2 else 0.0
            _topk_mass = float(_top_probs_k.sum().item())
            _ent_p = _step_probs_top.clamp_min(1e-12)
            _entropy = float(-(_ent_p * _ent_p.log()).sum().item())
            _token_signals = {
                "top1_prob": _top1_p,
                "entropy": _entropy,
                "margin": _top1_p - _top2_p,
                "topk_mass": _topk_mass,
            }
            if _sr_cem_p_correct is not None:
                # Inject under the active signal key (sr_cem_causal | wait_policy)
                # so the controller compares it with the right polarity.
                _token_signals[_sig_type] = float(_sr_cem_p_correct)

            # CTC-vs-attention disagreement signal (hybrid-decode native).
            # At THIS step compare the attention head's preferred next token to
            # the CTC prefix scorer's preferred next token. The CTC prefix score
            # marginalizes over frame alignments, so it is well-defined per step
            # and never desyncs from the autoregressive decoder (unlike a greedy
            # CTC sequence diff). Binary 1.0 (heads disagree) / 0.0 (agree).
            if _sig_type == "ctc_disagree":
                _dec_s = scores.get("decoder") if isinstance(scores, dict) else None
                _ctc_s = part_scores.get("ctc") if isinstance(part_scores, dict) else None
                if _dec_s is not None and _ctc_s is not None:
                    _a_tok = int(_dec_s[top1_hyp_idx].argmax().item())  # attention's choice
                    _ctc_row = _ctc_s[top1_hyp_idx]
                    if part_ids is not None:
                        _cand = part_ids[top1_hyp_idx]                 # CTC-scored candidates
                        _c_tok = int(_cand[_ctc_row[_cand].argmax()].item())
                    else:
                        _c_tok = int(_ctc_row.argmax().item())
                    _dis = (_a_tok != _c_tok)
                    _token_signals["ctc_disagree"] = 1.0 if _dis else 0.0

            _future_avail = getattr(self, "_dyn_fc_future_available", 0)
            # Branch A: key the budget on the *stuck token step* (process_idx),
            # NOT the audio/acoustic chunk. A defer drops one token and resumes
            # the SAME token next call with +1 chunk; process_idx is stable
            # across those resumes, so defer_count accumulates against the exact
            # position the model is stuck at and force-commits at
            # max_future_chunks. (An attn_center-based key was tried and reverted:
            # attn_center drifts forward with processed_block as audio arrives,
            # so the key changed under a stuck token and the budget never
            # accumulated -> unbounded deferral. Verified by audit.)
            if dyn_ctrl.mode == "token_chunk_resume":
                # Branch D: per-token EARLY-STOP trigger for B-style chunk
                # rollback. Stop the chunk decode at the FIRST low-conf token
                # (flag the branch reads); the branch then restores the chunk
                # snapshot and re-decodes WITH look-ahead (encoder nrc-future +
                # cross-attn window) one level deeper -- future chunk in BOTH
                # encoder and decoder, unlike the old causal D. The branch sets
                # _dyn_d_force_commit once the deep-deferral budget is spent,
                # after which we never defer (the chunk must commit).
                if not getattr(self, "_dyn_d_force_commit", False) and _force_tok is None:
                    # _force_tok is None: never defer ON a teacher-forced
                    # committed-prefix token (commit-stable-prefix); only the
                    # free suffix past the frozen prefix may trigger a defer.
                    # Generalized trigger: defer on whatever signal_type is
                    # configured (sr_cem_causal OR the calibrated decoder
                    # top1_prob), via the controller's token dispatch. Keeps
                    # SR-CEM-D behaviour identical (signal_type=sr_cem_causal)
                    # while enabling the temperature-scaled top1_prob variant.
                    _defer_d, _val_d = dyn_ctrl._signal_says_defer_token(_token_signals)
                    if _defer_d:
                        self._dyn_d_token_deferred = True
                        logger.info(
                            f"[dyn_fc_token_defer] D chunk={current_chunk} "
                            f"step={getattr(self, 'process_idx', 0)} "
                            f"signal={dyn_ctrl.config.signal_type} val={_val_d:.3f} "
                            f"-> stop chunk"
                        )
                        return running_hyps, "dyn_fc_token_defer"
                # confident, budget spent, or no probe -> commit this token.
            else:
                # Branch A (token_resume): per-token-step budget, drop one token.
                _stuck_step = int(getattr(self, "process_idx", 0))
                _decision = dyn_ctrl.decide_token(
                    chunk_idx=_stuck_step,
                    token_step=_stuck_step,
                    token_signals=_token_signals,
                    future_available=int(_future_avail),
                    is_final=block_is_final,
                )
                if _decision == "defer":
                    logger.info(
                        f"[dyn_fc_token_defer] chunk={current_chunk} "
                        f"step={getattr(self, 'process_idx', 0)} "
                        f"top1={_top1_p:.3f} ent={_entropy:.3f} "
                        f"margin={_top1_p - _top2_p:.3f} -> stop chunk"
                    )
                    return running_hyps, "dyn_fc_token_defer"

        # No issues — proceed with normal beam update
        non_eos_mask = (_next_tokens != self.eos)
        new_hyp_order = new_hyp_order_with_eos[non_eos_mask]

        _dec = self.full_scorers.get("decoder")
        if _dec is not None and hasattr(_dec, "_update_hyp_order"):
            _dec._update_hyp_order(new_hyp_order)

        for k, (
            full_prev_hyp_id,
            full_new_token_id,
            part_prev_hyp_id,
            part_new_token_id,
        ) in enumerate(zip(*r)):
            # Check ALL beam candidates for within-chunk repeat, not just top-1.
            # Bypass while teacher-forcing (commit-stable-prefix): the forced
            # token is the only candidate, so skipping it would empty the beam.
            if _force_tok is None and self._is_within_chunk_repeat(
                int(full_prev_hyp_id.item()), int(full_new_token_id.item())
            ):
                logger.debug(
                    f"[search] Skipping beam candidate {k}: token "
                    f"{int(full_new_token_id.item())} repeated in chunk {current_chunk}"
                )
                continue
            prev_hyp = prev_hyps[full_prev_hyp_id]

            if self.return_hs:
                new_hs = prev_hyp.hs + [hs[full_prev_hyp_id].squeeze(0)]
            else:
                new_hs = []
            # Collect top-k weighted scores for this timestep
            topk_output = weighted_scores[full_prev_hyp_id].topk(self.pre_beam_size)
            scores_dict = {
                str(idx.item()): val.item()
                for idx, val in zip(topk_output.indices, topk_output.values)
            }
            # Per-token softmax confidence (for Policy 3 rollback and diagnostics).
            if all_step_probs is not None:
                # Temperature-scale the per-step distribution (Guo 2017) so the
                # persisted per-token confidence is CALIBRATED — used by the
                # chunk-level top1_prob variants (B/C). T=1.0 is a no-op, so
                # confidence_list is byte-identical to before for all existing
                # runs; argmax/beam ranking are untouched either way.
                _conf_row = all_step_probs[full_prev_hyp_id]
                _conf_T = float(getattr(self, "trigger_temperature", 1.0))
                if _conf_T != 1.0:
                    _clp = _conf_row.clamp_min(1e-12).log() / _conf_T
                    _conf_row = (_clp - _clp.logsumexp(dim=-1)).exp()
                new_conf = float(_conf_row[full_new_token_id].item())
            else:
                new_conf = 1.0
            best_hyps.append(
                Hypothesis(
                    score=weighted_scores[full_prev_hyp_id, full_new_token_id],
                    yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
                    scores=self.merge_scores(
                        prev_hyp.scores,
                        {k: v[full_prev_hyp_id] for k, v in scores.items()},
                        full_new_token_id,
                        {k: v[part_prev_hyp_id] for k, v in part_scores.items()},
                        part_new_token_id,
                    ),
                    states=self.merge_states(
                        {
                            k: self.full_scorers[k].select_state(v, full_prev_hyp_id)
                            for k, v in states.items()
                        },
                        {
                            k: self.part_scorers[k].select_state(
                                v, part_prev_hyp_id, part_new_token_id
                            )
                            for k, v in part_states.items()
                        },
                        part_new_token_id,
                    ),
                    hs=new_hs,
                    scores_list=prev_hyp.scores_list + [scores_dict],
                    confidence_list=list(prev_hyp.confidence_list) + [new_conf],
                    TokenAgeIndex=torch.cat(
                        [
                            prev_hyp.TokenAgeIndex.to(x.device),
                            torch.tensor([0], device=x.device, dtype=torch.int32),
                        ]
                    ),
                    ChunkEmissionIndex=torch.cat(
                        [
                            prev_hyp.ChunkEmissionIndex.to(x.device),
                            torch.tensor([current_chunk], device=x.device, dtype=torch.int32),
                        ]
                    ),
                )
            )

        # If all candidates were repeats, stop the chunk
        if len(best_hyps) == 0:
            logger.debug(
                f"[search] All beam candidates repeated in chunk {current_chunk}, "
                "stopping chunk"
            )
            return running_hyps, "within_chunk_repeat"

        return self.batchfy(best_hyps), None

    def post_process(
        self,
        i: int,
        maxlen: int,
        minlen: int,
        maxlenratio: float,
        running_hyps: BatchHypothesis,
        ended_hyps: List[Hypothesis],
    ) -> BatchHypothesis:
        """Perform post-processing of beam search iterations.

        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (BatchHypothesis): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.

        Returns:
            BatchHypothesis: The new running hypotheses.

        """
        n_batch = running_hyps.yseq.shape[0]
        logger.debug(f"the number of running hypothes: {n_batch}")
        if self.token_list is not None:
            logger.debug(
                "best hypo: "
                + "".join(
                    [
                        self.token_list[x]
                        for x in running_hyps.yseq[0, 1 : running_hyps.length[0]]
                    ]
                )
            )
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logger.info("adding <eos> in the last position in the loop")
            yseq_eos = torch.cat(
                (
                    running_hyps.yseq,
                    torch.full(
                        (n_batch, 1),
                        self.eos,
                        device=running_hyps.yseq.device,
                        dtype=torch.int64,
                    ),
                ),
                1,
            )
            running_hyps.yseq.resize_as_(yseq_eos)
            running_hyps.yseq[:] = yseq_eos
            running_hyps.length[:] = yseq_eos.shape[1]

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a probmlem, number of hyps < beam)
        is_eos = (
            running_hyps.yseq[torch.arange(n_batch), running_hyps.length - 1]
            == self.eos
        )
        for b in torch.nonzero(is_eos, as_tuple=False).view(-1):
            hyp = self._select(running_hyps, b)
            # Apply each scorer's final_score at EOS (upstream batch mode omits this).
            for k, d in chain(self.full_scorers.items(), self.part_scorers.items()):
                s = d.final_score(hyp.states[k])
                hyp.scores[k] += s
                hyp = hyp._replace(score=hyp.score + self.weights[k] * s)
            if i >= minlen:
                ended_hyps.append(hyp)
        remained_ids = torch.nonzero(is_eos == 0, as_tuple=False).view(-1).cpu()
        return self._batch_select(running_hyps, remained_ids)
