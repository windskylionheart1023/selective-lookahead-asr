"""Dynamic future-chunks controller for streaming ASR inference.

Inference-only infrastructure that decides, per chunk (Branch B) or per
token (Branch A), whether the current decode state has enough future
context for acceptable quality, or whether the decoder should *wait* for
one more future chunk and re-do the work.

The branches are:

**Branch B — chunk-level rollback.** The encoder + beam-search run on the
chunk normally. After the chunk's tokens are emitted, a signal is checked
against a threshold. If the signal trips, the beam-search state is
restored to its pre-chunk snapshot, the chunk is unreleased from the
chunk tracker, and we wait for the next audio chunk before re-encoding
and re-decoding the same chunk with one more future-context chunk
available.

**Branch A — token-level stop and resume.** Inside the beam-search loop,
at every step a signal is checked. If the signal trips, the just-emitted
token is dropped, beam state is rolled back to the pre-step snapshot, and
the call returns. The next call (after another audio chunk has arrived)
resumes beam search from exactly that token step with a richer encoder
context.

**Branch D - token-triggered chunk resume.** Like Branch A the signal is
checked per token, but on a defer ALL tokens of the current chunk are
rolled back (not just one) and the chunk is re-decoded causally with the
next chunk as current context (no look-ahead window). Reuses Branch A's
causal path; the only difference lives in the per-step hook in
``batch_beam_search.py``.

There is also a baseline mode **pre-emptive** which evaluates the signal
*before* beam search runs (using CTC log-probs on the encoder output);
when the signal trips, beam search never executes for that chunk. This
is the cheapest mode and is useful as a control.

Default behaviour (controller disabled) is unchanged: every chunk is
released using the static ``num_right_chunks`` configured on the chunk
tracker.

This module is intentionally pure-Python (numpy only). It does not import
any model code; callers feed it log-probs they have already computed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# Metrics ---------------------------------------------------------------


@dataclass
class ConfidenceMetrics:
    """Aggregate confidence signals for one chunk of CTC log-probs.

    Aggregated over the *non-blank* frames in the chunk: a frame is
    non-blank when its argmax token id is not the CTC blank id. This
    avoids the blank-dominated frames (which always have near-1.0
    confidence) drowning out the few content-emitting frames that carry
    the recognition uncertainty.
    """

    n_frames: int
    n_nonblank: int
    top1_prob_mean: float
    top1_prob_min: float
    entropy_mean: float
    entropy_max: float
    margin_mean: float
    margin_min: float
    topk_mass_mean: float
    # SR-CEM Variant B (chunk_rollback) hook attaches a chunk-aggregated
    # calibrated p(correct) here. Populated by Speech2TextStreamingChunked.
    # _branch_chunk_rollback after a per-chunk feature build; None when
    # the SR-CEM scorer isn't loaded.
    sr_cem_chunk_p: Optional[float] = None


def compute_chunk_confidence(
    chunk_logp: np.ndarray,
    blank_id: int = 0,
    topk: int = 3,
) -> ConfidenceMetrics:
    """Compute aggregate confidence metrics for one chunk of CTC log-probs.

    Args:
        chunk_logp: ``(T, V)`` log-softmax over the vocabulary.
        blank_id: CTC blank id; frames whose argmax equals this are
            excluded from aggregation.
        topk: ``k`` for the cumulative top-k mass metric.

    A zero-length or all-blank chunk returns a trivially-confident
    metric set (top1=1, entropy=0, ...) so it never triggers a defer.
    """
    if chunk_logp.ndim != 2:
        raise ValueError(
            f"chunk_logp must be 2D (T, V), got shape {chunk_logp.shape}"
        )

    T = chunk_logp.shape[0]
    if T == 0:
        return _trivial_metrics(0, 0)

    p = np.exp(chunk_logp.astype(np.float64))
    row_sum = p.sum(axis=-1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    p = p / row_sum

    argmax = np.argmax(p, axis=-1)
    nonblank_mask = argmax != blank_id
    n_nonblank = int(nonblank_mask.sum())
    if n_nonblank == 0:
        return _trivial_metrics(T, 0)

    p_nb = p[nonblank_mask]
    V = p_nb.shape[-1]
    k = min(topk, V)
    topk_idx = np.argpartition(-p_nb, k - 1, axis=-1)[:, :k]
    topk_vals = np.take_along_axis(p_nb, topk_idx, axis=-1)
    topk_vals = -np.sort(-topk_vals, axis=-1)
    top1 = topk_vals[:, 0]
    top2 = topk_vals[:, 1] if k >= 2 else np.zeros_like(top1)
    topk_mass = topk_vals.sum(axis=-1)

    logp_nb = chunk_logp[nonblank_mask].astype(np.float64)
    plogp = np.where(p_nb > 0, p_nb * logp_nb, 0.0)
    entropy_per_frame = -plogp.sum(axis=-1)
    margin = top1 - top2

    return ConfidenceMetrics(
        n_frames=T,
        n_nonblank=n_nonblank,
        top1_prob_mean=float(top1.mean()),
        top1_prob_min=float(top1.min()),
        entropy_mean=float(entropy_per_frame.mean()),
        entropy_max=float(entropy_per_frame.max()),
        margin_mean=float(margin.mean()),
        margin_min=float(margin.min()),
        topk_mass_mean=float(topk_mass.mean()),
    )


def _trivial_metrics(T: int, n_nonblank: int) -> ConfidenceMetrics:
    return ConfidenceMetrics(
        n_frames=T,
        n_nonblank=n_nonblank,
        top1_prob_mean=1.0,
        top1_prob_min=1.0,
        entropy_mean=0.0,
        entropy_max=0.0,
        margin_mean=1.0,
        margin_min=1.0,
        topk_mass_mean=1.0,
    )


def token_signals_from_logp(
    step_logp: np.ndarray, temperature: float = 1.0
) -> Dict[str, float]:
    """Per-step confidence signals from a single decoder step's log-probs.

    Args:
        step_logp: 1-D log-softmax over the vocabulary at one decoder
            step. Shape ``(V,)``.
        temperature: Guo et al. (2017) temperature scaling applied to the
            log-softmax before signal extraction. ``T=1.0`` is the default
            no-op. ``T>1`` flattens (lower top1_prob, higher entropy);
            ``T<1`` sharpens. Argmax is preserved at any T, so the trigger's
            class ranking within a step is unchanged — only the across-step
            ordering of top1_prob/entropy/margin/topk_mass changes when the
            per-step distribution shape varies.

    Returns:
        dict with keys ``top1_prob``, ``entropy``, ``margin``,
        ``topk_mass``. All scalar floats. Used by token-level mode.
    """
    if step_logp.ndim != 1:
        raise ValueError(
            f"step_logp must be 1D (V,), got shape {step_logp.shape}"
        )
    if temperature != 1.0:
        from espnet2.asr_stream.temperature_scaling import (
            apply_temperature_to_logp,
        )
        step_logp = apply_temperature_to_logp(step_logp, temperature)
    p = np.exp(step_logp.astype(np.float64))
    s = p.sum()
    p = p / s if s > 0 else p
    # Sort descending; cheap enough at one step.
    order = np.argsort(-p)
    top1 = float(p[order[0]])
    top2 = float(p[order[1]]) if len(order) >= 2 else 0.0
    top3_mass = float(p[order[:3]].sum())
    logp = step_logp.astype(np.float64)
    plogp = np.where(p > 0, p * logp, 0.0)
    entropy = float(-plogp.sum())
    return {
        "top1_prob": top1,
        "entropy": entropy,
        "margin": top1 - top2,
        "topk_mass": top3_mass,
    }


# Configuration ---------------------------------------------------------


# Branches.
MODE_PRE_EMPTIVE = "pre_emptive"  # cheapest baseline: signal before beam search
MODE_CHUNK_ROLLBACK = "chunk_rollback"  # Branch B
MODE_TOKEN_RESUME = "token_resume"  # Branch A
# Branch D: per-token trigger (like A) but on defer roll back ALL tokens of the
# current chunk (not just one) and re-decode causally with the next chunk as
# CURRENT context (no look-ahead window). Reuses Branch A's causal branch; the
# only difference lives in the per-step hook (batch_beam_search.py).
MODE_TOKEN_CHUNK_RESUME = "token_chunk_resume"  # Branch D
_MODES = (
    MODE_PRE_EMPTIVE, MODE_CHUNK_ROLLBACK, MODE_TOKEN_RESUME,
    MODE_TOKEN_CHUNK_RESUME,
)

# Signal types. fake_random is for testing the plumbing without depending
# on the confidence logic being well-tuned; it samples Bernoulli per call.
SIGNAL_TOP1 = "top1_prob"
SIGNAL_ENTROPY = "entropy"
SIGNAL_MARGIN = "margin"
SIGNAL_TOPK_MASS = "topk_mass"
SIGNAL_FAKE_RANDOM = "fake_random"
# SR-CEM (Score-Rank Confidence Estimation Module, Jia & Van Hamme 2026)
# variants, streaming-adapted. See espnet2/asr_stream/sr_cem.py.
# sr_cem_causal: causal 7-feature variant for Branch A (drops S>t).
# sr_cem_chunk : chunk-local 8-feature variant for Branch B (uses
#                S>t over the just-decoded chunk's remaining tokens).
SIGNAL_SR_CEM_CAUSAL = "sr_cem_causal"
SIGNAL_SR_CEM_CHUNK = "sr_cem_chunk"
# Learned WAIT-POLICY (Jia & Van Hamme 2026): predicts p(wait)="deferring >=1
# chunk turns this token wrong->right" instead of SR-CEM's p(correct). Same MLP
# / checkpoint format / feature hook as SR-CEM; the ONLY difference is the
# controller comparison polarity (defer when p_wait > threshold, vs SR-CEM
# p_correct < threshold). wait_policy: token-level Variant A (Branch A/D);
# wait_policy_chunk: chunk-level Variant B (Branch B).
SIGNAL_WAIT_CAUSAL = "wait_policy"
SIGNAL_WAIT_CHUNK = "wait_policy_chunk"
# CTC-vs-attention disagreement (Jia & Van Hamme 2026): at each step the
# attention head's preferred next token is compared to the CTC prefix scorer's
# preferred next token; the just-emitted token is flagged (signal=1.0) when the
# two heads disagree. Binary signal; defer when value > threshold (0.5).
# Token-level (Branch A/D). See espnet/nets/batch_beam_search.py (inline
# CTC-prefix-scorer comparison); espnet2/asr_stream/ctc_disagree.py holds a
# superseded greedy-decode variant.
SIGNAL_CTC_DISAGREE = "ctc_disagree"
_SIGNALS = (
    SIGNAL_TOP1,
    SIGNAL_ENTROPY,
    SIGNAL_MARGIN,
    SIGNAL_TOPK_MASS,
    SIGNAL_FAKE_RANDOM,
    SIGNAL_SR_CEM_CAUSAL,
    SIGNAL_SR_CEM_CHUNK,
    SIGNAL_WAIT_CAUSAL,
    SIGNAL_WAIT_CHUNK,
    SIGNAL_CTC_DISAGREE,
)


@dataclass
class DynamicFutureChunksConfig:
    """Configuration for :class:`DynamicFutureChunksController`.

    Attributes:
        enabled: Master switch. When False the controller is a no-op.
        mode: Which branch to run. One of ``"pre_emptive"``,
            ``"chunk_rollback"``, ``"token_resume"``,
            ``"token_chunk_resume"``.
        max_future_chunks: Max chunks to wait per chunk-index before
            force-committing regardless of signal.
        signal_type: Which signal drives the decision. One of
            ``"top1_prob"``, ``"entropy"``, ``"margin"``,
            ``"topk_mass"``, ``"fake_random"``, ``"sr_cem_causal"``,
            ``"sr_cem_chunk"``, ``"wait_policy"``, ``"wait_policy_chunk"``,
            ``"ctc_disagree"``.
        top1_prob_threshold: Defer when ``top1_prob < threshold``.
        entropy_threshold: Defer when ``entropy > threshold`` (nats).
        margin_threshold: Defer when ``margin < threshold``.
        topk_mass_threshold: Defer when ``topk_mass < threshold``.
        fake_random_prob: Probability of a defer trigger in
            ``fake_random`` mode. Used to test the plumbing.
        fake_random_seed: Seed for ``fake_random`` so tests are reproducible.
        ctc_blank_id: CTC blank id used when filtering blank frames.
        sr_cem_threshold: SR-CEM: defer when calibrated p(correct) is below
            this threshold. Also the operating point for ``wait_policy``,
            which defers when p(wait) is above it (opposite polarity).
        ctc_disagree_threshold: CTC-vs-attention disagreement: defer when
            the (binary) signal exceeds this threshold.
    """

    enabled: bool = False
    mode: str = MODE_PRE_EMPTIVE
    max_future_chunks: int = 1
    signal_type: str = SIGNAL_TOP1
    top1_prob_threshold: float = 0.6
    entropy_threshold: float = 1.5
    margin_threshold: float = 0.3
    topk_mass_threshold: float = 0.8
    fake_random_prob: float = 0.5
    fake_random_seed: int = 1234
    ctc_blank_id: int = 0
    # SR-CEM: defer when calibrated p(correct) < threshold. The trained
    # scorer may also bundle its own learned threshold in the checkpoint;
    # the CLI flag overrides whichever is set there.
    sr_cem_threshold: float = 0.5
    # CTC-vs-attention disagreement: defer when signal > threshold. The signal
    # is binary (1.0 disagree / 0.0 agree), so any threshold in (0, 1) defers
    # all disagreements; intermediate behaviour is reserved for a future graded
    # variant.
    ctc_disagree_threshold: float = 0.5

    def __post_init__(self):
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {self.mode!r}")
        if self.signal_type not in _SIGNALS:
            raise ValueError(
                f"signal_type must be one of {_SIGNALS}, got "
                f"{self.signal_type!r}"
            )
        if self.max_future_chunks < 0:
            raise ValueError(
                f"max_future_chunks must be >= 0, got {self.max_future_chunks}"
            )
        if not (0.0 <= self.fake_random_prob <= 1.0):
            raise ValueError(
                f"fake_random_prob must be in [0,1], got {self.fake_random_prob}"
            )


# Decisions returned by the controller.
COMMIT = "commit"
DEFER = "defer"


class DynamicFutureChunksController:
    """Stateful controller for dynamic future-chunk decoding.

    Tracks, per chunk index, how many future chunks have already been
    consumed in defer decisions, and exposes two decision endpoints:

    * :meth:`decide_chunk`: invoked once per (chunk, encoder pass) by
      the chunk-level paths (``pre_emptive`` and ``chunk_rollback``).
    * :meth:`decide_token`: invoked at each beam-search step by the
      token-level path (``token_resume``).

    The same gating semantics apply to both:

    1. ``is_final=True`` always commits.
    2. ``defer_count[chunk_idx] >= max_future_chunks`` always commits.
    3. Every DEFER charges one budget slot; the caller must invoke
       :meth:`decide_chunk` at most once per encoder pass per chunk
       index so re-evaluations without new audio are not double-charged.
    4. Otherwise compare the signal against the configured threshold.
    """

    def __init__(self, config: DynamicFutureChunksConfig):
        self.config = config
        self.defer_count: Dict[int, int] = {}
        self.last_metrics: Dict[int, ConfidenceMetrics] = {}
        # For token-level: how many times decide_token has fired for the
        # current chunk (i.e. consumed defer budget within this chunk).
        # Reset on COMMIT of that chunk.
        # Introspection-only state: never gates a decision (gating uses
        # defer_count).
        self.token_defer_within_chunk: Dict[int, int] = {}
        # Decision log: tuples of
        # (chunk_idx, level, signal_name, signal_value, future_available,
        #  decision, reason). level is "chunk" or "token".
        self.decision_log: List[Tuple] = []
        # Local RNG for fake_random; isolated so other RNG state isn't
        # disturbed by toggling this mode.
        self._rng = random.Random(config.fake_random_seed)

    def reset(self) -> None:
        """Clear all per-utterance state (counters, log, fake_random RNG)."""
        self.defer_count.clear()
        self.last_metrics.clear()
        self.token_defer_within_chunk.clear()
        self.decision_log.clear()
        self._rng = random.Random(self.config.fake_random_seed)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def mode(self) -> str:
        return self.config.mode

    # ---- signal evaluation -----------------------------------------

    def _chunk_signal_value(self, metrics: ConfidenceMetrics) -> Optional[float]:
        st = self.config.signal_type
        if st == SIGNAL_TOP1:
            return metrics.top1_prob_mean
        if st == SIGNAL_ENTROPY:
            return metrics.entropy_mean
        if st == SIGNAL_MARGIN:
            return metrics.margin_mean
        if st == SIGNAL_TOPK_MASS:
            return metrics.topk_mass_mean
        if st == SIGNAL_FAKE_RANDOM:
            return None  # signal generated inside _signal_says_defer
        if st == SIGNAL_SR_CEM_CHUNK:
            return getattr(metrics, "sr_cem_chunk_p", None)
        if st == SIGNAL_SR_CEM_CAUSAL:
            return None  # token-level signal; no per-chunk scalar
        raise AssertionError(f"unknown signal_type {st!r}")

    def _signal_says_defer_chunk(self, metrics: ConfidenceMetrics) -> Tuple[bool, float]:
        cfg = self.config
        st = cfg.signal_type
        if st == SIGNAL_FAKE_RANDOM:
            r = self._rng.random()
            return r < cfg.fake_random_prob, r
        # SR-CEM-chunk is the only chunk-level CEM signal; its value lives
        # on the ConfidenceMetrics extension. Defer when calibrated p<thr.
        if st == SIGNAL_SR_CEM_CHUNK:
            # Attached by the chunk_rollback hook (see
            # _branch_chunk_rollback in asr_inference_streaming_modified.py).
            v = getattr(metrics, "sr_cem_chunk_p", None)
            if v is None:
                # Trained scorer not loaded; treat as confident (no defer).
                return False, 1.0
            return float(v) < cfg.sr_cem_threshold, float(v)
        # SR-CEM-causal is token-level only; trying to use it on a chunk
        # path is a configuration error.
        if st == SIGNAL_SR_CEM_CAUSAL:
            raise ValueError(
                "signal_type='sr_cem_causal' is for token_resume (Branch A); "
                "use 'sr_cem_chunk' for chunk_rollback (Branch B)."
            )
        v = self._chunk_signal_value(metrics)
        assert v is not None
        if st == SIGNAL_TOP1:
            return v < cfg.top1_prob_threshold, v
        if st == SIGNAL_ENTROPY:
            return v > cfg.entropy_threshold, v
        if st == SIGNAL_MARGIN:
            return v < cfg.margin_threshold, v
        if st == SIGNAL_TOPK_MASS:
            return v < cfg.topk_mass_threshold, v
        raise AssertionError(f"unknown signal_type {st!r}")

    def _signal_says_defer_token(
        self, token_signals: Optional[Dict[str, float]]
    ) -> Tuple[bool, float]:
        cfg = self.config
        st = cfg.signal_type
        if st == SIGNAL_FAKE_RANDOM:
            r = self._rng.random()
            return r < cfg.fake_random_prob, r
        # chunk-level signals on the token path are a configuration error.
        if st in (SIGNAL_SR_CEM_CHUNK, SIGNAL_WAIT_CHUNK):
            raise ValueError(
                f"signal_type={st!r} is chunk-level (Branch B / chunk_rollback); "
                "use 'sr_cem_causal' or 'wait_policy' for token_resume (Branch A)."
            )
        assert token_signals is not None, (
            "token_signals required for non-fake_random signal in token mode"
        )
        # SR-CEM-causal: defer when calibrated p(correct) < threshold. If
        # the trained scorer isn't loaded the hook leaves the key absent,
        # in which case we no-op (commit) gracefully.
        if st == SIGNAL_SR_CEM_CAUSAL:
            v = token_signals.get(SIGNAL_SR_CEM_CAUSAL)
            if v is None:
                return False, 1.0
            return float(v) < cfg.sr_cem_threshold, float(v)
        # Learned wait-policy: defer when p(wait) > threshold (OPPOSITE polarity
        # to SR-CEM). Reuses sr_cem_threshold as the swept operating point.
        # Scorer absent -> key missing -> no-op commit (p_wait treated as 0).
        if st == SIGNAL_WAIT_CAUSAL:
            v = token_signals.get(SIGNAL_WAIT_CAUSAL)
            if v is None:
                return False, 0.0
            return float(v) > cfg.sr_cem_threshold, float(v)
        # CTC-vs-attention disagreement: defer when the binary signal fires.
        # Producer absent -> key missing -> no-op commit (treated as agree).
        if st == SIGNAL_CTC_DISAGREE:
            v = token_signals.get(SIGNAL_CTC_DISAGREE)
            if v is None:
                return False, 0.0
            return float(v) > cfg.ctc_disagree_threshold, float(v)
        v = token_signals[st]
        if st == SIGNAL_TOP1:
            return v < cfg.top1_prob_threshold, v
        if st == SIGNAL_ENTROPY:
            return v > cfg.entropy_threshold, v
        if st == SIGNAL_MARGIN:
            return v < cfg.margin_threshold, v
        if st == SIGNAL_TOPK_MASS:
            return v < cfg.topk_mass_threshold, v
        raise AssertionError(f"unknown signal_type {st!r}")

    # ---- decision endpoints ----------------------------------------

    def decide_chunk(
        self,
        chunk_idx: int,
        metrics: ConfidenceMetrics,
        future_available: int,
        is_final: bool,
    ) -> str:
        """Decide commit/defer for a chunk-level evaluation.

        ``future_available`` = ``chunk_tracker.current_chunk_idx - chunk_idx``.

        Each call to this method represents one evaluation attempt under
        a (possibly) updated encoder pass; the caller is expected to
        invoke it at most once per encoder pass per chunk index.
        ``defer_count[chunk_idx]`` counts those defers; force-commit when
        that count reaches ``max_future_chunks``.
        """
        if not self.config.enabled:
            self.decision_log.append(
                (chunk_idx, "chunk", self.config.signal_type, None,
                 future_available, COMMIT, "disabled")
            )
            return COMMIT

        self.last_metrics[chunk_idx] = metrics
        spent = self.defer_count.get(chunk_idx, 0)

        if is_final:
            self._log_chunk(chunk_idx, metrics, future_available, COMMIT, "force:is_final")
            return COMMIT
        if spent >= self.config.max_future_chunks:
            self._log_chunk(chunk_idx, metrics, future_available, COMMIT, "force:budget_exhausted")
            return COMMIT

        defer, signal_value = self._signal_says_defer_chunk(metrics)
        if not defer:
            self._log_chunk(chunk_idx, metrics, future_available, COMMIT,
                            "confident", signal_value=signal_value)
            return COMMIT

        self.defer_count[chunk_idx] = spent + 1
        self._log_chunk(chunk_idx, metrics, future_available, DEFER,
                        "low_confidence", signal_value=signal_value)
        return DEFER

    def decide_token(
        self,
        chunk_idx: int,
        token_step: int,
        token_signals: Optional[Dict[str, float]],
        future_available: int,
        is_final: bool,
    ) -> str:
        """Decide commit/defer for a single beam-search step.

        Token-level mode (Branch A). On DEFER, the caller should drop
        the just-emitted token and return so the next call (with more
        encoder context) resumes from this step. The controller tracks
        how many defers have fired *within* the current chunk and
        force-commits when the per-chunk budget is exhausted.

        ``token_signals``: dict from :func:`token_signals_from_logp`,
        or None when ``signal_type=fake_random``.
        """
        if not self.config.enabled:
            self._log_token(chunk_idx, token_step, future_available, COMMIT, "disabled")
            return COMMIT

        spent_within = self.token_defer_within_chunk.get(chunk_idx, 0)
        spent_chunk = self.defer_count.get(chunk_idx, 0)

        if is_final:
            self._log_token(chunk_idx, token_step, future_available, COMMIT, "force:is_final")
            return COMMIT
        if spent_chunk >= self.config.max_future_chunks:
            self._log_token(chunk_idx, token_step, future_available, COMMIT, "force:budget_exhausted")
            return COMMIT

        defer, signal_value = self._signal_says_defer_token(token_signals)
        if not defer:
            self._log_token(chunk_idx, token_step, future_available, COMMIT,
                            "confident", signal_value=signal_value)
            return COMMIT

        # On a token-level defer, we consume *one* future-chunks budget
        # slot per chunk (the next call provides one more future chunk
        # to the encoder). Multiple defers within the same chunk during
        # a single call are not meaningful because we return after the
        # first defer; so we increment defer_count by one here as well.
        self.defer_count[chunk_idx] = spent_chunk + 1
        self.token_defer_within_chunk[chunk_idx] = spent_within + 1
        self._log_token(chunk_idx, token_step, future_available, DEFER,
                        "low_confidence", signal_value=signal_value)
        return DEFER

    # ---- bookkeeping for callers -----------------------------------

    def note_chunk_committed(self, chunk_idx: int) -> None:
        """Caller may call this when a chunk is finally committed to clear
        per-chunk token-level counters. Safe to omit; the counter just
        stays around without effect."""
        self.token_defer_within_chunk.pop(chunk_idx, None)

    # ---- logging helpers -------------------------------------------

    def _log_chunk(self, chunk_idx, metrics, future_available, decision,
                   reason, signal_value=None):
        if signal_value is None:
            signal_value = self._chunk_signal_value(metrics)
        self.decision_log.append((
            chunk_idx, "chunk", self.config.signal_type, signal_value,
            future_available, decision, reason,
        ))

    def _log_token(self, chunk_idx, token_step, future_available, decision,
                   reason, signal_value=None):
        self.decision_log.append((
            chunk_idx, "token", self.config.signal_type, signal_value,
            future_available, decision, f"step={token_step}:{reason}",
        ))

    def summary(self) -> dict:
        """Return the per-utterance decision summary (commit/defer counts,
        future chunks spent, mode and signal type). Logged once per
        utterance by the inference script's [dyn_fc] lines."""
        n = len(self.decision_log)
        n_defer = sum(1 for d in self.decision_log if d[5] == DEFER)
        return {
            "n_decisions": n,
            "n_commit": n - n_defer,
            "n_defer": n_defer,
            "total_future_chunks_spent": sum(self.defer_count.values()),
            "n_chunks_with_future": len(self.defer_count),
            "mode": self.config.mode,
            "signal_type": self.config.signal_type,
        }
