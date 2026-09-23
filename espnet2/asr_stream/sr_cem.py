"""Score-Rank Confidence Estimation Module (SR-CEM) — streaming-adapted.

Reuses the published SR-CEM Score_CEM model architecture
(``espnet2/asr/CEM/model.py`` in the original CEM training repo; a
one-hidden-layer MLP, hidden=64, ReLU, sigmoid output). This module
provides the *inference-time* glue for our streaming pipeline: a thin
torch scorer that loads a published-format checkpoint, plus
per-token feature builders for the two streaming variants.

DESIGN INVARIANT: SR-CEM produces a calibrated p(correct) per token
that is used ONLY to drive the dynamic-future-chunks defer/commit
trigger. It does NOT replace the raw softmax probability in the beam
search, nor the cumulative score that ranks hypotheses. Beam search
continues to operate on its native weighted_scores throughout; SR-CEM
features are READ from weighted_scores but never WRITTEN back.

Variant A — *causal* SR-CEM (Branch A, ``token_resume``):
    7-dim feature vector at decoder step t (all causally available):
        [score, rank, S_lt, top4_1, top4_2, top4_3, top4_4]
    The published paper's 8-feature variant minus S>t (succeeding
    cumulative score). The ablation in the paper (Table 4) shows
    dropping S>t is the largest single-feature hit, but causal SR-CEM
    is still ~3× more calibrated than raw softmax.

Variant B — *chunk-local* SR-CEM (Branch B, ``chunk_rollback``):
    8-dim feature vector at decoder step t (all causally available
    *after* the chunk's beam search completes):
        [score, rank, S_lt, S_gt_chunk, top4_1, top4_2, top4_3, top4_4]
    where S_gt_chunk = Σ_{k>t, k in current_chunk} score_k. Matches the
    published paper's 8-feature spec exactly, but with S>t restricted
    to within the just-decoded chunk so the feature is causal at
    Branch B's decision moment.

Both variants assume the training-time feature set drops the raw
'confidence' (softmax max-prob) input, matching the published
``train_token_ours.py`` setup (``remove_features = ['confidence']``).
Both share the same architecture (``Score_CEM``); only the input
dimension differs (7 vs 8; A_MARGIN expands to 9 via two rank-margin
features appended in ``predict()``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Feature schemas. Order matters: training and inference must agree.
FEAT_KEYS_A: Tuple[str, ...] = (
    "score", "rank", "S_lt",
    "top4_1", "top4_2", "top4_3", "top4_4",
)
FEAT_KEYS_B: Tuple[str, ...] = (
    "score", "rank", "S_lt", "S_gt_chunk",
    "top4_1", "top4_2", "top4_3", "top4_4",
)
# Variant A_MARGIN — Variant-A causal 7 features + two rank-margin features
# (top4_1-top4_2, top4_1-top4_3). The seam still builds the 7-dim causal
# vector; the scorer expands 7->9 in predict(), so no seam change is needed.
FEAT_KEYS_A_MARGIN: Tuple[str, ...] = FEAT_KEYS_A + ("margin12", "margin13")
INPUT_DIM_A = len(FEAT_KEYS_A)
INPUT_DIM_B = len(FEAT_KEYS_B)
INPUT_DIM_A_MARGIN = len(FEAT_KEYS_A_MARGIN)
TOPK = 4
HIDDEN = 64


def step_features_from_cumulative(
    selected_cum: float,
    prev_selected_cum: float,
    candidate_cums,
) -> Tuple[float, int, float, List[float]]:
    """Derive the paper-recipe per-step features from CUMULATIVE scores.

    Canonical semantics, mirroring the published training pipeline
    (``espnet2/asr/CEM/dataset_token_ours.py`` in the original repo):

      score = selected_cum - prev_selected_cum   # diffed cumulatives
                                                 # (``diff_scores`` there)
      rank  = 1 + #{c : cum_c > selected_cum}    # candidate order is
                                                 # basis-invariant (all
                                                 # candidates share the
                                                 # same prefix score)
      S_lt  = prev_selected_cum                  # == prefix sum of diffs
                                                 # (``prev_sum`` there)
      top4  = top-4 RAW cumulative candidate     # the published features
              scores, NOT diffed                 # use the raw stored
                                                 # step scores

    Every feature producer (decision-time hook, JSONL dump, Variant B)
    MUST go through this function: maintaining two implementations of
    the feature definitions is how the cumulative-vs-incremental
    train/inference mismatch happened.

    Args:
        selected_cum: cumulative (prefix-inclusive) score of the selected
            token at this step. MUST be taken from the same numeric
            representation as ``candidate_cums`` (e.g. the same fp32
            tensor row) — passing a higher-precision value against
            fp32-rounded candidates can flip the strict ``>`` in the rank
            computation for the selected token itself.
        prev_selected_cum: cumulative score of the previously selected
            token (0.0 at the first token).
        candidate_cums: cumulative candidate scores at this step — a
            list/iterable of floats or a 1-D torch tensor.

    Returns:
        (score, rank, S_lt, top4) per the published recipe.
    """
    if hasattr(candidate_cums, "tolist") and not isinstance(candidate_cums, list):
        vals = [float(v) for v in candidate_cums.tolist()]
    else:
        vals = [float(v) for v in candidate_cums]
    sel = float(selected_cum)
    prev = float(prev_selected_cum)
    score = sel - prev
    rank = 1 + sum(1 for v in vals if v > sel)
    top4 = sorted(vals, reverse=True)[:TOPK]
    while len(top4) < TOPK:
        top4.append(0.0)
    return score, rank, prev, top4


def build_features_causal(
    score: float,
    rank: int,
    S_lt: float,
    top4: List[float],
) -> List[float]:
    """Variant A: 7-dim causal feature vector."""
    t = list(top4) + [0.0] * (TOPK - len(top4))
    t = t[:TOPK]
    return [float(score), float(rank), float(S_lt), *(float(x) for x in t)]


def build_features_chunk(
    score: float,
    rank: int,
    S_lt: float,
    S_gt_chunk: float,
    top4: List[float],
) -> List[float]:
    """Variant B: 8-dim feature vector with chunk-local S>t."""
    t = list(top4) + [0.0] * (TOPK - len(top4))
    t = t[:TOPK]
    return [float(score), float(rank), float(S_lt), float(S_gt_chunk),
            *(float(x) for x in t)]


# -----------------------------------------------------------------
# Torch scorer (loads the published Score_CEM checkpoint format).
# -----------------------------------------------------------------


def _make_score_cem(input_size: int, hidden_size: int = HIDDEN):
    """Construct the published Score_CEM MLP without importing from the
    upstream code (so we don't add a hard dependency on the external
    CEM training branch). Architecture matches
    ``espnet2.asr.CEM.model.Score_CEM`` exactly:
        Linear(input, hidden) -> ReLU -> Linear(hidden, 1) -> Sigmoid.
    Loading a state_dict from the upstream checkpoint will succeed.
    """
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
        nn.Sigmoid(),
    )


@dataclass
class SRCemScorer:
    """Torch-based inference scorer for a trained SR-CEM checkpoint.

    Loads either a state_dict (.pt from ``train_token_ours.py``) or a
    full module checkpoint. Variant inferred from input_size when not
    explicitly given.

    When ``feat_mean`` / ``feat_std`` are set, z-score standardization
    is applied at inference. Saved at training time so train and
    inference use identical statistics.
    """

    model: object  # torch.nn.Module
    variant: str  # 'A' or 'B'
    threshold: float
    feat_keys: Tuple[str, ...]
    device: str = "cpu"
    feat_mean: object = None  # torch.Tensor[D] or None
    feat_std: object = None   # torch.Tensor[D] or None
    # Variant B chunk aggregation ('mean'/'min') the threshold was
    # calibrated against; stored in the checkpoint so inference cannot
    # silently use a different aggregation.
    chunk_agg: Optional[str] = None

    def __post_init__(self):
        if self.variant not in ("A", "B", "A_MARGIN"):
            raise ValueError(f"variant must be A, B or A_MARGIN, got {self.variant!r}")
        expected = (
            FEAT_KEYS_A_MARGIN if self.variant == "A_MARGIN"
            else FEAT_KEYS_A if self.variant == "A" else FEAT_KEYS_B
        )
        if self.feat_keys != expected:
            raise ValueError(
                f"feat_keys mismatch: expected {expected}, got {self.feat_keys}"
            )

    def predict(self, feat: List[float]) -> float:
        """Run the MLP on a single feature vector. Returns p in (0,1).

        Applies z-score standardization with the checkpoint's saved
        mean/std when present. For A_MARGIN, the seam passes the 7-dim
        causal vector; expand it to 9 (append the two rank-margins) here.
        """
        import torch
        if self.variant == "A_MARGIN" and len(feat) == INPUT_DIM_A:
            feat = list(feat) + [feat[3] - feat[4], feat[3] - feat[5]]
        x = torch.tensor(feat, dtype=torch.float32, device=self.device)
        if self.feat_mean is not None and self.feat_std is not None:
            x = (x - self.feat_mean.to(x.device)) / self.feat_std.to(x.device).clamp_min(1e-6)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            p = self.model(x)
        return float(p.squeeze().item())


def load_sr_cem_checkpoint(
    path: str,
    variant: str,
    threshold: Optional[float] = None,
    device: str = "cpu",
    chunk_agg: Optional[str] = None,
) -> Optional[SRCemScorer]:
    """Load a trained SR-CEM checkpoint.

    Args:
        path: ``.pt`` file produced by ``train_token_ours.py`` (state_dict)
            or by our adapted streaming trainer. Returns ``None`` (with a
            warning) if the file doesn't exist so callers can no-op
            gracefully when no checkpoint is configured.
        variant: 'A' (causal), 'B' (chunk-local) or 'A_MARGIN'
            (causal + two rank-margin features).
        threshold: defer when ``p_correct < threshold``.
        device: 'cpu' or 'cuda'.
        chunk_agg: Variant B chunk aggregation ('mean'/'min') the
            threshold was calibrated against; a caller value overrides
            the checkpoint's stored value.

    Returns:
        SRCemScorer or None.
    """
    import torch

    if not os.path.exists(path):
        logging.warning(
            "SR-CEM checkpoint %s does not exist; signal will no-op (COMMIT)",
            path,
        )
        return None

    input_size = (
        INPUT_DIM_A_MARGIN if variant == "A_MARGIN"
        else INPUT_DIM_A if variant == "A" else INPUT_DIM_B
    )
    model = _make_score_cem(input_size)
    state = torch.load(path, map_location=device)
    feat_mean = None
    feat_std = None
    ckpt_threshold = None
    ckpt_chunk_agg = None
    if isinstance(state, dict) and "state_dict" in state:
        # Wrapper dict {state_dict, threshold, variant, chunk_agg,
        # feat_mean, feat_std, ...}.
        if "threshold" in state:
            ckpt_threshold = float(state["threshold"])
        if "chunk_agg" in state:
            ckpt_chunk_agg = str(state["chunk_agg"])
        if "variant" in state:
            ckpt_variant = str(state["variant"])
            if ckpt_variant != variant:
                logging.warning(
                    "SR-CEM ckpt variant=%s but loader asked for variant=%s; "
                    "trusting loader.", ckpt_variant, variant,
                )
        if state.get("standardize") and "feat_mean" in state and "feat_std" in state:
            feat_mean = state["feat_mean"].to(device).float()
            feat_std = state["feat_std"].to(device).float()
            logging.info(
                "SR-CEM checkpoint includes z-score stats: "
                "mean=%s, std=%s",
                feat_mean.tolist(), feat_std.tolist(),
            )
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # Precedence: explicit caller value > checkpoint's calibrated value
    # > documented default (0.5 / 'mean'). The active source is logged.
    if threshold is not None:
        eff_threshold, thr_src = float(threshold), "caller/CLI"
    elif ckpt_threshold is not None:
        eff_threshold, thr_src = ckpt_threshold, "checkpoint"
    else:
        eff_threshold, thr_src = 0.5, "default"
    if chunk_agg is not None:
        eff_agg, agg_src = str(chunk_agg), "caller/CLI"
    elif ckpt_chunk_agg is not None:
        eff_agg, agg_src = ckpt_chunk_agg, "checkpoint"
    else:
        eff_agg, agg_src = "mean", "default"
    logging.info(
        "SR-CEM threshold=%.4f (source: %s); chunk_agg=%s (source: %s)",
        eff_threshold, thr_src, eff_agg, agg_src,
    )

    feat_keys = (
        FEAT_KEYS_A_MARGIN if variant == "A_MARGIN"
        else FEAT_KEYS_A if variant == "A" else FEAT_KEYS_B
    )
    return SRCemScorer(
        model=model,
        variant=variant,
        threshold=eff_threshold,
        feat_keys=feat_keys,
        device=device,
        feat_mean=feat_mean,
        feat_std=feat_std,
        chunk_agg=eff_agg,
    )


# -----------------------------------------------------------------
# Per-step / per-chunk inference helpers.
# -----------------------------------------------------------------


@dataclass
class CausalPrefixState:
    """Running state for Variant A: keeps S_lt (cumulative prefix score)
    across decoder steps within one beam search call. Reset between
    utterances via the controller's ``reset()``.

    Only the top-1 beam's prefix score sum is tracked, since the
    controller's per-step hook operates on the top-1 hypothesis.
    """

    s_lt: float = 0.0  # cumulative score of preceding tokens
    n_tokens: int = 0  # number of tokens accumulated

    def push(self, score: float) -> None:
        """Accumulate the current step's score into the running prefix sum."""
        self.s_lt += float(score)
        self.n_tokens += 1

    def s_lt_at_t(self) -> float:
        """Returns S_lt to feed into the CEM features for the CURRENT step
        (i.e. before pushing this step's score)."""
        return self.s_lt

    def reset(self) -> None:
        self.s_lt = 0.0
        self.n_tokens = 0


@dataclass
class ChunkScoresBuffer:
    """Running state for Variant B: per-chunk list of (token_idx, score)
    tuples, used to compute chunk-local S>t at chunk_rollback decision time.

    The buffer is populated during a chunk's beam search (one push per
    committed token via the same per-step hook as Variant A) and consumed
    once at chunk-rollback's decide_chunk call site.
    """

    scores: List[float] = field(default_factory=list)

    def push(self, score: float) -> None:
        """Append one committed token's score to the current chunk buffer."""
        self.scores.append(float(score))

    def s_gt_chunk_at(self, idx: int) -> float:
        """Sum of scores after index ``idx`` in the current chunk."""
        if idx + 1 >= len(self.scores):
            return 0.0
        return float(sum(self.scores[idx + 1:]))

    def __len__(self) -> int:
        return len(self.scores)

    def reset(self) -> None:
        self.scores = []
