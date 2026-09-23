"""CTC-vs-attention disagreement deferral signal.

A frame-synchronous CTC greedy decode of the same encoder output is compared to
the autoregressive attention hypothesis. When the attention decoder's just-emitted
token disagrees with the CTC view (a substitution/insertion, or the attention is
skipping a CTC token = a deletion), the token is flagged for deferral.

Computed at the beam-search producer seam (the encoder output ``x`` and the CTC
module are both in scope there). Pure-PyTorch / NumPy; no model retraining.

Superseded: inference computes the ``ctc_disagree`` signal inline via the CTC
prefix scorer in espnet/nets/batch_beam_search.py; this greedy-decode variant
is not imported at runtime and is kept for reference.
"""
from typing import List, Optional
import numpy as np
import torch


def ctc_greedy_tokens(ctc_module, enc: torch.Tensor, blank_id: int = 0) -> List[int]:
    """Greedy CTC decode of an encoder output -> collapsed token-id list.

    Args:
        ctc_module: the ASR model's CTC head (has ``.ctc_lo`` linear D->V, or
            falls back to ``.log_softmax``).
        enc: encoder output, shape (T, D) or (1, T, D).
        blank_id: CTC blank index (collapsed out).
    Returns:
        list of token ids after blank removal + repeat collapse.
    """
    if enc is None or enc.numel() == 0:
        return []
    if enc.dim() == 3:
        enc = enc[0]
    with torch.no_grad():
        if hasattr(ctc_module, "ctc_lo"):
            logits = ctc_module.ctc_lo(enc)
        else:  # CTCPrefixScorer-style wrapper or raw module
            inner = getattr(ctc_module, "ctc", ctc_module)
            logits = inner.ctc_lo(enc)
        ids = logits.argmax(dim=-1).tolist()
    out: List[int] = []
    prev = -1
    for i in ids:
        if i != prev and i != blank_id:
            out.append(i)
        prev = i
    return out


def _align_ops(attn: List[int], ctc: List[int]):
    """Levenshtein backtrace of attn vs ctc. Returns, per attn index, a tuple
    (op, ctc_skips_before) where op in {"=","S","I"} and ctc_skips_before is the
    number of unmatched ctc tokens immediately preceding this attn token
    (a deletion the attention is stepping over)."""
    n, m = len(attn), len(ctc)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if attn[i - 1] == ctc[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + c)
    # Backtrace (walks backward). A run of CTC deletions sits, in forward order,
    # immediately BEFORE the next attention token; in backward order we meet the
    # deletions AFTER that token, so we attribute a pending deletion run to the
    # most-recently-emitted attention token (the higher forward index).
    i, j = n, m
    op_of = {}          # attn_idx -> "=", "S", or "I"
    skips_before = {}   # attn_idx -> count of ctc deletions just before it
    pending_ctc_del = 0
    last_attn_idx = None
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (0 if attn[i - 1] == ctc[j - 1] else 1):
            op_of[i - 1] = "=" if attn[i - 1] == ctc[j - 1] else "S"
            if pending_ctc_del and last_attn_idx is not None:
                skips_before[last_attn_idx] = skips_before.get(last_attn_idx, 0) + pending_ctc_del
            pending_ctc_del = 0
            last_attn_idx = i - 1
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:   # attn insertion (no ctc match)
            op_of[i - 1] = "I"
            if pending_ctc_del and last_attn_idx is not None:
                skips_before[last_attn_idx] = skips_before.get(last_attn_idx, 0) + pending_ctc_del
            pending_ctc_del = 0
            last_attn_idx = i - 1
            i -= 1
        else:                                         # ctc deletion (attn skipped it)
            pending_ctc_del += 1
            j -= 1
    # leading ctc deletions (before the first attn token) attach to that token
    if pending_ctc_del and last_attn_idx is not None:
        skips_before[last_attn_idx] = skips_before.get(last_attn_idx, 0) + pending_ctc_del
    return op_of, skips_before


def last_token_disagrees(attn: List[int], ctc: List[int]) -> bool:
    """Does the LAST attention token disagree with the CTC view?

    True when the last attn token is a substitution/insertion vs CTC, OR when the
    attention skipped one or more CTC tokens immediately before it (deletion).
    """
    if not attn:
        return False
    if not ctc:
        return True  # attention emitted a token CTC has nothing for
    op_of, skips_before = _align_ops(attn, ctc)
    last = len(attn) - 1
    return op_of.get(last, "S") != "=" or skips_before.get(last, 0) > 0


def last_token_reason(attn: List[int], ctc: List[int]) -> str:
    """Debug: why the last attn token (dis)agrees. One of:
    'agree', 'sub'(content mismatch), 'ins'(no ctc match),
    'del_skip'(ctc tokens skipped before it), 'empty_ctc'."""
    if not attn:
        return "agree"
    if not ctc:
        return "empty_ctc"
    op_of, skips_before = _align_ops(attn, ctc)
    last = len(attn) - 1
    op = op_of.get(last, "S")
    if op == "S":
        return "sub"
    if op == "I":
        return "ins"
    if skips_before.get(last, 0) > 0:
        return "del_skip"
    return "agree"


def disagree_flags(attn: List[int], ctc: List[int]) -> List[bool]:
    """Per-attn-token disagreement flags (batch / analysis use)."""
    if not attn:
        return []
    if not ctc:
        return [True] * len(attn)
    op_of, skips_before = _align_ops(attn, ctc)
    return [op_of.get(i, "S") != "=" or skips_before.get(i, 0) > 0 for i in range(len(attn))]
