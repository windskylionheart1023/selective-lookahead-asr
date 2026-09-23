#!/usr/bin/env python3
"""Build the WAIT-POLICY training dataset from multi-lookahead streaming dumps.

The wait-policy predicts ``p_wait`` = "will deferring >=1 more chunk turn this
token from wrong -> right?" (vs SR-CEM's ``p_correct`` = "is this token right?").

Supervision is a level-conditional binary counterfactual built from STATIC
``num_right_chunks=k`` decodes for k = 0..K (K=4 for the SOFT R4MAX model).
This is exact because always-deferring-to-k is byte-identical to static nrc=k
(see memory: project_dfc_deep_deferral_fixed). No in-call-deepen needed for
offline labels.

Label (for a committed token of the pass-``k`` decode aligned to ref position j):

    y_wait = 1  iff  c_k(j) == 0  AND  exists k' in (k, K]: c_k'(j) == 1

where ``c_k(j) = 1`` iff the static nrc=k decode produces ref token j correctly.
Committing at level k is wrong, but a reachable deeper level fixes ref j.
``y = 0`` covers both "already correct at k" and "wrong and never fixed".

HYP insertions (over-emissions; no ref position) use a counterfactual:

    y_wait = 1  iff  exists k' in (k, K]: this insertion's REF-gap has NO
             insertion in pass k'  (i.e. waiting suppresses the over-emission).

Features are POOLED from passes k = 0..K-1 (every level the policy will be
re-queried at after a defer); pass K has no wait budget and contributes no
rows. ``utt_ids`` stay the plain utt_id so all levels of one utt share a
train/val split (no leakage).

Output (identical format to build_sr_cem_dataset.py, consumed unchanged by
train_sr_cem.py):
    <out_dir>/wait_policy_A_train.pt   # Variant A 7-dim, labels = y_wait
    <out_dir>/wait_policy_B_train.pt   # Variant B 8-dim, labels = y_wait

Usage:
    python build_wait_policy_dataset.py \\
        --feat 0=<dir0>/sr_cem_features.jsonl 1=<dir1>/... 2=... 3=... \\
        --ter  0=<dec0>/<set>/score_ter/result.txt 1=... 2=... 3=... 4=... \\
        --out_dir <out_dir>
(--feat for k=0..K-1, --ter for k=0..K. Keys join by utt_id directly.)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Reuse the SR-CEM builder's parsing/grouping helpers verbatim (same dir).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sr_cem_dataset import (  # noqa: E402
    parse_ter_result,
    load_feat_jsonl,
    group_by_utt,
    derive_per_chunk_s_gt,
)


# --------------------------------------------------------------------------
# sclite-alignment primitives. REF/HYP/Eval are position-aligned with '*'
# fillers: REF '*' = insertion (HYP over-emits), HYP '*' = deletion. A token
# is a filler iff it CONTAINS '*' (matches build_sr_cem_dataset's mask).
# --------------------------------------------------------------------------
def _is_tok(t: str) -> bool:
    return "*" not in t


def _aligned(block: dict) -> Tuple[List[str], List[str]]:
    ref, hyp = block["REF"], block["HYP"]
    if len(ref) != len(hyp):  # defend against truncation (should not happen)
        n = min(len(ref), len(hyp))
        ref, hyp = ref[:n], hyp[:n]
    return ref, hyp


def ref_correctness(block: dict) -> List[int]:
    """c(j) for each ref position j (REF token present): 1 iff the aligned HYP
    token equals it (case-insensitive); substitution/deletion -> 0."""
    ref, hyp = _aligned(block)
    out: List[int] = []
    for r, h in zip(ref, hyp):
        if not _is_tok(r):
            continue  # insertion position: no ref token here
        out.append(int(_is_tok(h) and h.upper() == r.upper()))
    return out


def hyp_targets(block: dict) -> List[Tuple[str, int]]:
    """Per kept (non-filler) HYP position, in emission order, its target:
    ('ref', ref_j) for sub/correct, or ('ins', gap_id) for an insertion, where
    gap_id = number of ref tokens emitted before the insertion. The kept-HYP
    order matches the feature-dump rows 1:1 (both strip blanks/fillers)."""
    ref, hyp = _aligned(block)
    out: List[Tuple[str, int]] = []
    ref_seen = 0
    for r, h in zip(ref, hyp):
        r_is, h_is = _is_tok(r), _is_tok(h)
        if h_is:
            out.append(("ref", ref_seen) if r_is else ("ins", ref_seen))
        if r_is:
            ref_seen += 1
    return out


def insertion_gaps(block: dict) -> Set[int]:
    """Set of gap_ids that contain >=1 HYP insertion. gap_id is consistent
    across passes because the REF is identical, so it identifies the same
    inter-reference slot in every decode."""
    ref, hyp = _aligned(block)
    gaps: Set[int] = set()
    ref_seen = 0
    for r, h in zip(ref, hyp):
        r_is, h_is = _is_tok(r), _is_tok(h)
        if (not r_is) and h_is:
            gaps.add(ref_seen)
        if r_is:
            ref_seen += 1
    return gaps


def wait_labels_for_utt(
    blocks_by_k: Dict[int, dict], k_base: int, K: int,
) -> List[int]:
    """Per kept-HYP-token y_wait for the pass-``k_base`` decode of one utt.

    blocks_by_k must contain every pass k_base..K for this utt (else the
    caller skips the utt). Returns a list parallel to the pass-k_base feature
    rows (one entry per kept HYP token, in step order).
    """
    base = blocks_by_k[k_base]
    c_base = ref_correctness(base)
    deeper = range(k_base + 1, K + 1)
    c_deep = {k: ref_correctness(blocks_by_k[k]) for k in deeper}
    igaps_deep = {k: insertion_gaps(blocks_by_k[k]) for k in deeper}

    labels: List[int] = []
    for kind, idx in hyp_targets(base):
        if kind == "ref":
            c_k = c_base[idx] if idx < len(c_base) else 0
            if c_k == 1:
                y = 0  # already correct at this level -> commit
            else:
                # substitution: waiting helps iff some deeper level gets ref idx right
                y = int(any(
                    idx < len(c_deep[k]) and c_deep[k][idx] == 1 for k in deeper
                ))
        else:  # ('ins', gap_id): over-emission with no ref position
            # counterfactual: waiting helps iff some deeper level has NO
            # insertion in this same REF gap (the over-emission is suppressed).
            y = int(any(idx not in igaps_deep[k] for k in deeper))
        labels.append(y)
    return labels


# --------------------------------------------------------------------------
def _lookup_block(ter: Dict[str, dict], utt_id: str) -> Optional[dict]:
    """Find an utt's sclite block, tolerating spk-utt prefixing (matches
    build_sr_cem_dataset's lookup)."""
    block = ter.get(utt_id) or ter.get(f"{utt_id}-{utt_id}")
    if block is None:
        for k in ter:
            if k.endswith(utt_id):
                block = ter[k]
                break
    return block


def build_datasets(
    feat_rows_by_k: Dict[int, List[dict]],
    ter_by_k: Dict[int, Dict[str, dict]],
    K: int,
) -> Tuple[dict, dict]:
    """Pool rows from passes k=0..K-1, label by the level-conditional
    counterfactual, and emit Variant A / B torch dicts."""
    import torch

    A_feats: List[List[float]] = []
    A_labels: List[int] = []
    A_utts: List[str] = []
    B_feats: List[List[float]] = []
    B_labels: List[int] = []
    B_utts: List[str] = []

    n_pos = 0
    n_ins_pos = 0
    per_pass_rows: Dict[int, int] = defaultdict(int)
    n_skip_len = 0
    n_skip_block = 0

    for k_base in sorted(feat_rows_by_k):  # commit levels with features provided
        by_utt = group_by_utt(feat_rows_by_k[k_base])
        for utt_id, utt_rows in by_utt.items():
            # Need this utt's block in every pass k_base..K for the lookup.
            blocks_by_k: Dict[int, dict] = {}
            ok = True
            for k in range(k_base, K + 1):
                blk = _lookup_block(ter_by_k[k], utt_id)
                if blk is None:
                    ok = False
                    break
                blocks_by_k[k] = blk
            if not ok:
                n_skip_block += 1
                continue

            labels = wait_labels_for_utt(blocks_by_k, k_base, K)
            if len(labels) != len(utt_rows):
                n_skip_len += 1
                continue

            s_gt = derive_per_chunk_s_gt(utt_rows)
            targets = hyp_targets(blocks_by_k[k_base])
            for r, lbl, sg, tgt in zip(utt_rows, labels, s_gt, targets):
                feat_a = list(r["feat"])
                feat_b = feat_a[:3] + [float(sg)] + feat_a[3:]
                A_feats.append(feat_a)
                A_labels.append(int(lbl))
                A_utts.append(utt_id)
                B_feats.append(feat_b)
                B_labels.append(int(lbl))
                B_utts.append(utt_id)
                per_pass_rows[k_base] += 1
                if lbl == 1:
                    n_pos += 1
                    if tgt[0] == "ins":
                        n_ins_pos += 1

    n_tot = len(A_feats)
    print(f"  pooled rows: {n_tot}  (per pass: "
          f"{dict(sorted(per_pass_rows.items()))})")
    print(f"  positives (y_wait=1): {n_pos} "
          f"({100.0 * n_pos / max(1, n_tot):.2f}%), of which insertions: {n_ins_pos}")
    print(f"  skipped utts: len_mismatch={n_skip_len}, missing_block={n_skip_block}")

    feat_keys_a = ("score", "rank", "S_lt",
                   "top4_1", "top4_2", "top4_3", "top4_4")
    feat_keys_b = ("score", "rank", "S_lt", "S_gt_chunk",
                   "top4_1", "top4_2", "top4_3", "top4_4")
    return (
        {
            "features": torch.tensor(A_feats, dtype=torch.float32),
            "labels": torch.tensor(A_labels, dtype=torch.float32).unsqueeze(1),
            "feat_keys": feat_keys_a,
            "utt_ids": A_utts,
        },
        {
            "features": torch.tensor(B_feats, dtype=torch.float32),
            "labels": torch.tensor(B_labels, dtype=torch.float32).unsqueeze(1),
            "feat_keys": feat_keys_b,
            "utt_ids": B_utts,
        },
    )


def _parse_kv(items: List[str], what: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--{what} expects K=PATH, got {it!r}")
        k, path = it.split("=", 1)
        out[int(k)] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", nargs="+", required=True,
                    help="K=PATH for sr_cem_features.jsonl, k=0..K-1 (rows).")
    ap.add_argument("--ter", nargs="+", required=True,
                    help="K=PATH for score_ter/result.txt, k=0..K (labels).")
    ap.add_argument("--max_k", type=int, default=4,
                    help="Deepest lookahead level K (SOFT R4MAX -> 4).")
    ap.add_argument("--commit_levels", type=int, nargs="+", default=None,
                    help="Commit levels to label (default 0..K-1). Use e.g. 0 for nrc0-only.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    K = args.max_k
    commit_levels = args.commit_levels if args.commit_levels is not None else list(range(0, K))
    feat_paths = _parse_kv(args.feat, "feat")
    ter_paths = _parse_kv(args.ter, "ter")

    missing_feat = [k for k in commit_levels if k not in feat_paths]
    missing_ter = [k for k in range(0, K + 1) if k not in ter_paths]
    if missing_feat:
        raise SystemExit(f"--feat missing commit levels {missing_feat}")
    if missing_ter:
        raise SystemExit(f"--ter missing levels {missing_ter} (need 0..{K})")

    feat_rows_by_k = {}
    for k in commit_levels:
        print(f"Loading features for nrc={k}: {feat_paths[k]}")
        feat_rows_by_k[k] = load_feat_jsonl(feat_paths[k])
        print(f"  {len(feat_rows_by_k[k])} rows")

    ter_by_k = {}
    for k in range(0, K + 1):
        print(f"Loading sclite result for nrc={k}: {ter_paths[k]}")
        ter_by_k[k] = parse_ter_result(ter_paths[k])
        print(f"  {len(ter_by_k[k])} utterances")

    dsA, dsB = build_datasets(feat_rows_by_k, ter_by_k, K)

    os.makedirs(args.out_dir, exist_ok=True)
    import torch
    pA = os.path.join(args.out_dir, "wait_policy_A_train.pt")
    pB = os.path.join(args.out_dir, "wait_policy_B_train.pt")
    torch.save(dsA, pA)
    torch.save(dsB, pB)
    print(f"Saved Variant A dataset to {pA} ({tuple(dsA['features'].shape)})")
    print(f"Saved Variant B dataset to {pB} ({tuple(dsB['features'].shape)})")


if __name__ == "__main__":
    main()
