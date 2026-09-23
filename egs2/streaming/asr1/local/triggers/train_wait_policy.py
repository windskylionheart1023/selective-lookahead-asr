#!/usr/bin/env python3
"""Train the learned WAIT-POLICY probe and benchmark it against confidence.

Same tiny MLP / checkpoint format as SR-CEM (so it drops into the controller
via signal_type='wait_policy'), but the label is y_wait (does deferring fix this
token) instead of correctness. Trains with BCEWithLogitsLoss + pos_weight for the
~5-8% positive imbalance; the saved state_dict loads into the sigmoid model
(_make_score_cem) used at inference.

CRUCIAL OUTPUT: held-out AUC/AP of the wait-policy vs CONFIDENCE baselines on the
SAME y_wait labels -- this is the direct test of "is confidence near-random at the
WAIT decision?". Baselines (all computable from the Variant-A features):
  - SR-CEM p_correct : score rows with an existing sr_cem_A scorer; wait-predictor = 1 - p_correct
  - raw step score   : feat[0] (selected_cum - prev_cum); wait-predictor = -score

Usage:
    python train_wait_policy.py --dataset wait_policy_A_train.pt --variant A \
        --out_ckpt wait_policy_A.pt [--srcem_ckpt <sr_cem_A_stdz.pt>] --epochs 40
"""
from __future__ import annotations
import argparse, os, sys


def auc_ap(y, s):
    """ROC-AUC and average precision; sklearn if present, else manual AUC."""
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    except Exception:
        import numpy as np
        y = np.asarray(y); s = np.asarray(s)
        order = np.argsort(-s); y = y[order]
        P = y.sum(); N = len(y) - P
        if P == 0 or N == 0:
            return float("nan"), float("nan")
        tp = np.cumsum(y); fp = np.cumsum(1 - y)
        tpr = tp / P; fpr = fp / N
        auc = float(np.trapz(tpr, fpr))
        prec = tp / (tp + fp); ap = float(np.trapz(prec, tpr))
        return auc, ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", required=True, choices=["A", "B"])
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--srcem_ckpt", default=None,
                    help="Existing sr_cem_{A,B}_stdz.pt for the confidence baseline.")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pos_weight", type=float, default=-1.0,
                    help="BCE pos_weight; <0 => auto neg/pos on train split.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from espnet2.asr_stream.sr_cem import (
        INPUT_DIM_A, INPUT_DIM_B, FEAT_KEYS_A, FEAT_KEYS_B, HIDDEN,
        load_sr_cem_checkpoint,
    )

    ds = torch.load(args.dataset, map_location="cpu")
    feats, labels = ds["features"], ds["labels"]
    utt_ids = ds.get("utt_ids", None)
    D = INPUT_DIM_A if args.variant == "A" else INPUT_DIM_B
    keys = FEAT_KEYS_A if args.variant == "A" else FEAT_KEYS_B
    assert feats.shape[1] == D, f"feat dim {feats.shape[1]} != {D}"
    pos = float(labels.mean())
    print(f"rows={len(feats)}  pos_frac(y_wait)={pos:.4f}  ({int(labels.sum())} positives)")

    # --- utterance-level split (no leakage across the pooled nrc levels) ---
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    uniq = sorted({str(u) for u in utt_ids})
    perm = torch.randperm(len(uniq), generator=g).tolist()
    uniq = [uniq[i] for i in perm]
    nval = max(1, int(len(uniq) * args.val_ratio))
    valset = set(uniq[:nval])
    vidx = [i for i, u in enumerate(utt_ids) if str(u) in valset]
    tidx = [i for i, u in enumerate(utt_ids) if str(u) not in valset]
    Xtr, Ytr = feats[tidx], labels[tidx]
    Xva, Yva = feats[vidx], labels[vidx]
    print(f"utt split: {len(uniq)-nval} train / {nval} val utts "
          f"({len(tidx)} / {len(vidx)} tokens); val pos_frac={float(Yva.mean()):.4f}")

    mean = Xtr.mean(0); std = Xtr.std(0).clamp_min(1e-6)
    Xtr_n = (Xtr - mean) / std; Xva_n = (Xva - mean) / std

    pw = args.pos_weight
    if pw < 0:
        npos = float(Ytr.sum()); nneg = len(Ytr) - npos
        pw = nneg / max(1.0, npos)
    print(f"pos_weight={pw:.2f}")

    dev = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    # logits model (sigmoid applied at inference by _make_score_cem); same Linear
    # layout/keys (0,2) so the state_dict loads into the SR-CEM sigmoid model.
    model = nn.Sequential(nn.Linear(D, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)).to(dev)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=dev))
    loader = DataLoader(TensorDataset(Xtr_n, Ytr), batch_size=args.batch_size, shuffle=True)

    def val_scores():
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(Xva_n.to(dev))).cpu().squeeze(1).numpy()

    best_auc, best_state = -1.0, None
    yva = Yva.squeeze(1).numpy()
    for ep in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        a, p = auc_ap(yva, val_scores())
        tag = ""
        if a > best_auc:
            best_auc = a
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            tag = " [BEST]"
        if ep % 5 == 0 or ep == args.epochs - 1 or tag:
            print(f"  epoch {ep+1:3d}/{args.epochs}: val AUC={a:.4f} AP={p:.4f}{tag}")

    # save best (SR-CEM ckpt format -> loadable by load_sr_cem_checkpoint)
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": best_state, "variant": args.variant, "feat_keys": list(keys),
        "input_size": int(D), "feat_mean": mean.cpu(), "feat_std": std.cpu(),
        "standardize": True,
    }, args.out_ckpt)
    wp_auc, wp_ap = auc_ap(yva, val_scores())
    print(f"\nSaved {args.out_ckpt}  (best val AUC={best_auc:.4f})")

    # ---- CONFIDENCE BASELINES on the SAME val labels ----
    print("\n=== WAIT-decision discrimination on held-out val (AUC / AP) ===")
    print(f"  learned wait-policy : AUC={wp_auc:.4f}  AP={wp_ap:.4f}")
    # raw step score: feat[0]; confident(high) => unlikely to need wait => predictor -score
    raw_score = -Xva[:, 0].numpy()
    a, p = auc_ap(yva, raw_score)
    print(f"  confidence: -raw_score : AUC={a:.4f}  AP={p:.4f}")
    if args.srcem_ckpt and os.path.exists(args.srcem_ckpt):
        sc = load_sr_cem_checkpoint(path=args.srcem_ckpt, variant=args.variant, device="cpu")
        if sc is not None:
            pc = [sc.predict(list(map(float, row))) for row in Xva.numpy()]
            import numpy as np
            a, p = auc_ap(yva, 1.0 - np.asarray(pc))  # low p_correct => wait
            print(f"  confidence: SR-CEM 1-p_correct : AUC={a:.4f}  AP={p:.4f}")
    print("\n(AUC ~0.5 = near-random; the learned trigger should be markedly higher.)")


if __name__ == "__main__":
    main()
