#!/usr/bin/env python3
"""Pool N shards of a sharded full-test decode into one result:
merged WER (word edit-distance over the cat'd hyp.trn/ref.trn) + pooled per-word
latency (dep = whole-chunk, prc = stable-prefix), mean/median/p90 ms.

Usage: CHUNK_MS=768 python analyze_csp_shards.py <shard_dir1> <shard_dir2> ...
  each shard_dir = .../<tag>_test_lib360_s4_K/test_lib360_s4_K  (the inner dset dir),
  or its parent (globs find score_wer/ and logdir/ underneath).
"""
import sys, glob, os, statistics as st
import per_word_latency_real as P
P.CHUNK_MS = float(os.environ.get("CHUNK_MS", 768))


def parse_trn(path):
    out = {}
    for ln in open(path):
        ln = ln.rstrip("\n")
        i = ln.rfind("(")
        if i < 0:
            continue
        txt = ln[:i].strip()
        uid = ln[i + 1:].rstrip(")").strip()
        out[uid] = txt.split()
    return out


def edit_counts(ref, hyp):
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1))
    return D[n][m], n


def main():
    shard_dirs = sys.argv[1:]
    # --- merged WER ---
    tot_err = tot_wrd = nutt = 0
    for d in shard_dirs:
        hp = glob.glob(f"{d}/**/score_wer/hyp.trn", recursive=True)
        rp = glob.glob(f"{d}/**/score_wer/ref.trn", recursive=True)
        if not hp or not rp:
            continue
        H, R = parse_trn(hp[0]), parse_trn(rp[0])
        for uid in R:
            if uid in H:
                e, w = edit_counts(R[uid], H[uid])
                tot_err += e; tot_wrd += w; nutt += 1
    wer = 100.0 * tot_err / max(1, tot_wrd)
    # --- pooled latency ---
    gt = P.load_gt()
    dep, prc = [], []
    for d in shard_dirs:
        lg = glob.glob(f"{d}/**/asr_inference.1.log", recursive=True)
        ky = glob.glob(f"{d}/**/keys.1.scp", recursive=True)
        if not lg or not ky:
            continue
        tr = P.parse_traces(lg[0], ky[0])
        for u, t in tr.items():
            if u not in gt:
                continue
            w = P.utt_rows(t)
            hyp = [x for x, _, _, _ in w]
            ref = [x for x, _ in gt[u]]
            tau = [e for _, e in gt[u]]
            mp = P.amap(hyp, ref)
            for hi, (ww, cc, Dc, dd) in enumerate(w):
                if hi in mp:
                    g = tau[mp[hi]] * 1000.0
                    dep.append((cc + Dc + 1) * P.CHUNK_MS - g)
                    prc.append((cc + dd + 1) * P.CHUNK_MS - g)

    def s(x):
        return f"{st.mean(x):.0f}/{P.pct(x,50):.0f}/{P.pct(x,90):.0f}" if x else "NA"
    print(f"WER={wer:.2f}%  utts={nutt}  Nwrd={tot_wrd}  |  "
          f"latency_words={len(dep)}  dep(mean/med/p90)={s(dep)}  prc={s(prc)}")


if __name__ == "__main__":
    main()
