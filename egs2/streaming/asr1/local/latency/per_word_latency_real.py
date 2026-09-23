"""MEASURED user-perceived latency from REAL Branch-D re-decode traces.

Replaces the static-nrc0..4 ladder of per_word_latency_exact.py with the actual
per-chunk per-depth hypotheses the running wait-policy decoder produced (the
`[redecode_trace] chunk=C depth=K toks=...` log lines added to
asr_inference_streaming_modified.py). Same metric:

  latency(word) = (max_token[chunk + depth] + 1)*CHUNK_MS - gt_end
  - as-deployed  : depth = D_c (whole chunk committed at its max depth)
  - user-perceived: depth = per-token settle depth (firstdiff within the chunk,
                    common-prefix rule) — a token unchanged by deferral keeps
                    depth 0 (it was already shown correctly).

Difference vs the proxy: the depth-k hypothesis here is the chunk's REAL re-decode
under the actual streaming prefix, not a static nrc=k whole-utterance decode.

Convention note: the per-token depth is the first trace depth at which the token's
final value appears. For token_chunk_resume (strategy D) each trace lists only the
tokens committed at that pass, so this IS the commit depth. For chunk_rollback the
whole provisional chunk is traced at depth 0, so report the whole-chunk (dep) latency.
Usage: python per_word_latency_real.py <decode_dir1> [<decode_dir2> ...]
  decode_dir = .../soft_dfc_waitD_t<THR>_b4_trace_95utt/small_test_lib360
"""
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

CHUNK_MS = float(os.environ.get("CHUNK_MS", 768))  # cs16=768, cs20=960, cs24=1152
# Public LibriSpeech MFA alignments (Lugosch et al.): directory containing <spk>/<chap>/*.alignment.txt
AL = os.environ.get("LIBRISPEECH_ALIGNMENTS", "LibriSpeech-Alignments/LibriSpeech/test-clean")


def load_gt():
    """utt -> [(WORD, end_sec), ...] from MFA alignments (space-split form,
    matching per_word_latency_exact.py)."""
    gt = {}
    for f in glob.glob(f"{AL}/*/*/*.alignment.txt"):
        for line in open(f):
            p = line.strip().split(" ")
            if len(p) < 3:
                continue
            gt[p[0]] = [
                (w.upper(), float(e))
                for w, e in zip(p[1].strip('"').split(","),
                                p[2].strip('"').split(","))
                if w
            ]
    return gt


def amap(a, b):
    """edit-distance align: a-index -> b-index for exact/sub matches."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + c)
    i, j, mp = n, m, {}
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1] and D[i][j] == D[i - 1][j - 1]:
            mp[i - 1] = j - 1; i -= 1; j -= 1
        elif D[i][j] == D[i - 1][j - 1] + 1:
            i -= 1; j -= 1
        elif D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return mp


def pct(xs, p):
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def parse_traces(logf, keysf):
    """utt -> {chunk: {depth: [tokens]}}, in utterance order from keys.scp."""
    utts = [l.split()[0] for l in open(keysf)]
    out, cur, ui = {}, defaultdict(dict), 0
    rx = re.compile(r"\[redecode_trace\] chunk=(\d+) depth=(\d+) toks=(.*)")
    for ln in open(logf):
        m = rx.search(ln)
        if m:
            c, k = int(m.group(1)), int(m.group(2))
            cur[c][k] = m.group(3).split()
        elif "best hypo:" in ln:
            if ui < len(utts):
                out[utts[ui]] = cur
            cur, ui = defaultdict(dict), ui + 1
    return out


def utt_rows(tr):
    """[(word, emission_chunk, D_c, d_perceived)] for one utt's traces."""
    chunks = sorted(tr.keys())
    # per-chunk final tokens (max-depth re-decode) + firstdiff per depth
    rows = []  # per-token: (token, chunk, D_c, d_perc)
    for c in chunks:
        depths = sorted(tr[c].keys())
        Dc = max(depths)
        final = tr[c][Dc]
        fd = {}  # depth k -> first position (in final) where depth k-1 vs k differ
        for k in range(1, Dc + 1):
            a, b = tr[c].get(k - 1, []), tr[c].get(k, [])
            ma, mb = amap(final, a), amap(final, b)
            for p in range(len(final)):
                av = a[ma[p]] if p in ma else None
                bv = b[mb[p]] if p in mb else None
                if av != bv:
                    fd[k] = p
                    break
        for p, t in enumerate(final):
            d = 0
            for k in range(1, Dc + 1):
                if k in fd and fd[k] <= p:
                    d = k
            rows.append((t, c, Dc, d))
    # group tokens into words (leading "_"); word finalizes at slowest token
    words = []
    cur_w, cur_c, cur_dep, cur_Dc = None, None, 0, 0
    for t, c, Dc, d in rows:
        if t.startswith("▁"):
            if cur_w is not None:
                words.append((cur_w.upper(), cur_c, cur_Dc, cur_dep))
            cur_w, cur_c, cur_dep, cur_Dc = t[1:], c, d, Dc
        else:
            cur_w = (cur_w or "") + t
            cur_c, cur_dep, cur_Dc = c, max(cur_dep, d), max(cur_Dc, Dc)
    if cur_w is not None:
        words.append((cur_w.upper(), cur_c, cur_Dc, cur_dep))
    return words


def main():
    gt = load_gt()
    print(f"{'cell':<30}{'dep_mn':>7}{'dep_p90':>8}{'dep_p95':>8}{'prc_mn':>7}{'prc_p90':>8}{'prc_p95':>8}{'nWord':>6}")
    for cell in sys.argv[1:]:
        logf = glob.glob(f"{cell}/**/asr_inference.1.log", recursive=True)
        keysf = glob.glob(f"{cell}/**/keys.1.scp", recursive=True)
        if not logf or not keysf:
            print(f"{cell.split('/')[-2]:<34} no log/keys"); continue
        tr = parse_traces(logf[0], keysf[0])
        dep, perc = [], []
        for u, t in tr.items():
            if u not in gt:
                continue
            words = utt_rows(t)
            hyp = [w for w, _, _, _ in words]
            ref = [w for w, _ in gt[u]]
            tau = [e for _, e in gt[u]]
            mp = amap(hyp, ref)  # hyp-idx -> ref-idx for correct words
            for hi, (w, c, Dc, d) in enumerate(words):
                if hi not in mp:
                    continue
                gte = tau[mp[hi]] * 1000.0
                dep.append((c + Dc + 1) * CHUNK_MS - gte)
                perc.append((c + d + 1) * CHUNK_MS - gte)
        name = cell.rstrip("/").split("/")[-2]
        if not dep:
            print(f"{name:<30} no matched words"); continue
        print(f"{name:<30}{st.mean(dep):>7.0f}{pct(dep,90):>8.0f}{pct(dep,95):>8.0f}"
              f"{st.mean(perc):>7.0f}{pct(perc,90):>8.0f}{pct(perc,95):>8.0f}{len(dep):>6}")


if __name__ == "__main__":
    main()
