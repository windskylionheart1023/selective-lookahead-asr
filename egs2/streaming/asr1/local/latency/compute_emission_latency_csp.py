"""CSP emission latency, MEASURED THE SAME WAY as compute_emission_latency.py
(token_chunk_map emission chunk + testfull REF word-ends + Levenshtein word
align) so Dynamic-CSP rows are directly comparable to the Static rows in the
paper's Table V. The ONLY change vs the whole-chunk number: the per-word
lookahead offset is the CSP per-word SETTLE DEPTH (from the [redecode_trace]),
not the whole-chunk chunk_future_map depth.

Pools N shards. Prints WER, and BOTH dep (whole-chunk) and prc (CSP) mean/p50/p90.

Usage: python compute_emission_latency_csp.py <shard_dir1> [shard_dir2 ...]
"""
import json, glob, sys, os, statistics as st
import compute_emission_latency as CE     # hyp_word_emissions, correct_pairs, REF, pct
import per_word_latency_real as P          # parse_traces, utt_rows

CHUNK_MS = float(os.environ.get("CHUNK_MS", 768))
P.CHUNK_MS = CHUNK_MS


def settle_depth_by_utt(cell):
    """utt -> [(word, settle_depth d, whole-chunk Dc)] in hyp order, from traces."""
    lg = glob.glob(f"{cell}/**/asr_inference.1.log", recursive=True)
    ky = glob.glob(f"{cell}/**/keys.1.scp", recursive=True)
    if not lg or not ky:
        return {}
    tr = P.parse_traces(lg[0], ky[0])
    out = {}
    for u, t in tr.items():
        out[u] = [(w.upper(), d, Dc) for (w, c, Dc, d) in P.utt_rows(t)]
    return out


def main():
    cells = sys.argv[1:]
    dep, prc = [], []
    tot_err = tot_wrd = 0
    for cell in cells:
        lf = (glob.glob(f"{cell}/**/1best_recog/latency", recursive=True)
              or glob.glob(f"{cell}/**/latency", recursive=True))
        if not lf:
            continue
        we, cfm_all = CE.hyp_word_emissions(lf[0])          # {utt:[(W,emit_chunk)]}
        sd = settle_depth_by_utt(cell)                      # {utt:[(W,d,Dc)]}
        for utt, hw in we.items():
            if utt not in CE.REF:
                continue
            hyp = [w for w, _ in hw]
            # align tcm-words -> trace-words to attach per-word (d, Dc)
            depth_for = {}
            if utt in sd:
                tw = [w for w, _, _ in sd[utt]]
                for ti, tj in CE.correct_pairs(hyp, tw):
                    depth_for[ti] = (sd[utt][tj][1], sd[utt][tj][2])  # (d, Dc)
            ref = [w.upper() for w, _ in CE.REF[utt]]
            tau = [t for _, t in CE.REF[utt]]
            cfm = cfm_all.get(utt, {})
            for hi, ri in CE.correct_pairs(hyp, ref):
                base = hw[hi][1]
                d, Dc = depth_for.get(hi, (cfm.get(base, 0), cfm.get(base, 0)))
                dep.append((base + Dc + 1) * CHUNK_MS - tau[ri])
                prc.append((base + d + 1) * CHUNK_MS - tau[ri])
    # WER via the recipe result.txt (sum over shards, exact)
    for cell in cells:
        for r in glob.glob(f"{cell}/**/score_wer/result.txt", recursive=True):
            for line in open(r):
                if "Sum/Avg" in line:
                    p = line.split("|")
                    w = float(p[2].split()[1]); e = float(p[3].split()[4])
                    tot_wrd += w; tot_err += w * e / 100.0
    wer = 100.0 * tot_err / max(1, tot_wrd)

    def s(x):
        return f"{st.mean(x):.0f}/{CE.pct(x,50):.0f}/{CE.pct(x,90):.0f}" if x else "NA"
    print(f"WER={wer:.2f}%  nWord={len(prc)}  dep(m/p50/p90)={s(dep)}  prc(CSP)={s(prc)}")


if __name__ == "__main__":
    main()
