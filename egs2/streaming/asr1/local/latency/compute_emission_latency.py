"""MFA-grounded emission latency for streaming decodes.

For each correctly-recognized word w:
    latency(w) = (emission_chunk(w)+1)*CHUNK_MS  -  tau(w)
where
  - emission_chunk(w) = chunk index of w's LAST subword token (when the word
    completes), read from the decode's `latency` file token_chunk_map;
  - tau(w) = MFA word-END time (ms) of the matched reference word.
Correct words are found by Levenshtein word-alignment of hyp vs reference.
Reports mean / p50 / p90 latency (ms) over correct words, plus WER.

Usage: python compute_emission_latency.py <cell_dir1> [cell_dir2 ...]
"""
import json, glob, sys, re, os, statistics as st

CHUNK_MS = 768.0  # cs16: 16 enc frames * 48 ms
# Reference = MFA word-end times of LibriSpeech test-clean (ms), shipped next to this
# script; set REF_WORD_ENDTIMES to use another file.
REF = json.load(open(os.environ.get(
    "REF_WORD_ENDTIMES",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_word_endtimes_testfull.json"))))


def hyp_word_emissions(latency_file):
    """utt -> [(WORD, emission_chunk_of_last_token), ...] in hypothesis order,
    and utt -> chunk_future_map {chunk: future_chunks_deferred}."""
    res, cfm_all = {}, {}
    for line in open(latency_file):
        utt, js = line.split(" ", 1)
        obj = json.loads(js)
        tcm = obj["token_chunk_map"]  # {idx: [token, chunk]}
        cfm_all[utt] = {int(k): int(v)
                        for k, v in obj.get("chunk_future_map", {}).items()}
        words, cur, ch = [], None, None
        for i in sorted(tcm, key=int):
            tok, c = tcm[i]
            if tok.startswith("▁"):          # word start
                if cur is not None:
                    words.append((cur, ch))
                cur, ch = tok[1:], c
            else:
                cur = (cur or "") + tok
                ch = c                             # last token's chunk
        if cur is not None:
            words.append((cur, ch))
        res[utt] = [(w.upper(), c) for w, c in words]
    return res, cfm_all


def correct_pairs(hyp, ref):
    """Levenshtein word-align; return matched (hyp_idx, ref_idx) for equal words."""
    n, m = len(hyp), len(ref)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if hyp[i - 1] == ref[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + c)
    i, j, out = n, m, []
    while i > 0 and j > 0:
        if hyp[i - 1] == ref[j - 1] and D[i][j] == D[i - 1][j - 1]:
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif D[i][j] == D[i - 1][j - 1] + 1:
            i -= 1; j -= 1
        elif D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def wer_of(cell):
    for r in glob.glob(f"{cell}/**/score_wer/result.txt", recursive=True):
        for line in open(r):
            if "Sum/Avg" in line:
                return float(line.split("|")[3].split()[4])
    return None


def pct(xs, p):
    xs = sorted(xs); k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


print(f"{'cell':<34} {'WER':>5} {'totLat':>7} {'medLat':>7} {'p90':>7} {'p95':>7} {'commit%':>8} {'nWord':>6}")
for cell in sys.argv[1:]:
    # 1-best only: decode dirs also hold 2best/3best_recog/latency, and glob order is arbitrary
    lf = (glob.glob(f"{cell}/**/1best_recog/latency", recursive=True)
          or glob.glob(f"{cell}/**/latency", recursive=True))
    if not lf:
        print(f"{cell.split('/')[-1]:<34}  no latency file"); continue
    # How many future chunks each token actually consumed (NOT reflected in
    # its recorded base/emission chunk):
    #  - static nrc=k: a constant k per token.
    #  - DFC: the per-chunk defer depth from chunk_future_map (the honest,
    #    variable lookahead). Falls back to 0 when the recording is absent
    #    (older decodes) -> undercounts deferral, as before.
    m = re.search(r"static_nrc(\d+)", cell)
    static_offset = int(m.group(1)) if m else None
    cs_m = re.search(r"_cs(\d+)", cell)            # cs20/cs24 chunks are wider
    chunk_ms = int(cs_m.group(1)) * 48 if cs_m else CHUNK_MS
    we, cfm_all = hyp_word_emissions(lf[0])
    lats = []      # total emission latency
    futs = []      # future-chunk latency only (chunking term removed)
    offs = []      # per-token deferral depth (0 = immediate commit)
    for utt, hw in we.items():
        if utt not in REF: continue
        hyp = [w for w, _ in hw]
        ref = [w.upper() for w, _ in REF[utt]]
        tau = [t for _, t in REF[utt]]
        cfm = cfm_all.get(utt, {})
        for hi, ri in correct_pairs(hyp, ref):
            base_chunk = hw[hi][1]
            tok_offset = (static_offset if static_offset is not None
                          else cfm.get(base_chunk, 0))    # per-token lookahead
            consumed_chunk = base_chunk + tok_offset      # audio chunks consumed
            lats.append((consumed_chunk + 1) * chunk_ms - tau[ri])
            acoustic_chunk = int(tau[ri] // chunk_ms)      # word's own chunk c(w)
            futs.append((consumed_chunk - acoustic_chunk) * chunk_ms)
            offs.append(tok_offset)
    if not lats:
        print(f"{cell.split('/')[-1]:<34}  no matched words"); continue
    name = cell.rstrip('/').split('/')[-1]
    print(f"{name:<34} {wer_of(cell)!s:>5} {st.mean(lats):>7.0f} "
          f"{st.median(lats):>7.0f} {pct(lats,90):>7.0f} {pct(lats,95):>7.0f} "
          f"{100*sum(1 for o in offs if o==0)/len(offs):>8.1f} {len(lats):>6}")
