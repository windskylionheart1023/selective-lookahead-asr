# Streaming latency: as-deployed vs achievable per-token

Two per-word latency metrics for the streaming SOFT model with the Branch-D wait-policy, measured against ground-truth word-end times. We keep exactly these two:

1. **As-deployed** — the latency the running code actually delivers.
2. **Achievable per-token** — the latency a commit-stable-prefix Branch-D *could* deliver (each token committed when its own value settles).

**Config:** model `asr_asr_train_SOFT_align_finetune_FROM_XCQDCCONV_R4MAX_BS32`, ckpt `valid.acc.ave_10best.pth`, cs16, R4MAX, Policy 3 on (`ROLLBACK=1.0`), Branch-D (`token_chunk_resume`, MFC=4), wait-policy `wait_policy_dev_lib360_P3/wait_policy_A.pt`. Eval: `small_test_lib360` (95 utts, 1807 reference words).

---

## 1. Shared quantities

**Chunk duration.** Frontend hop = 128 samples @ 16 kHz = 8 ms; encoder input_layer = `conv2d6` (÷6 subsampling) → 48 ms/encoder-frame; chunk = 16 frames →

```
T_chunk = 16 × 48 ms = 768 ms
```

**Ground truth.** MFA **word-level** forced alignments (LibriSpeech test-clean), `…/LibriSpeech-Alignments/…/<spk>-<chapter>.alignment.txt`, format `utt_id ",W1,W2,…" "end1,end2,…"` (end times in seconds; empty word = silence). Covers 97/100 small_test utts.

**Per-word latency (token-exact; both metrics share this).** The deferral acts on BPE **tokens**, but the gt is **word-level**. A word = consecutive ▁-tokens, and it is finalised only when its **slowest** token settles, so we take the max over the word's tokens:

```
latency(word) = ( max over the word's tokens t of [ chunk_t + depth_t ] + 1 ) × T_chunk  −  gt_end(word)
```
- `chunk_t` = the per-token commit chunk at nrc0 (from the nrc0 feature dump, mapped onto the token sequence). Acoustic, config-independent.
- `depth_t` = future chunks credited to token t — **the only thing the two metrics differ on.**
- `+1` = the chunk is processed only when it completes.
- `gt_end(word)` = MFA word-end time in ms.

This `max_t(chunk_t + depth_t)` is exact for words that straddle a chunk boundary (the deepest-settling token may sit in an earlier chunk than the word's last token). For single-chunk words it reduces to `(chunk + d_word)`, where `d_word` = the word's settle depth = max over its tokens.

**Reference word set** = words correct at nrc0 ∩ present in the MFA gt (1807 words). Latency is averaged over these; 95% CIs are bootstrap over words (2000 resamples).

**Base lag.** With `d_token = 0`, `latency = (natural_chunk+1)×768 − gt_end` = the nrc0 emission lag, measured **228 ms** (mean over the 1807 words). This is the model's intrinsic delay from a word's audio to its nrc0 commit.

**Floor property.** `d_token(w) ≥ 0` always — deferral only *adds* future chunks, never commits a token before its natural chunk. Therefore **every dynamic latency ≥ 228 ms (nrc0)**. The wait-policy can match nrc0's latency (by never deferring) but never beat it; its value is *better WER at ≈ nrc0 latency*, not lower latency.

---

## 2. Metric A — As-deployed latency

**Definition.** When a chunk defers, Branch-D rolls back the *whole* chunk (`restore_state(snap)`, snapshot taken before the chunk) and re-decodes it deeper, holding everything back until it commits at its deepest level. So **every token in a chunk is emitted together at the chunk's maximum defer depth**:

```
d_token(w) = D_c        for EVERY token in chunk c
```
where `D_c` = the maximum defer depth chunk c reached.

**How to compute `D_c`.** Parse the decode log (`…/logdir/asr_inference.1.log`):
1. Delimit utterances by `best hypo:` lines; utterance order = `…/logdir/keys.1.scp`.
2. Within each utterance, for every `[dyn_fc Branch B] chunk=N … defer_count=M` and `… k=X->Y` line, set `D[utt][N] = max(seen, M or Y)`.
3. `D_c = D[utt][natural_chunk(w)]` (0 if the chunk never deferred).

This is exact — it reads the chunk depths the running code actually used.

---

## 3. Metric B — Achievable per-token latency

**Definition.** Idealized commit-stable-prefix Branch-D: commit each token as soon as *its own* value settles, instead of dragging the whole chunk to the max depth. Within chunk c, the **common beginning** before the first change keeps `d=0`; each later token gets the depth at which **it** last changed (capped at `D_c`):

```
d_token(w) = max{ k ∈ [1 .. D_c] : firstdiff_c[k] ≤ pos(w) }   (0 if none)
```
- `firstdiff_c[k]` = the first word position in chunk c where the depth-(k−1) and depth-k re-decodes differ.
- `pos(w)` = w's position; the condition means "a change at depth k occurred at or before w in its chunk" → w is in the depth-k changed region.

**Per-depth re-decodes = the static nrc0–4 ladder.** Branch-D's re-decode of chunk c at depth k is byte-identical to static `nrc=k` for that chunk (validated), so we use the static-ladder hypotheses (`soft_cs16r{0..4}_P3`) as the depth-k decodes.

**How to compute:**
1. Align each static nrc=k hypothesis to the gt words → `arr_k[j]` = depth-k word at gt position j.
2. For each chunk c (gt positions with `natural_chunk == c`) and each depth k=1..D_c: `firstdiff_c[k]` = first j in c with `arr_{k-1}[j] ≠ arr_k[j]`.
3. `d_token(w) = max{ k ≤ D_c : firstdiff_c[k] ≤ pos(w) }`.

**Worked example** (one chunk, `D_c = 2`):
```
depth0 nrc0:  D E F
depth1 nrc1:  Z E F      (D→Z at depth 1)
depth2 nrc2:  Z Y X      (E→Y, F→X at depth 2)
as-deployed:  D,E,F  →  2,2,2     (whole chunk at the max depth)
per-token  :  Z→1, Y→2, X→2       (Z settled at depth 1, keeps 1)
```

---

## 4. Results (small_test, 95 utts)

Token-exact (`max_t(chunk_t + depth_t)`):

| wait-policy thr | **as-deployed (ms)** | **achievable per-token (ms)** [95% CI] | WER % |
|---|---|---|---|
| 0.50 | 1107 | 279 [259,300] | 7.1 |
| 0.65 | 953 | 278 [258,299] | 7.1 |
| 0.70 | 898 | 277 [257,298] | 7.2 |
| **0.81 (knee)** | **710** | **274 [254,295]** | 7.1 |
| 0.85 | 564 | 267 [248,287] | 7.6 |
| 0.88 | 516 | 262 [244,282] | 7.5 |
| 0.92 | 277 | 244 [227,262] | 8.6 |
| 0.96 | 229 | 229 | 8.8 |
| 0.99 | 229 | 229 | 8.3 |

**Anchors (per-word latency):** nrc0 = 229 ms / 8.4% (the floor) · static nrc2 = 1764 ms / 7.2% · static nrc4 = 3300 ms / 7.2% · offline = 4323 ms / 5.5%.

**Reading it (knee, t0.81, 7.1% WER):**
- **as-deployed 710 ms** — what the running code delivers.
- **achievable 274 ms** — commit-stable-prefix bound; only **+45 ms over the nrc0 floor (229 ms)** for −1.3 pp WER vs nrc0.
- The **710 → 274 ms gap (~435 ms)** is the latency Branch-D wastes by emitting whole chunks at the max depth instead of committing each token at its settle depth — a concrete improvement target.
- Both are floored at nrc0; the achievable curve is a near-vertical drop just right of nrc0 (it buys nrc2-level WER at ≈ nrc0 latency, not lower latency).
- (Word-level approximation `(last_token_chunk + d_word)` gave 706 / 283; the token-exact `max_t` corrects straddling words → 710 / 274.)

---

## 5. Scripts

- `compute_emission_latency.py`: per-word emission latency from a decode's 1-best `latency` file against `ref_word_endtimes_testfull.json` (Table V static rows).
- `compute_emission_latency_csp.py`: the same for dynamic decodes, pooled over all shards; prints `dep` (whole-chunk, the rollback rows) and `prc` (stable prefix) (Table V dynamic rows).
- `analyze_csp_shards.py` with `per_word_latency_real.py`: trace-based variant against the public LibriSpeech alignments (`LIBRISPEECH_ALIGNMENTS`); gives lower absolute values.

## 6. Values printed in the paper

Some printed latencies (mean/median/p90, ms) differ from what these scripts give; no conclusion changes.
- Table V static rows were read from the 3rd-best hypothesis. With the 1-best: R=0 233/234/686, R=1 816/862/1290, R=4 2483/2908/3508 (printed 240/238/696, 813/860/1298, 2490/2910/3518). The static cells of Tables IV and VI and the points of Fig. 5 shift by at most 16 ms for the same reason.
- Table V Random and Oracle rows were measured with `analyze_csp_shards.py`. With `compute_emission_latency_csp.py`, like the other dynamic rows: Random 1505/1294/3244 (printed 1410/1078/3248), Oracle 706/316/3050 (printed 696/292/3072).
