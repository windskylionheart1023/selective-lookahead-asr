# Streaming ASR with asymmetric chunked attention and dynamic future chunks

Streaming Conformer ASR recipe (LibriSpeech train-clean-360). The model is
trained with asymmetric chunked self-attention masks so that every layer sees
a bounded number of future chunks, and decoded with a chunked streaming
simulator that can defer uncertain tokens and re-decode them with more
lookahead (dynamic future chunks), optionally freezing the stable prefix
between passes (commit-stable-prefix). This is the recipe of the SLT 2026 paper
*Selective Lookahead for Attention-Based Streaming ASR* (Jia, Tamm, Van hamme); the
paper's terms map to the code as: bounded-lookahead chunk encoder = asymmetric chunk
mask + XCQ convolution; dynamic future-chunk decoding = DFC with strategy D
(`--dynamic_mode token_chunk_resume`); learned trigger = wait-policy probe.

## Method overview

- **Asymmetric chunk mask**: chunked self-attention with unlimited left
  context and a bounded, per-sample future window; the future receptive
  field stays depth-bounded so training matches streaming inference.
- **Causal / XCQ depthwise convolution**: the Conformer conv module is
  chunk-causal, with an optional per-query-slot bounded future window (XCQ)
  that mirrors the attention mask.
- **RoPE** in encoder self-attention and decoder cross-attention.
- **Alignment-supervised early emission (SOFT)**: per-token forced alignments
  give each token a soft deadline chunk; cross-attention and CTC mass after the
  deadline are penalized during fine-tuning.
- **Dynamic future chunks (DFC)**: at inference, a per-token trigger decides
  commit vs defer. A deferred chunk is re-decoded with one more future chunk,
  up to a budget (`--max_future_chunks`). Triggers: top-1 confidence
  (optionally temperature-scaled), learned wait-policy probe, SR-CEM calibrator.
- **Commit-stable-prefix** (`--commit_stable_prefix`): on each deferred
  re-decode, the longest prefix agreeing with the previous pass is frozen and
  only the divergent suffix is re-decoded, cutting tail latency.
- **Last-chunk handling**: decode-once with an offline-style final pass
  seeded with the full beam from the pre-final snapshot.
- **KV caches**: decoder self- and cross-attention caches are lossless and
  enabled by default; the encoder self-attention cache is structural
  (not bit-identical) and must stay off when DFC re-encodes chunks.

## Layout

Recipe entry points:

| File | Purpose |
| --- | --- |
| `run.sh` | Canonical driver: data, offline base training, streaming fine-tune, streaming decode |
| `asr.sh` | Standard ESPnet2 ASR pipeline (training stages) |
| `asr_streaming_sim.sh` | ASR pipeline fork whose stage 12 decodes with the chunked streaming simulator |
| `local/data.sh` | LibriSpeech preparation (train_lib360 / dev_lib360 / test_lib360) |
| `conf/train_asr_conformer_offline_rope_newfrontend.yaml` | Offline RoPE Conformer base model |
| `conf/train_asr_SOFT_align_finetune_FROM_XCQDCCONV_R4MAX_BS32.yaml` | Streaming fine-tune (asymmetric mask + XCQ conv + SOFT early emission): the paper's model |
| `local/triggers/` | Trigger checkpoints used in the paper (`wait_policy_A.pt`, `sr_cem_A_stdz.pt`) and the wait-policy dataset builder and trainer |
| `local/alignment/`, `local/bpe_unigram5000/`, `local/make_*.py` | The paper's training alignment and BPE model, the fine-tune init and the dev placeholder alignment (all applied by `run.sh`) |
| `local/latency/` | Per-word latency measurement against MFA word ends (see below) |
| `conf/decode_asr_bs20.yaml` | Beam-20 joint CTC/attention decoding |

Main code (relative to the repo root):

- `espnet2/bin/asr_inference_streaming_modified.py`: chunked streaming
  inference driver (`Speech2TextStreamingChunked`), DFC branches, triggers,
  commit-stable-prefix, KV-cache plumbing, last-chunk handling.
- `espnet2/asr_stream/`: streaming task model (`espnet_model.py`, dynamic
  chunk training and early-emission losses), `dynamic_future_chunks.py`
  (defer/commit controller), `sr_cem.py` (calibrated stop probe),
  `wait_aux.py` (learned wait-policy head), `temperature_scaling.py`
  (trigger calibration), `chunk_tracker.py`, `ctc_disagree.py` (ablation).
- `espnet/nets/batch_beam_search.py` and `batch_beam_search_online.py`:
  chunk-aware beam search with emission tracking, repetition backstops,
  defer signaling, forced-prefix decoding, seed-full-beam initialization.
- `espnet/nets/pytorch_backend/transformer/attention.py`,
  `conformer/convolution.py`, `nets_utils.py`: RoPE attention with KV
  caches, chunk-causal DCConv, batched chunked-mask builders.

## Usage

```bash
# 1. Data and features (installs the paper's BPE model)
./run.sh --run_data true

# 2. Offline base model
./run.sh --run_offline true

# 3. Streaming fine-tune from step 2 (offline weights except the decoder token
#    embedding, which is re-learned as in the paper)
./run.sh --run_streaming true

# 4a. Static streaming decode with 1 future chunk of lookahead
./run.sh --run_decode true --decode_mode static --nrc 1

# 4b. Dynamic future chunks: learned trigger, commit-stable-prefix
./run.sh --run_decode true --decode_mode dynamic --threshold 0.70
```

WER tables are generated from the sclite score files under each decode directory
(`exp/<model>/<tag>/<test_set>/score_wer/result.txt`).

## Reproducing the paper's rows

All rows use `asr_streaming_sim.sh` (stage 12) with `--stream_chunk_size 16
--stream_num_left_chunks -1 --use_asymmetric_mask_at_inference false` and the
`--inference_args` below (beam 20, `--attn_edge_stop_margin 2 --unmask_xattn_at_final true`).

| Row | `--stream_num_right_chunks` | extra `--inference_args` |
| --- | --- | --- |
| Static R | R | `--cross_attn_num_right_chunks R` |
| Dynamic, learned trigger | 4 | `--dynamic_future_chunks true --dynamic_mode token_chunk_resume --dynamic_signal_type wait_policy --sr_cem_ckpt local/triggers/wait_policy_A.pt --sr_cem_variant A --sr_cem_threshold 0.70 --max_future_chunks 4 --cross_attn_num_right_chunks 4 --commit_stable_prefix true` |
| Dynamic, SR-CEM trigger | 4 | as above with `--dynamic_signal_type sr_cem_causal --sr_cem_ckpt local/triggers/sr_cem_A_stdz.pt --sr_cem_threshold 0.85` |
| Dynamic, raw top-1 | 4 | as above with `--dynamic_signal_type top1_prob --dynamic_top1_prob_threshold 0.95` |
| Wait budget B | 4 | `--max_future_chunks B` |
| Random trigger (control) | 4 | `--dynamic_signal_type fake_random --dynamic_fake_random_prob 0.34` |
| Oracle trigger (control) | 4 | `--dynamic_mode chunk_rollback --dynamic_signal_type fake_random --dynamic_fake_random_prob 0.0 --max_future_chunks 4 --cross_attn_num_right_chunks 4` with env `ORACLE_REF_FILE=<kaldi text of the test set>`: defers exactly the chunks whose decode is wrong, whole-chunk commit (report the `dep` latency) |

The wait-policy probe is trained on paired static decodes of dev-clean at adjacent
lookaheads: `local/triggers/build_wait_policy_dataset.py` then
`local/triggers/train_wait_policy.py` (see their docstrings).

## Latency

`local/latency/` measures per-word emission latency against Montreal Forced Aligner
word ends. Table V: `compute_emission_latency.py` (static rows) and
`compute_emission_latency_csp.py` (dynamic rows, all shards: `dep` = rollback, `prc` =
stable prefix), on the test-clean reference `ref_word_endtimes_testfull.json`;
`analyze_csp_shards.py` is a trace-based variant on the public alignments
(`LIBRISPEECH_ALIGNMENTS`). `latency_methodology.md` explains the metric and lists
printed values that differ from these scripts. Keep the decoder's trace logging on (default).

On shared storage that drops out, `--stage_data_to_scratch true` (in `run.sh` or
`asr_streaming_sim.sh`) copies model, config and audio to node-local scratch first.

## Caveats

- The encoder self-attention KV cache changes results slightly (cuBLAS
  kernel dispatch differs between cached and uncached shapes); keep it off
  for exact reproducibility and always off with DFC.
- Long-form audio needs a bounded left context (about 8 chunks); unlimited
  left context degrades long recordings.
- Hand-merged decode dumps must be globally sorted before scoring, or
  hypothesis/reference pairing desyncs; the in-recipe scoring path already
  sorts.
