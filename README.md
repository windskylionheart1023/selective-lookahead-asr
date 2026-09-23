# Selective Lookahead for Attention-Based Streaming ASR

Code and recipe for

> Y. Jia, B. Tamm, and H. Van hamme, "Selective Lookahead for Attention-Based Streaming ASR,"
> *Proc. IEEE Spoken Language Technology Workshop (SLT)*, 2026.

Two mechanisms for streaming joint CTC/attention ASR: a **bounded-lookahead chunk encoder**
(one age-selection rule caps every chunk's future receptive field at a constant number of
chunks, independent of depth, for self-attention and the depthwise convolution) and
**dynamic future-chunk decoding** (a per-token trigger commits a token or waits and
re-decodes it under one more chunk; a learned trigger, with a confidence threshold as
fallback). On LibriSpeech test-clean the dynamic system matches the best static-lookahead
WER (6.5%) at a median per-word latency of 306 ms versus 860 ms for one-chunk static lookahead.

This is a fork of [ESPnet](https://github.com/espnet/espnet) (base commit
[`412aa11`](https://github.com/espnet/espnet/tree/412aa11f77ceeee405f75ee21aa5b33cad438fba)).
The first commit holds the unmodified upstream files and the second every change for
the paper. Everything specific to the paper lives in:

| Path | Contents |
| --- | --- |
| [`egs2/streaming/asr1/`](egs2/streaming/asr1/README.md) | Recipe: data, training configs, streaming decode, trigger checkpoints, latency tools, how to reproduce each table row |
| `espnet2/asr_stream/` | Streaming task model (dynamic chunked-mask training, early-emission losses), defer/commit controller, triggers |
| `espnet2/bin/asr_inference_streaming_modified.py` | Chunked streaming decoder: static lookahead, dynamic future chunks, commit-stable-prefix |
| `espnet/nets/pytorch_backend/{transformer/attention.py, conformer/convolution.py, nets_utils.py}` | Age-versioned asymmetric chunk mask, cross-age depthwise convolution, mask builders |
| `espnet/nets/batch_beam_search*.py` | Chunk-aware beam search with snapshot/restore and forced-prefix decoding |

Start with the [recipe README](egs2/streaming/asr1/README.md). Licensed under Apache-2.0
(see `LICENSE`), as ESPnet.

## Install

Same as ESPnet: create the environment (PyTorch 2.x, CUDA 12), then `pip install -e .`
from the repo root. Decoding needs `flex_attention`-capable PyTorch only when
`use_flex_attention` is enabled in a config (off by default). The recipe follows the
standard `egs2/*/asr1` layout (`run.sh`, `asr.sh`, `path.sh`, `db.sh`); scoring needs
sclite (`cd tools && ./installers/install_sctk.sh`).

```bibtex
@inproceedings{jia2026selective,
  title     = {Selective Lookahead for Attention-Based Streaming {ASR}},
  author    = {Jia, Yichen and Tamm, Bastiaan and Van hamme, Hugo},
  booktitle = {Proc. IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026}
}
```
