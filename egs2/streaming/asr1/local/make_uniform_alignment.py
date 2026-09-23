#!/usr/bin/env python3
"""Uniform placeholder token alignment for the validation set.

The early-emission losses read per-token encoder-frame indices (token_frame_mapping).
Training uses the forced alignment shipped in local/alignment/; the validation set
uses this placeholder, as in the paper: each utterance's N tokens are spread evenly
over its encoder frames (768 samples = 48 ms per frame), strictly increasing.

Usage: make_uniform_alignment.py <dump dir, e.g. dump/raw/dev_lib360> <bpe.model>
Writes <dump dir>/token_frame_mapping.
"""
import sys

import sentencepiece as spm

SAMPLES_PER_ENC_FRAME = 128 * 6  # hop length x subsampling factor

dump_dir, bpemodel = sys.argv[1], sys.argv[2]
sp = spm.SentencePieceProcessor(model_file=bpemodel)

n_samples = {}
for line in open(f"{dump_dir}/utt2num_samples"):
    utt, n = line.split()
    n_samples[utt] = int(n)
n_tokens = {}
for line in open(f"{dump_dir}/text"):
    utt, _, text = line.rstrip("\n").partition(" ")
    n_tokens[utt] = len(sp.encode(text, out_type=str))

with open(f"{dump_dir}/token_frame_mapping", "w") as f:
    for utt in sorted(n_samples.keys() & n_tokens.keys()):
        n_tok = n_tokens[utt]
        if n_tok <= 0:
            continue
        n_frames = max(n_samples[utt] // SAMPLES_PER_ENC_FRAME, n_tok + 1)
        indices = []
        for i in range(n_tok):
            idx = max(0, min(n_frames - 1, int((i + 0.5) * n_frames / n_tok)))
            if indices and idx <= indices[-1]:
                idx = indices[-1] + 1
            indices.append(idx)
        f.write(f"{utt} " + " ".join(map(str, indices)) + "\n")
