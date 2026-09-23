#!/usr/bin/env python3
"""Build the streaming fine-tune initialization from the offline checkpoint.

Reproduces the paper's initialization: every offline weight is loaded except the
decoder token embedding, which the streaming fine-tune learns from scratch. In the
paper's run the embedding was stored under a key the streaming model does not have,
so ESPnet's --ignore_init_mismatch dropped it; deleting it here has the same effect.

Usage: make_finetune_init.py <offline valid.acc.ave_10best.pth> <output.pth>
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
state = torch.load(src, map_location="cpu")
del state["decoder.embed.0.weight"]
torch.save(state, dst)
print(f"Wrote {dst}: {len(state)} tensors (decoder token embedding re-initialized)")
