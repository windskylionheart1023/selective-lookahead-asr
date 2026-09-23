"""Tests for the dynamic chunked convolution (DCConv) in ConvolutionModule.

Covers the heterogeneous chunk-size fallback of forward()'s 4-D path: when
chunk sizes differ across batch items, DCConv runs per sample and the outputs
must match the vectorized single-sample path, with invalid c_q slots passing
the input through unchanged.
"""

import torch

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.nets_utils import ChunkedMaskConfig


def _make_module(channels=8, kernel_size=5):
    torch.manual_seed(0)
    module = ConvolutionModule(channels, kernel_size)
    module.eval()  # freeze BatchNorm running stats: batched == per-sample
    return module


@torch.no_grad()
def test_heterogeneous_chunk_size_fallback_matches_per_sample():
    """Mixed chunk sizes in one batch: fallback == per-sample vectorized path."""
    module = _make_module()
    chunk_sizes = [4, 8]
    nrcs = [1, 2]  # per-sample num_right_chunks -> different valid c_q counts
    B, T, D = 2, 16, 8
    Cq = max(nrcs) + 1  # 3 slots; sample 0 has one invalid slot (c_q=2)
    torch.manual_seed(1)
    x = torch.randn(B, T, Cq, D)

    cfg = ChunkedMaskConfig(
        chunk_size=chunk_sizes,
        num_left_chunks=[-1, -1],
        num_right_chunks=nrcs,
        use_asymmetric_mask=[True, True],
        full_attention=[False, False],
    )
    y = module(x, chunked_mask_config=cfg)
    assert y.shape == (B, T, Cq, D)

    for b in range(B):
        cfg_b = ChunkedMaskConfig(
            chunk_size=chunk_sizes[b],
            num_left_chunks=-1,
            num_right_chunks=nrcs[b],
            use_asymmetric_mask=True,
            full_attention=False,
        )
        y_b = module(x[b : b + 1], chunked_mask_config=cfg_b)
        torch.testing.assert_close(y[b : b + 1], y_b)

    # The invalid c_q slot of sample 0 (c_q=2 > nrc=1) passes through the input.
    assert torch.equal(y[0, :, 2, :], x[0, :, 2, :])
    # Valid slots are actually convolved (not pass-through).
    assert not torch.equal(y[0, :, 0, :], x[0, :, 0, :])


@torch.no_grad()
def test_uniform_chunk_size_batch_matches_per_sample():
    """Uniform chunk sizes (vectorized branch) also match per-sample decode."""
    module = _make_module()
    B, T, D = 2, 16, 8
    nrc = 1
    Cq = nrc + 1
    torch.manual_seed(2)
    x = torch.randn(B, T, Cq, D)
    cfg = ChunkedMaskConfig(
        chunk_size=4,
        num_left_chunks=-1,
        num_right_chunks=nrc,
        use_asymmetric_mask=True,
        full_attention=False,
    )
    y = module(x, chunked_mask_config=cfg)
    for b in range(B):
        y_b = module(x[b : b + 1], chunked_mask_config=cfg)
        torch.testing.assert_close(y[b : b + 1], y_b)
