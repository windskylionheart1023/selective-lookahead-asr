# -*- coding: utf-8 -*-

"""Network related utility tools."""

from dataclasses import dataclass
import logging
import math
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

import random

def to_device(m, x):
    """Send tensor into the device of the module.

    Args:
        m (torch.nn.Module): Torch module.
        x (Tensor): Torch tensor.

    Returns:
        Tensor: Torch tensor located in the same place as torch module.

    """
    if isinstance(m, torch.nn.Module):
        device = next(m.parameters()).device
    elif isinstance(m, torch.Tensor):
        device = m.device
    else:
        raise TypeError(
            "Expected torch.nn.Module or torch.tensor, " f"bot got: {type(m)}"
        )
    return x.to(device)


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    # If the input dimension is 2 or 3,
    # then we use ESPnet-ONNX based implementation for tracable modeling.
    # otherwise we use the traditional implementation for research use.
    if isinstance(lengths, list):
        logging.warning(
            "Using make_pad_mask with a list of lengths is not tracable. "
            + "If you try to trace this function with type(lengths) == list, "
            + "please change the type of lengths to torch.LongTensor."
        )

    if (
        (xs is None or xs.dim() in (2, 3))
        and length_dim <= 2
        and (not isinstance(lengths, list) and lengths.dim() == 1)
    ):
        return _make_pad_mask_traceable(lengths, xs, length_dim, maxlen)
    else:
        return _make_pad_mask(lengths, xs, length_dim, maxlen)


def _make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None, "When maxlen is specified, xs must not be specified."
        assert maxlen >= int(
            max(lengths)
        ), f"maxlen {maxlen} must be >= max(lengths) {max(lengths)}"

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert (
            xs.size(0) == bs
        ), f"The size of x.size(0) {xs.size(0)} must match the batch size {bs}"

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def _make_pad_mask_traceable(lengths, xs, length_dim, maxlen=None):
    """Make mask tensor containing indices of padded part.

    This is a simplified implementation of make_pad_mask without the xs input
    that supports JIT tracing for applications like exporting models to ONNX.
    Dimension length of xs should be 2 or 3
    This function will create torch.ones(maxlen, maxlen).triu(diagonal=1) and
    select rows to create mask tensor.
    """
    if xs is None:
        device = lengths.device
    else:
        device = xs.device

    if xs is not None and len(xs.shape) == 3:
        if length_dim == 1:
            lengths = lengths.unsqueeze(1).expand(*xs.transpose(1, 2).shape[:2])
        else:
            # Then length_dim is 2 or -1.
            if length_dim not in (-1, 2):
                logging.warning(
                    f"Invalid length_dim {length_dim}."
                    + "We set it to -1, which is the default value."
                )
                length_dim = -1
            lengths = lengths.unsqueeze(1).expand(*xs.shape[:2])

    if maxlen is not None:
        assert xs is None
        assert maxlen >= lengths.max()
    elif xs is not None:
        maxlen = xs.shape[length_dim]
    else:
        maxlen = lengths.max()

    # clip max(length) to maxlen
    lengths = torch.clamp(lengths, max=maxlen).type(torch.long)

    mask = torch.ones(maxlen + 1, maxlen + 1, dtype=torch.bool, device=device)
    mask = triu_onnx(mask)[1:, :-1]  # onnx cannot handle diagonal argument.
    mask = mask[lengths - 1][..., :maxlen]

    if xs is not None and len(xs.shape) == 3 and length_dim == 1:
        return mask.transpose(1, 2)
    else:
        return mask


def triu_onnx(x):
    """Make TriU for ONNX."""
    arange = torch.arange(x.size(0), device=x.device)
    mask = arange.unsqueeze(-1).expand(-1, x.size(0)) <= arange
    return x * mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    return ~make_pad_mask(lengths, xs, length_dim)


def mask_by_length(xs, lengths, fill=0):
    """Mask tensor according to length.

    Args:
        xs (Tensor): Batch of input tensor (B, `*`).
        lengths (LongTensor or List): Batch of lengths (B,).
        fill (int or float): Value to fill masked part.

    Returns:
        Tensor: Batch of masked input tensor (B, `*`).

    Examples:
        >>> x = torch.arange(5).repeat(3, 1) + 1
        >>> x
        tensor([[1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]])
        >>> lengths = [5, 3, 2]
        >>> mask_by_length(x, lengths)
        tensor([[1, 2, 3, 4, 5],
                [1, 2, 3, 0, 0],
                [1, 2, 0, 0, 0]])

    """
    assert xs.size(0) == len(lengths)
    ret = xs.data.new(*xs.size()).fill_(fill)
    for i, l in enumerate(lengths):
        ret[i, :l] = xs[i, :l]
    return ret

def build_chunked_mask(
    T: int,
    chunk_size: torch.Tensor,
    num_left_chunks: torch.Tensor,
    num_right_chunks: torch.Tensor,
    num_global_tokens: torch.Tensor,
    attention_between_global_tokens: torch.Tensor,
    full_attention: torch.Tensor,
    device=None,
) -> torch.Tensor:
    """Create a chunked self-attention mask with support for global tokens (batched).

    The mask allows each rolling query to attend to keys within a fixed number
    of left/right chunks and to all global tokens. Global queries can be restricted
    to only attend to themselves (diagonal) or to all global tokens.

    Args:
        T (int): Total sequence length (global + rolling tokens).
        chunk_size (Tensor[int]): Number of tokens per chunk. Shape: (B,).
            Use chunk_size > 1 for block-wise attention (streaming encoders).
            Use chunk_size = 1 for token-wise attention (streaming AR decoders).
        num_left_chunks (Tensor[int]): Max left (history) chunks visible to each rolling token (-1 = unlimited).
            Shape: (B,).
        num_right_chunks (Tensor[int]): Max right (future) chunks visible to each rolling token
            (-1 = unlimited, 0 = causal). Shape: (B,).
        num_global_tokens (Tensor[int]): Number of global tokens at sequence start (always attendable).
            Note: if this is batch-dependent, then the user is expected to append num_global_tokens.max()
            tokens to every sample in the batch: tokens beyond num_global_tokens[b] are considered dummy
            tokens and are always masked. Shape: (B,).
        attention_between_global_tokens (Tensor[bool]): If False, global tokens can only attend to themselves (diagonal).
            If True (default), global tokens can attend to all global tokens. Shape: (B,).
        full_attention (Tensor[bool]): If True, ignore all limits and allow every token to attend every other
            token for that sample. Shape: (B,).
        device: PyTorch device (optional).

    Returns:
        torch.Tensor[bool]: Boolean mask of shape (B, T, T), where True means "mask" (disallow attention).
            If all args are scalars, B=1 and the output shape is (1, T, T).

    Examples:
        Single-sample mask (B = 1)
        3 global tokens, 3 chunks of size 2, 1 left chunk, 0 right chunks (causal)
        >>> cfg = ChunkedMaskConfig(chunk_size=2, num_left_chunks=1, num_right_chunks=0, num_global_tokens=3)
        >>> mask = build_chunked_mask_from_config(cfg, T=9)[0].int()
        >>> print(mask)
        tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0]], dtype=torch.int32)

        3 global tokens, 3 chunks of size 2, 1 left chunk, 0 right chunks (causal), no inter-global attention
        >>> cfg = ChunkedMaskConfig(
        ...     chunk_size=2, num_left_chunks=1, num_right_chunks=0, num_global_tokens=3,
        ...     attention_between_global_tokens=False,
        ... )
        >>> mask = build_chunked_mask_from_config(cfg, T=9)[0].int()
        >>> print(mask)
        tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0]], dtype=torch.int32)

        Batched mask (B = 2)
        >>> cfg = ChunkedMaskConfig(
        ...     chunk_size=[2, 1],
        ...     num_left_chunks=[1, -1],
        ...     num_right_chunks=[0, 0],
        ...     num_global_tokens=[2, 0],
        ... )
        >>> mask = build_chunked_mask_from_config(cfg, T=6).int()
        >>> print(mask.shape)
        torch.Size([2, 6, 6])
        >>> print(mask[0])
        tensor([[0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]], dtype=torch.int32)
        >>> print(mask[1])
        tensor([[1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0]], dtype=torch.int32)
    """
    assert (num_global_tokens <= T).all(), f"num_global_tokens must be in [0,T] with T={T}"
    B = chunk_size.numel()
    
    if T == 0:
        # Return an empty mask of shape (B, 0, 0) with correct device and dtype
        return torch.empty(B, 0, 0, dtype=torch.bool, device=device)

    idx = torch.arange(T, device=device)                       # (T,)
    idx_b = idx.unsqueeze(0).expand(B, T)                      # (B, T)
    G = num_global_tokens.max()
    is_global = idx_b < G                                      # (B, T)
    is_rolling = ~is_global                                    # (B, T)

    roll_idx = (idx_b - num_global_tokens.unsqueeze(1)).clamp(min=0)
    roll_chunk = torch.div(roll_idx, chunk_size.unsqueeze(1), rounding_mode='trunc')  # (B, T)

    # Chunk distance: For each (batch, query, key), gives (chunk_q - chunk_k)
    dist_chunk = roll_chunk.unsqueeze(2) - roll_chunk.unsqueeze(1)  # (B, T, T)

    # Initialize mask: 0 = allowed, 1 = masked
    mask = torch.zeros(B, T, T, dtype=torch.bool, device=device)

    # Mask rolling queries to rolling keys beyond allowed chunk limits
    rolling_to_rolling = is_rolling.unsqueeze(2) & is_rolling.unsqueeze(1)  # (B, T, T)
    _left = num_left_chunks.view(B, 1, 1)
    _right = num_right_chunks.view(B, 1, 1)
    if (num_left_chunks >= 0).any():
        mask |= (rolling_to_rolling & (_left >= 0) & (dist_chunk > _left))
    if (num_right_chunks >= 0).any():
        mask |= (rolling_to_rolling & (_right >= 0) & (dist_chunk < -_right))

    # Mask global queries to rolling keys (global queries see only global keys)
    global_to_rolling = is_global.unsqueeze(2) & is_rolling.unsqueeze(1)     # (B, T, T)
    mask |= global_to_rolling

    # Optionally restrict global-global submatrix to only the diagonal (self-attention)
    if (~attention_between_global_tokens).any():
        global_to_global = is_global.unsqueeze(2) & is_global.unsqueeze(1)  # (B, T, T)
        diagonal = torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0)  # (1, T, T)
        mask_off_diag = global_to_global & ~diagonal  # (B, T, T)

        disable = (~attention_between_global_tokens).view(B, 1, 1)  # (B,1,1)
        batch_mask = disable & mask_off_diag  # (B, T, T)
        mask |= batch_mask

    # Mask attention to the dummy global states.
    if G > 0:
        idx_g = idx_b[:, :G]                                        # (B, G)
        g_dummy_mask = idx_g >= num_global_tokens.unsqueeze(1)      # (B, G)
        mask[:, :, :G] |= g_dummy_mask.unsqueeze(1)                 # (B, 1, G)

    # Allow full attention for selected batch items (overrides all masking rules)
    mask &= (~full_attention).view(B, 1, 1)

    # (rolling queries to global keys, and global queries to global keys [if allowed], are always allowed)
    return mask  # (B, T, T)

@dataclass
class ChunkedMaskConfig:
    """
    Batch-capable configuration for chunked / global-token attention masks.

    This class defines chunk-wise attention and emission behavior for streaming models,
    allowing fine-grained control over left/right context and compute trade-offs.

    Each argument may be
        • a scalar  -> same value for every batch item
        • a 1-D list / tuple / NumPy array -> per-batch values
        • a 1-D torch.Tensor -> per-batch values (device/ dtype ignored here)

    Args:
        chunk_size (int | Sequence | Tensor):
            Number of tokens per chunk. Defines the block size for streaming attention.
            Use chunk_size > 1 for block-wise attention (streaming encoder).
            Use chunk_size = 1 for token-wise attention (streaming AR decoder).
        num_left_chunks (int | Sequence | Tensor):
            Max left (history) chunks visible to each rolling token (-1 = unlimited).
            Default: -1.
        num_right_chunks (int | Sequence | Tensor):
            Max right (future) chunks visible to each rolling token (-1 = unlimited,
            0 = causal). Default: 0.
        use_asymmetric_mask (bool | Sequence | Tensor):
            When True, activates the C-axis asymmetric attention mask that prevents the
            growing right receptive field problem. Each chunk gets C = num_right_chunks + 1
            versions, where the oldest version (c=0) has no right context baked in, and
            progressively newer versions have more. This bounds the effective right
            receptive field to exactly num_right_chunks, regardless of encoder depth.
            Has no effect when num_right_chunks <= 0.
            **Note:** Doubles (or more) encoder memory and compute during training.
            Default: True.
        num_global_tokens (int | Sequence | Tensor):
            Number of "global" tokens always visible at the sequence start. These are
            always attendable by rolling queries. Default: 0.
        attention_between_global_tokens (bool | Sequence | Tensor):
            If False, global tokens can only attend to themselves (diagonal). If
            True, global tokens can attend to all global tokens. Default: True.
        full_attention (bool | Sequence | Tensor):
            If True, disables all chunking and masking logic; every token attends to all others.
            Default: False.

    All units are in *chunks*, except chunk_size (in tokens).
    """

    # ---- Chunking and context window parameters ----
    chunk_size: Union[int, Sequence[int], torch.Tensor]
    num_left_chunks: Union[int, Sequence[int], torch.Tensor] = -1
    num_right_chunks: Union[int, Sequence[int], torch.Tensor] = 0
    use_asymmetric_mask: Union[bool, Sequence[bool], torch.Tensor] = True
    num_global_tokens: Union[int, Sequence[int], torch.Tensor] = 0
    attention_between_global_tokens: Union[bool, Sequence[bool], torch.Tensor] = True
    full_attention: Union[bool, Sequence[bool], torch.Tensor] = False

    def resolve(self, batch_size: int=None, device=None) -> "ResolvedChunkedMaskConfig":
        """
        Build per-batch 1D tensors for all configuration fields.

        Args:
            batch_size (int, optional): Number of items in batch. If omitted, the batch size
                will be inferred through the field shapes.
            device (torch.device, optional): Target device for tensors.

        Returns:
            ResolvedChunkedMaskConfig: Mask configuration as per-batch tensors.
        """
        def _to_tensor(x, dtype) -> torch.Tensor:
            if torch.is_tensor(x):
                x = x.to(device=device, dtype=dtype)
            elif isinstance(x, (list, tuple)):
                x = torch.tensor(x, dtype=dtype, device=device)
            else:
                x = torch.tensor([x], dtype=dtype, device=device)  # scalar -> (1,)
            return x

        chunk  = _to_tensor(self.chunk_size, torch.long)
        left   = _to_tensor(self.num_left_chunks, torch.long)
        right  = _to_tensor(self.num_right_chunks, torch.long)
        glob   = _to_tensor(self.num_global_tokens, torch.long)
        asym   = _to_tensor(self.use_asymmetric_mask, torch.bool)
        attn_glob = _to_tensor(self.attention_between_global_tokens, torch.bool)
        full_attn = _to_tensor(self.full_attention, torch.bool)

        # Infer batch size.
        _err_str = "input"
        if batch_size is None:
            batch_size = max(
                chunk.numel(),
                left.numel(),
                right.numel(),
                glob.numel(),
                asym.numel(),
                attn_glob.numel(),
                full_attn.numel(),
            )
            _err_str = "inferred"
        if chunk.numel() not in {1, batch_size}:
            raise Exception(f"chunk_size {chunk.shape} is not compatible with {_err_str} batch size {batch_size}.")
        if left.numel() not in {1, batch_size}:
            raise Exception(f"num_left_chunks {left.shape} is not compatible with {_err_str} batch size {batch_size}.")
        if right.numel() not in {1, batch_size}:
            raise Exception(f"num_right_chunks {right.shape} is not compatible with {_err_str} batch size {batch_size}.")
        if glob.numel() not in {1, batch_size}:
            raise Exception(f"num_global_tokens {glob.shape} is not compatible with {_err_str} batch size {batch_size}.")
        if asym.numel() not in {1, batch_size}:
            raise Exception(f"use_asymmetric_mask {asym.shape} is not compatible with {_err_str} batch size {batch_size}.")
        if attn_glob.numel() not in {1, batch_size}:
            raise Exception(f"attention_between_global_tokens {attn_glob.shape} is not compatible with {_err_str} batch size {batch_size}.")
        if full_attn.numel() not in {1, batch_size}:
            raise Exception(f"full_attention {full_attn.shape} is not compatible with {_err_str} batch size {batch_size}.")

        # Broadcast to (batch size,)
        chunk = chunk.expand(batch_size)
        left = left.expand(batch_size)
        right = right.expand(batch_size)
        glob = glob.expand(batch_size)
        asym = asym.expand(batch_size)
        attn_glob = attn_glob.expand(batch_size)
        full_attn = full_attn.expand(batch_size)

        # Disable asymmetric mask when full_attention is True or num_right_chunks <= 0
        asym = asym & (~full_attn) & (right > 0)

        # ---- range assertions -------------------------------------------------
        assert (chunk > 0).all(),          "chunk_size must be > 0"
        assert (left  >= -1).all(),        "num_left_chunks  must be >= -1"
        assert (right >= -1).all(),        "num_right_chunks must be >= -1"
        assert (glob  >= 0).all(),         "num_global_tokens must be >= 0"

        return ResolvedChunkedMaskConfig(
            chunk_size=chunk,
            num_left_chunks=left,
            num_right_chunks=right,
            num_global_tokens=glob,
            use_asymmetric_mask=asym,
            attention_between_global_tokens=attn_glob,
            full_attention=full_attn,
        )

@dataclass
class ResolvedChunkedMaskConfig:
    """Per-batch tensor form of :class:`ChunkedMaskConfig`.

    Produced by ``ChunkedMaskConfig.resolve``; every field is a 1D tensor of
    shape (B,) so mask builders can operate batched. Also provides
    ``build_age_mask`` for the C-axis asymmetric attention mask.
    """

    chunk_size: torch.Tensor                       # (B,)
    num_left_chunks: torch.Tensor                  # (B,)
    num_right_chunks: torch.Tensor                 # (B,)
    num_global_tokens: torch.Tensor                # (B,)
    use_asymmetric_mask: torch.Tensor              # (B,) bool
    attention_between_global_tokens: torch.Tensor  # (B,)
    full_attention: torch.Tensor                   # (B,)

    def build_age_mask(
        self,
        Tq: int,
        Tk: Optional[int] = None,
        query_offset: int = 0,
    ) -> torch.Tensor:
        """
        Build a boolean mask for the asymmetric C-axis attention.

        Maintains C = num_right_chunks + 1 versions of each chunk. Version c_q
        can only read version c_k of a key chunk, where c_k is determined by the
        "age" (chunk distance + c_q). This prevents information from leaking
        through encoder layers beyond num_right_chunks.

        For each (b, q, c_q, k):
            age_index = clamp(floor((q+query_offset)/cs) - floor(k/cs) + c_q,
                              min=-1, max=nrc)
            -1 → masked (illegal future key), otherwise c_k = age_index.

        Args:
            Tq (int): Number of query tokens.
            Tk (int, optional): Number of key tokens. If None, defaults to Tq.
            query_offset (int): Absolute position offset for queries (Stage B
                streaming cache). Queries occupy absolute positions
                ``[query_offset, query_offset + Tq)``. Keys always start at 0.

        Returns:
            torch.BoolTensor of shape (B, Tq, C, Tk, C) where C = max(num_right_chunks) + 1.
        """
        B = self.chunk_size.size(0)
        device = self.chunk_size.device
        if Tk is None:
            Tk = Tq

        # 1) chunk distance: (B, Tq, Tk)
        q_idx = torch.arange(query_offset, query_offset + Tq, device=device).view(1, Tq, 1)
        k_idx = torch.arange(Tk, device=device).view(1, 1, Tk)
        chunk_size = self.chunk_size.view(B, 1, 1)
        q_chunk = q_idx // chunk_size      # (B, Tq, 1)
        k_chunk = k_idx // chunk_size      # (B, 1, Tk)
        raw_age = q_chunk - k_chunk        # (B, Tq, Tk)

        # 2) per-version offset: age = raw_age + c_q
        C = int(self.num_right_chunks.max().item()) + 1
        age_offset = torch.arange(C, device=device).view(1, C)  # (1, C)

        raw_age = (
            raw_age.unsqueeze(2)              # (B, Tq, 1, Tk)
            + age_offset.view(1, 1, C, 1)     # (1, 1, C, 1)
        )  # (B, Tq, C, Tk)

        # 3) clamp into [-1 … num_right_chunks[b]]
        age = raw_age.clamp(min=-1)
        age = torch.minimum(age, self.num_right_chunks.view(B, 1, 1, 1).long())
        age = age.long()  # (B, Tq, C, Tk)

        # 4) build boolean mask via scatter
        age_idx = age.clamp(min=0)  # (B, Tq, C, Tk)
        mask = torch.zeros(B, Tq, C, Tk, C, dtype=torch.bool, device=device)
        mask.scatter_(
            dim=-1,
            index=age_idx.unsqueeze(-1),
            src=torch.ones_like(age_idx, dtype=torch.bool).unsqueeze(-1),
        )

        # zero out where age was -1 (illegal future key)
        mask &= (age.unsqueeze(-1) >= 0)

        # full_attention override: collapse to c_k=0 only
        full = self.full_attention.view(B, 1, 1, 1, 1)
        mask &= ~full
        mask[..., 0] |= full.squeeze(-1)

        return mask  # (B, Tq, C, Tk, C)


def build_streaming_attn_mask_flat(
    config: Union[ChunkedMaskConfig, ResolvedChunkedMaskConfig],
    T_tail: int,
    T_cached: int,
    query_offset: int,
    device=None,
    n_pad_keys: int = 0,
) -> torch.Tensor:
    """Build the flattened attention mask for Stage B encoder-streaming self-attn.

    Produces a ``(1, T_tail * C, (T_cached + T_tail) * C)`` boolean mask where
    ``True`` = attend. Queries correspond to the unfinalized tail at absolute
    positions ``[query_offset, query_offset + T_tail)``. Keys span the full
    virtual sequence ``[0, T_cached + T_tail)`` (cached prefix + fresh tail).

    Only asymmetric (C > 1) mode is supported here. Per-query left-chunk
    bounding is applied for finite ``num_left_chunks`` (>= 0), matching the
    full path (``build_chunked_mask``); ``num_left_chunks=-1`` disables it and
    yields a bit-identical mask to the pre-bounding behavior.

    Args:
        config: ChunkedMaskConfig or resolved form. Must have ``use_asymmetric_mask=True``.
        T_tail: Number of query positions.
        T_cached: Number of cached key positions (finalized prefix).
        query_offset: Absolute position of the first query (n_finalized * chunk_size).
        device: Target device.

    Returns:
        Bool tensor, shape ``(1, T_tail * C, (T_cached + T_tail) * C)``, True = attend.
    """
    if isinstance(config, ChunkedMaskConfig):
        r = config.resolve(device=device)
    else:
        r = config

    if not r.use_asymmetric_mask.any() and r.num_right_chunks.max().item() > 0:
        raise ValueError(
            "build_streaming_attn_mask_flat requires use_asymmetric_mask=True"
        )

    Tk = T_cached + T_tail
    # (B=1, Tq=T_tail, C, Tk, C), True = attend
    mask = r.build_age_mask(Tq=T_tail, Tk=Tk, query_offset=query_offset)
    B, _, C, _, _ = mask.shape

    # Per-query left-chunk bound, matching the full path (build_chunked_mask,
    # nets_utils.py:509-510: it masks keys with (q_chunk - k_chunk) > L when
    # L >= 0). The caller passes a CACHE-RELATIVE query_offset: the first tail
    # query sits at shifted chunk query_offset // cs, and key index 0 is the
    # OLDEST RETAINED cached chunk (utterance chunk n_fin - n_in_cache), not
    # utterance chunk 0. Queries and keys are shifted by the SAME amount
    # (n_fin - n_in_cache chunks), so the chunk distance (q_chunk - k_chunk) is
    # invariant under the shift; bounding on the shifted distance equals bounding
    # on the absolute distance. The cache trim alone bounds only the oldest tail
    # query chunk; later tail chunks (q_chunk up to +F) would otherwise attend up
    # to F extra left chunks vs the full path. L=-1 is a bit-identical no-op.
    if (r.num_left_chunks >= 0).any():
        cs = r.chunk_size.view(B, 1, 1, 1, 1)
        left = r.num_left_chunks.view(B, 1, 1, 1, 1)
        q_chunk = (
            torch.arange(query_offset, query_offset + T_tail, device=mask.device)
            .view(1, T_tail, 1, 1, 1) // cs
        )                                              # (B, Tq, 1, 1, 1)
        k_chunk = (
            torch.arange(Tk, device=mask.device).view(1, 1, 1, Tk, 1) // cs
        )                                              # (B, 1, 1, Tk, 1)
        dist_chunk = q_chunk - k_chunk                 # (B, Tq, 1, Tk, 1)
        # Forbid keys beyond the finite left window (broadcast over both C dims).
        mask &= ~((left >= 0) & (dist_chunk > left))

    if n_pad_keys > 0:
        # Trailing key positions [Tk - n_pad_keys, Tk) are zero-pad frames
        # appended at is_final to complete the final chunk. Forbid every query
        # from attending to them (they carry no real audio).
        mask[:, :, :, Tk - n_pad_keys:, :] = False
    # Flatten (C dim merged into T dim): (B, Tq*C, Tk*C)
    mask_flat = mask.reshape(B, T_tail * C, Tk * C)
    return mask_flat


def build_chunked_mask_from_config(
    config: Union[ChunkedMaskConfig, ResolvedChunkedMaskConfig],
    T: int,
    device=None,
) -> torch.Tensor:
    """
    Wrapper to build a chunked self-attention mask from a ChunkedMaskConfig.

    Args:
        config (ChunkedMaskConfig or ResolvedChunkedMaskConfig): Configuration object with mask parameters.
        T (int): Total sequence length (global + rolling tokens).
        device (torch.device, optional): Device to construct mask on.

    Returns:
        torch.Tensor: Boolean mask of shape (B, T, T), where True means "mask" (disallow attention).
    """
    if isinstance(config, ChunkedMaskConfig):
        r = config.resolve(device=device)
    else:
        r = config
    return build_chunked_mask(
        T=T,
        chunk_size=r.chunk_size,
        num_left_chunks=r.num_left_chunks,
        num_right_chunks=r.num_right_chunks,
        num_global_tokens=r.num_global_tokens,
        attention_between_global_tokens=r.attention_between_global_tokens,
        full_attention=r.full_attention,
        device=device,
    )


def pad_to_streaming_length(
    x: torch.Tensor,
    buffer_samples: int,
    buffer_overlap: int,
    time_dim: int = -1,
) -> torch.Tensor:
    """
    Right-pad an input tensor along the time dimension so the final streaming
    buffer (window) is always fully filled.

    This ensures that, when splitting the sequence into overlapping streaming
    chunks of length ``buffer_samples`` and overlap ``buffer_overlap``,
    every chunk is complete.

    Args:
        x (torch.Tensor): Input tensor, shape (..., time, ...).
            The time dimension to pad should be specified by ``time_dim``.
        buffer_samples (int): Number of samples per streaming chunk (window size).
        buffer_overlap (int): Number of samples of overlap between consecutive chunks.
            Effective chunk stride is ``buffer_samples - buffer_overlap``.
        time_dim (int): Axis that corresponds to time. May be negative (default: -1).

    Returns:
        torch.Tensor: Padded tensor. Same shape as input, except the
            ``time_dim`` is extended to ensure the last chunk is full.

    Example:
        >>> import torch
        >>> x = torch.arange(950)  # (time,)
        >>> y = pad_to_streaming_length(x, buffer_samples=250, buffer_overlap=50)
        >>> print(y.shape)
        torch.Size([1050])
    """
    stride = buffer_samples - buffer_overlap
    assert stride > 0, "buffer_samples must be > buffer_overlap"

    if time_dim < 0:
        time_dim = x.dim() + time_dim
    assert 0 <= time_dim < x.dim(), "time_dim out of range"
    length = x.size(time_dim)

    # Compute smallest k such that k*stride + buffer_samples >= length
    k = max(0, math.ceil((length - buffer_samples) / stride))
    target_len = k * stride + buffer_samples
    pad_needed = target_len - length
    if pad_needed <= 0:
        return x  # already aligned

    # pad format: (..., left_time, right_time)
    pad = [0, 0] * (x.dim() - time_dim - 1) + [0, pad_needed]
    return torch.nn.functional.pad(x, pad)


def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax, D).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)
    ).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


def to_torch_tensor(x):
    """Change to torch.Tensor or ComplexTensor from numpy.ndarray.

    Args:
        x: Inputs. It should be one of numpy.ndarray, Tensor, ComplexTensor, and dict.

    Returns:
        Tensor or ComplexTensor: Type converted inputs.

    Examples:
        >>> xs = np.ones(3, dtype=np.float32)
        >>> xs = to_torch_tensor(xs)
        tensor([1., 1., 1.])
        >>> xs = torch.ones(3, 4, 5)
        >>> assert to_torch_tensor(xs) is xs
        >>> xs = {'real': xs, 'imag': xs}
        >>> to_torch_tensor(xs)
        ComplexTensor(
        Real:
        tensor([1., 1., 1.])
        Imag;
        tensor([1., 1., 1.])
        )

    """
    # If numpy, change to torch tensor
    if isinstance(x, np.ndarray):
        if x.dtype.kind == "c":
            # Dynamically importing because torch_complex requires python3
            from torch_complex.tensor import ComplexTensor

            return ComplexTensor(x)
        else:
            return torch.from_numpy(x)

    # If {'real': ..., 'imag': ...}, convert to ComplexTensor
    elif isinstance(x, dict):
        # Dynamically importing because torch_complex requires python3
        from torch_complex.tensor import ComplexTensor

        if "real" not in x or "imag" not in x:
            raise ValueError("has 'real' and 'imag' keys: {}".format(list(x)))
        # Relative importing because of using python3 syntax
        return ComplexTensor(x["real"], x["imag"])

    # If torch.Tensor, as it is
    elif isinstance(x, torch.Tensor):
        return x

    else:
        error = (
            "x must be numpy.ndarray, torch.Tensor or a dict like "
            "{{'real': torch.Tensor, 'imag': torch.Tensor}}, "
            "but got {}".format(type(x))
        )
        try:
            from torch_complex.tensor import ComplexTensor
        except Exception:
            # If PY2
            raise ValueError(error)
        else:
            # If PY3
            if isinstance(x, ComplexTensor):
                return x
            else:
                raise ValueError(error)


def get_subsample(train_args, mode, arch):
    """Parse the subsampling factors from the args for the specified `mode` and `arch`.

    Args:
        train_args: argument Namespace containing options.
        mode: one of ('asr', 'mt', 'st')
        arch: one of ('rnn', 'rnn-t', 'rnn_mix', 'rnn_mulenc', 'transformer')

    Returns:
        np.ndarray / List[np.ndarray]: subsampling factors.
    """
    if arch == "transformer":
        return np.array([1])

    elif mode == "mt" and arch == "rnn":
        # +1 means input (+1) and layers outputs (train_args.elayer)
        subsample = np.ones(train_args.elayers + 1, dtype=np.int64)
        logging.warning("Subsampling is not performed for machine translation.")
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif (
        (mode == "asr" and arch in ("rnn", "rnn-t"))
        or (mode == "mt" and arch == "rnn")
        or (mode == "st" and arch == "rnn")
    ):
        subsample = np.ones(train_args.elayers + 1, dtype=np.int64)
        if train_args.etype.endswith("p") and not train_args.etype.startswith("vgg"):
            ss = train_args.subsample.split("_")
            for j in range(min(train_args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                "Subsampling is not performed for vgg*. "
                "It is performed in max pooling layers at CNN."
            )
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif mode == "asr" and arch == "rnn_mix":
        subsample = np.ones(
            train_args.elayers_sd + train_args.elayers + 1, dtype=np.int64
        )
        if train_args.etype.endswith("p") and not train_args.etype.startswith("vgg"):
            ss = train_args.subsample.split("_")
            for j in range(
                min(train_args.elayers_sd + train_args.elayers + 1, len(ss))
            ):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                "Subsampling is not performed for vgg*. "
                "It is performed in max pooling layers at CNN."
            )
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif mode == "asr" and arch == "rnn_mulenc":
        subsample_list = []
        for idx in range(train_args.num_encs):
            subsample = np.ones(train_args.elayers[idx] + 1, dtype=np.int64)
            if train_args.etype[idx].endswith("p") and not train_args.etype[
                idx
            ].startswith("vgg"):
                ss = train_args.subsample[idx].split("_")
                for j in range(min(train_args.elayers[idx] + 1, len(ss))):
                    subsample[j] = int(ss[j])
            else:
                logging.warning(
                    "Encoder %d: Subsampling is not performed for vgg*. "
                    "It is performed in max pooling layers at CNN.",
                    idx + 1,
                )
            logging.info("subsample: " + " ".join([str(x) for x in subsample]))
            subsample_list.append(subsample)
        return subsample_list

    else:
        raise ValueError("Invalid options: mode={}, arch={}".format(mode, arch))


def rename_state_dict(
    old_prefix: str, new_prefix: str, state_dict: Dict[str, torch.Tensor]
):
    """Replace keys of old prefix with new prefix in state dict."""
    # need this list not to break the dict iterator
    old_keys = [k for k in state_dict if k.startswith(old_prefix)]
    if len(old_keys) > 0:
        logging.warning(f"Rename: {old_prefix} -> {new_prefix}")
    for k in old_keys:
        v = state_dict.pop(k)
        new_k = k.replace(old_prefix, new_prefix)
        state_dict[new_k] = v


def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import
    from espnet.nets.pytorch_backend.conformer.swish import Swish

    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": Swish,
    }

    return activation_funcs[act]()


def trim_by_ctc_posterior(
    h: torch.Tensor,
    ctc_probs: torch.Tensor,
    masks: torch.Tensor,
    pos_emb: torch.Tensor = None,
):
    """Trim the encoder hidden output using CTC posterior.

    The continuous frames in the tail that confidently represent
    blank symbols are trimmed.
    """
    # Empirical settings
    frame_tolerance = 5
    conf_tolerance = 0.95
    blank_id = 0

    assert masks.size(1) == 1
    masks = masks.squeeze(1)
    hlens = masks.sum(dim=1)
    assert h.size()[:2] == ctc_probs.size()[:2]
    assert h.size(0) == hlens.size(0)

    # blank frames
    max_values, max_indices = ctc_probs.max(dim=2)
    blank_masks = torch.logical_and(
        max_values > conf_tolerance, max_indices == blank_id
    )

    # plus ignored frames
    joint_masks = torch.logical_or(blank_masks, ~masks)

    # lengths after the trimming
    B, T, _ = h.size()
    frame_idx = torch.where(
        joint_masks, -1, torch.arange(T).unsqueeze(0).repeat(B, 1).to(h.device)
    )
    after_lens = torch.where(
        frame_idx.max(dim=-1)[0] + frame_tolerance + 1 < hlens,
        frame_idx.max(dim=-1)[0] + frame_tolerance + 1,
        hlens,
    )

    h = h[:, : max(after_lens)]
    masks = ~make_pad_mask(after_lens).to(h.device).unsqueeze(1)

    if pos_emb is None:
        pos_emb = None
    elif (hlens.max() * 2 - 1).item() == pos_emb.size(1):  # RelPositionalEncoding
        pos_emb = pos_emb[
            :, pos_emb.size(1) // 2 - h.size(1) + 1 : pos_emb.size(1) // 2 + h.size(1)
        ]
    else:
        pos_emb = pos_emb[:, : h.size(1)]

    return h, masks, pos_emb


def roll_tensor(
    x: torch.Tensor,
    lengths: torch.Tensor,
    roll_amounts: Optional[torch.Tensor] = None,
    fixed_intervals: Optional[int] = None,
) -> torch.Tensor:
    """Left-roll tensor x by roll_amounts, only within lengths and optionally quantized.

    Args:
        x: input tensor (B, T, D)
        lengths: lengths of each sequence (B,)
        roll_amounts: random shift amounts (B,). If None, random shift
            amounts are generated.
        fixed_intervals: if not None, roll_amounts are quantized to
            multiples of this.
    Returns:
        rolled_x: rolled tensor (B, T, D)
    Useful to apply roll augmentation to the input, while considering
    the input length for each sample.
    """
    B, T, D = x.shape

    indices = torch.arange(T).unsqueeze(0).expand(B, T).to(x.device)  # (B, T)
    lengths = lengths.unsqueeze(1)  # (B, 1)

    if roll_amounts is None:
        roll_amounts = torch.randint(0, lengths.max(), (B,), device=x.device)
    if fixed_intervals is not None:
        roll_amounts = (roll_amounts // fixed_intervals) * fixed_intervals
    roll_indices = (indices - roll_amounts.unsqueeze(1)) % lengths  # (B, T)
    roll_indices = roll_indices.unsqueeze(2).expand(-1, -1, D)  # (B, T, D)

    mask = indices < lengths  # (B, T), True if position is valid
    rolled_x = torch.empty_like(x)
    rolled_x[mask] = x.gather(1, roll_indices)[mask]
    rolled_x[~mask] = x[~mask]
    return rolled_x

@dataclass
class DynamicChunkConfigSampler:
    """
    Dynamic chunk configuration sampler for streaming ASR training.

    During training (self.training = True):
      - With probability `chunkwise_prob`, sample a random chunk size and left context.
      - Otherwise, return full-attention (non-streaming) config.

    During evaluation (self.training = False):
      - Always return a fixed deterministic config for stable inference/validation.
    """

    # --- Randomization probabilities ---
    chunkwise_prob: float = 0.6
    limited_left_context_prob: float = 0.75

    # --- Ranges (in encoder frames/chunks) ---
    chunk_size_min: int = 8
    chunk_size_max: int = 32
    left_context_chunks_min: int = 2
    left_context_chunks_max: int = 16

    # --- Dynamic right (future) context ---
    dynamic_right_context: bool = False       # master switch (off by default)
    right_context_chunks_max: int = 2         # F sampled from [1, right_context_chunks_max]
    right_context_prob: float = 0.5           # prob of F > 0 within streaming batches

    # --- Fixed config for eval ---
    eval_chunk_size: int = 32
    eval_num_left_chunks: int = 16
    eval_num_right_chunks: int = 0            # eval right context (keep 0 = causal)

    # --- Asymmetric mask ---
    use_asymmetric_mask: bool = True

    # --- Internal training mode flag ---
    training: bool = True

    def train(self, mode: bool = True):
        """Set training/eval mode (like torch.nn.Module.train())."""
        self.training = mode

    def eval(self):
        """Alias for self.train(False)."""
        self.train(False)

    def __call__(self, batch_size: int = 1) -> "ChunkedMaskConfig":
        """
        Sample a ChunkedMaskConfig according to mode and probabilities.
        """
        if not self.training:
            # Evaluation mode: deterministic config
            return ChunkedMaskConfig(
                chunk_size=self.eval_chunk_size,
                num_left_chunks=self.eval_num_left_chunks,
                num_right_chunks=self.eval_num_right_chunks,
                use_asymmetric_mask=self.use_asymmetric_mask,
                full_attention=False,
            )

        # ---------------- TRAIN MODE ----------------
        configs = []
        for _ in range(batch_size):
            if random.random() > self.chunkwise_prob:
                # with probability (1 - chunkwise_prob) -> full attention (non-streaming)
                cfg = ChunkedMaskConfig(
                    chunk_size=1,
                    num_left_chunks=-1,
                    num_right_chunks=-1,
                    use_asymmetric_mask=self.use_asymmetric_mask,
                    full_attention=True,
                )
            else:
                # with probability chunkwise_prob -> chunked streaming
                chunk_size = random.randint(self.chunk_size_min, self.chunk_size_max)
                if random.random() < self.limited_left_context_prob:
                    num_left_chunks = random.randint(
                        self.left_context_chunks_min, self.left_context_chunks_max
                    )
                else:
                    num_left_chunks = -1  # unlimited left context

                # Sample right context (future chunks)
                if self.dynamic_right_context and random.random() < self.right_context_prob:
                    num_right_chunks = random.randint(1, self.right_context_chunks_max)
                else:
                    num_right_chunks = 0

                cfg = ChunkedMaskConfig(
                    chunk_size=chunk_size,
                    num_left_chunks=num_left_chunks,
                    num_right_chunks=num_right_chunks,
                    use_asymmetric_mask=self.use_asymmetric_mask,
                    full_attention=False,
                )
            configs.append(cfg)

        # Return a single config if batch_size == 1
        if batch_size == 1:
            return configs[0]

        # Otherwise merge per-batch lists
        return ChunkedMaskConfig(
            chunk_size=[c.chunk_size for c in configs],
            num_left_chunks=[c.num_left_chunks for c in configs],
            num_right_chunks=[c.num_right_chunks for c in configs],
            use_asymmetric_mask=[c.use_asymmetric_mask for c in configs],
            full_attention=[c.full_attention for c in configs],
        )

from torch.nn.attention.flex_attention import create_nested_block_mask

def build_flex_blockmask_with_future_chunks(
    C: int,              # Number of chunks
    L: int,              # Length of each chunk (must be fixed across all batches)
    G: int,              # Number of global tokens at the start of each chunk
    block_size: int,     # Size of each normal block inside each chunk
    num_prev_blocks: int,# How many previous blocks each block can attend (intra-chunk)
    B: int,              # Batch size (FlexAttention requires this parameter)
    H: int,              # Number of attention heads (FlexAttention requires this parameter)
):
    """
    Build a static FlexAttention BlockMask for a sequence consisting of `C` chunks,
    each of fixed length `L`, where:

    --------------------------------------------------------------------
    1. GLOBAL TOKEN RULES (per chunk)
    --------------------------------------------------------------------
        - The first `G` positions in each chunk are *global tokens*.
        - Global tokens can attend to ALL positions within the SAME chunk.
        - All positions in the SAME chunk can attend to global tokens.
        - No global-to-global or global-to-normal attention across different chunks.

    --------------------------------------------------------------------
    2. NORMAL TOKEN RULES (per chunk)
    --------------------------------------------------------------------
        - Positions G .. L-1 are normal tokens.
        - Normal tokens are grouped into blocks of size `block_size`.
        - Within the same chunk:
              a) full attention inside the same block
              b) a block may attend to up to `num_prev_blocks` previous blocks
                 (e.g., num_prev_blocks = 1 → attend to immediate previous block only)

    --------------------------------------------------------------------
    3. CROSS-CHUNK FUTURE ATTENTION (normal tokens only)
    --------------------------------------------------------------------
        - Let chunk index = q_chunk = q_idx // L   (0-based)
        - A normal token in chunk q_chunk can attend to normal tokens in
          FUTURE chunks, with the following rule:

              chunk 0 → can attend to 0 future chunks
              chunk 1 → can attend to 1 future chunk
              chunk 2 → can attend to 2 future chunks
              ...
              chunk k → can attend to k future chunks

          BUT cannot exceed total available chunks.

    --------------------------------------------------------------------
    4. STATIC PATTERN REQUIREMENT
    --------------------------------------------------------------------
    - All parameters (C, L, G, block_size, num_prev_blocks) MUST be fixed
      at model compile time.
    - No pad masks or per-example variable lengths are allowed.

    """

    # ------------------------------------------------------------------
    # Basic validation of static parameters
    # ------------------------------------------------------------------
    assert 0 <= G < L, "G must be in the range [0, L-1]"
    assert (L - G) % block_size == 0, "Normal region size (L-G) must be divisible by block_size"

    total_len = C * L                         # Total sequence length
    blocks_per_chunk = (L - G) // block_size  # Number of normal blocks inside each chunk

    # Container for nested masks for create_nested_block_mask
    regions = []


    # ==================================================================
    # 1. GLOBAL TOKEN MASK MODIFIER
    # ==================================================================
    def global_mask_mod(score, b, h, q_idx, kv_idx):
        """
        Handles attention involving global tokens ONLY.

        Rules implemented:
            • Global tokens attend to ALL tokens in the SAME chunk.
            • All tokens attend to global tokens (again SAME chunk).
            • Never allow cross-chunk global attention.

        If neither q_idx nor kv_idx is a global token, return False and
        let the normal-region mask_mod handle it.
        """

        # Identify chunk index of query and key
        q_chunk = q_idx // L
        k_chunk = kv_idx // L

        # Must be within the same chunk → global tokens are chunk-local
        if q_chunk != k_chunk:
            return False

        # Compute local positions inside the chunk
        local_q = q_idx % L
        local_k = kv_idx % L

        # If *query* is a global token → allow all tokens of this chunk
        if local_q < G:
            return True

        # If *key* is a global token → allow all queries of this chunk
        if local_k < G:
            return True

        # Otherwise let normal_mask_mod handle it
        return False

    # Register global attention region
    regions.append({
        "q_start": 0,
        "q_len": total_len,
        "kv_start": 0,
        "kv_len": total_len,
        "mask_mod": global_mask_mod,
    })


    # ==================================================================
    # 2. NORMAL TOKEN MASK MODIFIER
    # ==================================================================
    def normal_mask_mod(score, b, h, q_idx, kv_idx):
        """
        Handles attention between *normal tokens* (excluding global tokens).

        Implements:
            (A) Intra-chunk:
                    • full attention inside same block
                    • attend up to `num_prev_blocks` previous blocks
            (B) Cross-chunk future attention:
                    • chunk k can attend to up to k future chunks
        """

        q_chunk = q_idx // L
        k_chunk = kv_idx // L
        local_q = q_idx % L
        local_k = kv_idx % L

        # Skip if either query or key is global; handled already
        if local_q < G or local_k < G:
            return False

        # ------------------------------------------------------------------
        # (A) INTRA-CHUNK ATTENTION
        # ------------------------------------------------------------------
        if q_chunk == k_chunk:

            # Block indices among normal tokens
            q_block = (local_q - G) // block_size
            k_block = (local_k - G) // block_size

            # (A1) Full attention inside the same block
            if q_block == k_block:
                return True

            # (A2) Attend up to num_prev_blocks previous blocks
            if 0 <= k_block <= q_block and (q_block - k_block) <= num_prev_blocks:
                return True

            return False


        # ------------------------------------------------------------------
        # (B) CROSS-CHUNK FUTURE ATTENTION
        # ------------------------------------------------------------------
        if k_chunk > q_chunk:
            # q_chunk can attend up to q_chunk future chunks
            max_future_chunks = min(C - 1 - q_chunk, q_chunk)

            # Is k_chunk within this allowable future range?
            if 1 <= (k_chunk - q_chunk) <= max_future_chunks:
                return True

        # No backward or out-of-range cross-chunk attention
        return False

    # Register normal-token attention region
    regions.append({
        "q_start": 0,
        "q_len": total_len,
        "kv_start": 0,
        "kv_len": total_len,
        "mask_mod": normal_mask_mod,
    })


    # ==================================================================
    # 3. CREATE THE NESTED BLOCK MASK
    # ==================================================================
    block_mask = create_nested_block_mask(
        nested_masks=regions,   # list of regions (global + normal)
        B=B,                    # batch size (constant)
        H=H,                    # number of heads (constant)
        Q_LEN=total_len,        # full sequence length
        KV_LEN=total_len,       # full key length (same here)
        BLOCK_SIZE=block_size,  # block granularity for FlexAttention
        _compile=False,         # True to compile kernel immediately
    )

    return block_mask


def build_flex_block_mask_for_encoder(
    resolved_cfg: "ResolvedChunkedMaskConfig",
    T: int,
    C: int,
    nonpad_mask: torch.Tensor,
    H: int,
    device=None,
    BLOCK_SIZE: int = 128,
):
    """Build a FlexAttention BlockMask for encoder self-attention.

    Encodes the same mask semantics as
    ``build_chunked_mask_from_config`` AND ``ResolvedChunkedMaskConfig.build_age_mask``
    combined, plus the sequence pad mask.

    Flat index layout: query/key indices run over the C-expanded sequence of
    length ``T * C``; index ``i`` decomposes as ``t = i // C`` (time position)
    and ``c = i % C`` (C-axis version), matching the T-major C-minor reshape
    in ``MultiHeadedAttention.forward``.

    Args:
        resolved_cfg: Per-batch mask configuration.
        T: Sequence length (already includes global tokens).
        C: C-axis size (1 if no asymmetric mask).
        nonpad_mask: Bool tensor of shape ``(B, T)``. True = non-pad.
        H: Number of attention heads.
        device: Target device.
        BLOCK_SIZE: FlexAttention block size (must be 128 in current PyTorch).

    Returns:
        BlockMask ready for ``flex_attention(..., block_mask=block_mask)``.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    B = resolved_cfg.chunk_size.shape[0]
    dev = device if device is not None else nonpad_mask.device

    chunk_size = resolved_cfg.chunk_size.to(device=dev, dtype=torch.long)
    num_left_chunks = resolved_cfg.num_left_chunks.to(device=dev, dtype=torch.long)
    num_right_chunks = resolved_cfg.num_right_chunks.to(device=dev, dtype=torch.long)
    num_global_tokens = resolved_cfg.num_global_tokens.to(device=dev, dtype=torch.long)
    use_asymmetric_mask = resolved_cfg.use_asymmetric_mask.to(device=dev)
    attention_between_global_tokens = resolved_cfg.attention_between_global_tokens.to(device=dev)
    full_attention = resolved_cfg.full_attention.to(device=dev)
    nonpad = nonpad_mask.to(device=dev, dtype=torch.bool)

    C_const = int(C)

    def mask_mod(b, h, q_idx, kv_idx):
        t_q = q_idx // C_const
        c_q = q_idx % C_const
        t_k = kv_idx // C_const
        c_k = kv_idx % C_const

        # Match baseline semantics: pad mask is key-only (broadcast from (B, 1, T)).
        # Pad query rows still compute attention over valid keys; downstream
        # consumers ignore those rows. Gating queries here would zero them out,
        # producing a large divergence in the first layer's output.
        nonpad_ok = nonpad[b, t_k]

        G = num_global_tokens[b]
        cs = chunk_size[b]
        L = num_left_chunks[b]
        R = num_right_chunks[b]
        asym = use_asymmetric_mask[b]
        attn_gg = attention_between_global_tokens[b]
        full = full_attention[b]

        q_is_global = t_q < G
        k_is_global = t_k < G
        q_is_rolling = ~q_is_global
        k_is_rolling = ~k_is_global

        gg_ok = q_is_global & k_is_global & (attn_gg | (t_q == t_k))
        rg_ok = q_is_rolling & k_is_global

        roll_q = t_q - G
        roll_k = t_k - G
        q_chunk = roll_q // cs
        k_chunk = roll_k // cs
        dist = q_chunk - k_chunk
        left_ok = (L < 0) | (dist <= L)
        right_ok = (R < 0) | (dist >= -R)
        rr_ok = q_is_rolling & k_is_rolling & left_ok & right_ok

        chunked_ok = gg_ok | rg_ok | rr_ok

        # C-axis age mask. Match build_age_mask: age is clamped to [-1, R];
        # if age < 0 → masked, else c_k == min(age, R) is the attendable slot.
        q_chunk_abs = t_q // cs
        k_chunk_abs = t_k // cs
        raw_age = q_chunk_abs - k_chunk_abs + c_q
        age_clamped = torch.minimum(raw_age, R)
        c_age_ok = (raw_age >= 0) & (c_k == age_clamped)
        c_ok = (asym & c_age_ok) | (~asym)

        full_ok = c_k == 0

        return nonpad_ok & ((full & full_ok) | (~full & chunked_ok & c_ok))

    block_mask = create_block_mask(
        mask_mod=mask_mod,
        B=B,
        H=H,
        Q_LEN=T * C_const,
        KV_LEN=T * C_const,
        device=str(dev),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return block_mask

