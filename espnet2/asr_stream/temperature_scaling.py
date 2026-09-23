"""Temperature scaling for DFC trigger signals.

Applies Guo et al. (2017) temperature scaling to the per-step log-softmax that
feeds the DFC trigger (`token_signals_from_logp`). The calibration is a single
scalar T fit on a held-out dev set by NLL minimization against forced-aligned
true labels.

Maths note: input is already log-softmax `logp[j] = z[j] - lse(z)`, not raw
logits. Because `(z/T) - lse(z/T) = (logp/T) - lse(logp/T)` (the original
lse(z) shift cancels), the calibration can be applied directly on logp:

    logp_cal = logp / T - logsumexp(logp / T)

This is mathematically identical to recomputing softmax(logits / T) and taking
its log. We never need the raw logits.

Used for the dynamic-future-chunks trigger only. Decoder/beam scores are NOT
touched — T is a pure trigger-recalibration parameter that does not affect
argmax, beam ranking, or final WER (only the defer/commit decisions).
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp as _logsumexp_scipy
except Exception:  # pragma: no cover
    minimize_scalar = None
    _logsumexp_scipy = None


def _logsumexp(x: np.ndarray, axis: Optional[int] = -1) -> np.ndarray:
    """Stable logsumexp. Uses scipy if available, else numpy fallback."""
    if _logsumexp_scipy is not None:
        return _logsumexp_scipy(x, axis=axis)
    m = np.max(x, axis=axis, keepdims=True)
    out = m.squeeze(axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))
    return out


def apply_temperature_to_logp(
    logp: np.ndarray, temperature: float
) -> np.ndarray:
    """Apply temperature scaling to a log-softmax vector or batch.

    Args:
        logp: log-softmax tensor of any shape; last axis is the vocab.
        temperature: positive scalar. T=1.0 is a no-op (returns input).

    Returns:
        Temperature-scaled log-softmax with the same shape as ``logp``.
    """
    if temperature == 1.0:
        return logp
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = logp.astype(np.float64) / temperature
    lse = _logsumexp(scaled, axis=-1)
    return (scaled - np.expand_dims(lse, axis=-1)).astype(logp.dtype)


def fit_temperature(
    logp_iter: Iterable[np.ndarray],
    true_label_iter: Iterable[int],
    *,
    T_init: float = 1.0,
    bounds: Tuple[float, float] = (0.5, 5.0),
    tol: float = 1e-4,
) -> Tuple[float, float]:
    """Fit a single scalar T by minimizing NLL on (logp, true_label) pairs.

    Args:
        logp_iter: iterable of 1-D log-softmax vectors (one per step).
        true_label_iter: iterable of int true-label indices, aligned with
            ``logp_iter``.
        T_init: initial guess for T. Note: with the bounded optimizer used
            below, SciPy ignores the bracket, so this value has no effect
            on the result; it is kept for API stability.
        bounds: (lo, hi) search interval for T.
        tol: optimizer tolerance.

    Returns:
        Tuple ``(T_star, nll_star)``.
    """
    if minimize_scalar is None:
        raise RuntimeError(
            "scipy is required for fit_temperature; pip install scipy"
        )
    # Materialize into stacked arrays. logp_iter may be large but we need
    # the whole set for repeated NLL evaluations.
    logps = []
    labels = []
    for lp, y in zip(logp_iter, true_label_iter):
        if lp.ndim != 1:
            raise ValueError(f"each logp must be 1D, got shape {lp.shape}")
        logps.append(lp)
        labels.append(int(y))
    if not logps:
        raise ValueError("empty (logp, label) iter")
    logp_mat = np.stack(logps, axis=0).astype(np.float64)  # (N, V)
    label_arr = np.asarray(labels, dtype=np.int64)  # (N,)
    N, V = logp_mat.shape
    rows = np.arange(N)

    def nll_at_T(T: float) -> float:
        if T <= 0:
            return float("inf")
        # logp_cal[i, y] = logp[i, y] / T - logsumexp(logp[i, :] / T)
        scaled = logp_mat / T
        lse = _logsumexp(scaled, axis=-1)  # (N,)
        return float(np.mean(lse - scaled[rows, label_arr]))

    # NOTE: SciPy's method="bounded" ignores the bracket argument, so
    # T_init does not influence the optimization; only bounds/tol matter.
    res = minimize_scalar(
        nll_at_T,
        bracket=(bounds[0], T_init, bounds[1]),
        bounds=bounds,
        method="bounded",
        options={"xatol": tol},
    )
    T_star = float(res.x)
    nll_star = float(res.fun)
    logging.info(
        "fit_temperature: T_star=%.4f  NLL_star=%.4f  (N=%d, V=%d)",
        T_star,
        nll_star,
        N,
        V,
    )
    return T_star, nll_star
