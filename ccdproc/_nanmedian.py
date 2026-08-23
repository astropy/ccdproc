# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A NaN-aware median written only in terms of the array API standard.

``median``/``nanmedian`` are not part of the array API standard, so this
module provides a fallback that works on any conforming namespace
(``array-api-strict``, ``jax``, ``dask``, ``numpy``, ...). It is used by
`ccdproc.combiner.Combiner.median_combine` when the selected namespace
does not provide ``nanmedian``.
"""

import operator

import array_api_compat

__all__ = ["nanmedian"]


def nanmedian(x, /, *, axis=0, xp=None):
    """
    Median along an axis, ignoring NaNs, using only array-API functions.

    NaNs are replaced by ``+inf`` and the result is sorted along ``axis``,
    so that the first ``n`` positions hold the non-NaN values in order no
    matter where the namespace's ``sort`` places NaNs (the standard leaves
    that implementation-defined). The number of non-NaN entries ``n`` is
    counted, and the elements at positions ``(n - 1) // 2`` and ``n // 2``
    are gathered and averaged. Slices that are entirely NaN yield NaN.

    Parameters
    ----------
    x : array
        Input array. Integer and boolean inputs are promoted to the
        namespace's default real floating dtype.
    axis : int, optional
        Axis along which to compute the median. Default is 0. Booleans,
        ``None`` and tuples of axes are not supported; numpy integer
        scalars are accepted.
    xp : array namespace, optional
        Namespace to use. Defaults to ``array_api_compat.array_namespace(x)``.

    Returns
    -------
    array
        Median of ``x`` along ``axis``, with that axis removed.

    Notes
    -----
    The implementation relies on a full sort so its cost is O(n log n) along
    ``axis``, compared with the O(n) selection used by ``numpy.nanmedian``
    or ``bottleneck.nanmedian``. Prefer a native ``nanmedian`` when the
    namespace offers one.
    """
    # bool subclasses int -- axis=True would silently mean axis 1 -- so it is
    # rejected explicitly, while operator.index accepts the numpy integer
    # scalars that isinstance(axis, int) would refuse.
    if axis is None or isinstance(axis, bool):
        raise NotImplementedError(
            "nanmedian fallback supports only a single integer axis."
        )
    try:
        axis = operator.index(axis)
    except TypeError:
        raise NotImplementedError(
            "nanmedian fallback supports only a single integer axis."
        ) from None

    if xp is None:
        xp = array_api_compat.array_namespace(x)

    ndim = x.ndim
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    axis = axis % ndim

    device = array_api_compat.device(x)

    if not xp.isdtype(x.dtype, "real floating"):
        # Promote to the namespace's default real dtype rather than hardcoding
        # float64: jax without JAX_ENABLE_X64 has no float64 and warns when one
        # is requested, which pytest's filterwarnings turns into an error.
        info = xp.__array_namespace_info__()
        x = xp.astype(x, info.default_dtypes(device=device)["real floating"])

    # Replacing NaNs with +inf keeps them past every real value regardless of
    # how the namespace orders NaNs in ``sort``. Genuine +inf entries compare
    # equal to the sentinels, so positions below ``n`` are unaffected.
    s = xp.sort(
        xp.where(xp.isnan(x), xp.asarray(xp.inf, dtype=x.dtype, device=device), x),
        axis=axis,
    )

    # Number of non-NaN values along the axis, kept broadcastable.
    n = xp.sum(xp.astype(~xp.isnan(x), xp.int32), axis=axis, keepdims=True)
    # For odd ``n`` these collapse to the same index, so the middle entry is
    # picked twice and averaged with itself -- exact, bar overflow when the
    # value exceeds half the dtype's maximum (numpy.nanmedian overflows there
    # too). For ``n == 0`` ``lo`` is -1, which matches no index; see below.
    lo = (n - 1) // 2
    hi = n // 2

    # Index along ``axis``, shaped to broadcast against ``s``.
    shape = [1] * ndim
    shape[axis] = x.shape[axis]
    idx = xp.reshape(xp.arange(x.shape[axis], device=device), tuple(shape))

    zero = xp.asarray(0, dtype=s.dtype, device=device)
    lo_val = xp.sum(xp.where(idx == lo, s, zero), axis=axis)
    hi_val = xp.sum(xp.where(idx == hi, s, zero), axis=axis)
    result = (lo_val + hi_val) / 2

    # This guard is load-bearing, not defensive: for an all-NaN slice ``n`` is
    # 0, so ``lo`` is -1 and matches no index (``lo_val`` sums to zero) while
    # ``hi`` is 0 and picks s[0], which is one of the +inf sentinels above.
    # ``result`` is therefore +inf rather than NaN, and only this ``where``
    # makes an all-NaN slice yield NaN. Do not remove it as redundant.
    nan = xp.asarray(xp.nan, dtype=s.dtype, device=device)
    return xp.where(xp.squeeze(n, axis=axis) == 0, nan, result)
