# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A NaN-aware median written only in terms of the array API standard.

``median``/``nanmedian`` are not part of the array API standard, so this
module provides a fallback that works on any conforming namespace
(``array-api-strict``, ``jax``, ``dask``, ``numpy``, ...). It is used by
`ccdproc.combiner.Combiner.median_combine` when the selected namespace
does not provide ``nanmedian``.
"""

import array_api_compat

__all__ = ["nanmedian"]


def nanmedian(x, /, *, axis=0, xp=None):
    """
    Median along an axis, ignoring NaNs, using only array-API functions.

    The input is sorted along ``axis`` (NaNs sort to the end per the array
    API specification), the number of non-NaN entries ``n`` is counted, and
    the elements at positions ``(n - 1) // 2`` and ``n // 2`` are gathered
    and averaged. Slices that are entirely NaN yield NaN.

    Parameters
    ----------
    x : array
        Input array. Integer inputs are promoted to ``float64``.
    axis : int, optional
        Axis along which to compute the median. Default is 0.
        ``None`` and tuples of axes are not supported.
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
    if axis is None or not isinstance(axis, int):
        raise NotImplementedError(
            "nanmedian fallback supports only a single integer axis."
        )

    xp = xp or array_api_compat.array_namespace(x)

    if not xp.isdtype(x.dtype, "real floating"):
        x = xp.astype(x, xp.float64)

    ndim = x.ndim
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    axis = axis % ndim

    device = array_api_compat.device(x)
    s = xp.sort(x, axis=axis)

    # Number of non-NaN values along the axis, kept broadcastable.
    n = xp.sum(xp.astype(~xp.isnan(x), xp.int64), axis=axis, keepdims=True)
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

    nan = xp.asarray(xp.nan, dtype=s.dtype, device=device)
    return xp.where(xp.squeeze(n, axis=axis) == 0, nan, result)
