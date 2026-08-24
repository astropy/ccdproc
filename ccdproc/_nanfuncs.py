# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
NaN-aware sum/mean/standard deviation/median, and a NaN-propagating median,
written only in terms of the array API.

``nansum``/``nanmean``/``nanstd``/``nanmedian`` are not part of the array
API standard, so this module provides fallbacks that work on any conforming
namespace (``array-api-strict``, ``jax``, ``dask``, ``numpy``, ...). They
are used by `ccdproc.combiner.Combiner` when the selected namespace does
not provide the native versions. ``median`` is not part of the standard
either, and this module also provides a fallback for it, built on
``nanmedian``, used by `ccdproc.core.subtract_overscan` when the selected
namespace does not provide the native version.

All five functions promote integer and boolean input to the namespace's
default real floating dtype, which is where they part company with
``numpy.nansum``: numpy preserves an integer dtype, these do not. Every
caller in `ccdproc` combines floating point image data, and the promotion
keeps the five functions consistent with each other.
"""

import operator

import array_api_compat

__all__ = ["median", "nanmean", "nanmedian", "nanstd", "nansum"]


def _setup(x, axis, xp):
    """
    Validate ``axis``, resolve the namespace and device, promote to float.

    Parameters
    ----------
    x : array
        Input array.
    axis : int
        Axis along which the caller will reduce. Booleans, ``None`` and
        tuples of axes are rejected; anything else goes through
        `operator.index`, so numpy integer scalars are accepted.
    xp : array namespace or None
        Namespace to use. ``None`` resolves it from ``x``.

    Returns
    -------
    x : array
        The input, promoted if necessary to the namespace's default real
        floating dtype.
    axis : int
        The axis, normalised to a non-negative integer.
    xp : array namespace
        The resolved namespace.
    device : device
        The device ``x`` lives on.

    Raises
    ------
    NotImplementedError
        If ``axis`` is not a single integer.
    ValueError
        If ``axis`` is out of bounds for ``x``.
    """
    # bool subclasses int -- axis=True would silently mean axis 1 -- so it is
    # rejected explicitly, while operator.index accepts the numpy integer
    # scalars that isinstance(axis, int) would refuse.
    if axis is None or isinstance(axis, bool):
        raise NotImplementedError(
            "NaN-aware reduction fallbacks support only a single integer axis."
        )
    try:
        axis = operator.index(axis)
    except TypeError:
        raise NotImplementedError(
            "NaN-aware reduction fallbacks support only a single integer axis."
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

    return x, axis, xp, device


def _sum_and_count(x, axis, xp, device, *, keepdims):
    """
    NaN-free sum along ``axis`` and the number of non-NaN entries in it.

    Parameters
    ----------
    x : array
        Input array, already promoted to a real floating dtype.
    axis : int
        Axis to reduce, already normalised to a non-negative integer.
    xp : array namespace
        Namespace to use.
    device : device
        Device on which to build scalar constants.
    keepdims : bool
        Whether the reduced axis is kept, with size one, in both outputs.

    Returns
    -------
    total : array
        Sum of the non-NaN entries along ``axis``.
    count : array
        Number of non-NaN entries along ``axis``, in the dtype of ``x`` so
        that it can divide ``total`` without triggering a promotion the
        namespace might not allow. A float32 ``count`` is exact only up to
        2**24 non-NaN entries per slice -- unreachable when reducing over
        the image axis of a combiner.
    """
    isnan = xp.isnan(x)
    zero = xp.asarray(0, dtype=x.dtype, device=device)
    total = xp.sum(xp.where(isnan, zero, x), axis=axis, keepdims=keepdims)
    count = xp.sum(xp.astype(~isnan, x.dtype), axis=axis, keepdims=keepdims)
    return total, count


def _safe_divide(total, count, xp, device):
    """
    ``total / count``, yielding NaN wherever ``count`` is zero.

    Parameters
    ----------
    total : array
        Numerator.
    count : array
        Denominator, in the same dtype as ``total``. Zero entries mark
        slices that had no non-NaN values.
    xp : array namespace
        Namespace to use.
    device : device
        Device on which to build scalar constants.

    Returns
    -------
    array
        ``total / count`` where ``count`` is non-zero, NaN elsewhere.

    Notes
    -----
    The zeros are swapped for ones *before* the division rather than being
    patched up afterwards: 0/0 on a numpy-backed namespace emits an "invalid
    value encountered" `RuntimeWarning`, and a filter around this expression
    would not survive a lazy backend -- under dask the warning only surfaces
    at compute time, outside any ``catch_warnings`` block placed here.
    """
    one = xp.asarray(1, dtype=total.dtype, device=device)
    nan = xp.asarray(xp.nan, dtype=total.dtype, device=device)
    quotient = total / xp.where(count == 0, one, count)
    return xp.where(count == 0, nan, quotient)


def nansum(x, /, *, axis=0, xp=None):
    """
    Sum along an axis, ignoring NaNs, using only array-API functions.

    Parameters
    ----------
    x : array
        Input array. Integer and boolean inputs are promoted to the
        namespace's default real floating dtype.
    axis : int, optional
        Axis along which to sum. Default is 0. ``None`` and tuples of axes
        are not supported.
    xp : array namespace, optional
        Namespace to use. Defaults to ``array_api_compat.array_namespace(x)``.

    Returns
    -------
    array
        Sum of ``x`` along ``axis``, with that axis removed. Slices that are
        entirely NaN sum to zero, matching `numpy.nansum`.
    """
    x, axis, xp, device = _setup(x, axis, xp)
    total, _ = _sum_and_count(x, axis, xp, device, keepdims=False)
    return total


def nanmean(x, /, *, axis=0, xp=None):
    """
    Mean along an axis, ignoring NaNs, using only array-API functions.

    Parameters
    ----------
    x : array
        Input array. Integer and boolean inputs are promoted to the
        namespace's default real floating dtype.
    axis : int, optional
        Axis along which to average. Default is 0. ``None`` and tuples of
        axes are not supported.
    xp : array namespace, optional
        Namespace to use. Defaults to ``array_api_compat.array_namespace(x)``.

    Returns
    -------
    array
        Mean of ``x`` along ``axis``, with that axis removed. Slices that
        are entirely NaN yield NaN silently, matching ``bottleneck.nanmean``
        (the numpy-backend default); `numpy.nanmean` warns here, but a fully
        masked pixel is a routine input for the combiner, not an anomaly.
    """
    x, axis, xp, device = _setup(x, axis, xp)
    total, count = _sum_and_count(x, axis, xp, device, keepdims=False)
    return _safe_divide(total, count, xp, device)


def nanstd(x, /, *, axis=0, xp=None):
    """
    Standard deviation along an axis, ignoring NaNs, via array-API functions.

    The deviation is the population one (``ddof=0``), which is what
    `numpy.nanstd` and ``bottleneck.nanstd`` compute by default and what
    `ccdproc.combiner.Combiner` expects.

    Parameters
    ----------
    x : array
        Input array. Integer and boolean inputs are promoted to the
        namespace's default real floating dtype.
    axis : int, optional
        Axis along which to compute the deviation. Default is 0. ``None``
        and tuples of axes are not supported.
    xp : array namespace, optional
        Namespace to use. Defaults to ``array_api_compat.array_namespace(x)``.

    Returns
    -------
    array
        Standard deviation of ``x`` along ``axis``, with that axis removed.
        Slices that are entirely NaN yield NaN silently, matching
        ``bottleneck.nanstd`` (the numpy-backend default); `numpy.nanstd`
        warns here, but a fully masked pixel is a routine input for the
        combiner, not an anomaly. A slice with a single non-NaN entry
        yields zero.

    Notes
    -----
    This is the two-pass form: the mean is computed first and then subtracted,
    rather than accumulating ``sum(x**2) - sum(x)**2 / n``. The extra pass
    costs one more reduction but avoids the catastrophic cancellation the
    single-pass form suffers when the values are large relative to their
    spread, which is not unusual for CCD counts.
    """
    x, axis, xp, device = _setup(x, axis, xp)

    isnan = xp.isnan(x)
    zero = xp.asarray(0, dtype=x.dtype, device=device)

    total, count = _sum_and_count(x, axis, xp, device, keepdims=True)
    mean = _safe_divide(total, count, xp, device)

    # NaN entries contribute a NaN deviation, so they are zeroed out again
    # after the subtraction; an all-NaN slice has a NaN ``mean``, but its
    # deviations are zeroed here too and the NaN is restored by the final
    # division against a zero ``count``.
    deviation = xp.where(isnan, zero, x - mean)
    variance = _safe_divide(
        xp.sum(deviation * deviation, axis=axis, keepdims=True), count, xp, device
    )
    return xp.squeeze(xp.sqrt(variance), axis=axis)


def nanmedian(x, /, *, axis=0, xp=None):
    """
    Median along an axis, ignoring NaNs, using only array-API functions.

    NaNs are replaced by ``+inf`` and the result is sorted along ``axis``,
    so that the first ``n`` positions hold the non-NaN values in order no
    matter where the namespace's ``sort`` places NaNs (the standard leaves
    that implementation-defined). The number of non-NaN entries ``n`` is
    counted, and the elements at positions ``(n - 1) // 2`` and ``n // 2``
    are gathered and averaged.

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
        Median of ``x`` along ``axis``, with that axis removed. Slices that
        are entirely NaN yield NaN silently, matching
        ``bottleneck.nanmedian`` (the numpy-backend default);
        `numpy.nanmedian` warns here, but a fully masked pixel is a routine
        input for the combiner, not an anomaly.

    Notes
    -----
    The implementation relies on a full sort so its cost is O(n log n) along
    ``axis``, compared with the O(n) selection used by ``numpy.nanmedian``
    or ``bottleneck.nanmedian``. Prefer a native ``nanmedian`` when the
    namespace offers one.
    """
    x, axis, xp, device = _setup(x, axis, xp)
    ndim = x.ndim

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


def median(x, /, *, axis=0, xp=None):
    """
    Median along an axis, using only array-API functions.

    On input with no NaNs this is exactly `nanmedian` -- the same sorting
    and index-picking algorithm is used, so the values agree bit for bit.
    The two differ only in how a NaN in the reduced slice is handled: this
    function propagates it to the result, matching `numpy.median`, while
    `nanmedian` ignores it. That NaN-propagating behaviour is restored here
    with a final `where` over whether any NaN is present along ``axis``,
    since `nanmedian` alone would silently drop NaNs instead.

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
        Median of ``x`` along ``axis``, with that axis removed. Slices that
        contain any NaN yield NaN, matching `numpy.median`; this is the
        difference from `nanmedian`, which ignores NaNs entirely.
    """
    x, axis, xp, device = _setup(x, axis, xp)
    nan = xp.asarray(xp.nan, dtype=x.dtype, device=device)
    return xp.where(xp.any(xp.isnan(x), axis=axis), nan, nanmedian(x, axis=axis, xp=xp))
