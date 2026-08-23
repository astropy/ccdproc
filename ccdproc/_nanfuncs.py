# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
NaN-aware sum/mean/standard deviation written only in terms of the array API.

``nansum``/``nanmean``/``nanstd`` are not part of the array API standard, so
this module provides fallbacks that work on any conforming namespace
(``array-api-strict``, ``jax``, ``dask``, ``numpy``, ...). They are used by
`ccdproc.combiner.Combiner` when the selected namespace does not provide the
native versions. The NaN-aware median lives next door in
`ccdproc._nanmedian`, which needs a sort and so is a good deal more
involved than anything here.

All three functions promote integer and boolean input to the namespace's
default real floating dtype, which is where they part company with
``numpy.nansum``: numpy preserves an integer dtype, these do not. Every
caller in `ccdproc` combines floating point image data, and the promotion
keeps the three functions consistent with each other and with
`ccdproc._nanmedian.nanmedian`.
"""

import operator

import array_api_compat

__all__ = ["nanmean", "nanstd", "nansum"]


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
