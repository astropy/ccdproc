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

import math
import operator
import textwrap
from functools import partial

import array_api_compat

# Host-side axis handling: normalize_axis_tuple operates on python ints
# only, never on array data, and np.bool_ appears only in the guards that
# reject boolean axes, so neither ties the fallbacks to numpy.
import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

__all__ = ["median", "nanmad", "nanmean", "nanmedian", "nanstd", "nansum"]

# The ``x``/``axis``/``xp`` parameters mean the same thing for every public
# function here, so their docstring entries are written once and filled into
# each docstring's ``{params}`` placeholder by ``_fill_doc``; only the axis
# action phrase differs. Function-specific behaviour (what an all-NaN slice
# yields, NaN propagation, ...) stays inline in each Returns section.
_COMMON_PARAMS = """\
x : array
    Input array. Integer and boolean inputs are promoted to the
    namespace's default real floating dtype.
axis : int, tuple of int, list of int or None, optional
    Axis or axes along which {action}. Default is 0. ``None`` reduces
    over every axis; a tuple or list over all the listed axes at once.
    Booleans are rejected, numpy integer scalars are accepted.
xp : array namespace, optional
    Namespace to use. Defaults to ``array_api_compat.array_namespace(x)``.\
"""


def _fill_doc(**substitutions):
    """
    Fill a function docstring's ``{params}`` placeholder with
    `_COMMON_PARAMS`, applying ``substitutions`` to the template first.
    """

    def decorator(func):
        # ``python -OO`` strips docstrings; there is nothing to fill then.
        if func.__doc__:
            params = _COMMON_PARAMS.format(**substitutions)
            func.__doc__ = func.__doc__.format(
                params=textwrap.indent(params, "    ").lstrip()
            )
        return func

    return decorator


def _promote_to_real(x, xp, device):
    """
    Promote integer and boolean ``x`` to the namespace's default real
    floating dtype; a real floating ``x``, including float32, passes
    through unchanged.

    Parameters
    ----------
    x : array
        Input array.
    xp : array namespace
        Namespace to use.
    device : device
        Device on which to resolve the default real floating dtype.

    Returns
    -------
    array
        ``x``, promoted if necessary.
    """
    if xp.isdtype(x.dtype, "real floating"):
        return x
    # Promote to the namespace's default real dtype rather than hardcoding
    # float64: jax without JAX_ENABLE_X64 has no float64 and warns when one
    # is requested, which pytest's filterwarnings turns into an error.
    info = xp.__array_namespace_info__()
    return xp.astype(x, info.default_dtypes(device=device)["real floating"])


def _setup(x, axis, xp):
    """
    Normalize ``axis``, resolve the namespace and device, promote to float.

    Parameters
    ----------
    x : array
        Input array.
    axis : int, tuple of int, list of int or None
        Axis or axes along which the caller will reduce. Booleans are
        rejected -- bool subclasses int, so ``axis=True`` would silently
        mean axis 1 -- while numpy integer scalars are accepted. Negative
        values count from the last axis.
    xp : array namespace or None
        Namespace to use. ``None`` resolves it from ``x``.

    Returns
    -------
    x : array
        The input, promoted if necessary to the namespace's default real
        floating dtype, flattened when ``axis`` is ``None``, and with the
        listed axes moved to the end and merged into one when ``axis`` is
        a tuple or list.
    axis : int
        The single axis of the returned ``x`` to reduce, normalized to a
        non-negative integer.
    xp : array namespace
        The resolved namespace.
    device : device
        The device ``x`` lives on.
    restore : callable
        Maps an array shaped like the returned ``x`` back to the layout of
        the input ``x``; the identity for a single integer ``axis``.
        Reductions remove the reduced axis and never need it;
        ``ccdproc.combiner._sigma_clip_mask`` keeps the full shape and
        uses it to hand its mask back in the caller's layout.

    Raises
    ------
    TypeError
        If ``axis``, or an entry of a tuple/list ``axis``, is a bool or
        not an integer.
    ValueError
        If ``axis``, or an entry of a tuple/list ``axis``, is out of
        bounds for ``x``, or a tuple/list names an axis more than once
        (including via a negative alias).

    Notes
    -----
    ``axis`` may be a single integer, ``None`` or a tuple/list of integers.
    ``None`` flattens ``x`` so the caller reduces over everything; a tuple
    or list moves the listed axes to the end and merges them into one, so
    the caller's single-axis reduction reduces over all of them at once.
    Either way the caller only ever sees a single non-negative integer
    axis.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)
    device = array_api_compat.device(x)
    x = _promote_to_real(x, xp, device)
    ndim = x.ndim

    if axis is None:
        # Reducing over everything is the same as naming every axis.
        axis = tuple(range(ndim))

    if isinstance(axis, tuple | list):
        # normalize_axis_tuple would treat a bool as an axis: operator.index
        # turns True into 1, and on the oldest supported numpy (2.0) it
        # still accepts np.bool_ too, with only a DeprecationWarning.
        if any(isinstance(ax, bool | np.bool_) for ax in axis):
            raise TypeError("axis entries must be integers, not bool")
        # Host-side validation and normalization in one call: entries go
        # through operator.index, negatives are wrapped mod ndim,
        # out-of-bounds raises AxisError, and a duplicate (even via a
        # negative alias) raises ValueError.
        axes = normalize_axis_tuple(axis, ndim)
        # Move the reduced axes to the end and merge them into one trailing
        # axis, so that a single-axis reduction reduces over all of them at
        # once. How the merge interleaves elements is irrelevant: every
        # reduction here is order-insensitive within the reduced set.
        kept = tuple(ax for ax in range(ndim) if ax not in axes)
        order = kept + axes
        permuted_shape = tuple(x.shape[ax] for ax in order)
        # The merged length is spelled out because reshape cannot infer it
        # from -1 when a kept axis has size 0 (total size 0 is ambiguous);
        # numpy returns an empty result there, and so does this.
        merged = math.prod(permuted_shape[len(kept) :])
        x = xp.reshape(
            xp.permute_dims(x, order), permuted_shape[: len(kept)] + (merged,)
        )
        # ``inverse`` undoes ``order``; ``restore`` maps a full-shape array
        # in the permuted-merged layout back to the caller's layout by
        # un-merging (reshape) and un-permuting.
        inverse = tuple(order.index(ax) for ax in range(ndim))

        def restore(a):
            return xp.permute_dims(xp.reshape(a, permuted_shape), inverse)

        return x, len(kept), xp, device, restore

    # bool subclasses int -- axis=True would silently mean axis 1 -- and on
    # numpy 2.0 operator.index still accepts np.bool_ as well, so both are
    # rejected explicitly, while numpy integer scalars (which
    # isinstance(axis, int) would refuse) are accepted.
    if isinstance(axis, bool | np.bool_):
        raise TypeError("axis must be an integer, not bool")
    try:
        axis = operator.index(axis)
    except TypeError:
        raise TypeError(
            f"axis must be an integer, a tuple or list of integers, or None, "
            f"got {axis!r}"
        ) from None

    # normalize_axis_tuple wraps a negative axis and raises AxisError -- a
    # ValueError subclass with numpy's own message -- when it is out of
    # bounds, exactly as the tuple branch above does for entries.
    return x, normalize_axis_tuple(axis, ndim)[0], xp, device, lambda a: a


def _sum_and_count(x, axis, xp, device, *, keepdims):
    """
    NaN-free sum along ``axis`` and the number of non-NaN entries in it.

    Parameters
    ----------
    x : array
        Input array, already promoted to a real floating dtype.
    axis : int
        Axis to reduce, already normalized to a non-negative integer.
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


@_fill_doc(action="to sum")
def nansum(x, /, *, axis=0, xp=None):
    """
    Sum along an axis, ignoring NaNs, using only array-API functions.

    Parameters
    ----------
    {params}

    Returns
    -------
    array
        Sum of ``x`` along ``axis``, with the reduced axes removed (0-d
        when ``axis`` is ``None``). Slices that are entirely NaN sum to
        zero, matching `numpy.nansum`.
    """
    x, axis, xp, device, _ = _setup(x, axis, xp)
    total, _ = _sum_and_count(x, axis, xp, device, keepdims=False)
    return total


@_fill_doc(action="to average")
def nanmean(x, /, *, axis=0, xp=None):
    """
    Mean along an axis, ignoring NaNs, using only array-API functions.

    Parameters
    ----------
    {params}

    Returns
    -------
    array
        Mean of ``x`` along ``axis``, with the reduced axes removed (0-d
        when ``axis`` is ``None``). Slices that are entirely NaN yield
        NaN silently, matching ``bottleneck.nanmean``
        (the numpy-backend default); `numpy.nanmean` warns here, but a fully
        masked pixel is a routine input for the combiner, not an anomaly.
    """
    x, axis, xp, device, _ = _setup(x, axis, xp)
    total, count = _sum_and_count(x, axis, xp, device, keepdims=False)
    return _safe_divide(total, count, xp, device)


@_fill_doc(action="to compute the deviation")
def nanstd(x, /, *, axis=0, xp=None):
    """
    Standard deviation along an axis, ignoring NaNs, via array-API functions.

    The deviation is the population one (``ddof=0``), which is what
    `numpy.nanstd` and ``bottleneck.nanstd`` compute by default and what
    `ccdproc.combiner.Combiner` expects.

    Parameters
    ----------
    {params}

    Returns
    -------
    array
        Standard deviation of ``x`` along ``axis``, with the reduced axes
        removed (0-d when ``axis`` is ``None``).
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
    x, axis, xp, device, _ = _setup(x, axis, xp)

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


@_fill_doc(action="to compute the median")
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
    {params}

    Returns
    -------
    array
        Median of ``x`` along ``axis``, with the reduced axes removed (0-d
        when ``axis`` is ``None``). Slices that are entirely NaN yield NaN
        silently, matching
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
    x, axis, xp, device, _ = _setup(x, axis, xp)
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


@_fill_doc(action="to compute the median")
def median(x, /, *, axis=0, xp=None):
    """
    Median along an axis, using only array-API functions.

    Parameters
    ----------
    {params}

    Returns
    -------
    array
        Median of ``x`` along ``axis``, with the reduced axes removed (0-d
        when ``axis`` is ``None``). Slices that contain any NaN yield NaN,
        matching `numpy.median`; this is the
        difference from `nanmedian`, which ignores NaNs entirely.

    Notes
    -----
    On input with no NaNs this is exactly `nanmedian` -- the same sorting
    and index-picking algorithm is used, so the values agree bit for bit.
    The two differ only in how a NaN in the reduced slice is handled: this
    function propagates it to the result, matching `numpy.median`, while
    `nanmedian` ignores it. That NaN-propagating behaviour is restored here
    with a final `where` over whether any NaN is present along ``axis``,
    since `nanmedian` alone would silently drop NaNs instead.
    """
    x, axis, xp, device, _ = _setup(x, axis, xp)
    nan = xp.asarray(xp.nan, dtype=x.dtype, device=device)
    return xp.where(xp.any(xp.isnan(x), axis=axis), nan, nanmedian(x, axis=axis, xp=xp))


@_fill_doc(action="to compute the deviation")
def nanmad(x, /, *, axis=0, xp=None, median=None):
    """
    Median absolute deviation along ``axis``, ignoring NaNs.

    Parameters
    ----------
    {params}
    median : callable, optional
        Reduction used for both medians, called as ``median(x, axis=axis)``,
        always with a single integer ``axis``: a ``None`` or tuple/list
        ``axis`` has already been flattened or merged away by `_setup`.
        Default is `nanmedian`. A keyword rather than a module-level tier
        (as `ccdproc.combiner._default_median` provides) so this module has
        no dependency on `ccdproc.combiner`.

    Returns
    -------
    array
        ``median(|x - median(x)|)`` along ``axis``, with the reduced axes
        removed (0-d when ``axis`` is ``None``).
        Unscaled: multiply by ``1.482602218505602`` for an estimate of the
        standard deviation, as `astropy.stats.mad_std` does.
    """
    x, axis, xp, device, _ = _setup(x, axis, xp)
    if median is None:
        median = partial(nanmedian, xp=xp)
    center = xp.expand_dims(median(x, axis=axis), axis=axis)
    return median(xp.abs(x - center), axis=axis)
