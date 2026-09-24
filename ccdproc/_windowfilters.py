# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Local window (moving-window) filters written only in terms of the array API.

`scipy.ndimage`'s ``median_filter``, ``percentile_filter``,
``maximum_filter`` and ``generic_filter`` are numpy-only: handing one of
them a jax, dask, CuPy or array-api-strict array either copies it silently
to the host or fails outright. This module provides the four window
reductions `ccdproc.core` needs -- an order statistic, a median, a boolean
"any value in the window", and a general reduction -- for any conforming
namespace, so that `ccdproc.core` can keep calling ndimage for numpy input
and use these everywhere else.

The implementation is deliberately literal: the array is padded once, then
each window offset is taken as a shifted slice and the offsets are stacked
into a new trailing axis, which turns a window reduction into an ordinary
reduction over that axis. That is what makes it expressible in the standard
at all -- there is no ``as_strided``, no ``sliding_window_view``, not even a
``pad`` -- and it is also what makes it expensive: the stack holds
``prod(size)`` copies of the input, and the rank filters sort it, costing
O(k**2 log k**2) per pixel for a k-by-k window where ndimage's selection
costs O(k**2).

Two divergences from ndimage are deliberate, and documented on the
functions themselves: integer input is promoted to a floating dtype, and
only the ``'reflect'`` and ``'nearest'`` boundary modes are implemented.

One dask ``PerformanceWarning`` is deliberately swallowed, in
`_stack_from_padded`; see there.
"""

import itertools
import math
import textwrap
import warnings

import array_api_compat
from astropy.utils.exceptions import AstropyUserWarning

from ._nanfuncs import _nanrank, _promote_to_real

__all__ = ["window_any", "window_median", "window_rank", "window_reduce"]

# Boundary modes implemented here, in ndimage's naming. ndimage's 'mirror',
# 'constant' and 'wrap' are each a one-line variant of the two below, but
# nothing in ccdproc asks for them and an unimplemented mode has to raise
# rather than quietly behave like 'reflect'.
_MODES = ("reflect", "nearest")

# Memory budget for one band of the window stack, in bytes. The stack for a
# whole 2048x2048 float64 image is 3.8 GiB at an 11x11 window, and the sort
# in ``window_rank`` doubles that; banding the output into groups of rows
# holds the peak near this figure instead, at no cost in accuracy -- see
# ``_windowed``.
_BAND_BUDGET_BYTES = 256 * 1024 * 1024

# The ``x``/``size``/``mode``/``band_rows``/``xp`` parameters mean the same
# thing for every function here, so their docstring entries are written once
# and filled into each docstring's ``{params}`` placeholder by
# ``_window_doc``. ``{extra}`` is where a function's own extra parameter
# goes, so that the rendered Parameters section keeps signature order.
_COMMON_PARAMS = """\
x : array
    Input array of any rank.
{dtype}size : int or sequence of int
    Window shape: an integer uses the same length along every axis, a
    sequence gives one length per axis of ``x``. The window for index
    ``i`` along an axis whose window length is ``k`` spans ``i - k // 2``
    through ``i + (k - 1 - k // 2)``, so an even ``k`` sits asymmetrically
    about ``i`` -- `scipy.ndimage`'s placement, reproduced here.
{extra}mode : str, optional
    How ``x`` is extended past its edges, in `scipy.ndimage`'s naming.
    ``'reflect'``, the default here and there, mirrors about the edge
    sample (``d c b a | a b c d | d c b a``); ``'nearest'`` repeats it
    (``a a a a | a b c d | d d d d``). No other mode is implemented.
band_rows : int, optional
    Number of output rows whose windows are stacked at a time. The default
    is the largest number that keeps a band's stack within a {budget} MiB
    budget. Every value gives the same result -- this trades peak memory
    against the number of passes -- so it is here for that trade and for
    the test that pins it, and no `ccdproc.core` function exposes it. On
    dask a band is also what is collapsed into a single chunk; see
    `_windowed`.
xp : array namespace, optional
    Namespace to use. Defaults to ``array_api_compat.array_namespace(x)``.\
"""

_PROMOTED_DTYPE = """\
    Integer and boolean input is promoted to the namespace's default
    real floating dtype, which `scipy.ndimage` does not do.
"""

_PERCENTILE_PARAM = """\
percentile : float
    Percentile to take, in ``[0, 100]``.
"""

_FUNC_PARAM = """\
func : callable
    Called as ``func(stack, axis=-1)`` with an array whose trailing axis
    holds each pixel's window values, and must reduce that axis away. It
    sees a whole band of pixels at once rather than one window, so
    `scipy.ndimage.generic_filter`'s scalar per-window callables do not
    fit; an array-API reduction such as `ccdproc.core.sigma_func` does.
"""


def _window_doc(dtype=_PROMOTED_DTYPE, extra=""):
    """Fill one function's ``{params}`` placeholder from `_COMMON_PARAMS`."""
    params = _COMMON_PARAMS.format(
        dtype=dtype,
        extra=extra,
        budget=_BAND_BUDGET_BYTES // 1024**2,
    )
    indented = textwrap.indent(params, "    ").lstrip()

    def decorator(func):
        # ``python -OO`` strips docstrings; there is nothing to fill then.
        if func.__doc__:
            func.__doc__ = func.__doc__.format(params=indented)
        return func

    return decorator


def _normalize_size(size, ndim):
    """
    Turn ``size`` into a tuple holding one positive integer per axis.

    Parameters
    ----------
    size : int or sequence of int
        Window shape as the caller gave it.
    ndim : int
        Number of axes of the array being filtered.

    Returns
    -------
    tuple of int
        Window length along each axis.

    Raises
    ------
    ValueError
        If a sequence ``size`` does not have one entry per axis, or an
        entry is not a positive integer.
    """
    if isinstance(size, int):
        sizes = (size,) * ndim
    else:
        sizes = tuple(size)
        if len(sizes) != ndim:
            raise ValueError(
                f"size must have one entry per axis: got {len(sizes)} for an "
                f"array with {ndim} axes"
            )
    if any(not isinstance(k, int) or k < 1 for k in sizes):
        raise ValueError(f"size entries must be positive integers, got {size!r}")
    return sizes


def _slice_along(x, axis, start, stop):
    """
    ``x`` sliced to ``start:stop`` along ``axis``, left whole elsewhere.

    Parameters
    ----------
    x : array
        Array to slice.
    axis : int
        Non-negative axis to slice along.
    start, stop : int or None
        Bounds of the slice, as for ordinary Python slicing.

    Returns
    -------
    array
        The slice.
    """
    index = [slice(None)] * x.ndim
    index[axis] = slice(start, stop)
    return x[tuple(index)]


def _pad_axis(x, before, after, axis, mode, xp):
    """
    Extend ``x`` along ``axis`` past its edges, in `scipy.ndimage`'s naming.

    ``'reflect'`` mirrors about the edge sample, ``d c b a | a b c d | d c
    b a`` (numpy calls that ``'symmetric'``); ``'nearest'`` repeats it,
    ``a a a a | a b c d | d d d d``.

    Parameters
    ----------
    x : array
        Array to pad.
    before, after : int
        Number of elements to add before the start and after the end.
    axis : int
        Non-negative axis to pad.
    mode : str
        One of `_MODES`. Anything that is not ``'reflect'`` is treated as
        ``'nearest'``; `_pad_windows` is what rejects an unknown mode.
    xp : array namespace
        Namespace to use.

    Returns
    -------
    array
        ``x`` with ``before + after`` extra elements along ``axis``.

    Raises
    ------
    ValueError
        For ``'reflect'``, if more elements are asked for than the axis
        holds. ndimage keeps reflecting back and forth in that case;
        matching that is not worth the code, since a window more than
        twice as wide as the image is not a filter anyone means to apply.
        For ``'nearest'``, if ``axis`` is empty and any padding was asked
        for: there is no edge sample to repeat.
    """
    length = x.shape[axis]
    if mode == "reflect":
        if before > length or after > length:
            raise ValueError(
                f"window is too large for axis {axis}, whose length is {length}: "
                f"reflecting needs {max(before, after)} elements of padding, which "
                f"would have to re-reflect"
            )
        head = [xp.flip(_slice_along(x, axis, 0, before), axis=axis)] if before else []
        tail = (
            [xp.flip(_slice_along(x, axis, length - after, None), axis=axis)]
            if after
            else []
        )
    else:
        if length == 0 and (before or after):
            raise ValueError(f"cannot pad axis {axis}, which is empty")
        # ``concat`` of repeated edge slices rather than ``repeat``, which
        # only entered the standard in 2023.12 and which array-api-compat's
        # dask wrapper does not provide.
        head = [_slice_along(x, axis, 0, 1)] * before
        tail = [_slice_along(x, axis, length - 1, None)] * after

    pieces = [*head, x, *tail]
    return xp.concat(pieces, axis=axis) if len(pieces) > 1 else x


def _pad_windows(x, size, mode, xp):
    """
    Pad every axis of ``x`` by the amount its window overhangs the edges.

    Parameters
    ----------
    x : array
        Array to pad.
    size : tuple of int
        Window length along each axis, as `_normalize_size` returns.
    mode : str
        One of `_MODES`.
    xp : array namespace
        Namespace to use.

    Returns
    -------
    array
        ``x`` grown by ``k - 1`` along each axis, so that the window of
        output index ``i`` is the padded slice ``i : i + k``.

    Raises
    ------
    ValueError
        If ``mode`` is not one of `_MODES`.
    """
    # This check is the only thing standing between a typo and silently
    # wrong padding, now that `_pad_axis` treats anything that is not
    # ``'reflect'`` as ``'nearest'``.
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    for axis, k in enumerate(size):
        # ndimage places an even window one element above the index, so the
        # overhangs are k // 2 before and k - 1 - k // 2 after.
        x = _pad_axis(x, k // 2, k - 1 - k // 2, axis, mode, xp)
    return x


def _stack_from_padded(padded, size, shape, xp):
    """
    Stack every window offset of ``padded`` into a new trailing axis.

    Parameters
    ----------
    padded : array
        Array already padded by `_pad_windows`, that is, ``k - 1`` longer
        than the wanted output along each axis.
    size : tuple of int
        Window length along each axis.
    shape : tuple of int
        Shape of the output this covers.
    xp : array namespace
        Namespace to use.

    Returns
    -------
    array
        Shape ``shape + (prod(size),)``; entry ``[..., w]`` of a pixel
        holds one of the values in that pixel's window. The order along
        the new axis is the row-major order of the offsets, which nothing
        depends on: every reduction applied to it is order-insensitive.
    """
    offsets = itertools.product(*(range(k) for k in size))
    windows = [
        padded[tuple(slice(o, o + n) for o, n in zip(offset, shape, strict=True))]
        for offset in offsets
    ]
    with warnings.catch_warnings():
        if array_api_compat.is_dask_namespace(xp):
            # dask warns that this stack multiplies the chunk count by
            # prod(size). It does, deliberately: that is how the window
            # reduction is expressed at all, and there is nothing a caller
            # could change in response short of using a smaller window.
            # Matched by message rather than by class so that dask need not
            # be imported here.
            warnings.filterwarnings("ignore", message="Increasing number of chunks")
        return xp.stack(windows, axis=-1)


def _itemsize(dtype, xp):
    """
    Bytes per element of ``dtype``, for the band-size estimate.

    Notes
    -----
    The array API exposes no itemsize, but ``finfo``/``iinfo`` report the
    width in bits, which is the same thing for every dtype these filters
    can be handed. ``bool`` is the one dtype neither of them covers.
    """
    if dtype == xp.bool:
        return 1
    info = xp.finfo if xp.isdtype(dtype, "real floating") else xp.iinfo
    return info(dtype).bits // 8


def _default_band_rows(x, size, xp):
    """
    Largest number of output rows whose stack fits `_BAND_BUDGET_BYTES`.

    Parameters
    ----------
    x : array
        Array about to be filtered.
    size : tuple of int
        Window length along each axis.
    xp : array namespace
        Namespace to use.

    Returns
    -------
    int
        At least 1.

    Warns
    -----
    AstropyUserWarning
        If even a single row's stack exceeds the budget, the one case
        banding cannot fix. There is no size cap: the filter is still
        computed, the caller is only told that it will need more memory
        than the budget allows for.
    """
    row_bytes = math.prod(x.shape[1:]) * _itemsize(x.dtype, xp) * math.prod(size)
    if row_bytes == 0:
        # A zero-width image has nothing to band; one band covers it.
        return max(x.shape[0], 1)

    band_rows = _BAND_BUDGET_BYTES // row_bytes
    if band_rows < 1:
        warnings.warn(
            f"one row of the window stack for a "
            f"{'x'.join(str(k) for k in size)} window needs "
            f"{row_bytes / 1024**2:.0f} MiB, more than the "
            f"{_BAND_BUDGET_BYTES // 1024**2} MiB this filter budgets for one "
            f"band; the filter is still computed, but it cannot be split any "
            f"finer",
            AstropyUserWarning,
            stacklevel=2,
        )
        return 1
    return band_rows


def _windowed(x, size, reduction, *, mode, band_rows, xp):
    """
    Apply ``reduction`` to the window stack of ``x``, a band of rows at a
    time.

    Parameters
    ----------
    x : array
        Input array, already in whatever dtype the reduction wants.
    size : tuple of int
        Window length along each axis, as `_normalize_size` returns.
    reduction : callable
        Called as ``reduction(stack)`` with an array shaped
        ``band_shape + (prod(size),)``, and must reduce that trailing axis
        away. It must depend on nothing but the values within each window.
    mode : str
        Boundary mode, one of `_MODES`.
    band_rows : int or None
        Rows per band; ``None`` asks `_default_band_rows`.
    xp : array namespace
        Namespace to use.

    Returns
    -------
    array
        Shape ``x.shape``.

    Notes
    -----
    Banding is exact, not an approximation. The array is padded once, up
    front, and a band is a slice of that *padded* array which is
    ``size[0] - 1`` rows taller than the output rows it produces, so every
    output pixel sees the values it would have seen in the whole-image
    stack. Consecutive bands therefore overlap in input rows and never in
    output rows, and concatenating their outputs reassembles the image
    exactly. ``test_windowfilters.py`` pins that.

    On dask a band is additionally collapsed into a single chunk before it
    is sliced. The ``prod(size)`` window slices are each offset by a
    different amount, so against a chunked band every one of them lands
    across the chunk boundaries differently and dask has to realign them
    all: a 21x21 window over a 100x100 image builds a graph of 585,000
    tasks that way, and 912 this way -- three orders of magnitude, and the
    difference between a minute and a tenth of a second. The band budget is
    what makes collapsing safe, since it is the band, not the image, that
    has to fit in memory.
    """
    shape = x.shape
    padded = _pad_windows(x, size, mode, xp)
    dask = array_api_compat.is_dask_namespace(xp)

    if band_rows is None:
        band_rows = _default_band_rows(x, size, xp)

    if band_rows >= shape[0]:
        if dask:
            padded = padded.rechunk(-1)
        return reduction(_stack_from_padded(padded, size, shape, xp))

    overhang = size[0] - 1
    reduced = []
    for start in range(0, shape[0], band_rows):
        rows = min(band_rows, shape[0] - start)
        band = _slice_along(padded, 0, start, start + rows + overhang)
        if dask:
            # One chunk per band before it is sliced into window offsets;
            # see Notes.
            band = band.rechunk(-1)
        # Reduce inside the loop: holding the band stacks and reducing
        # afterwards would rebuild the whole-image stack the banding exists
        # to avoid.
        reduced.append(
            reduction(_stack_from_padded(band, size, (rows,) + shape[1:], xp))
        )
    return xp.concat(reduced, axis=0)


@_window_doc(extra=_PERCENTILE_PARAM)
def window_rank(x, size, percentile, *, mode="reflect", band_rows=None, xp=None):
    """
    Order statistic over a moving window, using only array-API functions.

    This is the array-API counterpart of
    `scipy.ndimage.percentile_filter`, and reproduces it exactly on finite
    input: both take the element at rank
    ``int(prod(size) * percentile / 100)`` of the sorted window.

    Parameters
    ----------
    {params}

    Returns
    -------
    array
        Same shape as ``x``, in ``x``'s namespace and on its device, in a
        real floating dtype.

    Raises
    ------
    ValueError
        If ``percentile`` is outside ``[0, 100]``, if ``mode`` is not
        implemented, if ``size`` does not match ``x``, or if a window is
        more than twice as wide as the axis it slides along.

    Notes
    -----
    NaNs are excluded rather than sorted to one end: the rank is taken
    among the ``n`` non-NaN values of each window, and a window holding no
    non-NaN value at all yields NaN. That is what lets a caller exclude
    masked pixels by substituting NaN for them. ndimage instead sorts NaNs
    in with the rest of the window, so the two agree only on all-finite
    input -- which is the only input `ccdproc.core` sends down its numpy,
    ndimage-backed path.

    For a window with an even number of valid values ``percentile=50``
    gives the upper-middle value, ndimage's convention, and not the
    average of the middle two that `numpy.nanmedian` gives.

    The rank is found by sorting the whole window stack, so the cost is
    O(k**2 log k**2) per pixel for a k-by-k window, against ndimage's
    O(k**2) selection.
    """
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile!r}")

    if xp is None:
        xp = array_api_compat.array_namespace(x)
    x = _promote_to_real(x, xp, array_api_compat.device(x))
    size = _normalize_size(size, x.ndim)

    # ndimage's rank is int(size * percentile / 100); _nanrank takes the
    # same product, in the same order, against the count of non-NaN values
    # in each window.
    return _windowed(
        x,
        size,
        lambda stack: _nanrank(stack, percentile, -1, xp),
        mode=mode,
        band_rows=band_rows,
        xp=xp,
    )


@_window_doc()
def window_median(x, size, *, mode="reflect", band_rows=None, xp=None):
    """
    Median over a moving window, using only array-API functions.

    This is the array-API counterpart of `scipy.ndimage.median_filter`,
    and reproduces it exactly on finite input.

    Parameters
    ----------
    {params}

    Returns
    -------
    array
        Same shape as ``x``, in ``x``'s namespace and on its device, in a
        real floating dtype.

    Notes
    -----
    Exactly `window_rank` at ``percentile=50``; see there for how NaNs are
    treated, for the even-window convention (the upper-middle value, as
    ndimage takes, not the average of the middle two) and for the cost.
    """
    return window_rank(x, size, 50.0, mode=mode, band_rows=band_rows, xp=xp)


@_window_doc(dtype="    Cast to boolean; any non-zero value counts as true.\n")
def window_any(x, size, *, mode="reflect", band_rows=None, xp=None):
    """
    Whether any value in a moving window is true, via array-API functions.

    On boolean input this is `scipy.ndimage.maximum_filter`, which is how
    `ccdproc.core.cosmicray_median` grows a cosmic-ray flag into its
    neighbourhood.

    Parameters
    ----------
    {params}

    Returns
    -------
    array
        Boolean, same shape as ``x``, in ``x``'s namespace and on its
        device.

    Notes
    -----
    Alone among the filters here this one does no dtype promotion, and it
    reduces with ``any`` rather than by sorting, so it costs a single pass
    over the window stack.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)
    x = xp.astype(x, xp.bool)
    size = _normalize_size(size, x.ndim)

    return _windowed(
        x,
        size,
        lambda stack: xp.any(stack, axis=-1),
        mode=mode,
        band_rows=band_rows,
        xp=xp,
    )


@_window_doc(extra=_FUNC_PARAM)
def window_reduce(x, size, func, *, mode="reflect", band_rows=None, xp=None):
    """
    Reduce each moving window with ``func``, using only array-API functions.

    This is the array-API counterpart of `scipy.ndimage.generic_filter`
    for the case where the per-window callable is a vectorized reduction
    rather than a scalar function of one window at a time.

    Parameters
    ----------
    {params}

    Returns
    -------
    array
        Same shape as ``x``, in ``x``'s namespace and on its device, in
        whatever dtype ``func`` returns.

    Notes
    -----
    Handing ``func`` a whole band at once is the point: it can then be an
    ordinary array-API reduction, where `generic_filter` calls its callable
    once per pixel, in Python, and is correspondingly slow.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)
    x = _promote_to_real(x, xp, array_api_compat.device(x))
    size = _normalize_size(size, x.ndim)

    return _windowed(
        x,
        size,
        lambda stack: func(stack, axis=-1),
        mode=mode,
        band_rows=band_rows,
        xp=xp,
    )
