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
import numbers
import textwrap
import warnings
from functools import partial

import array_api_compat
from astropy.utils.exceptions import AstropyUserWarning

from ._blocks import _UNKNOWN_SHAPE_MESSAGE
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

    Notes
    -----
    A scalar is anything that cannot be iterated, and an integer is
    anything registered as `numbers.Integral`, so a numpy integer is
    accepted either way -- ``ccdmask(ratio, ncmed=np.int64(7))`` reaches
    here. ``3.0`` is still rejected, as `scipy.ndimage` rejects it;
    ``ccdproc._blocks._block_size`` accepts an integral float instead
    because `astropy.nddata` does, which is why the two validators share
    only this shape and not their bodies.
    """
    try:
        sizes = tuple(size)
    except TypeError:
        sizes = (size,) * ndim
    if len(sizes) != ndim:
        raise ValueError(
            f"size must have one entry per axis: got {len(sizes)} for an "
            f"array with {ndim} axes"
        )
    if any(not isinstance(k, numbers.Integral) or k < 1 for k in sizes):
        raise ValueError(f"size entries must be positive integers, got {size!r}")
    return tuple(int(k) for k in sizes)


def _cast_real(x, xp):
    """
    Promote ``x`` for a rank or a general reduction, or refuse its dtype.

    Parameters
    ----------
    x : array
        Input array.
    xp : array namespace
        Namespace to use.

    Returns
    -------
    array
        ``x``, promoted to a real floating dtype if it was integer or
        boolean.

    Raises
    ------
    TypeError
        For complex input, which `xp.sort` is not defined for and which
        `scipy.ndimage` refuses as well -- numpy would otherwise take the
        real part with a ``ComplexWarning`` and array-api-strict would
        raise from inside ``astype``. Also for a floating dtype narrower
        than ``float32``: ndimage refuses 2-D ``float16`` outright, and
        the rank arithmetic would be wrong there anyway, since a count
        above 2048 does not survive a round trip through binary16.
    """
    if xp.isdtype(x.dtype, "complex floating"):
        raise TypeError("complex input is not supported by the window filters")
    if xp.isdtype(x.dtype, "real floating") and xp.finfo(x.dtype).bits < 32:
        raise TypeError(f"{x.dtype} input is not supported by the window filters")
    return _promote_to_real(x, xp, array_api_compat.device(x))


def _cast_bool(x, xp):
    """``x`` as booleans; any non-zero value counts as true."""
    return xp.astype(x, xp.bool)


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


def _window_offsets(padded, size, shape):
    """
    Yield each window offset of ``padded`` as a view shaped ``shape``.

    Parameters
    ----------
    padded : array
        Array already padded by `_pad_windows`.
    size : tuple of int
        Window length along each axis.
    shape : tuple of int
        Shape of the output these views cover.

    Yields
    ------
    array
        One shifted slice per window position, in the row-major order of
        the offsets. Nothing depends on that order: both consumers here,
        `_stack_from_padded` and `_band_any`, are order-insensitive.
    """
    for offset in itertools.product(*(range(k) for k in size)):
        yield padded[tuple(slice(o, o + n) for o, n in zip(offset, shape, strict=True))]


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
    windows = list(_window_offsets(padded, size, shape))
    with warnings.catch_warnings():
        if array_api_compat.is_dask_namespace(xp):
            # dask warns that this stack multiplies the chunk count by
            # prod(size). It does, deliberately: that is how the window
            # reduction is expressed at all, and there is nothing a caller
            # could change in response short of using a smaller window.
            from dask.array import PerformanceWarning

            warnings.filterwarnings("ignore", category=PerformanceWarning)
        return xp.stack(windows, axis=-1)


def _band_rank(padded, size, shape, xp, *, percentile):
    """The order statistic of every window in one padded band."""
    return _nanrank(_stack_from_padded(padded, size, shape, xp), percentile, -1, xp)


def _band_reduce(padded, size, shape, xp, *, func):
    """``func`` applied to the window stack of one padded band."""
    return func(_stack_from_padded(padded, size, shape, xp), axis=-1)


def _band_any(padded, size, shape, xp):
    """
    Whether any value in each window of one padded band is true.

    Notes
    -----
    The rank filters have to hold a whole window at once to sort it; an
    ``or`` does not, so this accumulates the shifted slices into one
    output-sized array instead of stacking ``prod(size)`` of them. On a
    1024x1024 boolean image with a 21x21 window that is 0.03 s and 3 MiB
    against 2.3 s and 258 MiB for the stack.

    It still runs inside `_windowed`'s bands, and so behind the same dask
    rechunk. Accumulating over the whole padded array instead would drop
    that rechunk, and the ``prod(size)`` differently-offset slices would
    then land across dask's chunk boundaries every which way: measured at
    five million tasks and ten minutes for the image above, against three
    thousand tasks and under a second here.
    """
    result = None
    for window in _window_offsets(padded, size, shape):
        result = window if result is None else xp.logical_or(result, window)
    return result


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
    if x.ndim == 0:
        # There are no rows to band, and no ``x.shape[0]`` to report.
        return 1

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


def _windowed(x, size, reduce_band, *, cast, mode, band_rows, xp):
    """
    Apply ``reduce_band`` to the windows of ``x``, a band of rows at a time.

    This is the whole body of every public function here: they differ only
    in the validation they do first and in the two callables they pass.

    Parameters
    ----------
    x : array
        Input array, in the dtype the caller gave.
    size : int or sequence of int
        Window shape as the caller gave it; normalized here.
    reduce_band : callable
        Called as ``reduce_band(padded, size, band_shape, xp)`` with one
        padded band, and must return that band's output. It is given the
        padded band rather than the window stack so that `_band_any` can
        accumulate the window offsets instead of stacking them; the other
        three go through `_stack_from_padded` themselves. It must depend
        on nothing but the values within each window.
    cast : callable
        Called as ``cast(x, xp)`` before anything else touches ``x``.
        This is where a dtype the filters do not support is refused.
    mode : str
        Boundary mode, one of `_MODES`.
    band_rows : int or None
        Rows per band; ``None`` asks `_default_band_rows`.
    xp : array namespace or None
        Namespace to use. ``None`` resolves it from ``x``.

    Returns
    -------
    array
        Shape ``x.shape``.

    Raises
    ------
    ValueError
        If ``x``'s shape is not fully known, which on dask means chunk
        sizes that have not been computed. Without this the failure comes
        out of ``concat`` several frames down, naming a shape with a
        ``nan`` in it and no remedy.

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
    if xp is None:
        xp = array_api_compat.array_namespace(x)
    if not all(isinstance(length, int) for length in x.shape):
        raise ValueError(_UNKNOWN_SHAPE_MESSAGE)
    x = cast(x, xp)
    size = _normalize_size(size, x.ndim)

    shape = x.shape
    padded = _pad_windows(x, size, mode, xp)
    dask = array_api_compat.is_dask_namespace(xp)

    if band_rows is None:
        band_rows = _default_band_rows(x, size, xp)

    # A 0-d array has no rows to band, and no ``shape[0]`` to compare with.
    if x.ndim == 0 or band_rows >= shape[0]:
        if dask:
            padded = padded.rechunk(-1)
        return reduce_band(padded, size, shape, xp)

    overhang = size[0] - 1
    reduced = []
    for start in range(0, shape[0], band_rows):
        rows = min(band_rows, shape[0] - start)
        band = _slice_along(padded, 0, start, start + rows + overhang)
        if dask:
            # One chunk per band before it is sliced into window offsets;
            # see Notes.
            band = band.rechunk(-1)
        # Reduce inside the loop: holding the bands and reducing afterwards
        # would rebuild the whole-image stack the banding exists to avoid.
        reduced.append(reduce_band(band, size, (rows,) + shape[1:], xp))
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
        implemented, if ``size`` does not match ``x``, if ``x``'s shape is
        not fully known, or if a window is more than twice as wide as the
        axis it slides along.
    TypeError
        If ``x`` is complex or narrower than ``float32``; `scipy.ndimage`
        refuses both as well. See `_cast_real`.

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

    # ndimage's rank is int(size * percentile / 100); _nanrank takes the
    # same product, in the same order, against the count of non-NaN values
    # in each window.
    return _windowed(
        x,
        size,
        partial(_band_rank, percentile=percentile),
        cast=_cast_real,
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
    never builds the window stack: an ``or`` needs no more than one
    output-sized accumulator, so the window offsets are folded into one as
    they are taken. See `_band_any`.
    """
    return _windowed(
        x,
        size,
        _band_any,
        cast=_cast_bool,
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
    return _windowed(
        x,
        size,
        partial(_band_reduce, func=func),
        cast=_cast_real,
        mode=mode,
        band_rows=band_rows,
        xp=xp,
    )
