# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the array-API-native window filters in ``ccdproc._windowfilters``.

The filters exist to reproduce four `scipy.ndimage` filters on namespaces
ndimage cannot handle, so most of what is pinned here is agreement with
ndimage, run against whichever backend ``CCDPROC_ARRAY_LIBRARY`` selects.
The reference is always computed with numpy on host data; only the input
under test is converted into the active namespace.
"""

import array_api_compat
import array_api_extra as xpx
import numpy as np
import pytest
from astropy.nddata import CCDData
from astropy.utils.exceptions import AstropyUserWarning
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from ccdproc import core
from ccdproc._windowfilters import (
    _default_band_rows,
    _itemsize,
    _pad_windows,
    _stack_from_padded,
    _window_any,
    _window_median,
    _window_rank,
    _window_reduce,
)
from ccdproc.conftest import testing_array_device as xp_device
from ccdproc.conftest import testing_array_library as xp
from ccdproc.core import _dispatch_median_filter, _dispatch_percentile_filter
from ccdproc.tests.pytest_fixtures import assert_same_namespace_and_device
from ccdproc.tests.pytest_fixtures import to_xp as _as_test_array

_rng = np.random.default_rng(20260907)

# Deliberately not square, and not a multiple of any window size below, so
# that a wrong slice or a wrong band boundary shows up as a shape or a value
# error rather than cancelling out.
_IMAGE = _rng.normal(size=(23, 17))

# A boolean image with a handful of set pixels, for the maximum filter that
# grows a cosmic-ray flag.
_FLAGS = _rng.random((23, 17)) > 0.9

# Odd, even, rectangular, degenerate (1x1, which must be the identity),
# one window wider than it is tall, and 10, which with percentile 29 is a
# window whose rank depends on the order the arithmetic is grouped in.
_SIZES = [3, 4, 5, 1, 10, (3, 7), (4, 2), (1, 5)]

_MODES = ["reflect", "nearest"]

# The two percentiles ccdmask asks for, the ends and the middle, and 29,
# which at n = 100 is int(n * p / 100) == 29 but int(n * (p / 100)) == 28.
_PERCENTILES = [0.0, 29.0, 30.9, 50.0, 69.1, 100.0]


# One call per public filter, with the dtype kind its result must have.
_CALLS = [
    pytest.param(lambda data: _window_median(data, 3), "real floating", id="median"),
    pytest.param(lambda data: _window_rank(data, 3, 69.1), "real floating", id="rank"),
    pytest.param(lambda data: _window_any(data, 3), "bool", id="any"),
    pytest.param(
        lambda data: _window_reduce(data, 3, xp.std), "real floating", id="reduce"
    ),
]


def _assert_matches(result, expected):
    """Assert an array from the namespace under test equals a numpy one."""
    expected = _as_test_array(np.asarray(expected, dtype=float))
    assert result.shape == expected.shape
    assert bool(xp.all(xpx.isclose(result, expected, equal_nan=True)))


def _rank_reference(data, size, percentile, mode="reflect"):
    """
    NaN-aware window rank, as a reference for `_window_rank`.

    Notes
    -----
    The windows and the boundary handling come from
    `scipy.ndimage.generic_filter` itself, so the only thing written out
    here is the NaN-aware rank ndimage has no mode for: each window's
    non-NaN values are sorted and the element at
    ``min(int(n * percentile / 100), n - 1)`` taken, with a window holding
    no non-NaN value giving NaN. Re-deriving the padding and the window
    extraction instead would mirror the implementation under test, which
    is how the ``percentile / 100`` regrouping bug survived a 70-case
    parity grid.
    """

    def pick(window):
        values = np.sort(window[~np.isnan(window)])
        if values.size == 0:
            return np.nan
        return values[min(int(values.size * percentile / 100), values.size - 1)]

    return ndimage.generic_filter(data, pick, size=size, mode=mode)


@pytest.mark.parametrize("size", [3, 4], ids=str)
def test_window_median_matches_ndimage(size):
    """
    ``_window_median`` reproduces `scipy.ndimage.median_filter`.

    Notes
    -----
    ``_window_median`` is a one-line delegate to ``_window_rank`` at 50.0,
    which ``test_window_rank_matches_ndimage`` already covers over the
    whole size and mode grid, so these two cases are here for what that
    grid cannot see: that the public name really does ask for 50.0 and
    not 0.5, and that ``median_filter`` and ``percentile_filter(50.0)``
    agree -- the scipy invariant the delegation rests on. An even size is
    kept alongside an odd one because that is where the two conventions
    could differ.
    """
    result = _window_median(_as_test_array(_IMAGE), size)

    _assert_matches(result, ndimage.median_filter(_IMAGE, size=size))


@pytest.mark.parametrize("percentile", _PERCENTILES)
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("size", _SIZES, ids=str)
def test_window_rank_matches_ndimage(size, mode, percentile):
    """
    ``_window_rank`` reproduces `scipy.ndimage.percentile_filter` exactly.

    The 30.9 and 69.1 percentiles are the ones `ccdproc.core.ccdmask` asks
    for; 0 and 100 pin the ends, where the rank formula
    ``int(size * percentile / 100)`` would otherwise be off by one (at 100
    ndimage's own rank runs off the end of the window and it raises, while
    this clamps, so the two are compared only through the values they do
    both produce).
    """
    result = _window_rank(_as_test_array(_IMAGE), size, percentile, mode=mode)

    if percentile == 100.0:
        # ndimage refuses rank == size; the maximum is the same thing.
        expected = ndimage.maximum_filter(_IMAGE, size=size, mode=mode)
    else:
        expected = ndimage.percentile_filter(_IMAGE, percentile, size=size, mode=mode)
    _assert_matches(result, expected)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("size", _SIZES, ids=str)
def test_window_any_matches_ndimage_maximum(size, mode):
    """
    ``_window_any`` reproduces `scipy.ndimage.maximum_filter` on booleans.

    That is the substitution `ccdproc.core.cosmicray_median` makes for its
    ``gbox`` growth step, where the maximum of a boolean window is just
    whether any pixel in it is flagged. The result must stay boolean:
    ``cosmicray_median`` goes straight on to combine it with the input mask
    using ``&``.
    """
    result = _window_any(_as_test_array(_FLAGS), size, mode=mode)

    assert xp.isdtype(result.dtype, "bool")
    expected = _as_test_array(ndimage.maximum_filter(_FLAGS, size=size, mode=mode))
    assert bool(xp.all(result == expected))


@pytest.mark.parametrize("size", [3, (3, 5)], ids=str)
def test_window_reduce_matches_ndimage_generic_filter(size):
    """
    ``_window_reduce`` reproduces `scipy.ndimage.generic_filter`.

    ``generic_filter`` calls its callable once per pixel with that window
    flattened, while ``_window_reduce`` calls it once per band with the
    window values on a trailing axis; this pins that the two see the same
    values, which is what lets `ccdproc.core.background_deviation_filter`
    hand the same ``sigma_func`` to either.
    """
    result = _window_reduce(_as_test_array(_IMAGE), size, xp.std)

    _assert_matches(result, ndimage.generic_filter(_IMAGE, np.std, size=size))


@pytest.mark.parametrize("percentile", [30.9, 50.0, 69.1])
@pytest.mark.parametrize("size", [3, 4, (3, 5)], ids=str)
def test_nan_windows_match_an_explicit_rank_reference(size, percentile):
    """
    NaNs are excluded from the rank rather than sorted to one end.

    ndimage sorts NaNs in with the rest of the window, so it is no use as a
    reference here; the reference is an explicit `sliding_window_view` sort
    that drops the NaNs. This behaviour is what will let issue #984 exclude
    masked pixels from ``cosmicray_median``'s median simply by substituting
    NaN for them, so it is pinned now even though nothing depends on it yet.
    """
    data = np.array(_IMAGE, copy=True)
    # Scattered NaNs, plus a block wide enough that some windows see only
    # NaNs and some see a single value.
    data[[0, 5, 11, 22], [0, 3, 16, 9]] = np.nan
    data[7:12, 6:11] = np.nan

    result = _window_rank(_as_test_array(data), size, percentile)

    _assert_matches(result, _rank_reference(data, size, percentile))


def test_window_of_only_nan_is_nan():
    """
    A window with no non-NaN value at all yields NaN, not a sentinel.

    ``_nanrank`` sorts NaNs to ``+inf`` before selecting, so the failure
    mode this guards against is a window of NaNs coming back as ``+inf``
    -- silently plausible, and poisonous downstream.
    """
    data = np.full((5, 5), np.nan)
    data[0, 0] = 1.0

    result = _window_median(_as_test_array(data), 3)

    assert bool(result[0, 0] == 1.0)
    # The far corner's 3x3 window is entirely NaN.
    assert bool(xp.isnan(result[4, 4]))


@pytest.mark.parametrize("band_rows", [1, 2, 7, 22, 23, 1000])
@pytest.mark.parametrize("size", [3, 4, (5, 3)], ids=str)
def test_banded_matches_unbanded(size, band_rows):
    """
    Splitting the output into bands of rows changes nothing.

    Banding is what keeps the window stack -- ``prod(size)`` copies of the
    image -- inside a memory budget, and it is only sound because each band
    is cut from the *padded* array with the window overhang included. An
    off-by-one there would corrupt exactly the rows at the band seams,
    which is why the band sizes here divide 23 evenly, unevenly, exactly,
    and not at all.
    """
    data = _as_test_array(_IMAGE)

    banded = _window_median(data, size, band_rows=band_rows)
    unbanded = _window_median(data, size, band_rows=_IMAGE.shape[0])

    assert banded.shape == unbanded.shape
    assert bool(xp.all(banded == unbanded))


def test_banded_matches_unbanded_for_window_any():
    """
    Banding is exact for the boolean filter too.

    ``_window_any`` reduces with ``any`` rather than by sorting and so takes
    a different route through ``_windowed``; the seams are the same, but
    the concatenation of boolean bands is not covered by the float test.
    """
    data = _as_test_array(_FLAGS)

    banded = _window_any(data, 5, band_rows=3)
    unbanded = _window_any(data, 5, band_rows=_FLAGS.shape[0])

    assert bool(xp.all(banded == unbanded))


@pytest.mark.parametrize(("call", "dtype_kind"), _CALLS)
def test_result_keeps_the_namespace_device_and_promotes_the_dtype(call, dtype_kind):
    """
    Integer input comes back in the caller's namespace, on the caller's
    device, and in the dtype each filter documents.

    Notes
    -----
    Namespace and device are the point of the module -- the ndimage calls
    it replaces returned numpy arrays on the host -- and the device half
    is what the array-api-strict run, whose arrays live on a non-default
    device, exists to catch. The dtype is a deliberate divergence:
    `scipy.ndimage.median_filter` keeps an integer dtype where these
    promote, because the sort machinery shared with `ccdproc._nanfuncs`
    needs a NaN for "no value". ``_window_any`` is in the same list with
    ``bool`` as its kind rather than left out of the dtype half, because
    ``cosmicray_median`` combines its result with ``&``.
    """
    data = xp.asarray(np.arange(35).reshape(5, 7), device=xp_device)

    result = call(data)

    assert_same_namespace_and_device(result, data)
    assert xp.isdtype(result.dtype, dtype_kind)


def test_window_stack_holds_each_pixels_window():
    """
    The window stack holds exactly one window per pixel on a trailing axis.

    Every filter here is a reduction over that axis, so if the stack were
    wrong -- a slice off by one, an offset omitted -- the parity tests
    would fail without saying why. The reference is `sliding_window_view`
    on a numpy copy of the same padded array, which shares no machinery
    with the shifted slices under test; the windows are sorted before
    comparison because the order along the stacked axis is deliberately
    unspecified.
    """
    size = (3, 3)
    data = np.arange(20.0).reshape(4, 5)
    padded = np.pad(data, [(1, 1), (1, 1)], mode="symmetric")

    stack = _stack_from_padded(
        _pad_windows(_as_test_array(data), size, "reflect", xp), size, data.shape, xp
    )

    assert stack.shape == (4, 5, 9)
    expected = np.sort(sliding_window_view(padded, size).reshape(4, 5, 9), axis=-1)
    _assert_matches(xp.sort(stack, axis=-1), expected)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda data: _window_median(data, 3), id="_window_median"),
        pytest.param(lambda data: _window_rank(data, 3, 69.1), id="_window_rank"),
        pytest.param(lambda data: _window_reduce(data, 3, xp.std), id="_window_reduce"),
    ],
)
def test_complex_input_is_rejected(call):
    """
    Complex input raises rather than having its imaginary part dropped.

    `scipy.ndimage` refuses complex input outright, so the numpy path was
    already an error; without a check of its own the native path answered
    instead, with the median of the real parts on numpy (behind a
    ``ComplexWarning``) and a `TypeError` from inside ``astype`` on
    array-api-strict. Three backends, three answers, none of them
    ndimage's.
    """
    data = xp.astype(_as_test_array(_IMAGE), xp.complex128)

    with pytest.raises(TypeError, match="complex input is not supported"):
        call(data)


def test_float16_input_is_rejected():
    """
    Floating input narrower than ``float32`` raises, as ndimage's does.

    `scipy.ndimage` answers ``array type not supported`` for 2-D
    ``float16``, and the rank arithmetic could not honour it anyway: the
    window count itself stops being exact in binary16 above 2048, so a
    45x47 window would rank at 1058 of 2115 rather than 1057. Refusing is
    the only answer that is the same on every backend.
    """
    float16 = getattr(xp, "float16", None)
    if float16 is None:
        pytest.skip("this array library does not provide float16")
    data = xp.astype(_as_test_array(_IMAGE), float16)

    with pytest.raises(TypeError, match="is not supported by the window filters"):
        _window_median(data, 3)


@pytest.mark.parametrize("size", [np.int64(3), (np.int64(3), np.int64(5))], ids=str)
def test_numpy_integer_sizes_are_accepted(size):
    """
    A numpy integer window size is accepted, scalar or in a tuple.

    `scipy.ndimage` takes one, so ``ccdmask(ratio, ncmed=np.int64(7))``
    works on numpy; rejecting it here made the same call fail on every
    other array library. ``3.0`` stays rejected -- ndimage rejects it too
    -- which ``test_bad_size_raises`` pins.
    """
    result = _window_median(_as_test_array(_IMAGE), size)

    _assert_matches(result, ndimage.median_filter(_IMAGE, size=size))


def test_zero_dimensional_input_matches_ndimage():
    """
    A 0-d array is filtered, not rejected with an ``IndexError``.

    ``size=1`` on a 0-d array is the identity for
    `scipy.ndimage.median_filter`, and the module documents ``x`` as an
    array of any rank; before the guard in ``_windowed`` the band
    arithmetic asked for ``x.shape[0]`` and died on the empty shape.
    """
    result = _window_median(_as_test_array(np.asarray(1.0)), 1)

    assert result.shape == ()
    assert float(result) == float(ndimage.median_filter(np.asarray(1.0), size=1))


@pytest.mark.backend_skip(
    "numpy",
    "jax",
    "array-api-strict",
    "cupy",
    reason="only dask produces arrays whose shape is not fully known",
)
def test_unknown_shape_is_rejected_with_a_clear_message():
    """
    A dask array with unknown chunk sizes is rejected with the message the
    block functions use, which names the fix.

    Without it an unknown leading axis surfaced as dask's own "chunk sizes
    are unknown" from inside the padding helper, and an unknown trailing
    axis as ``Tried to concatenate arrays with unknown shape (1, nan)``,
    which names neither the filter nor the remedy.
    """
    unknown = _as_test_array(_IMAGE)
    unknown = unknown[unknown[:, 0] > -1e9]
    assert any(not isinstance(length, int) for length in unknown.shape)

    with pytest.raises(ValueError, match="fully known shape"):
        _window_median(unknown, 3)


def test_unimplemented_mode_raises():
    """
    A boundary mode this module does not implement is rejected.

    ndimage has five; only ``'reflect'`` and ``'nearest'`` are implemented,
    and silently treating ``'wrap'`` as ``'reflect'`` would be a wrong
    answer rather than a missing feature.
    """
    with pytest.raises(ValueError, match="mode must be one of"):
        _window_median(_as_test_array(_IMAGE), 3, mode="wrap")


def test_window_wider_than_twice_the_axis_raises():
    """
    A window needing more padding than the axis holds is rejected.

    ndimage re-reflects to fill such a window; this does not, so it has to
    say so rather than produce a differently-defined answer from the numpy
    path.
    """
    with pytest.raises(ValueError, match="window is too large for axis"):
        _window_median(_as_test_array(np.ones((4, 4))), 11)


@pytest.mark.parametrize("percentile", [-1.0, 101.0])
def test_percentile_outside_the_range_raises(percentile):
    """
    A percentile outside ``[0, 100]`` is rejected rather than clamped.

    `scipy.ndimage.percentile_filter` accepts a negative percentile as an
    offset from 100; not reproducing that quietly would make the two paths
    disagree, so it is refused outright.
    """
    with pytest.raises(ValueError, match=r"percentile must be in \[0, 100\]"):
        _window_rank(_as_test_array(_IMAGE), 3, percentile)


@pytest.mark.parametrize(
    ("size", "match"),
    [
        ((3, 3, 3), "one entry per axis"),
        ((3,), "one entry per axis"),
        (0, "positive integers"),
        ((3, -1), "positive integers"),
        ((3, 2.5), "positive integers"),
    ],
    ids=str,
)
def test_bad_size_raises(size, match):
    """
    A window shape that does not fit the array, or is not a positive
    integer, is rejected with a message naming the problem.

    Left to itself a short sequence would silently be zipped against the
    axes it does have and a zero-length window would build an empty stack
    whose median is NaN everywhere -- both wrong answers rather than
    errors.
    """
    with pytest.raises(ValueError, match=match):
        _window_median(_as_test_array(_IMAGE), size)


def test_band_rows_default_fits_the_budget():
    """
    The default band is the largest that keeps a band's stack in budget.

    Pinned because everything about banding is invisible in the results:
    without this, the default could silently become "one row" (correct but
    slow) or "the whole image" (correct until the image is large).
    """
    data = xp.asarray(np.zeros((512, 512)), device=xp_device)

    band_rows = _default_band_rows(data, (11, 11), xp)

    row_bytes = 512 * 8 * 11 * 11
    assert band_rows == (256 * 1024**2) // row_bytes
    assert band_rows >= 1

    # The widths the estimate above is built on, read out of the
    # namespace's own ``finfo``/``iinfo`` rather than a table here.
    assert _itemsize(xp.bool, xp) == 1
    assert _itemsize(xp.float32, xp) == 4
    assert _itemsize(xp.float64, xp) == 8


def test_warns_when_a_single_row_exceeds_the_budget():
    """
    One row too big for the band budget warns, and still computes.

    This is the one case banding cannot fix, and the deliberate choice is
    to tell the caller rather than to refuse the filter or silently use
    much more memory than the budget names.
    """
    # Shaped, not filled: only its shape and dtype reach the estimate.
    data = xp.asarray(np.zeros((2, 100_000)), device=xp_device)

    with pytest.warns(AstropyUserWarning, match="more than the 256 MiB"):
        assert _default_band_rows(data, (25, 25), xp) == 1


# ccdmask's default window shapes: (nlmed, ncmed) for the median it
# subtracts, and (nlsig, ncsig) for the two percentiles it takes of the
# residual.
_CCDMASK_SIZES = {50.0: (7, 7), 30.9: (15, 15), 69.1: (15, 15)}


def _flat_ratio_with_non_finite_pixels():
    """
    A flat-ratio image of the kind ``ccdmask`` is given, carrying the
    non-finite pixels a ratio of two flats really does contain: an
    isolated 0/0, a dead block wider than half the median window, and a
    divide-by-zero of each sign.
    """
    ratio = _rng.normal(loc=1.0, scale=0.02, size=(31, 29))
    ratio[4, 5] = np.nan
    ratio[10:15, 8:13] = np.nan
    ratio[20, 3] = np.inf
    ratio[22, 25] = -np.inf
    return ratio


@pytest.mark.parametrize("percentile", sorted(_CCDMASK_SIZES), ids=str)
def test_ccdmask_window_filters_exclude_nan_off_numpy(percentile):
    """
    On a non-numpy namespace ``ccdmask``'s window filters drop NaNs out of
    each window; on numpy they keep ndimage's ordering, which sorts NaNs
    in with the values.

    ``ccdmask`` is the one caller whose input routinely carries non-finite
    pixels -- it opens by masking them -- so it is the one place the
    divergence documented in ``docs/array_api.rst`` is reachable, and this
    pins it rather than leaving it to be discovered. The nan-aware side is
    checked against an explicit `sliding_window_view` rank reference,
    since ndimage cannot produce it; the two references are asserted to
    disagree first, or the test would pass whichever way the dispatcher
    went. Infinities are not part of the divergence: both implementations
    treat them as ordinary large values, and they are here only because a
    real flat ratio has them.
    """
    size = _CCDMASK_SIZES[percentile]
    ratio = _flat_ratio_with_non_finite_pixels()
    data = _as_test_array(ratio)

    if percentile == 50.0:
        result = _dispatch_median_filter(data, size, xp=xp)
        from_ndimage = ndimage.median_filter(ratio, size=size)
    else:
        result = _dispatch_percentile_filter(data, percentile, size, xp=xp)
        from_ndimage = ndimage.percentile_filter(ratio, percentile, size=size)

    nan_aware = _rank_reference(ratio, size, percentile)
    assert not np.allclose(nan_aware, from_ndimage, equal_nan=True)

    if array_api_compat.is_numpy_namespace(xp):
        _assert_matches(result, from_ndimage)
    else:
        _assert_matches(result, nan_aware)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda data: core.median_filter(data, 3), id="positional-size"),
        pytest.param(lambda data: core.median_filter(data, size=3), id="keyword-size"),
        pytest.param(
            lambda data: core.median_filter(data, (3, 5), mode="nearest"),
            id="rectangular-nearest",
        ),
    ],
)
@pytest.mark.parametrize("wrap", [False, True], ids=["array", "CCDData"])
def test_public_median_filter_matches_ndimage(call, wrap):
    """
    The public `ccdproc.median_filter` gives ndimage's answer for a bare
    array and for a `~astropy.nddata.CCDData`, in the caller's namespace.

    ``test_wrapped_external_funcs`` only ever hands it numpy, so this is
    the one place the off-numpy dispatch through ``_median_filter_array``
    (size positional or keyword, rectangular window, ``mode``) is
    exercised end to end; on numpy it pins the ndimage passthrough.
    """
    expected = call(_IMAGE)
    data = _as_test_array(_IMAGE)
    if wrap:
        data = CCDData(data, unit="adu")

    result = call(data)

    if wrap:
        assert isinstance(result, CCDData)
        assert result.unit == "adu"
        result = result.data
    assert array_api_compat.array_namespace(result) is xp
    _assert_matches(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(_IMAGE.tolist(), id="list"),
        pytest.param(tuple(row.tolist() for row in _IMAGE), id="tuple-of-lists"),
    ],
)
def test_public_median_filter_still_takes_a_plain_array_like(data):
    """
    `ccdproc.median_filter` keeps accepting a nested list or tuple, filtering
    it with ndimage exactly as it did before the native filters existed.

    ``median_filter`` documents itself as a passthrough for
    `scipy.ndimage.median_filter`, which takes any array-like. Resolving the
    namespace with a bare ``array_api_compat.array_namespace`` would raise
    ``TypeError: list is not a supported array type`` for these, so
    ``_median_filter_array`` falls back to numpy for an input no array
    library claims -- the same guard `~ccdproc.core._block_dispatch` uses.
    This runs on every backend: the input is numpy-ish whatever
    ``CCDPROC_ARRAY_LIBRARY`` says, so it must always take the ndimage path.
    """
    result = core.median_filter(data, size=3)

    assert array_api_compat.is_numpy_namespace(array_api_compat.array_namespace(result))
    np.testing.assert_allclose(result, ndimage.median_filter(_IMAGE, size=3))


@pytest.mark.parametrize(
    ("function", "argument"),
    [
        pytest.param(core.background_deviation_filter, 3, id="filter"),
        pytest.param(core.background_deviation_box, 5, id="box"),
    ],
)
def test_background_deviation_still_takes_a_plain_array_like(function, argument):
    """
    Both ``background_deviation`` functions keep accepting a nested list,
    as their docstrings promise.

    Notes
    -----
    Each resolves the namespace of its input, and a bare
    ``array_api_compat.array_namespace`` raises ``TypeError: list is not a
    supported array type`` for a list. ``background_deviation_filter`` used
    to hand the list straight to `scipy.ndimage.generic_filter`, which
    takes one; ``background_deviation_box`` never did, which is a bug of
    the same shape. Both now fall back to numpy for input no array library
    claims, the guard `ccdproc.median_filter` already used. This runs on
    every backend: the input is numpy-ish whatever
    ``CCDPROC_ARRAY_LIBRARY`` says, so it must always take the numpy path.
    """
    result = function(_IMAGE.tolist(), argument)

    assert array_api_compat.is_numpy_namespace(array_api_compat.array_namespace(result))
    np.testing.assert_allclose(result, function(_IMAGE, argument))


def test_public_median_filter_reraises_for_a_broken_array():
    """
    An object that advertises ``__array_namespace__`` but whose namespace
    lookup fails still raises, rather than being quietly handed to ndimage.

    The array-like fallback above must not swallow a genuine namespace
    failure: ndimage would coerce such an object into a 0-d object array and
    report something unrelated. This is the second half of the
    `~ccdproc.core._block_dispatch` guard.
    """

    class Broken:
        __array_namespace__ = None

    with pytest.raises(TypeError):
        core.median_filter(Broken(), size=3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"size": 3, "cval": 1.0}, "not 'cval'", id="cval"),
        pytest.param(
            {"size": 3, "origin": 1, "output": None},
            "not 'origin', 'output'",
            id="two-unsupported",
        ),
        pytest.param({"footprint": np.ones((3, 3))}, "not 'footprint'", id="footprint"),
        pytest.param({}, "requires a size", id="no-size"),
    ],
)
def test_public_median_filter_rejects_ndimage_only_arguments_off_numpy(kwargs, match):
    """
    Off numpy, `ccdproc.median_filter` refuses any ndimage argument the
    native filter cannot honour, naming the argument, and insists on
    ``size`` when nothing else is wrong.

    Silently ignoring ``cval`` or ``origin`` would give a different answer
    from the numpy path with no warning; the error is the documented
    contract. A ``footprint`` is reported as an unsupported argument
    rather than as a missing ``size``, so the missing-size error is only
    reachable with no arguments at all. numpy is skipped because there
    every argument is passed through to ndimage unchanged.
    """
    if array_api_compat.is_numpy_namespace(xp):
        pytest.skip("numpy input is passed through to ndimage unchanged")
    with pytest.raises(TypeError, match=match):
        core.median_filter(_as_test_array(_IMAGE), **kwargs)


def test_nearest_padding_of_an_empty_axis_raises():
    """
    ``'nearest'`` padding of an empty axis is refused: there is no edge
    sample to repeat.

    ``'reflect'`` on the same input is already refused by the
    too-large-window check; this pins that ``'nearest'`` does not slip
    through to a confusing error from ``concat`` instead.
    """
    with pytest.raises(ValueError, match="cannot pad axis 0, which is empty"):
        _window_median(_as_test_array(np.ones((0, 5))), 3, mode="nearest")


def test_zero_width_image_is_one_band():
    """
    An image with no columns costs nothing per row, so the whole of it is
    a single band rather than a division by zero.

    Pinned because ``_default_band_rows`` divides the budget by the row
    cost, and a zero-width input is the one shape that makes that cost
    zero.
    """
    assert _default_band_rows(_as_test_array(np.zeros((5, 0))), (3, 3), xp) == 5
    assert _default_band_rows(_as_test_array(np.zeros((0, 0))), (3, 3), xp) == 1
