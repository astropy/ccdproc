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
from astropy.utils.exceptions import AstropyUserWarning
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from ccdproc._windowfilters import (
    _default_band_rows,
    _normalize_size,
    _window_stack,
    window_any,
    window_median,
    window_rank,
    window_reduce,
)
from ccdproc.conftest import testing_array_device as xp_device
from ccdproc.conftest import testing_array_library as xp

_rng = np.random.default_rng(20260907)

# Deliberately not square, and not a multiple of any window size below, so
# that a wrong slice or a wrong band boundary shows up as a shape or a value
# error rather than cancelling out.
_IMAGE = _rng.normal(size=(23, 17))

# A boolean image with a handful of set pixels, for the maximum filter that
# grows a cosmic-ray flag.
_FLAGS = _rng.random((23, 17)) > 0.9

# Odd, even, rectangular, degenerate (1x1, which must be the identity) and
# one window wider than it is tall.
_SIZES = [3, 4, 5, 1, (3, 7), (4, 2), (1, 5)]

_MODES = ["reflect", "nearest"]

# The two percentiles ccdmask asks for, plus the ends and the middle.
_PERCENTILES = [0.0, 30.9, 50.0, 69.1, 100.0]

# numpy's names for the two boundary modes ndimage calls "reflect" and
# "nearest"; numpy's own "reflect" is ndimage's "mirror", which is a
# different thing and is not implemented.
_NUMPY_PAD_MODES = {"reflect": "symmetric", "nearest": "edge"}


def _as_test_array(data):
    """The input converted into the namespace and device under test."""
    return xp.asarray(data, device=xp_device)


def _assert_matches(result, expected):
    """Assert an array from the namespace under test equals a numpy one."""
    expected = _as_test_array(np.asarray(expected, dtype=float))
    assert result.shape == expected.shape
    assert bool(xp.all(xpx.isclose(result, expected, equal_nan=True)))


def _rank_reference(data, size, percentile, mode="reflect"):
    """
    NaN-aware window rank computed with `sliding_window_view`, as an
    independent reference for `window_rank`.

    Each window's non-NaN values are sorted and the element at
    ``min(floor(n * percentile / 100), n - 1)`` taken; a window with no
    non-NaN value gives NaN.
    """
    size = _normalize_size(size, data.ndim)
    padded = np.pad(
        data,
        [(k // 2, k - 1 - k // 2) for k in size],
        mode=_NUMPY_PAD_MODES[mode],
    )
    windows = sliding_window_view(padded, size).reshape(data.shape + (-1,))
    ordered = np.sort(np.where(np.isnan(windows), np.inf, windows), axis=-1)
    n = np.count_nonzero(~np.isnan(windows), axis=-1)
    index = np.minimum(np.trunc(n * (percentile / 100)).astype(int), n - 1)
    picked = np.take_along_axis(ordered, np.maximum(index, 0)[..., None], axis=-1)
    return np.where(n == 0, np.nan, picked[..., 0])


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("size", _SIZES, ids=str)
def test_window_median_matches_ndimage(size, mode):
    """
    ``window_median`` reproduces `scipy.ndimage.median_filter` exactly.

    This is the whole contract of the module: `ccdproc.core` sends numpy
    input to ndimage and everything else here, so any disagreement would
    make a reduction's result depend on which array library the user
    happens to run. Even window lengths are in the grid because ndimage
    places them asymmetrically and takes the upper-middle element rather
    than averaging, which is easy to get wrong in both places at once.
    """
    result = window_median(_as_test_array(_IMAGE), size, mode=mode)

    _assert_matches(result, ndimage.median_filter(_IMAGE, size=size, mode=mode))


@pytest.mark.parametrize("percentile", _PERCENTILES)
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("size", _SIZES, ids=str)
def test_window_rank_matches_ndimage(size, mode, percentile):
    """
    ``window_rank`` reproduces `scipy.ndimage.percentile_filter` exactly.

    The 30.9 and 69.1 percentiles are the ones `ccdproc.core.ccdmask` asks
    for; 0 and 100 pin the ends, where the rank formula
    ``int(size * percentile / 100)`` would otherwise be off by one (at 100
    ndimage's own rank runs off the end of the window and it raises, while
    this clamps, so the two are compared only through the values they do
    both produce).
    """
    result = window_rank(_as_test_array(_IMAGE), size, percentile, mode=mode)

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
    ``window_any`` reproduces `scipy.ndimage.maximum_filter` on booleans.

    That is the substitution `ccdproc.core.cosmicray_median` makes for its
    ``gbox`` growth step, where the maximum of a boolean window is just
    whether any pixel in it is flagged. The result must stay boolean:
    ``cosmicray_median`` goes straight on to combine it with the input mask
    using ``&``.
    """
    result = window_any(_as_test_array(_FLAGS), size, mode=mode)

    assert xp.isdtype(result.dtype, "bool")
    expected = _as_test_array(ndimage.maximum_filter(_FLAGS, size=size, mode=mode))
    assert bool(xp.all(result == expected))


@pytest.mark.parametrize("size", [3, (3, 5)], ids=str)
def test_window_reduce_matches_ndimage_generic_filter(size):
    """
    ``window_reduce`` reproduces `scipy.ndimage.generic_filter`.

    ``generic_filter`` calls its callable once per pixel with that window
    flattened, while ``window_reduce`` calls it once per band with the
    window values on a trailing axis; this pins that the two see the same
    values, which is what lets `ccdproc.core.background_deviation_filter`
    hand the same ``sigma_func`` to either.
    """
    result = window_reduce(_as_test_array(_IMAGE), size, xp.std)

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

    result = window_rank(_as_test_array(data), size, percentile)

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

    result = window_median(_as_test_array(data), 3)

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

    banded = window_median(data, size, band_rows=band_rows)
    unbanded = window_median(data, size, band_rows=_IMAGE.shape[0])

    assert banded.shape == unbanded.shape
    assert bool(xp.all(banded == unbanded))


def test_banded_matches_unbanded_for_window_any():
    """
    Banding is exact for the boolean filter too.

    ``window_any`` reduces with ``any`` rather than by sorting and so takes
    a different route through ``_windowed``; the seams are the same, but
    the concatenation of boolean bands is not covered by the float test.
    """
    data = _as_test_array(_FLAGS)

    banded = window_any(data, 5, band_rows=3)
    unbanded = window_any(data, 5, band_rows=_FLAGS.shape[0])

    assert bool(xp.all(banded == unbanded))


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda data: window_median(data, 3), id="window_median"),
        pytest.param(lambda data: window_rank(data, 3, 69.1), id="window_rank"),
        pytest.param(lambda data: window_any(data, 3), id="window_any"),
        pytest.param(lambda data: window_reduce(data, 3, xp.std), id="window_reduce"),
    ],
)
def test_result_stays_in_the_input_namespace_and_device(call):
    """
    The result comes back in the caller's namespace, on the caller's device.

    This is the point of the module -- the ndimage calls it replaces
    returned numpy arrays on the host -- and the device half of it is what
    the array-api-strict run, which puts its arrays on a non-default
    device, exists to catch.
    """
    data = _as_test_array(_IMAGE)

    result = call(data)

    assert array_api_compat.array_namespace(result) is (
        array_api_compat.array_namespace(data)
    )
    assert array_api_compat.device(result) == array_api_compat.device(data)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda data: window_median(data, 3), id="window_median"),
        pytest.param(lambda data: window_rank(data, 3, 69.1), id="window_rank"),
        pytest.param(lambda data: window_reduce(data, 3, xp.std), id="window_reduce"),
    ],
)
def test_integer_input_is_promoted_to_a_real_floating_dtype(call):
    """
    Integer input comes back as floats, a documented divergence.

    `scipy.ndimage.median_filter` keeps an integer dtype; these filters
    promote, because the sort machinery they share with
    `ccdproc._nanfuncs` needs a NaN to represent "no value" and because
    every ccdproc caller filters image data it goes on to do float
    arithmetic with. Pinned so the divergence cannot drift back silently.
    """
    data = xp.asarray(np.arange(35).reshape(5, 7), device=xp_device)

    result = call(data)

    assert xp.isdtype(result.dtype, "real floating")


def test_window_stack_holds_each_pixels_window():
    """
    ``_window_stack`` puts exactly one window per pixel on a trailing axis.

    Every other function here is a reduction over that axis, so if the
    stack were wrong -- a slice off by one, an offset omitted -- the
    parity tests would fail without saying why. This checks the stack
    itself, against the window read directly out of a padded copy.
    """
    data = np.arange(20.0).reshape(4, 5)
    padded = np.pad(data, [(1, 1), (1, 1)], mode="symmetric")

    stack = _window_stack(_as_test_array(data), 3)

    assert stack.shape == (4, 5, 9)
    for row in range(4):
        for col in range(5):
            expected = np.sort(padded[row : row + 3, col : col + 3].reshape(-1))
            got = np.sort(np.asarray(_to_host(stack[row, col, :])))
            assert np.array_equal(got, expected)


def _to_host(array):
    """
    A small array as numpy, for tests that compare element by element.

    Goes through `float` per element rather than ``np.asarray`` because
    array-api-strict refuses to export an array on a non-default device.
    """
    return np.array([float(array[i]) for i in range(array.shape[0])])


def test_unimplemented_mode_raises():
    """
    A boundary mode this module does not implement is rejected.

    ndimage has five; only ``'reflect'`` and ``'nearest'`` are implemented,
    and silently treating ``'wrap'`` as ``'reflect'`` would be a wrong
    answer rather than a missing feature.
    """
    with pytest.raises(ValueError, match="mode must be one of"):
        window_median(_as_test_array(_IMAGE), 3, mode="wrap")


def test_window_wider_than_twice_the_axis_raises():
    """
    A window needing more padding than the axis holds is rejected.

    ndimage re-reflects to fill such a window; this does not, so it has to
    say so rather than produce a differently-defined answer from the numpy
    path.
    """
    with pytest.raises(ValueError, match="window is too large for axis"):
        window_median(_as_test_array(np.ones((4, 4))), 11)


@pytest.mark.parametrize("percentile", [-1.0, 101.0])
def test_percentile_outside_the_range_raises(percentile):
    """
    A percentile outside ``[0, 100]`` is rejected rather than clamped.

    `scipy.ndimage.percentile_filter` accepts a negative percentile as an
    offset from 100; not reproducing that quietly would make the two paths
    disagree, so it is refused outright.
    """
    with pytest.raises(ValueError, match=r"percentile must be in \[0, 100\]"):
        window_rank(_as_test_array(_IMAGE), 3, percentile)


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
        window_median(_as_test_array(_IMAGE), size)


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
