# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings

import array_api_extra as xpx
import numpy as np
import pytest
from astropy.stats import median_absolute_deviation

from ccdproc._nanfuncs import median, nanmad, nanmean, nanmedian, nanstd, nansum
from ccdproc.conftest import testing_array_device as xp_device
from ccdproc.conftest import testing_array_library as xp

_rng = np.random.default_rng(986)
_some_nan = _rng.normal(size=(4, 3))
_some_nan[[0, 1, 2, 3], [1, 2, 0, 2]] = np.nan

# Values large compared with their spread: the single-pass
# ``sum(x**2) - sum(x)**2 / n`` form of the variance loses every significant
# digit here, so this pins ``nanstd`` to the two-pass form it actually uses.
_ill_conditioned = np.array([1e8, 1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0])

_FUNCS = [
    pytest.param(nansum, np.nansum, id="nansum"),
    pytest.param(nanmean, np.nanmean, id="nanmean"),
    # Both default to the population deviation (ddof=0), as does
    # ``bottleneck.nanstd``; any other ddof fails the differential test on
    # every multi-element float row below.
    pytest.param(nanstd, np.nanstd, id="nanstd"),
    pytest.param(nanmedian, np.nanmedian, id="nanmedian"),
    pytest.param(median, np.median, id="median"),
]

_DATA = [
    *[(_rng.normal(size=(n, 7)), 0) for n in range(1, 7)],  # odd/even lengths
    (np.array([3.0, 1.0, 2.0, 4.0]), 0),  # 1-D
    (_rng.normal(size=(5, 4, 3)), 0),  # 3-D
    (_some_nan, 0),  # NaNs scattered through the reduced axis
    (_some_nan, 1),  # a non-zero axis
    (_some_nan, -1),  # a negative axis
    (_some_nan, np.int64(1)),  # a numpy integer axis
    (np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]]), 0),  # all-NaN column
    (np.array([[1.0, np.nan], [np.nan, np.nan]]), 0),  # single non-NaN in a slice
    (np.array([np.nan, np.nan, np.nan]), 0),  # every value NaN
    (_ill_conditioned, 0),
    (np.array([[1, 4], [2, 3], [5, 6], [4, 1]]), 0),  # integer input
    (np.array([[True, False], [False, True], [True, True]]), 0),  # boolean input
]


# The ignore marks let the numpy reference warn on all-NaN slices where the
# fallbacks deliberately do not; ``test_no_warning_on_all_nan_slice`` is the
# explicit owner of the fallbacks' silence.
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.parametrize(("func", "reference"), _FUNCS)
@pytest.mark.parametrize(("data", "axis"), _DATA)
def test_matches_numpy(func, reference, data, axis):
    """The fallback reproduces its numpy counterpart, in shape and value."""
    converted = xp.asarray(data, device=xp_device)
    if data is _ill_conditioned and bool(xp.all(converted == converted[0])):
        # A float32-default backend (jax without JAX_ENABLE_X64) collapses
        # these values to a single one on conversion; the fallback's answer
        # of 0 would then be correct but the float64 reference disagrees.
        pytest.skip("backend's default real dtype cannot resolve these values")

    expected_np = np.asarray(reference(data, axis=axis), dtype=float)

    result = func(converted, axis=axis)
    expected = xp.asarray(expected_np, device=xp_device)

    assert result.shape == expected.shape
    # Integer and boolean input is promoted to the namespace's default real
    # dtype, which is where ``nansum`` parts company with ``numpy.nansum``.
    assert xp.isdtype(result.dtype, "real floating")
    assert xp.all(xpx.isclose(result, expected, equal_nan=True))


@pytest.mark.parametrize("func", [nansum, nanmean, nanstd, nanmedian, median])
def test_no_warning_on_all_nan_slice(func):
    """
    All-NaN slices are handled silently.

    Most of the numpy counterparts warn here, and ccdproc's pytest
    configuration turns warnings into errors, so a fallback that warned would
    fail every ``Combiner`` test with a fully masked pixel. ``numpy.median``
    itself does not warn on NaN input, but ``median`` is included here too
    since it shares the ``_setup``/``nanmedian`` machinery with the other
    fallbacks.
    """
    data = xp.asarray(
        np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]]), device=xp_device
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        func(data, axis=0)


def test_nansum_all_nan_slice_is_zero():
    """An all-NaN slice sums to zero, as `numpy.nansum` does -- not to NaN."""
    data = xp.asarray(np.array([[1.0, np.nan], [2.0, np.nan]]), device=xp_device)

    result = nansum(data, axis=0)

    assert xp.all(xpx.isclose(result, xp.asarray([3.0, 0.0], device=xp_device)))


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_nanmad_matches_astropy(axis):
    """``nanmad`` reproduces ``median_absolute_deviation(ignore_nan=True)``."""
    result = nanmad(xp.asarray(_some_nan, device=xp_device), axis=axis)
    expected = xp.asarray(
        median_absolute_deviation(_some_nan, axis=axis, ignore_nan=True),
        device=xp_device,
    )
    assert result.shape == expected.shape
    assert bool(xp.all(xpx.isclose(result, expected, equal_nan=True)))


@pytest.mark.parametrize("func", [nansum, nanmean, nanstd, nanmedian, median])
@pytest.mark.parametrize(
    ("axis", "error"),
    [
        (None, NotImplementedError),
        (True, NotImplementedError),  # bool subclasses int; reject it anyway
        ((0, 1), NotImplementedError),
        (2, ValueError),
        (-3, ValueError),
    ],
)
def test_bad_axis(func, axis, error):
    """Axes the fallbacks cannot handle are rejected instead of silently wrong."""
    with pytest.raises(error):
        func(xp.asarray(np.ones((2, 2)), device=xp_device), axis=axis)
