# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
from contextlib import contextmanager

import array_api_extra as xpx
import numpy as np
import pytest

from ccdproc._nanfuncs import nanmean, nanstd, nansum
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
    pytest.param(nanstd, np.nanstd, id="nanstd"),
]

_DATA = [
    *[(_rng.normal(size=(n, 7)), 0) for n in range(1, 4)],
    (np.array([3.0, 1.0, 2.0, 4.0]), 0),  # 1-D
    (_rng.normal(size=(5, 4, 3)), 0),  # 3-D
    (_some_nan, 0),  # NaNs scattered through the reduced axis
    (_some_nan, 1),  # a non-zero axis
    (_some_nan, -1),  # a negative axis
    (np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]]), 0),  # all-NaN column
    (np.array([[1.0, np.nan], [np.nan, np.nan]]), 0),  # single non-NaN in a slice
    (np.array([np.nan, np.nan, np.nan]), 0),  # every value NaN
    (_ill_conditioned, 0),
    (np.array([[1, 4], [2, 3], [5, 6], [4, 1]]), 0),  # integer input
    (np.array([[True, False], [False, True], [True, True]]), 0),  # boolean input
]


@contextmanager
def _numpy_reference():
    """
    Let the numpy reference warn where the fallbacks deliberately do not.

    ``numpy.nanmean`` and ``numpy.nanstd`` raise "Mean of empty slice" and
    "Degrees of freedom <= 0 for slice" on an all-NaN slice. The values they
    return there are still the ones to match, so the reference is evaluated
    with the warnings suppressed -- something the fallbacks never need.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


@pytest.mark.parametrize(("func", "reference"), _FUNCS)
@pytest.mark.parametrize(("data", "axis"), _DATA)
def test_matches_numpy(func, reference, data, axis):
    """The fallback reproduces its numpy counterpart, in shape and value."""
    with _numpy_reference():
        expected_np = np.asarray(reference(data, axis=axis), dtype=float)

    result = func(xp.asarray(data, device=xp_device), axis=axis)
    expected = xp.asarray(expected_np, device=xp_device)

    assert result.shape == expected.shape
    # Integer and boolean input is promoted to the namespace's default real
    # dtype, which is where ``nansum`` parts company with ``numpy.nansum``.
    assert xp.isdtype(result.dtype, "real floating")
    assert xp.all(xpx.isclose(result, expected, equal_nan=True))


@pytest.mark.parametrize("func", [nansum, nanmean, nanstd])
def test_no_warning_on_all_nan_slice(func):
    """
    All-NaN slices are handled silently.

    Two of the numpy counterparts warn here, and ccdproc's pytest
    configuration turns warnings into errors, so a fallback that warned would
    fail every ``Combiner`` test with a fully masked pixel.
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


@pytest.mark.parametrize("func", [nanmean, nanstd])
def test_mean_and_std_all_nan_slice_is_nan(func):
    """An all-NaN slice yields NaN, and leaves its neighbours alone."""
    data = xp.asarray(np.array([[1.0, np.nan], [2.0, np.nan]]), device=xp_device)

    result = func(data, axis=0)

    assert not xp.any(xp.isnan(result[0:1]))
    assert xp.all(xp.isnan(result[1:2]))


def test_nanstd_is_population_deviation():
    """``ddof`` is 0, matching `numpy.nanstd` and ``bottleneck.nanstd``."""
    result = nanstd(xp.asarray(np.array([1.0, 2.0, 3.0, np.nan]), device=xp_device))

    expected = xp.asarray(np.std(np.array([1.0, 2.0, 3.0])), device=xp_device)
    assert xp.all(xpx.isclose(result, expected))


@pytest.mark.parametrize("func", [nansum, nanmean, nanstd])
@pytest.mark.parametrize(
    ("axis", "error"),
    [
        (None, NotImplementedError),
        ((0, 1), NotImplementedError),
        (2, ValueError),
        (-3, ValueError),
    ],
)
def test_bad_axis(func, axis, error):
    """Axes the fallbacks cannot handle are rejected instead of silently wrong."""
    with pytest.raises(error):
        func(xp.asarray(np.ones((2, 2)), device=xp_device), axis=axis)
