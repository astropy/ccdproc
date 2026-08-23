# Licensed under a 3-clause BSD style license - see LICENSE.rst
import array_api_extra as xpx
import numpy as np
import pytest

from ccdproc._nanmedian import nanmedian
from ccdproc.conftest import testing_array_device as xp_device
from ccdproc.conftest import testing_array_library as xp

pytestmark = pytest.mark.filterwarnings(
    "ignore:All-NaN slice encountered:RuntimeWarning"
)


def _check(np_data, axis=0):
    """Compare the array-API nanmedian with numpy's, without leaving ``xp``."""
    arr = xp.asarray(np_data, device=xp_device)
    result = nanmedian(arr, axis=axis)
    expected = xp.asarray(np.nanmedian(np_data, axis=axis), device=xp_device)
    assert result.shape == expected.shape
    assert xp.isdtype(result.dtype, "real floating")
    assert xp.all(xpx.isclose(result, expected, equal_nan=True))


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6])
def test_nanmedian_odd_and_even_lengths(length):
    rng = np.random.default_rng(length)
    _check(rng.normal(size=(length, 7)))


def test_nanmedian_1d():
    _check(np.array([3.0, 1.0, 2.0, 4.0]))


def test_nanmedian_3d():
    rng = np.random.default_rng(3)
    _check(rng.normal(size=(5, 4, 3)))


def test_nanmedian_some_nan():
    data = np.array(
        [
            [1.0, np.nan, 5.0],
            [2.0, 4.0, np.nan],
            [np.nan, 3.0, 1.0],
            [4.0, 2.0, np.nan],
        ]
    )
    _check(data)


def test_nanmedian_all_nan_column():
    data = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    arr = xp.asarray(data, device=xp_device)
    result = nanmedian(arr, axis=0)
    assert float(result[0]) == 2.0
    assert bool(xp.isnan(result[1]))


def test_nanmedian_integer_input():
    data = np.array([[1, 4], [2, 3], [5, 6], [4, 1]])
    _check(data)


def test_nanmedian_axis_1():
    rng = np.random.default_rng(11)
    data = rng.normal(size=(3, 6))
    data[1, 2] = np.nan
    _check(data, axis=1)
    _check(data, axis=-1)


def test_nanmedian_unsupported_axis():
    arr = xp.asarray(np.ones((2, 2)), device=xp_device)
    with pytest.raises(NotImplementedError):
        nanmedian(arr, axis=None)
    with pytest.raises(NotImplementedError):
        nanmedian(arr, axis=(0, 1))


@pytest.mark.parametrize("axis", [2, -3])
def test_nanmedian_axis_out_of_bounds(axis):
    arr = xp.asarray(np.ones((2, 2)), device=xp_device)
    with pytest.raises(ValueError, match="out of bounds"):
        nanmedian(arr, axis=axis)
