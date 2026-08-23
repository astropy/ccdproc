# Licensed under a 3-clause BSD style license - see LICENSE.rst
import array_api_extra as xpx
import numpy as np
import pytest

from ccdproc._nanmedian import nanmedian
from ccdproc.conftest import testing_array_device as xp_device
from ccdproc.conftest import testing_array_library as xp

_rng = np.random.default_rng(906)
_some_nan = _rng.normal(size=(4, 3))
_some_nan[[0, 1, 2, 3], [1, 2, 0, 2]] = np.nan


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.parametrize(
    ("data", "axis"),
    [
        *[(_rng.normal(size=(n, 7)), 0) for n in range(1, 7)],  # odd/even lengths
        (np.array([3.0, 1.0, 2.0, 4.0]), 0),  # 1-D
        (_rng.normal(size=(5, 4, 3)), 0),  # 3-D
        (_some_nan, 0),  # NaNs scattered through the reduced axis
        (_some_nan, 1),  # a non-zero axis
        (_some_nan, -1),  # a negative axis
        (_some_nan, np.int64(1)),  # a numpy integer axis
        (np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]]), 0),  # all-NaN column
        (np.array([[1, 4], [2, 3], [5, 6], [4, 1]]), 0),  # integer input
    ],
)
def test_nanmedian_matches_numpy(data, axis):
    """The fallback reproduces ``numpy.nanmedian``, in shape, dtype and value."""
    result = nanmedian(xp.asarray(data, device=xp_device), axis=axis)
    expected = xp.asarray(np.nanmedian(data, axis=axis), device=xp_device)
    assert result.shape == expected.shape
    assert xp.isdtype(result.dtype, "real floating")
    assert xp.all(xpx.isclose(result, expected, equal_nan=True))


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
def test_nanmedian_bad_axis(axis, error):
    """Axes the fallback cannot handle are rejected instead of silently wrong."""
    with pytest.raises(error):
        nanmedian(xp.asarray(np.ones((2, 2)), device=xp_device), axis=axis)
