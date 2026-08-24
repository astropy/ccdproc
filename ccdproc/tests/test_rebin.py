# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Not updating to array API in here because rebin will be removed
# in version 3.0 of ccdproc.
import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.utils.exceptions import AstropyDeprecationWarning

from ccdproc.conftest import testing_array_library as xp
from ccdproc.core import rebin
from ccdproc.tests.pytest_fixtures import ccd_data as ccd_data_func


# test rebinning ndarray
def test_rebin_ndarray():
    with pytest.raises(TypeError), pytest.warns(AstropyDeprecationWarning):
        rebin(1, (5, 5))


# test rebinning dimensions
def test_rebin_dimensions():
    ccd_data = ccd_data_func(data_size=10)
    with pytest.raises(ValueError), pytest.warns(AstropyDeprecationWarning):
        rebin(ccd_data.data, (5,))


# test rebinning dimensions
def test_rebin_ccddata_dimensions():
    ccd_data = ccd_data_func(data_size=10)
    with pytest.raises(ValueError), pytest.warns(AstropyDeprecationWarning):
        rebin(ccd_data, (5,))


# test rebinning works
def test_rebin_larger():
    ccd_data = ccd_data_func(data_size=10)
    a = ccd_data.data
    with pytest.warns(AstropyDeprecationWarning):
        try:
            b = rebin(a, (20, 20))
        except TypeError as e:
            if "does not support this method of rebinning" in str(e):
                pytest.skip("Rebinning not supported for this data type")

    assert b.shape == (20, 20)
    np.testing.assert_almost_equal(b.sum(), 4 * a.sum())


# test rebinning is invariant
def test_rebin_smaller():
    ccd_data = ccd_data_func(data_size=10)
    a = ccd_data.data
    with pytest.warns(AstropyDeprecationWarning):
        try:
            b = rebin(a, (20, 20))
            c = rebin(b, (10, 10))
        except TypeError as e:
            if "does not support this method of rebinning" in str(e):
                pytest.skip("Rebinning not supported for this data type")

    assert c.shape == (10, 10)
    assert (c - a).sum() == 0


# test rebinning with ccddata object
@pytest.mark.parametrize("mask_data, uncertainty", [(False, False), (True, True)])
def test_rebin_ccddata(mask_data, uncertainty):
    ccd_data = ccd_data_func(data_size=10)
    if mask_data:
        ccd_data.mask = xp.zeros_like(ccd_data.data)
    if uncertainty:
        err = np.random.default_rng().normal(size=ccd_data.shape)
        ccd_data.uncertainty = StdDevUncertainty(err)

    with pytest.warns(AstropyDeprecationWarning):
        try:
            b = rebin(ccd_data, (20, 20))
        except TypeError as e:
            if "does not support this method of rebinning" in str(e):
                pytest.skip("Rebinning not supported for this data type")

    assert b.shape == (20, 20)
    if mask_data:
        assert b.mask.shape == (20, 20)
    if uncertainty:
        assert b.uncertainty.array.shape == (20, 20)


def test_rebin_does_not_change_input():
    ccd_data = ccd_data_func()
    original = ccd_data.copy()
    with pytest.warns(AstropyDeprecationWarning):
        try:
            _ = rebin(ccd_data, (20, 20))
        except TypeError as e:
            if "does not support this method of rebinning" in str(e):
                pytest.skip("Rebinning not supported for this data type")

    np.testing.assert_allclose(original.data, ccd_data.data)
    assert original.unit == ccd_data.unit


def test_rebin_keeps_array_namespace():
    # Regression test for #967: rebin used the numpy-only ``.astype`` method
    # on the index array, which fails for array-API backends that do not
    # provide it (e.g. array-api-strict).
    import array_api_compat

    from ccdproc.conftest import testing_array_device as xp_device
    from ccdproc.conftest import testing_array_library as xp

    a = xp.asarray(np.arange(16.0).reshape(4, 4), device=xp_device)
    with pytest.warns(AstropyDeprecationWarning):
        try:
            b = rebin(a, (8, 8))
        except TypeError as e:
            if "does not support this method of rebinning" in str(e):
                pytest.skip("Rebinning not supported for this data type")

    assert array_api_compat.array_namespace(b) is array_api_compat.array_namespace(a)
    assert array_api_compat.device(b) == array_api_compat.device(a)
    assert b.shape == (8, 8)
    # Every input pixel is replicated into a 2x2 block of the output.
    assert bool(xp.all(b[::2, ::2] == a))
