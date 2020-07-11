# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from astropy.nddata import StdDevUncertainty

from astropy.tests.helper import catch_warnings
from astropy.utils.exceptions import AstropyDeprecationWarning

from ccdproc.core import rebin
from ccdproc.tests.pytest_fixtures import ccd_data as ccd_data_func


# test rebinning ndarray
def test_rebin_ndarray():
    with pytest.raises(TypeError), catch_warnings(AstropyDeprecationWarning):
        rebin(1, (5, 5))


# test rebinning dimensions
def test_rebin_dimensions():
    ccd_data = ccd_data_func(data_size=10)
    with pytest.raises(ValueError), catch_warnings(AstropyDeprecationWarning):
            rebin(ccd_data.data, (5,))


# test rebinning dimensions
def test_rebin_ccddata_dimensions():
    ccd_data = ccd_data_func(data_size=10)
    with pytest.raises(ValueError), catch_warnings(AstropyDeprecationWarning):
        rebin(ccd_data, (5,))


# test rebinning works
def test_rebin_larger():
    ccd_data = ccd_data_func(data_size=10)
    a = ccd_data.data
    with catch_warnings(AstropyDeprecationWarning) as w:
        b = rebin(a, (20, 20))
    assert len(w) >= 1

    assert b.shape == (20, 20)
    np.testing.assert_almost_equal(b.sum(), 4 * a.sum())


# test rebinning is invariant
def test_rebin_smaller():
    ccd_data = ccd_data_func(data_size=10)
    a = ccd_data.data
    with catch_warnings(AstropyDeprecationWarning) as w:
        b = rebin(a, (20, 20))
        c = rebin(b, (10, 10))
    assert len(w) >= 1

    assert c.shape == (10, 10)
    assert (c - a).sum() == 0


# test rebinning with ccddata object
@pytest.mark.parametrize('mask_data, uncertainty', [
                         (False, False),
                         (True, True)])
def test_rebin_ccddata(mask_data, uncertainty):
    ccd_data = ccd_data_func(data_size=10)
    if mask_data:
        ccd_data.mask = np.zeros_like(ccd_data)
    if uncertainty:
        err = np.random.normal(size=ccd_data.shape)
        ccd_data.uncertainty = StdDevUncertainty(err)

    with catch_warnings(AstropyDeprecationWarning) as w:
        b = rebin(ccd_data, (20, 20))
    assert len(w) >= 1

    assert b.shape == (20, 20)
    if mask_data:
        assert b.mask.shape == (20, 20)
    if uncertainty:
        assert b.uncertainty.array.shape == (20, 20)


def test_rebin_does_not_change_input():
    ccd_data = ccd_data_func()
    original = ccd_data.copy()
    with catch_warnings(AstropyDeprecationWarning) as w:
        _ = rebin(ccd_data, (20, 20))
    assert len(w) >= 1
    np.testing.assert_array_equal(original.data, ccd_data.data)
    assert original.unit == ccd_data.unit
