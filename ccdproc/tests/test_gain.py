# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

import astropy.units as u

from ..core import create_deviation, gain_correct, Keyword
from .pytest_fixtures import ccd_data as ccd_data_func


# tests for gain
@pytest.mark.parametrize('gain', [
                         3.0,
                         3.0 * u.photon / u.adu,
                         3.0 * u.electron / u.adu,
                         Keyword('gainval', unit=u.electron / u.adu)])
def test_linear_gain_correct(gain):
    ccd_data = ccd_data_func()
    # The data values should be positive, so the poisson noise calculation
    # works without throwing warnings
    ccd_data.data = np.absolute(ccd_data.data)
    ccd_data = create_deviation(ccd_data, readnoise=1.0 * u.adu)
    ccd_data.meta['gainval'] = 3.0
    orig_data = ccd_data.data
    ccd = gain_correct(ccd_data, gain)
    if isinstance(gain, Keyword):
        gain = gain.value   # convert to Quantity...
    try:
        gain_value = gain.value
    except AttributeError:
        gain_value = gain

    np.testing.assert_array_almost_equal_nulp(ccd.data, gain_value * orig_data)
    np.testing.assert_array_almost_equal_nulp(
        ccd.uncertainty.array, gain_value * ccd_data.uncertainty.array)

    if isinstance(gain, u.Quantity):
        assert ccd.unit == ccd_data.unit * gain.unit
    else:
        assert ccd.unit == ccd_data.unit


# test gain with gain_unit
def test_linear_gain_unit_keyword():
    ccd_data = ccd_data_func()
    # The data values should be positive, so the poisson noise calculation
    # works without throwing warnings
    ccd_data.data = np.absolute(ccd_data.data)

    ccd_data = create_deviation(ccd_data, readnoise=1.0 * u.adu)
    orig_data = ccd_data.data
    gain = 3.0
    gain_unit = u.electron / u.adu
    ccd = gain_correct(ccd_data, gain, gain_unit=gain_unit)
    np.testing.assert_array_almost_equal_nulp(ccd.data, gain * orig_data)
    np.testing.assert_array_almost_equal_nulp(
        ccd.uncertainty.array, gain * ccd_data.uncertainty.array)
    assert ccd.unit == ccd_data.unit * gain_unit
