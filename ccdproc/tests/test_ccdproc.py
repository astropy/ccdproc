# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

import numpy as np
from astropy.io import fits
from astropy.modeling import models
from astropy.units.quantity import Quantity
import astropy.units as u

from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest

from ..ccddata import CCDData, electrons, adu
from ..ccdproc import *

# tests for overscan


def test_subtract_overscan_mean(ccd_data):
    # create the overscan region
    oscan = 300.0
    ccd_data.data = ccd_data.data + oscan
    oscan = ccd_data[:, 0:10]
    ccd_data = subtract_overscan(ccd_data, oscan, median=False, model=None)
    assert abs(ccd_data.data.mean()) < 0.1


def test_subtract_overscan_median(ccd_data):
    # create the overscan region
    oscan = 300.0
    ccd_data.data = ccd_data.data + oscan
    oscan = ccd_data[:, 0:10]
    ccd_data = subtract_overscan(ccd_data, oscan, median=True, model=None)
    assert abs(ccd_data.data.mean()) < 0.1

# tests for gain correction


def test_subtract_overscan_model(ccd_data):
    # create the overscan region
    size = ccd_data.shape[0]
    yscan, xscan = np.mgrid[0:size, 0:size] / 10.0 + 300.0
    ccd_data.data = ccd_data.data + yscan
    oscan = ccd_data[:, 0:10]
    ccd_data = subtract_overscan(
        ccd_data, oscan, median=False, model=models.Polynomial1D(2))
    assert abs(ccd_data.data.mean()) < 0.1


def test_subtract_overscan_ccd_fails():
    # do we get an error if the *image* is neither an nor an array?
    with pytest.raises(TypeError):
        subtract_overscan(3, np.zeros((5, 5)))
    # do we get an error if the *overscan* is not an image or an array?
    with pytest.raises(TypeError):
        subtract_overscan(np.zeros((10, 10)), 3, median=False, model=None)


# test for flat correction
@pytest.mark.data_scale(10)
def test_flat_correct(ccd_data):
    size = ccd_data.shape[0]

    # create the flat
    data = 2 * np.ones((size, size))
    flat = CCDData(data, meta=fits.header.Header(), unit=ccd_data.unit)
    ccd_data = flat_correct(ccd_data, flat)

# test for variance and for flat correction


@pytest.mark.data_scale(10)
@pytest.mark.data_mean(300)
def test_flat_correct_variance(ccd_data):
    size = ccd_data.shape[0]
    ccd_data.unit = electrons
    ccd_data.create_variance(5)
    # create the flat
    data = 2 * np.ones((size, size))
    flat = CCDData(data, meta=fits.header.Header(), unit=ccd_data.unit)
    flat.create_variance(0.5)
    ccd_data = flat_correct(ccd_data, flat)


# tests for gain correction
def test_gain_correct(ccd_data):
    init_data = ccd_data.data
    ccd_data = gain_correct(ccd_data, gain=3)
    assert_array_equal(ccd_data.data, 3 * init_data)


def test_gain_correct_quantity(ccd_data):
    init_data = ccd_data.data
    g = Quantity(3, electrons / u.adu)
    ccd_data = gain_correct(ccd_data, gain=g)

    assert_array_equal(ccd_data.data, 3 * init_data)
    assert ccd_data.unit == electrons
