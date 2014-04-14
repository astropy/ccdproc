# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

import os

import numpy as np
from astropy.io import fits
from astropy.modeling import models
from astropy.units.quantity import Quantity
import astropy.units as u

from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest

from ..ccddata import CCDData, electrons, adu
from ..ccdproc import *


# test creating variance
# success expected if u_image * u_gain = u_readnoise
@pytest.mark.parametrize('u_image,u_gain,u_readnoise,expect_succes', [
                         (electrons, None, electrons, True),
                         (electrons, electrons, electrons, False),
                         (u.adu, electrons / u.adu, electrons, True),
                         (electrons, None, u.dimensionless_unscaled, False),
                         (electrons, u.dimensionless_unscaled, electrons, True),
                         (u.adu, u.dimensionless_unscaled, electrons, False),
                         (u.adu, u.photon / u.adu, electrons, False),
                         ])
@pytest.mark.data_size(10)
def test_create_variance(ccd_data, u_image, u_gain, u_readnoise,
                         expect_succes):
    ccd_data.unit = u_image
    if u_gain:
        gain = 2.0 * u_gain
    else:
        gain = None
    readnoise = 5 * u_readnoise
    if expect_succes:
        ccd_var = create_variance(ccd_data, gain=gain, readnoise=readnoise)
        assert ccd_var.uncertainty.array.shape == (10, 10)
        assert ccd_var.uncertainty.array.size == 100
        assert ccd_var.uncertainty.array.dtype == np.dtype(float)
        if gain:
            expected_var = np.sqrt(2 * ccd_data.data + 5 ** 2) / 2
        else:
            expected_var = np.sqrt(ccd_data.data + 5 ** 2)
        np.testing.assert_array_equal(ccd_var.uncertainty.array,
                                      expected_var)
        assert ccd_var.unit == ccd_data.unit
    else:
        with pytest.raises(u.UnitsError):
            ccd_var = create_variance(ccd_data, gain=gain, readnoise=readnoise)


def test_create_variance_keywords_must_have_unit(ccd_data):
    # gain must have units if provided
    with pytest.raises(TypeError):
        create_variance(ccd_data, gain=3)
    # readnoise must have units
    with pytest.raises(TypeError):
        create_variance(ccd_data, readnoise=5)
    # readnoise must be provided
    with pytest.raises(ValueError):
        create_variance(ccd_data)


# tests for overscan
def test_subtract_overscan_mean(ccd_data):
    # create the overscan region
    oscan = 300.0
    ccd_data.data = ccd_data.data + oscan
    ccd_data = subtract_overscan(ccd_data, section='[:, 0:10]',
                                 median=False, model=None)
    assert abs(ccd_data.data.mean()) < 0.1


def test_subtract_overscan_median(ccd_data):
    # create the overscan region
    oscan = 300.0
    ccd_data.data = ccd_data.data + oscan
    ccd_data = subtract_overscan(ccd_data, section='[:, 0:10]',
                                 median=True, model=None)
    assert abs(ccd_data.data.mean()) < 0.1

# tests for gain correction


def test_subtract_overscan_model(ccd_data):
    # create the overscan region
    size = ccd_data.shape[0]
    yscan, xscan = np.mgrid[0:size, 0:size] / 10.0 + 300.0
    ccd_data.data = ccd_data.data + yscan
    ccd_data = subtract_overscan(ccd_data, section='[:, 0:10]',
                                 median=False, model=models.Polynomial1D(2))
    assert abs(ccd_data.data.mean()) < 0.1


def test_subtract_overscan_ccd_fails():
    # do we get an error if the *image* is neither an nor an array?
    with pytest.raises(TypeError):
        subtract_overscan(3, np.zeros((5, 5)))
    # do we get an error if the *overscan* is not an image or an array?
    with pytest.raises(TypeError):
        subtract_overscan(np.zeros((10, 10)), 3, median=False, model=None)


def test_trim_image_requires_section(ccd_data):
    with pytest.raises(ValueError):
        trim_image(ccd_data)


@pytest.mark.data_size(50)
def test_trim_image(ccd_data):
    trimmed = trim_image(ccd_data, section='[20:40,:]')
    assert trimmed.shape == (20, 50)
    np.testing.assert_array_equal(trimmed.data, ccd_data[20:40, :])


# this xfail needs to get pulled out ASAP...
@pytest.mark.xfail('TRAVIS' in os.environ, reason='needs astropy fix')
# test for flat correction
@pytest.mark.data_scale(10)
def test_flat_correct(ccd_data):
    size = ccd_data.shape[0]

    # create the flat
    data = 2 * np.ones((size, size))
    flat = CCDData(data, meta=fits.header.Header(), unit=ccd_data.unit)
    ccd_data = flat_correct(ccd_data, flat)


# test for variance and for flat correction

# this xfail needs to get pulled out ASAP...
@pytest.mark.xfail('TRAVIS' in os.environ, reason='needs astropy fix')
@pytest.mark.data_scale(10)
@pytest.mark.data_mean(300)
def test_flat_correct_variance(ccd_data):
    size = ccd_data.shape[0]
    ccd_data.unit = electrons
    ccd_data = create_variance(ccd_data, readnoise=5 * electrons)
    # create the flat
    data = 2 * np.ones((size, size))
    flat = CCDData(data, meta=fits.header.Header(), unit=ccd_data.unit)
    flat = create_variance(flat, readnoise=0.5 * electrons)
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
