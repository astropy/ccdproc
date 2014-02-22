# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

import numpy as np
from astropy.io import fits
from astropy.modeling import models
from astropy.units.quantity import Quantity

from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest, raises
from astropy.utils import NumpyRNGContext

from ..ccddata import CCDData, electrons, adu, fromFITS, toFITS
from ..ccdproc import *

# tests for overscan


def test_subtract_overscan_mean():
    with NumpyRNGContext(125):
        size = 100
        scale = 1
        # create the basic data array
        data = np.random.normal(loc=0, size=(size, size),  scale=scale)
        ccd = CCDData(data, meta=fits.header.Header())
        # create the overscan region
        oscan = 300.0
        ccd.data = ccd.data + oscan
    oscan = ccd[:, 0:10]
    ccd = subtract_overscan(ccd, oscan, median=False, model=None)
    assert abs(ccd.data.mean()) < 0.1


def test_subtract_overscan_median():
    with NumpyRNGContext(125):
        size = 100
        scale = 1
        # create the basic data array
        data = np.random.normal(loc=0, size=(size, size),  scale=scale)
        ccd = CCDData(data, meta=fits.header.Header())
        # create the overscan region
        oscan = 300.0
        ccd.data = ccd.data + oscan
    oscan = ccd[:, 0:10]
    ccd = subtract_overscan(ccd, oscan, median=True, model=None)
    assert abs(ccd.data.mean()) < 0.1

# tests for gain correction


def test_subtract_overscan_model():
    with NumpyRNGContext(125):
        size = 100
        scale = 1
        # create the basic data array
        data = np.random.normal(loc=0, size=(size, size),  scale=scale)
        ccd = CCDData(data, meta=fits.header.Header())
        # create the overscan region
        yscan, xscan = np.mgrid[0:size, 0:size] / 10.0 + 300.0
        ccd.data = ccd.data + yscan
    oscan = ccd[:, 0:10]
    ccd = subtract_overscan(
        ccd, oscan, median=False, model=models.Polynomial1D(2))
    assert abs(ccd.data.mean()) < 0.1


@raises(TypeError)
def test_sutract_overscan_ccd_failt():
    subtract_overscan(3, oscan, median=False, model=None)


@raises(TypeError)
def test_sutract_overscan_ccd_failt():
    subtract_overscan(np.zeros((10, 10)), 3, median=False, model=None)


# test for flat correction
def test_flat_correct():
    with NumpyRNGContext(125):
        size = 100
        scale = 10
        # create the basic data array
        data = np.random.normal(loc=0, size=(size, size),  scale=scale)
        ccd = CCDData(data, meta=fits.header.Header())
        # create the flat
        data = 2 * np.ones((size, size))
        flat = CCDData(data, meta=fits.header.Header())
    ccd = flat_correct(ccd, flat)

# test for variance and for flat correction


def test_flat_correct_variance():
    with NumpyRNGContext(125):
        size = 100
        scale = 10
        # create the basic data array
        data = np.random.normal(loc=300, size=(size, size),  scale=scale)
        ccd = CCDData(data, meta=fits.header.Header(), unit=electrons)
        ccd.create_variance(5)
        # create the flat
        data = 2 * np.ones((size, size))
        flat = CCDData(data, meta=fits.header.Header(), unit=electrons)
        flat.create_variance(0.5)
    ccd = flat_correct(ccd, flat)


# tests for gain correction
def test_gain_correct():
    with NumpyRNGContext(125):
        size = 100
        scale = 1
        # create the basic data array
        data = np.random.normal(loc=0, size=(size, size),  scale=scale)
        ccd = CCDData(data, meta=fits.header.Header())
        # create the overscan region
    ccd = gain_correct(ccd, gain=3)
    assert_array_equal(ccd.data, 3 * data)


def test_gain_correct_quantity():
    with NumpyRNGContext(125):
        size = 100
        scale = 1
        # create the basic data array
        data = np.random.normal(loc=0, size=(size, size),  scale=scale)
        ccd = CCDData(data, meta=fits.header.Header(), unit=adu)
        # create the overscan region
    g = Quantity(3, electrons / adu)
    ccd = gain_correct(ccd, gain=g)
    assert_array_equal(ccd.data, 3 * data)
    assert ccd.unit == electrons
