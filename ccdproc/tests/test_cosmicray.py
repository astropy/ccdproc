# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

import numpy as np
from astropy.io import fits

from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.utils import NumpyRNGContext

from ..core import *

import os

DATA_SCALE = 5.3
NCRAYS = 30


def add_cosmicrays(data, scale, threshold, ncrays=NCRAYS):
    size = data.shape[0]
    with NumpyRNGContext(125):
        crrays = np.random.random_integers(0, size - 1, size=(ncrays, 2))
        # use (threshold + 1) below to make sure cosmic ray is well above the
        # threshold no matter what the random number generator returns
        crflux = (10 * scale * np.random.random(NCRAYS) +
                  (threshold + 1) * scale)
        for i in range(ncrays):
            y, x = crrays[i]
            data.data[y, x] = crflux[i]


@pytest.mark.data_scale(DATA_SCALE)
@pytest.mark.parametrize("background_type", [
                         (DATA_SCALE),
                         (None)])
def test_cosmicray_clean_scalar_background(ccd_data, background_type):
    scale = DATA_SCALE  # yuck. Maybe use pytest.parametrize?
    threshold = 5
    add_cosmicrays(ccd_data, scale, threshold, ncrays=NCRAYS)
    testdata = 1.0 * ccd_data.data
    cc = cosmicray_clean(ccd_data, 5, cosmicray_median, crargs=(11,),
                         background=background_type, bargs=(), rbox=11, gbox=0)
    assert abs(cc.data.std() - scale) < 0.1
    assert ((testdata - cc.data) > 0).sum() == NCRAYS

@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_clean_gbox(ccd_data):
    scale = DATA_SCALE  # yuck. Maybe use pytest.parametrize?
    threshold = 5
    add_cosmicrays(ccd_data, scale, threshold, ncrays=NCRAYS)
    testdata = 1.0 * ccd_data.data
    cc = ccd_data  # currently here because no copy command for NDData
    cc = cosmicray_clean(cc, 5.0, cosmicray_median, crargs=(11,),
                             background=background_variance_box, bargs=(25,),
                             rbox=0, gbox=5)
    data = np.ma.masked_array(cc.data, cc.mask)
    assert abs(data.std() - scale) < 0.1
    assert cc.mask.sum() > NCRAYS



@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_clean(ccd_data):
    scale = DATA_SCALE  # yuck. Maybe use pytest.parametrize?
    threshold = 5
    add_cosmicrays(ccd_data, scale, threshold, ncrays=NCRAYS)
    testdata = 1.0 * ccd_data.data
    cc = ccd_data  # currently here because no copy command for NDData
    for i in range(5):
        cc = cosmicray_clean(cc, 5.0, cosmicray_median, crargs=(11,),
                             background=background_variance_box, bargs=(25,),
                             rbox=11)
    assert abs(cc.data.std() - scale) < 0.1
    assert (testdata - cc.data > 0).sum() == NCRAYS


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_clean_rbox_zero_replaces_no_pixels(ccd_data):
    scale = DATA_SCALE  # yuck. Maybe use pytest.parametrize?
    threshold = 5
    add_cosmicrays(ccd_data, scale, threshold, ncrays=NCRAYS)

    testdata = 1.0 * ccd_data.data
    cc = cosmicray_clean(ccd_data, 5, cosmicray_median, crargs=(11,),
                         background=scale, bargs=(), rbox=0, gbox=0)
    assert_allclose(cc, testdata)


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_median(ccd_data):
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    crarr = cosmicray_median(ccd_data.data, 5, mbox=11, background=DATA_SCALE)

    # check the number of cosmic rays detected
    assert crarr.sum() == NCRAYS

@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_median_masked(ccd_data):
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    data = np.ma.masked_array(ccd_data.data, (ccd_data.data>-1e6))
    crarr = cosmicray_median(data, 5, mbox=11, background=DATA_SCALE)

    # check the number of cosmic rays detected
    assert crarr.sum() == NCRAYS


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_median_background_None(ccd_data):
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    crarr = cosmicray_median(ccd_data.data, 5, mbox=11, background=None)

    # check the number of cosmic rays detected
    assert crarr.sum() == NCRAYS




def test_background_variance_box():
    with NumpyRNGContext(123):
        scale = 5.3
        cd = np.random.normal(loc=0, size=(100, 100), scale=scale)
    bd = background_variance_box(cd, 25)
    assert abs(bd.mean() - scale) < 0.10


def test_background_variance_box_fail():
    with NumpyRNGContext(123):
        scale = 5.3
        cd = np.random.normal(loc=0, size=(100, 100), scale=scale)
    with pytest.raises(ValueError):
        bd = background_variance_box(cd, 0.5)


def test_background_variance_filter():
    with NumpyRNGContext(123):
        scale = 5.3
        cd = np.random.normal(loc=0, size=(100, 100), scale=scale)
    bd = background_variance_filter(cd, 25)
    assert abs(bd.mean() - scale) < 0.10


def test_background_variance_filter_fail():
    with NumpyRNGContext(123):
        scale = 5.3
        cd = np.random.normal(loc=0, size=(100, 100), scale=scale)
    with pytest.raises(ValueError):
        bd = background_variance_filter(cd, 0.5)
