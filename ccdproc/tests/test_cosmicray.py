# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from numpy.testing import assert_allclose
import pytest
from astropy.utils import NumpyRNGContext
from astropy.nddata import StdDevUncertainty


from ..core import (cosmicray_lacosmic, cosmicray_median,
                    background_deviation_box, background_deviation_filter)

DATA_SCALE = 5.3
NCRAYS = 30


def add_cosmicrays(data, scale, threshold, ncrays=NCRAYS):
    size = data.shape[0]
    with NumpyRNGContext(125):
        crrays = np.random.randint(0, size, size=(ncrays, 2))
        # use (threshold + 1) below to make sure cosmic ray is well above the
        # threshold no matter what the random number generator returns
        crflux = (10 * scale * np.random.random(NCRAYS) +
                  (threshold + 5) * scale)
        for i in range(ncrays):
            y, x = crrays[i]
            data.data[y, x] = crflux[i]


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_lacosmic(ccd_data):
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * np.ones_like(ccd_data.data)
    data, crarr = cosmicray_lacosmic(ccd_data.data, sigclip=5)

    # check the number of cosmic rays detected
    # currently commented out while checking on issues
    # in astroscrappy
    # assert crarr.sum() == NCRAYS


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_lacosmic_ccddata(ccd_data):
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * np.ones_like(ccd_data.data)
    ccd_data.uncertainty = noise
    nccd_data = cosmicray_lacosmic(ccd_data, sigclip=5)

    # check the number of cosmic rays detected
    # currently commented out while checking on issues
    # in astroscrappy
    # assert nccd_data.mask.sum() == NCRAYS


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_lacosmic_check_data(ccd_data):
    with pytest.raises(TypeError):
        noise = DATA_SCALE * np.ones_like(ccd_data.data)
        cosmicray_lacosmic(10, noise)


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_median_check_data():
    with pytest.raises(TypeError):
        ndata, crarr = cosmicray_median(10, thresh=5, mbox=11,
                                        error_image=DATA_SCALE)


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_median(ccd_data):
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    ndata, crarr = cosmicray_median(ccd_data.data, thresh=5, mbox=11,
                                    error_image=DATA_SCALE)

    # check the number of cosmic rays detected
    assert crarr.sum() == NCRAYS


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_median_ccddata(ccd_data):
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    ccd_data.uncertainty = ccd_data.data*0.0+DATA_SCALE
    nccd = cosmicray_median(ccd_data, thresh=5, mbox=11,
                            error_image=None)

    # check the number of cosmic rays detected
    assert nccd.mask.sum() == NCRAYS


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_median_masked(ccd_data):
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    data = np.ma.masked_array(ccd_data.data, (ccd_data.data > -1e6))
    ndata, crarr = cosmicray_median(data, thresh=5, mbox=11,
                                    error_image=DATA_SCALE)

    # check the number of cosmic rays detected
    assert crarr.sum() == NCRAYS


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_median_background_None(ccd_data):
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    data, crarr = cosmicray_median(ccd_data.data, thresh=5, mbox=11,
                                   error_image=None)

    # check the number of cosmic rays detected
    assert crarr.sum() == NCRAYS


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_median_gbox(ccd_data):
    scale = DATA_SCALE  # yuck. Maybe use pytest.parametrize?
    threshold = 5
    add_cosmicrays(ccd_data, scale, threshold, ncrays=NCRAYS)
    error = ccd_data.data*0.0+DATA_SCALE
    data, crarr = cosmicray_median(ccd_data.data, error_image=error,
                                   thresh=5, mbox=11, rbox=0, gbox=5)
    data = np.ma.masked_array(data, crarr)
    assert crarr.sum() > NCRAYS
    assert abs(data.std() - scale) < 0.1


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_median_rbox(ccd_data):
    scale = DATA_SCALE  # yuck. Maybe use pytest.parametrize?
    threshold = 5
    add_cosmicrays(ccd_data, scale, threshold, ncrays=NCRAYS)
    error = ccd_data.data*0.0+DATA_SCALE
    data, crarr = cosmicray_median(ccd_data.data, error_image=error,
                                   thresh=5, mbox=11, rbox=21, gbox=5)
    assert data[crarr].mean() < ccd_data.data[crarr].mean()
    assert crarr.sum() > NCRAYS


@pytest.mark.data_scale(DATA_SCALE)
def test_cosmicray_median_background_deviation(ccd_data):
    with pytest.raises(TypeError):
        cosmicray_median(ccd_data.data, thresh=5, mbox=11,
                         error_image='blank')


def test_background_deviation_box():
    with NumpyRNGContext(123):
        scale = 5.3
        cd = np.random.normal(loc=0, size=(100, 100), scale=scale)
    bd = background_deviation_box(cd, 25)
    assert abs(bd.mean() - scale) < 0.10


def test_background_deviation_box_fail():
    with NumpyRNGContext(123):
        scale = 5.3
        cd = np.random.normal(loc=0, size=(100, 100), scale=scale)
    with pytest.raises(ValueError):
        background_deviation_box(cd, 0.5)


def test_background_deviation_filter():
    with NumpyRNGContext(123):
        scale = 5.3
        cd = np.random.normal(loc=0, size=(100, 100), scale=scale)
    bd = background_deviation_filter(cd, 25)
    assert abs(bd.mean() - scale) < 0.10


def test_background_deviation_filter_fail():
    with NumpyRNGContext(123):
        scale = 5.3
        cd = np.random.normal(loc=0, size=(100, 100), scale=scale)
    with pytest.raises(ValueError):
        background_deviation_filter(cd, 0.5)
