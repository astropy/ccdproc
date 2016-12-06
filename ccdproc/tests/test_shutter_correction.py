# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from astropy.extern.six import moves
from scipy.ndimage.filters import median_filter

from ..core import *
from ..shutter_correction import *


def model_image(exptime=1.0, bias=0.0, shape=(16,32), gradient=0.005):
    '''
    Generate simple model image data given an exposure time.
    '''
    flux = np.ones(shape)*1000
    for line in range(flux.shape[0]):
        flux[line,:] -= 20*line
    exptime_arr = np.ones(shape)*(exptime+bias)
    # Model shutter takes time to open by sliding across image columns and
    # closes instantaneously.  Leads to a 0.05 second gradient per pixel.
    for col in range(exptime_arr.shape[1]):
        exptime_arr[:,col] -= gradient*col
    data = np.array(flux*exptime_arr, dtype=np.int)
    correct_im = CCDData(flux, unit='adu',
                         meta={'EXPTIME': exptime, 'GAIN': 1.0})
    im = CCDData(data, unit='adu',
                 meta={'EXPTIME': exptime, 'GAIN': 1.0})
    return im, correct_im


def test_GaladiEnriqez1995():
    input_data = [model_image(exptime=e) for e in moves.range(2,11,2)]
    flats = [x[0] for x in input_data]
    correct = [x[1] for x in input_data]
    shutter_map = GaladiEnriqez1995(flats)
    for i in range(len(flats)):
        repaired = apply_shutter_map(flats[i], shutter_map, exptimekey='EXPTIME')
        assert_allclose(repaired.data, correct[i].data, rtol=1e-3)


def test_GaladiEnriqez1995_with_bias():
    shutter_bias = 0.125
    input_data = [model_image(exptime=e, bias=shutter_bias)
                  for e in moves.range(2,11,2)]
    flats = [x[0] for x in input_data]
    correct = [x[1] for x in input_data]
    shutter_map = GaladiEnriqez1995(flats)
    for i in range(len(flats)):
        repaired = apply_shutter_map(flats[i], shutter_map, exptimekey='EXPTIME')
        assert_allclose(repaired.data, correct[i].data, rtol=1e-3)


def test_fit_shutter_bias():
    shutter_bias = 0.125
    input_data = [model_image(exptime=e, bias=shutter_bias)
                  for e in moves.range(2,11,2)]
    flats = [x[0] for x in input_data]
    measured_bias = fit_shutter_bias(flats)
    assert_almost_equal(measured_bias, shutter_bias, decimal=5)


def test_fit_shutter_bias_with_normalizer():
    shutter_bias = 0.125
    input_data = [model_image(exptime=e, bias=shutter_bias, shape=(64,64), gradient=0.001)
                  for e in moves.range(2,11,2)]
    flats = [x[0] for x in input_data]
    measured_bias = fit_shutter_bias(flats,
        normalizer=lambda f: median_filter(f.data, size=(3,3)).max())
    assert_almost_equal(measured_bias, shutter_bias, decimal=2)


def test_Surma1993():
    input_data = [model_image(exptime=e) for e in moves.range(2,11,2)]
    flats = [x[0] for x in input_data]
    correct = [x[1] for x in input_data]
    shutter_map = Surma1993(flats)
    for i in range(len(flats)):
        repaired = apply_shutter_map(flats[i], shutter_map, exptimekey='EXPTIME')
        assert_allclose(repaired.data, correct[i].data, rtol=1e-3)


def test_Surma1993_with_bias():
    shutter_bias = 0.125
    input_data = [model_image(exptime=e, bias=shutter_bias)
                  for e in moves.range(2,11,2)]
    flats = [x[0] for x in input_data]
    correct = [x[1] for x in input_data]
    measured_bias = fit_shutter_bias(flats)
    assert_almost_equal(measured_bias, shutter_bias, decimal=5)
    shutter_map = Surma1993(flats, shutter_bias=measured_bias)
    for i in range(len(flats)):
        repaired = apply_shutter_map(flats[i], shutter_map, exptimekey='EXPTIME')
        assert_allclose(repaired.data, correct[i].data, rtol=1e-3)
