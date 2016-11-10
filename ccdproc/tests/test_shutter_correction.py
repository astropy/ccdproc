# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_allclose

from ..core import *
from ..shutter_correction import *

def test_shutter_correction_algorithm():
    def model_image(exptime=1.0):
        shape = (16,32)
        flux = np.ones(shape)*1000
        for line in range(flux.shape[0]):
            flux[line,:] -= 20*line
        exptime_arr = np.ones(shape)*exptime
        # Model shutter takes time to open by sliding across image columns and
        # closes instantaneously.  Leads to a 0.05 second gradient per pixel
        for col in range(exptime_arr.shape[1]):
            exptime_arr[:,col] -= 0.050*col
        data = np.array(flux*exptime_arr, dtype=np.int)
        correct_im = CCDData(flux, unit='adu',
                             meta={'EXPTIME': exptime, 'GAIN': 1.0})
        im = CCDData(data, unit='adu',
                     meta={'EXPTIME': exptime, 'GAIN': 1.0})
        return im, correct_im

    input_data = [model_image(exptime=e) for e in range(2,11,2)]
    flats = [x[0] for x in input_data]
    correct = [x[1] for x in input_data]
    shutter_map = shutter_correction_algorithm(flats)
    for i in range(len(flats)):
        repaired = apply_shutter_correction(flats[i], shutter_map, exptimekey='EXPTIME')
        assert_allclose(repaired.data, correct[i].data, rtol=1e-3)
