# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

import numpy as np
from astropy.io import fits
from astropy import modeling as models

from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest
from astropy.utils import NumpyRNGContext
from astropy.units.quantity import Quantity

from ..ccddata import CCDData, adu, electrons, fromFITS, toFITS
from ..ccdproc import *


def writeout(cd, outfile):
    import os
    hdu = toFITS(cd)
    if os.path.isfile(outfile):
        os.remove(outfile)
    hdu.writeto(outfile)


# tests for overscan
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


test_gain_correct()
test_gain_correct_quantity()
