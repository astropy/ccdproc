# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

import numpy as np
from astropy.io import fits
from astropy import modeling as models

from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest
from astropy.utils import NumpyRNGContext
from astropy.units.quantity import Quantity
import astropy.units as u

from ..ccddata import electron
from ..ccdproc import *


# tests for overscan
def test_gain_correct(ccd_data):
    orig_data = ccd_data.data
    ccd = gain_correct(ccd_data, gain=3)
    assert_array_equal(ccd.data, 3 * orig_data)


def test_gain_correct_quantity(ccd_data):
    orig_data = ccd_data.data
    g = Quantity(3, electron / u.adu)
    ccd = gain_correct(ccd_data, gain=g)
    assert_array_equal(ccd.data, 3 * orig_data)
    assert ccd.unit == electron
