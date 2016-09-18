# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

import astropy.units as u
from astropy.stats import median_absolute_deviation as mad

from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.compat import NUMPY_LT_1_9
from astropy.wcs import WCS

from ..astrometry import *


#test that the Combiner raises error if empty
def test_remove_duplicates_exact():
    x = np.array([1, 2, 2, 3, 2, 4, 3, 7])
    y = np.array([7, 2, 2, 4, 2, 5, 4, 8])

    x, y = remove_duplicates(x, y, 0.1)

    np.testing.assert_array_equal(x,np.array([1, 2, 3, 4, 7]))
    np.testing.assert_array_equal(y, np.array([7, 2, 4, 5, 8]))

#test that the Combiner raises error if empty
def test_remove_duplicates_tol():
    x = np.array([1.1, 2.0, 2.3, 11.1, 2.8, 4.2, 3.8, 7.2])
    y = np.array([7.5, 2.0, 2.2, 47.1, 2.7, 5.1, 4.9, 8.2])

    x, y = remove_duplicates(x, y, 1.0)

    np.testing.assert_array_equal(x,np.array([1.1, 2.0, 11.1, 2.8, 4.2, 7.2]))
    np.testing.assert_array_equal(y, np.array([7.5, 2.0, 47.1, 2.7, 5.1, 8.2]))

    
