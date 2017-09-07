# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

import astropy.units as u
from astropy.stats import median_absolute_deviation as mad

from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

from ..astrometry import *
from ..astrometry import _calc_ratio, _get_index


# test that the Combiner raises error if empty
def test_remove_duplicates_exact():
    x = np.array([1, 2, 2, 3, 2, 4, 3, 7])
    y = np.array([7, 2, 2, 4, 2, 5, 4, 8])

    x, y = remove_duplicates(x, y, 0.1)

    np.testing.assert_array_equal(x, np.array([1, 2, 3, 4, 7]))
    np.testing.assert_array_equal(y, np.array([7, 2, 4, 5, 8]))


# test that the Combiner raises error if empty
def test_remove_duplicates_tol():
    x = np.array([1.1, 2.0, 2.3, 11.1, 2.8, 4.2, 3.8, 7.2])
    y = np.array([7.5, 2.0, 2.2, 47.1, 2.7, 5.1, 4.9, 8.2])

    x, y = remove_duplicates(x, y, 1.0)

    np.testing.assert_array_equal(x, np.array([1.1, 2.0, 11.1, 2.8, 4.2, 7.2]))
    np.testing.assert_array_equal(y, np.array([7.5, 2.0, 47.1, 2.7, 5.1, 8.2]))


def test_distance():
    assert 5 == distance(1, 1, 4, 5)


def test__calc_ratio():
    x = np.array([1, 4, 6])
    y = np.array([1, 5, 13])

    assert 5.0/13.0 == _calc_ratio(x, y, 0, 1, 2)


def test__get_index():
    assert _get_index(3, 3) == (1, 2, 0)


def test_distance_ratio():
    x = np.array([1, 4, 6])
    y = np.array([1, 5, 13])

    r = distance_ratios(x, y)

    assert len(r) == 6
    result = [0.38461538, 2.6, 0.60633906, 1.64924225, 1.57648156, 0.63432394]
    np.testing.assert_almost_equal(r, result)
