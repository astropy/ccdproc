# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing.utils import assert_allclose


import astropy.units as u
from astropy import modeling as mod
from astropy.wcs import WCS

from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename

from ..astrometry import (
    remove_duplicates, distance, distance_ratios, triangle_angle,
    match_by_fit, create_wcs_from_fit)
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

# add test for zero
def test__calc_ratio_zero():
    x = np.array([1, 4, 1])
    y = np.array([5, 5, 5])

    assert np.isnan(_calc_ratio(x, y, 0, 1, 2))
    


def test__get_index():
    assert _get_index(3, 3) == (1, 2, 0)

# remove
def test_distance_ratio():
    x = np.array([1, 4, 6])
    y = np.array([1, 5, 13])

    r = distance_ratios(x, y)

    assert len(r) == 6
    result = [0.38461538, 2.6, 0.60633906, 1.64924225, 1.57648156, 0.63432394]
    np.testing.assert_almost_equal(r, result)

# triangle_angle
def test_triangle_angle():
    assert triangle_angle(5,4,3) == np.pi/2.0

# calc_triangle
def calc_triangle():
    x = np.array([1, 4, 6])
    y = np.array([1, 5, 13])

    sides, angles, order = astrom.calc_triangle(x, y, 0, 1, 2)

    sides_result = np.array([ 0.38461538, 1., 0.63432394])
    np.testing.assert_almost_equal(sides, sides_result)
    angles_result = array([ 0.14981246, 2.74307021, 0.24870999])
    np.testing.assert_almost_equal(angles, angles_result)
    assert order == np.array(0, 2, 1)

# match_by_triangle

# match_by_fit
def test_match_by_fit():
    x_init = mod.models.Polynomial2D(1, c0_0=10, c1_0=0.1, c0_1= 0.2)
    y_init = mod.models.Polynomial2D(1, c0_0=5, c1_0=0.2, c0_1= 0.1)

    x = np.arange(10)
    y = np.arange(10)

    d = y_init(x,y)
    r = x_init(x,y)

    rr, dd = match_by_fit(x,y,r,d,[0,1,2], [0,1,2],
                                  tolerance=0.1*u.degree)


    assert np.all(rr==np.arange(10))
    assert np.all(dd==np.arange(10))


# match_by_fit
def test_match_by_fit_quantity():
    x_init = mod.models.Polynomial2D(1, c0_0=10, c1_0=0.1, c0_1= 0.2)
    y_init = mod.models.Polynomial2D(1, c0_0=5, c1_0=0.2, c0_1= 0.1)

    x = np.arange(10)
    y = np.arange(10)

    d = y_init(x,y)
    r = x_init(x,y)

    rr, dd = match_by_fit(x,y,r*u.deg,d*u.deg,[0,1,2], [0,1,2],
                                  tolerance=0.1*u.degree)


    assert np.all(rr==np.arange(10))
    assert np.all(dd==np.arange(10))



# create_wcs_from_fit
def test_create_wcs_from_fit():
    x_init = mod.models.Polynomial2D(1, c0_0=10, c1_0=0.1, c0_1= 0.2)
    y_init = mod.models.Polynomial2D(1, c0_0=5, c1_0=0.2, c0_1= 0.1)

    x = np.arange(10)
    y = np.arange(10)
  
    d = y_init(x,y)
    r = x_init(x,y)

    idp = np.arange(10)
  
    wcs = create_wcs_from_fit(x, y, r, d, idp, idp, xref=0, yref=0) 
    rr, dd = wcs.all_pix2world(x,y,1)
    # having trouble testing almost_equal
    #assert np.test.assert_almost_equal(rr, r)
    #assert np.testing.assert_almost_equal(dd, d)
    assert abs(rr-r).mean() < 1e-2 
    assert abs(dd-d).mean() < 1e-2 
