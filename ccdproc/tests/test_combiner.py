# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

import numpy as np
from astropy.io import fits
from astropy.modeling import models
from astropy.units.quantity import Quantity
import astropy.units as u

from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest

from ..ccddata import CCDData, electron, adu
from ..combiner import *

from astropy.stats import median_absolute_deviation as mad

from pytest_fixtures import ccd_data


#test that the Combiner raises error if empty
def test_combiner_empty():
    with pytest.raises(TypeError):
        Combiner()  # empty initializer should fail

#test that the Combiner raises error if empty if ccd_list is None
def test_combiner_empty():
    with pytest.raises(TypeError):
        Combiner(None)  # empty initializer should fail


#test that Combiner throws an error if input
#objects are not ccddata objects
def test_ccddata_combiner_objects(ccd_data):
    ccd_list = [ccd_data, ccd_data, None]
    with pytest.raises(TypeError):
        Combiner(ccd_list)  # different objects should fail


#test that Combiner throws an error if input
#objects do not have the same size
def test_ccddata_combiner_size(ccd_data):
    ccd_large = CCDData(np.zeros((200, 100)), unit=u.adu)
    ccd_list = [ccd_data, ccd_data, ccd_large]
    with pytest.raises(TypeError):
        Combiner(ccd_list)  # arrays of different sizes should fail


#test that Combiner throws an error if input
#objects do not have the same units
def test_ccddata_combiner_units(ccd_data):
    ccd_large = CCDData(np.zeros((100, 100)), unit=u.second)
    ccd_list = [ccd_data, ccd_data, ccd_large]
    with pytest.raises(TypeError):
        Combiner(ccd_list)

#test if mask and data array are created
def test_combiner_create(ccd_data):
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    assert c.data_arr.shape == (3, 100, 100)
    assert c.data_arr.mask.shape == (3, 100, 100)

#test mask is created from ccd.data
def test_combiner_mask(ccd_data):
    data = np.zeros((10, 10))
    data[5,5] = 1
    mask = (data == 0)
    ccd = CCDData(data, unit=u.adu, mask=mask)
    ccd_list = [ccd, ccd, ccd]
    c = Combiner(ccd_list)
    assert c.data_arr.shape == (3, 10, 10)
    assert c.data_arr.mask.shape == (3, 10, 10)
    assert c.data_arr.mask[0,5,5] == False



def test_weights(ccd_data):
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    with pytest.raises(TypeError):
        c.weights = 1


def test_weights_shape(ccd_data):
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    with pytest.raises(ValueError):
        c.weights = ccd_data.data


#test the min-max rejection
def test_combiner_minmax(ccd_data):
    ccd_list = [CCDData(np.zeros((10, 10)), unit=u.adu),
                CCDData(np.zeros((10, 10)) - 1000, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 1000, unit=u.adu)]

    c = Combiner(ccd_list)
    c.minmax_clipping(min_clip=-500, max_clip=500)
    ccd = c.median_combine()
    assert ccd.data.mean() == 0


def test_combiner_minmax_max(ccd_data):
    ccd_list = [CCDData(np.zeros((10, 10)), unit=u.adu),
                CCDData(np.zeros((10, 10)) - 1000, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 1000, unit=u.adu)]

    c = Combiner(ccd_list)
    c.minmax_clipping(min_clip=None, max_clip=500)
    assert c.data_arr[2].mask.all()


def test_combiner_minmax_min(ccd_data):
    ccd_list = [CCDData(np.zeros((10, 10)), unit=u.adu),
                CCDData(np.zeros((10, 10)) - 1000, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 1000, unit=u.adu)]

    c = Combiner(ccd_list)
    c.minmax_clipping(min_clip=-500, max_clip=None)
    assert c.data_arr[1].mask.all()


def test_combiner_sigmaclip_high():
    ccd_list = [CCDData(np.zeros((10, 10)), unit=u.adu),
                CCDData(np.zeros((10, 10)) - 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) - 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 1000, unit=u.adu)]

    c = Combiner(ccd_list)
    #using mad for more rubust statistics vs. std
    c.sigma_clipping(high_thresh=3, low_thresh=None, func=np.median,
                     dev_func=mad)
    assert c.data_arr[5].mask.all()


def test_combiner_sigmaclip_low():
    ccd_list = [CCDData(np.zeros((10, 10)), unit=u.adu),
                CCDData(np.zeros((10, 10)) - 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) - 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) - 1000, unit=u.adu)]

    c = Combiner(ccd_list)
    #using mad for more rubust statistics vs. std
    c.sigma_clipping(high_thresh=None, low_thresh=3, func=np.median,
                     dev_func=mad)
    assert c.data_arr[5].mask.all()


#test that the average combination works and returns a ccddata object
def test_combiner_median(ccd_data):
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.average_combine()
    assert isinstance(ccd, CCDData)
    assert ccd.shape == (100, 100)
    assert ccd.unit == u.adu
    assert ccd.meta['NCOMBINE'] == len(ccd_list)


#test that the median combination works and returns a ccddata object
def test_combiner_average(ccd_data):
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.average_combine()
    assert isinstance(ccd, CCDData)
    assert ccd.shape == (100, 100)
    assert ccd.unit == u.adu
    assert ccd.meta['NCOMBINE'] == len(ccd_list)

#test data combined with mask is created correctly
def test_combiner_mask_average(ccd_data):
    data = np.zeros((10, 10))
    data[5,5] = 1
    mask = (data == 0)
    ccd = CCDData(data, unit=u.adu, mask=mask)
    ccd_list = [ccd, ccd, ccd]
    c = Combiner(ccd_list)
    ccd = c.average_combine()
    assert ccd.data[0,0] == 0
    assert ccd.data[5,5] == 1
    assert ccd.mask[0,0] == True 
    assert ccd.mask[5,5] == False

#test data combined with mask is created correctly
def test_combiner_mask_media(ccd_data):
    data = np.zeros((10, 10))
    data[5,5] = 1
    mask = (data == 0)
    ccd = CCDData(data, unit=u.adu, mask=mask)
    ccd_list = [ccd, ccd, ccd]
    c = Combiner(ccd_list)
    ccd = c.median_combine()
    assert ccd.data[0,0] == 0
    assert ccd.data[5,5] == 1
    assert ccd.mask[0,0] == True 
    assert ccd.mask[5,5] == False

    
    

