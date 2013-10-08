# Licensed under a 3-clause BSD style license - see LICENSE.rst
#This module implements the base CCDData class.

import numpy as np
from astropy.io import fits

from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest, raises
from astropy.utils import NumpyRNGContext

from ..ccddata import CCDData, electrons, fromFITS, toFITS
from ..ccdproc import *

import os

def writeout(cd, outfile):
    hdu = toFITS(cd)
    if os.path.isfile(outfile): os.remove(outfile)
    hdu.writeto(outfile)

def test_cosmicray_clean_scalarbackground():
    with NumpyRNGContext(125):
        scale=5.3
        size=100
        data=np.random.normal(loc=0, size=(size,size),  scale=scale)
        cd = CCDData(data, meta=fits.header.Header())
        ncrays=30
        crrays=np.random.random_integers(0, size-1, size=(ncrays,2))
        crflux=10*scale*np.random.random(30)+5*scale
        for i in range(ncrays):
           y,x=crrays[i]
           cd.data[y,x]=crflux[i]
        testdata=1.0*cd.data
    cc=cosmicray_clean(cd, 5, cosmicray_median, crargs=(11,), background=scale, bargs=(),rbox=11, gbox=0)
    assert abs(cc.data.std()-scale)<0.1
    assert (testdata-cc.data > 0).sum()==ncrays

def test_cosmicray_clean():
    with NumpyRNGContext(125):
        scale=5.3
        size=100
        data=np.random.normal(loc=0, size=(size,size),  scale=scale)
        cd = CCDData(data, meta=fits.header.Header())
        ncrays=30
        crrays=np.random.random_integers(0, size-1, size=(ncrays,2))
        crflux=1000*scale*np.random.random(30)+5*scale
        for i in range(ncrays):
           y,x=crrays[i]
           cd.data[y,x]=crflux[i]
        testdata=1.0*cd.data
    cc=cd #currently here because of lacking a copy command for NDData
    for i in range(5):
        cc=cosmicray_clean(cc, 5.0, cosmicray_median, crargs=(11,), background=background_variance_box, bargs=(25,), rbox=11)
    assert abs(cc.data.std()-scale)<0.1
    assert (testdata-cc.data > 0).sum()==30

def test_cosmicray_median():
    with NumpyRNGContext(125):
        scale=5.3
        size=100
        cd = np.random.normal(loc=0, size=(size,size), scale=scale)
        ncrays=30
        crrays=np.random.random_integers(0, size-1, size=(ncrays,2))
        crflux=10*scale*np.random.random(30)+5*scale
        for i in range(ncrays):
           y,x=crrays[i]
           cd[y,x]=crflux[i]

    crarr=cosmicray_median(cd, 5, mbox=11, background=scale)

    #check the number of cosmic rays detected
    assert crarr.sum()==ncrays



def test_background_variance_box():
    with NumpyRNGContext(123):
        scale=5.3
        cd = np.random.normal(loc=0, size=(100, 100), scale=scale)
    bd=background_variance_box(cd, 25)
    assert abs(bd.mean()-scale)<0.10

@raises(ValueError)
def test_background_variance_box_fail():
    with NumpyRNGContext(123):
        scale=5.3
        cd = np.random.normal(loc=0, size=(100, 100), scale=scale)
    bd=background_variance_box(cd, 0.5)
    
def test_background_variance_filter():
    with NumpyRNGContext(123):
        scale=5.3
        cd = np.random.normal(loc=0, size=(100, 100), scale=scale)
    bd=background_variance_filter(cd, 25)
    assert abs(bd.mean()-scale)<0.10

@raises(ValueError)
def test_background_variance_filter_fail():
    with NumpyRNGContext(123):
        scale=5.3
        cd = np.random.normal(loc=0, size=(100, 100), scale=scale)
    bd=background_variance_filter(cd, 0.5)


