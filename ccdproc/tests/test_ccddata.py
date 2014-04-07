# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

import numpy as np
from astropy.io import fits

from astropy.tests.helper import pytest
from astropy.utils import NumpyRNGContext

from ..ccddata import CCDData, electrons, fromFITS, toFITS


def test_ccddata_empty():
    with pytest.raises(TypeError):
        CCDData()  # empty initializer should fail


@pytest.mark.data_size(10)
def test_ccddata_simple(ccd_data):
    assert ccd_data.shape == (10, 10)
    assert ccd_data.size == 100
    assert ccd_data.dtype == np.dtype(float)


@pytest.mark.data_size(10)
def test_fromFITS(ccd_data):
    hdu = fits.PrimaryHDU(ccd_data)
    hdulist = fits.HDUList([hdu])
    cd = fromFITS(hdulist)
    assert cd.shape == (10, 10)
    assert cd.size == 100
    assert cd.dtype == np.dtype(float)
    assert cd.meta == hdu.header


def test_fromMEF(ccd_data):
    hdu = fits.PrimaryHDU(ccd_data)
    hdulist = fits.HDUList([hdu, hdu])
    with pytest.raises(ValueError):
        cd = fromFITS(hdulist)


def test_metafromheader(ccd_data):
    hdr = fits.header.Header()
    hdr.set('observer', 'Edwin Hubble')
    hdr.set('exptime', '3600')

    d1 = CCDData(np.ones((5, 5)), meta=hdr)
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'
    assert d1.header['OBSERVER'] == 'Edwin Hubble'


def test_metafromdict():
    dic = {'OBSERVER': 'Edwin Hubble', 'EXPTIME': 3600}
    d1 = CCDData(np.ones((5, 5)), meta=dic)
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'


def test_header2meta():
    hdr = fits.header.Header()
    hdr.set('observer', 'Edwin Hubble')
    hdr.set('exptime', '3600')

    d1 = CCDData(np.ones((5, 5)))
    d1.header = hdr
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'
    assert d1.header['OBSERVER'] == 'Edwin Hubble'


def test_metafromstring_fail():
    hdr = 'this is not a valid header'
    with pytest.raises(TypeError):
        d1 = CCDData(np.ones((5, 5)), meta=hdr)


@pytest.mark.data_size(10)
def test_create_variance(ccd_data):
    ccd_data.unit = electrons
    ccd_data.create_variance(5)
    assert ccd_data.uncertainty.array.shape == (10, 10)
    assert ccd_data.uncertainty.array.size == 100
    assert ccd_data.uncertainty.array.dtype == np.dtype(float)


def test_setting_bad_uncertainty_raises_error(ccd_data):
    with pytest.raises(TypeError):
        # Uncertainty is supposed to be an instance of NDUncertainty
        ccd_data.uncertainty = 10


def test_create_variance_with_bad_image_units_raises_error(ccd_data):
    with pytest.raises(TypeError):
        ccd_data.create_variance(10)


def test_toFITS(ccd_data):
    ccd_data.meta = {'observer': 'Edwin Hubble'}
    fits_hdulist = toFITS(ccd_data)
    assert isinstance(fits_hdulist, fits.HDUList)


if __name__ == '__main__':
    test_ccddata_empty()
    test_ccddata_simple()
    test_fromFITS()
    test_fromMEF()
    test_metafromheader()
    test_metafromstring_fail()
    test_metafromdict()
    test_header2meta()
    test_create_variance()
