# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

import numpy as np
from astropy.io import fits

from astropy.tests.helper import pytest
from astropy.utils import NumpyRNGContext

from ..ccddata import CCDData, electrons


def test_ccddata_empty():
    with pytest.raises(TypeError):
        CCDData()  # empty initializer should fail


def test_ccddata_must_have_unit():
    with pytest.raises(ValueError):
        CCDData(np.zeros([100, 100]))


@pytest.mark.data_size(10)
def test_ccddata_simple(ccd_data):
    assert ccd_data.shape == (10, 10)
    assert ccd_data.size == 100
    assert ccd_data.dtype == np.dtype(float)


@pytest.mark.data_size(10)
def test_initialize_from_FITS(ccd_data, tmpdir):
    hdu = fits.PrimaryHDU(ccd_data)
    hdulist = fits.HDUList([hdu])
    filename = tmpdir.join('afile.fits').strpath
    hdulist.writeto(filename)
    cd = CCDData.read(filename, unit=electrons)
    assert cd.shape == (10, 10)
    assert cd.size == 100
    assert np.issubdtype(cd.data.dtype, np.float)
    assert cd.meta == hdu.header


def test_initialize_from_FITS_bad_keyword_raises_error(ccd_data, tmpdir):
    # There are two fits.open keywords that are not permitted in ccdpro:
    #     do_not_scale_image_data and scale_back
    filename = tmpdir.join('test.fits').strpath
    ccd_data.write(filename)

    with pytest.raises(TypeError):
        CCDData.read(filename, unit=ccd_data.unit,
                     do_not_scale_image_data=True)
    with pytest.raises(TypeError):
        CCDData.read(filename, unit=ccd_data.unit, scale_back=True)


def test_ccddata_writer(ccd_data, tmpdir):
    filename = tmpdir.join('test.fits').strpath
    ccd_data.write(filename)

    ccd_disk = CCDData.read(filename, unit=ccd_data.unit)
    np.testing.assert_array_equal(ccd_data.data, ccd_disk.data)


def test_ccddata_meta_is_fits_header(ccd_data):
    ccd_data.meta = {'OBSERVER': 'Edwin Hubble'}
    assert isinstance(ccd_data.meta, fits.Header)


def test_fromMEF(ccd_data, tmpdir):
    hdu = fits.PrimaryHDU(ccd_data)
    hdu2 = fits.PrimaryHDU(2 * ccd_data.data)
    hdulist = fits.HDUList(hdu)
    hdulist.append(hdu2)
    filename = tmpdir.join('afile.fits').strpath
    hdulist.writeto(filename)
    # by default, we reading from the first extension
    cd = CCDData.read(filename, unit=electrons)
    np.testing.assert_array_equal(cd.data, ccd_data.data)
    # but reading from the second should work too
    cd = CCDData.read(filename, hdu=1, unit=electrons)
    np.testing.assert_array_equal(cd.data, 2 * ccd_data.data)


def test_metafromheader(ccd_data):
    hdr = fits.header.Header()
    hdr.set('observer', 'Edwin Hubble')
    hdr.set('exptime', '3600')

    d1 = CCDData(np.ones((5, 5)), meta=hdr, unit=electrons)
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'
    assert d1.header['OBSERVER'] == 'Edwin Hubble'


def test_metafromdict():
    dic = {'OBSERVER': 'Edwin Hubble', 'EXPTIME': 3600}
    d1 = CCDData(np.ones((5, 5)), meta=dic, unit=electrons)
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'


def test_header2meta():
    hdr = fits.header.Header()
    hdr.set('observer', 'Edwin Hubble')
    hdr.set('exptime', '3600')

    d1 = CCDData(np.ones((5, 5)), unit=electrons)
    d1.header = hdr
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'
    assert d1.header['OBSERVER'] == 'Edwin Hubble'


def test_metafromstring_fail():
    hdr = 'this is not a valid header'
    with pytest.raises(TypeError):
        d1 = CCDData(np.ones((5, 5)), meta=hdr)


def test_setting_bad_uncertainty_raises_error(ccd_data):
    with pytest.raises(TypeError):
        # Uncertainty is supposed to be an instance of NDUncertainty
        ccd_data.uncertainty = 10


def test_to_hdu(ccd_data):
    ccd_data.meta = {'observer': 'Edwin Hubble'}
    fits_hdulist = ccd_data.to_hdu()
    assert isinstance(fits_hdulist, fits.HDUList)


def test_copy(ccd_data):
    ccd_copy = ccd_data.copy()
    np.testing.assert_array_equal(ccd_copy.data, ccd_data.data)
    assert ccd_copy.unit == ccd_data.unit
    assert ccd_copy.meta == ccd_data.meta


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
