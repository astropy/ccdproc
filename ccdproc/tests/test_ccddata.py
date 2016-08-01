# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.io import fits

from astropy.tests.helper import pytest
from astropy.nddata import StdDevUncertainty
from astropy import units as u
from astropy.extern import six
from astropy import log
from astropy.wcs import WCS

from ..ccddata import CCDData
from .. import subtract_dark


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


def test_ccddata_init_with_string_electron_unit():
    ccd = CCDData(np.zeros((10, 10)), unit="electron")
    assert ccd.unit is u.electron


@pytest.mark.data_size(10)
def test_initialize_from_FITS(ccd_data, tmpdir):
    hdu = fits.PrimaryHDU(ccd_data)
    hdulist = fits.HDUList([hdu])
    filename = tmpdir.join('afile.fits').strpath
    hdulist.writeto(filename)
    cd = CCDData.read(filename, unit=u.electron)
    assert cd.shape == (10, 10)
    assert cd.size == 100
    assert np.issubdtype(cd.data.dtype, np.float)
    for k, v in hdu.header.items():
        assert cd.meta[k] == v


def test_initialize_from_fits_with_unit_in_header(tmpdir):
    fake_img = np.random.random(size=(100, 100))
    hdu = fits.PrimaryHDU(fake_img)
    hdu.header['bunit'] = u.adu.to_string()
    filename = tmpdir.join('afile.fits').strpath
    hdu.writeto(filename)
    ccd = CCDData.read(filename)
    # ccd should pick up the unit adu from the fits header...did it?
    assert ccd.unit is u.adu

    # An explicit unit in the read overrides any unit in the FITS file
    ccd2 = CCDData.read(filename, unit="photon")
    assert ccd2.unit is u.photon


def test_initialize_from_fits_with_ADU_in_header(tmpdir):
    fake_img = np.random.random(size=(100, 100))
    hdu = fits.PrimaryHDU(fake_img)
    hdu.header['bunit'] = 'ADU'
    filename = tmpdir.join('afile.fits').strpath
    hdu.writeto(filename)
    ccd = CCDData.read(filename)
    # ccd should pick up the unit adu from the fits header...did it?
    assert ccd.unit is u.adu


def test_initialize_from_fits_with_data_in_different_extension(tmpdir):
    fake_img = np.random.random(size=(100, 100))
    new_hdul = fits.HDUList()
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU(fake_img)
    hdus = fits.HDUList([hdu1, hdu2])
    filename = tmpdir.join('afile.fits').strpath
    hdus.writeto(filename)
    ccd = CCDData.read(filename, unit='adu')
    # ccd should pick up the unit adu from the fits header...did it?
    np.testing.assert_array_equal(ccd.data, fake_img)
    # check that the header is the combined header
    assert hdu1.header + hdu2.header == ccd.header


def test_initialize_from_fits_with_extension(tmpdir):
    fake_img1 = np.random.random(size=(100, 100))
    fake_img2 = np.random.random(size=(100, 100))
    new_hdul = fits.HDUList()
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(fake_img1)
    hdu2 = fits.ImageHDU(fake_img2)
    hdus = fits.HDUList([hdu0, hdu1, hdu2])
    filename = tmpdir.join('afile.fits').strpath
    hdus.writeto(filename)
    ccd = CCDData.read(filename, hdu=2, unit='adu')
    # ccd should pick up the unit adu from the fits header...did it?
    np.testing.assert_array_equal(ccd.data, fake_img2)


def test_write_unit_to_hdu(ccd_data, tmpdir):
    ccd_unit = ccd_data.unit
    hdulist = ccd_data.to_hdu()
    assert 'bunit' in hdulist[0].header
    assert hdulist[0].header['bunit'] == ccd_unit.to_string()


def test_initialize_from_FITS_bad_keyword_raises_error(ccd_data, tmpdir):
    # There are two fits.open keywords that are not permitted in ccdproc:
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


def test_ccddata_meta_is_case_sensitive(ccd_data):
    key = 'SoMeKEY'
    ccd_data.meta[key] = 10
    assert key.lower() not in ccd_data.meta
    assert key.upper() not in ccd_data.meta
    assert key in ccd_data.meta


def test_ccddata_meta_is_not_fits_header(ccd_data):
    ccd_data.meta = {'OBSERVER': 'Edwin Hubble'}
    assert not isinstance(ccd_data.meta, fits.Header)


def test_fromMEF(ccd_data, tmpdir):
    hdu = fits.PrimaryHDU(ccd_data)
    hdu2 = fits.PrimaryHDU(2 * ccd_data.data)
    hdulist = fits.HDUList(hdu)
    hdulist.append(hdu2)
    filename = tmpdir.join('afile.fits').strpath
    hdulist.writeto(filename)
    # by default, we reading from the first extension
    cd = CCDData.read(filename, unit=u.electron)
    np.testing.assert_array_equal(cd.data, ccd_data.data)
    # but reading from the second should work too
    cd = CCDData.read(filename, hdu=1, unit=u.electron)
    np.testing.assert_array_equal(cd.data, 2 * ccd_data.data)


def test_metafromheader(ccd_data):
    hdr = fits.header.Header()
    hdr['observer'] = 'Edwin Hubble'
    hdr['exptime'] = '3600'

    d1 = CCDData(np.ones((5, 5)), meta=hdr, unit=u.electron)
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'
    assert d1.header['OBSERVER'] == 'Edwin Hubble'


def test_metafromdict():
    dic = {'OBSERVER': 'Edwin Hubble', 'EXPTIME': 3600}
    d1 = CCDData(np.ones((5, 5)), meta=dic, unit=u.electron)
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'


def test_header2meta():
    hdr = fits.header.Header()
    hdr['observer'] = 'Edwin Hubble'
    hdr['exptime'] = '3600'

    d1 = CCDData(np.ones((5, 5)), unit=u.electron)
    d1.header = hdr
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'
    assert d1.header['OBSERVER'] == 'Edwin Hubble'


def test_metafromstring_fail():
    hdr = 'this is not a valid header'
    with pytest.raises(TypeError):
        CCDData(np.ones((5, 5)), meta=hdr, unit=u.adu)


def test_setting_bad_uncertainty_raises_error(ccd_data):
    with pytest.raises(TypeError):
        # Uncertainty is supposed to be an instance of NDUncertainty
        ccd_data.uncertainty = 10


def test_setting_uncertainty_with_array(ccd_data):
    ccd_data.uncertainty = None
    fake_uncertainty = np.sqrt(np.abs(ccd_data.data))
    ccd_data.uncertainty = fake_uncertainty.copy()
    np.testing.assert_array_equal(ccd_data.uncertainty.array, fake_uncertainty)


def test_setting_uncertainty_wrong_shape_raises_error(ccd_data):
    with pytest.raises(ValueError):
        ccd_data.uncertainty = np.random.random(size=2 * ccd_data.shape)


def test_to_hdu(ccd_data):
    ccd_data.meta = {'observer': 'Edwin Hubble'}
    fits_hdulist = ccd_data.to_hdu()
    assert isinstance(fits_hdulist, fits.HDUList)
    for k, v in ccd_data.meta.items():
        assert fits_hdulist[0].header[k] == v
    np.testing.assert_array_equal(fits_hdulist[0].data, ccd_data.data)


def test_to_hdu_long_metadata_item(ccd_data):
    # There is no attempt to try to handle the general problem of
    # a long keyword (that requires HIERARCH) with a long string value
    # (that requires CONTINUE).
    # However, a long-ish keyword with a long value can happen because of
    # auto-logging, and we are supposed to handle that.

    # So, a nice long command:
    from ..core import subtract_dark, _short_names

    dark = CCDData(np.zeros_like(ccd_data.data), unit="adu")
    result = subtract_dark(ccd_data, dark, dark_exposure=30 * u.second,
                           data_exposure=15 * u.second, scale=True)
    assert 'subtract_dark' in result.header
    hdulist = result.to_hdu()
    header = hdulist[0].header
    assert header['subtract_dark'] == _short_names['subtract_dark']
    args_value = header[_short_names['subtract_dark']]
    # Yuck -- have to hand code the ".0" to the numbers to get this to pass...
    assert "dark_exposure={0} {1}".format(30.0, u.second) in args_value
    assert "data_exposure={0} {1}".format(15.0, u.second) in args_value
    assert "scale=True" in args_value


def test_copy(ccd_data):
    ccd_copy = ccd_data.copy()
    np.testing.assert_array_equal(ccd_copy.data, ccd_data.data)
    assert ccd_copy.unit == ccd_data.unit
    assert ccd_copy.meta == ccd_data.meta


@pytest.mark.parametrize('operation,affects_uncertainty', [
                         ("multiply", True),
                         ("divide", True),
                         ])
@pytest.mark.parametrize('operand', [
                         2.0,
                         2 * u.dimensionless_unscaled,
                         2 * u.photon / u.adu,
                         ])
@pytest.mark.parametrize('with_uncertainty', [
                         True,
                         False])
@pytest.mark.data_unit(u.adu)
def test_mult_div_overload(ccd_data, operand, with_uncertainty,
                           operation, affects_uncertainty):
    if with_uncertainty:
        ccd_data.uncertainty = StdDevUncertainty(np.ones_like(ccd_data))
    method = ccd_data.__getattribute__(operation)
    np_method = np.__getattribute__(operation)
    result = method(operand)
    assert result is not ccd_data
    assert isinstance(result, CCDData)
    assert (result.uncertainty is None or
            isinstance(result.uncertainty, StdDevUncertainty))
    try:
        op_value = operand.value
    except AttributeError:
        op_value = operand

    np.testing.assert_array_equal(result.data,
                                  np_method(ccd_data.data, op_value))
    if with_uncertainty:
        if affects_uncertainty:
            np.testing.assert_array_equal(result.uncertainty.array,
                                          np_method(ccd_data.uncertainty.array,
                                                    op_value))
        else:
            np.testing.assert_array_equal(result.uncertainty.array,
                                          ccd_data.uncertainty.array)
    else:
        assert result.uncertainty is None

    if isinstance(operand, u.Quantity):
        # Need the "1 *" below to force arguments to be Quantity to work around
        # astropy/astropy#2377
        expected_unit = np_method(1 * ccd_data.unit, 1 * operand.unit).unit
        assert result.unit == expected_unit
    else:
        assert result.unit == ccd_data.unit


@pytest.mark.parametrize('operation,affects_uncertainty', [
                         ("add", False),
                         ("subtract", False),
                         ])
@pytest.mark.parametrize('operand,expect_failure', [
                         (2.0, u.UnitsError),  # fail--units don't match image
                         (2 * u.dimensionless_unscaled, u.UnitsError),  # same
                         (2 * u.adu, False),
                         ])
@pytest.mark.parametrize('with_uncertainty', [
                         True,
                         False])
@pytest.mark.data_unit(u.adu)
def test_add_sub_overload(ccd_data, operand, expect_failure, with_uncertainty,
                          operation, affects_uncertainty):
    if with_uncertainty:
        ccd_data.uncertainty = StdDevUncertainty(np.ones_like(ccd_data))
    method = ccd_data.__getattribute__(operation)
    np_method = np.__getattribute__(operation)
    if expect_failure:
        with pytest.raises(expect_failure):
            result = method(operand)
        return
    else:
        result = method(operand)
    assert result is not ccd_data
    assert isinstance(result, CCDData)
    assert (result.uncertainty is None or
            isinstance(result.uncertainty, StdDevUncertainty))
    try:
        op_value = operand.value
    except AttributeError:
        op_value = operand

    np.testing.assert_array_equal(result.data,
                                  np_method(ccd_data.data, op_value))
    if with_uncertainty:
        if affects_uncertainty:
            np.testing.assert_array_equal(result.uncertainty.array,
                                          np_method(ccd_data.uncertainty.array,
                                                    op_value))
        else:
            np.testing.assert_array_equal(result.uncertainty.array,
                                          ccd_data.uncertainty.array)
    else:
        assert result.uncertainty is None

    if isinstance(operand, u.Quantity):
        assert (result.unit == ccd_data.unit and result.unit == operand.unit)
    else:
        assert result.unit == ccd_data.unit


def test_arithmetic_overload_fails(ccd_data):
    with pytest.raises(TypeError):
        ccd_data.multiply("five")

    with pytest.raises(TypeError):
        ccd_data.divide("five")

    with pytest.raises(TypeError):
        ccd_data.add("five")

    with pytest.raises(TypeError):
        ccd_data.subtract("five")


def test_arithmetic_overload_ccddata_operand(ccd_data):
    ccd_data.uncertainty = StdDevUncertainty(np.ones_like(ccd_data))
    operand = ccd_data.copy()
    result = ccd_data.add(operand)
    assert len(result.meta) == 0
    np.testing.assert_array_equal(result.data,
                                  2 * ccd_data.data)
    np.testing.assert_array_equal(result.uncertainty.array,
                                  np.sqrt(2) * ccd_data.uncertainty.array)

    result = ccd_data.subtract(operand)
    assert len(result.meta) == 0
    np.testing.assert_array_equal(result.data,
                                  0 * ccd_data.data)
    np.testing.assert_array_equal(result.uncertainty.array,
                                  np.sqrt(2) * ccd_data.uncertainty.array)

    result = ccd_data.multiply(operand)
    assert len(result.meta) == 0
    np.testing.assert_array_equal(result.data,
                                  ccd_data.data ** 2)
    expected_uncertainty = (np.sqrt(2) * np.abs(ccd_data.data) *
                            ccd_data.uncertainty.array)
    np.testing.assert_allclose(result.uncertainty.array,
                               expected_uncertainty)

    result = ccd_data.divide(operand)
    assert len(result.meta) == 0
    np.testing.assert_array_equal(result.data,
                                  np.ones_like(ccd_data.data))
    expected_uncertainty = (np.sqrt(2) / np.abs(ccd_data.data) *
                            ccd_data.uncertainty.array)
    np.testing.assert_allclose(result.uncertainty.array,
                               expected_uncertainty)


def test_arithmetic_overload_differing_units():
    a = np.array([1, 2, 3]) * u.m
    b = np.array([1, 2, 3]) * u.cm
    ccddata = CCDData(a)

    # TODO: Could also be parametrized.
    res = ccddata.add(b)
    np.testing.assert_array_almost_equal(res.data, np.add(a, b).value)
    assert res.unit == np.add(a, b).unit

    res = ccddata.subtract(b)
    np.testing.assert_array_almost_equal(res.data, np.subtract(a, b).value)
    assert res.unit == np.subtract(a, b).unit

    res = ccddata.multiply(b)
    np.testing.assert_array_almost_equal(res.data, np.multiply(a, b).value)
    assert res.unit == np.multiply(a, b).unit

    res = ccddata.divide(b)
    np.testing.assert_array_almost_equal(res.data, np.divide(a, b).value)
    assert res.unit == np.divide(a, b).unit


def test_ccddata_header_does_not_corrupt_fits(ccd_data, tmpdir):
    # This test is for the problem described in astropy/ccdproc#165
    # The issue comes up when a long FITS keyword value is in a header
    # that is read in and then converted to a non-fits.Header object
    # that is dict-like, and then you try to write that out again as
    # FITS. Certainly FITS files out to be able to round-trip, and
    # this test checks for that.

    fake_dark = ccd_data.copy()
    # This generates a nice long log entry in the header.
    ccd = subtract_dark(ccd_data, fake_dark, dark_exposure=30*u.second,
                        data_exposure=30*u.second)
    # The write below succeeds...
    long_key = tmpdir.join('long_key.fit').strpath
    ccd.write(long_key)

    # And this read succeeds...
    ccd_read = CCDData.read(long_key, unit="adu")

    # This write failed in astropy/ccdproc#165 but should not:
    rewritten = tmpdir.join('should_work.fit').strpath
    ccd_read.write(rewritten)

    # If all is well then reading the file we just wrote should result in an
    # identical header.
    ccd_reread = CCDData.read(rewritten, unit="adu")
    assert ccd_reread.header == ccd_read.header


def test_ccddata_with_fits_header_as_meta_works_with_autologging(ccd_data,
                                                                 tmpdir):
    tmp_file = tmpdir.join('tmp.fits')
    hdr = fits.Header(ccd_data.header)
    ccd_data.header = hdr
    fake_dark = ccd_data.copy()
    # The combination below will generate a long keyword ('subtract_dark')
    # and a long value (the function signature) in autlogging.
    ccd2 = subtract_dark(ccd_data, fake_dark,
                         dark_exposure=30*u.second,
                         data_exposure=15*u.second,
                         scale=True)
    # This should not fail....
    ccd2.write(tmp_file.strpath)
    # And the header on ccd2 should be a subset of the written header; they
    # do not match exactly because the written header contains information
    # about the array size that is the hdr we created manually.
    ccd2_read = CCDData.read(tmp_file.strpath, unit=u.adu)
    for k, v in six.iteritems(ccd2.header):
        assert ccd2_read.header[k] == v


def test_history_preserved_if_metadata_is_fits_header(tmpdir):
    fake_img = np.random.random(size=(100, 100))
    hdu = fits.PrimaryHDU(fake_img)
    hdu.header['history'] = 'one'
    hdu.header['history'] = 'two'
    hdu.header['history'] = 'three'
    assert len(hdu.header['history']) == 3
    tmp_file = tmpdir.join('temp.fits').strpath
    hdu.writeto(tmp_file)

    ccd_read = CCDData.read(tmp_file, unit="adu")
    assert ccd_read.header['history'] == hdu.header['history']


def test_infol_logged_if_unit_in_fits_header(ccd_data, tmpdir):
    tmpfile = tmpdir.join('temp.fits')
    ccd_data.write(tmpfile.strpath)
    log.setLevel('INFO')
    explicit_unit_name = "photon"
    with log.log_to_list() as log_list:
        ccd_from_disk = CCDData.read(tmpfile.strpath, unit=explicit_unit_name)
        assert explicit_unit_name in log_list[0].message


def test_wcs_attribute(ccd_data, tmpdir):
    """
    Check that WCS attribute gets added to header, and that if a CCDData
    object is created from a FITS file with a header, and the WCS attribute
    is modified, then the CCDData object is turned back into an hdu, the
    WCS object overwrites the old WCS information in the header.
    """
    tmpfile = tmpdir.join('temp.fits')
    # This wcs example is taken from the astropy.wcs docs.
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = np.array(ccd_data.shape)/2
    wcs.wcs.cdelt = np.array([-0.066667, 0.066667])
    wcs.wcs.crval = [0, -90]
    wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    wcs.wcs.set_pv([(2, 1, 45.0)])
    ccd_data.header = ccd_data.to_hdu()[0].header
    ccd_data.header.extend(wcs.to_header(), useblanks=False)
    ccd_data.write(tmpfile.strpath)
    ccd_new = CCDData.read(tmpfile.strpath)
    original_header_length = len(ccd_new.header)
    # WCS attribute should be set for ccd_new
    assert ccd_new.wcs is not None
    # WCS attribute should be equal to wcs above.
    assert ccd_new.wcs.wcs == wcs.wcs

    # Converting CCDData object with wcs to an hdu shouldn't
    # create duplicate wcs-related entries in the header.
    ccd_new_hdu = ccd_new.to_hdu()[0]
    assert len(ccd_new_hdu.header) == original_header_length

    # Making a CCDData with WCS (but not WCS in the header) should lead to
    # WCS information in the header when it is converted to an HDU.
    ccd_wcs_not_in_header = CCDData(ccd_data.data, wcs=wcs, unit="adu")
    hdu = ccd_wcs_not_in_header.to_hdu()[0]
    wcs_header = wcs.to_header()
    for k in wcs_header.keys():
        # Skip these keywords if they are in the WCS header because they are
        # not WCS-specific.
        if k in ['', 'COMMENT', 'HISTORY']:
            continue
        # No keyword from the WCS should be in the header.
        assert k not in ccd_wcs_not_in_header.header
        # Every keyword in the WCS should be in the header of the HDU
        assert hdu.header[k] == wcs_header[k]

    # Now check that if WCS of a CCDData is modified, then the CCDData is
    # converted to an HDU, the WCS keywords in the header are overwritten
    # with the appropriate keywords from the header.
    #
    # ccd_new has a WCS and WCS keywords in the header, so try modifying
    # the WCS.
    ccd_new.wcs.wcs.cdelt *= 2
    ccd_new_hdu_mod_wcs = ccd_new.to_hdu()[0]
    assert ccd_new_hdu_mod_wcs.header['CDELT1'] == ccd_new.wcs.wcs.cdelt[0]
    assert ccd_new_hdu_mod_wcs.header['CDELT2'] == ccd_new.wcs.wcs.cdelt[1]


def test_header(ccd_data):
    a = {'Observer': 'Hubble'}
    ccd = CCDData(ccd_data, header=a)
    assert ccd.meta == a


def test_wcs_arithmetic(ccd_data):
    ccd_data.wcs = 5
    result = ccd_data.multiply(1.0)
    assert result.wcs == 5


@pytest.mark.parametrize('operation',
                         ['multiply', 'divide', 'add', 'subtract'])
def test_wcs_arithmetic_ccd(ccd_data, operation):
    ccd_data2 = ccd_data.copy()
    ccd_data.wcs = 5
    method = ccd_data.__getattribute__(operation)
    result = method(ccd_data2)
    assert result.wcs == ccd_data.wcs
    assert ccd_data2.wcs is None

@pytest.mark.parametrize('operation',
                         ['multiply', 'divide', 'add', 'subtract'])
def test_mask_arithmetic_ccd(ccd_data, operation):
    ccd_data2 = ccd_data.copy()
    ccd_data.mask = (ccd_data.data > 0)
    method = ccd_data.__getattribute__(operation)
    result = method(ccd_data2)
    np.testing.assert_equal(result.mask, ccd_data.mask)



def test_write_read_multiextensionfits_mask_default(ccd_data, tmpdir):
    # Test that if a mask is present the mask is saved and loaded by default.
    ccd_data.mask = ccd_data.data > 10
    filename = tmpdir.join('afile.fits').strpath
    ccd_data.write(filename)
    ccd_after = CCDData.read(filename)
    assert ccd_after.mask is not None
    np.testing.assert_array_equal(ccd_data.mask, ccd_after.mask)


def test_write_read_multiextensionfits_uncertainty_default(ccd_data, tmpdir):
    # Test that if a uncertainty is present it is saved and loaded by default.
    ccd_data.uncertainty = StdDevUncertainty(ccd_data.data * 10)
    filename = tmpdir.join('afile.fits').strpath
    ccd_data.write(filename)
    ccd_after = CCDData.read(filename)
    assert ccd_after.uncertainty is not None
    np.testing.assert_array_equal(ccd_data.uncertainty.array,
                                  ccd_after.uncertainty.array)


def test_write_read_multiextensionfits_not(ccd_data, tmpdir):
    # Test that writing mask and uncertainty can be disabled
    ccd_data.mask = ccd_data.data > 10
    ccd_data.uncertainty = StdDevUncertainty(ccd_data.data * 10)
    filename = tmpdir.join('afile.fits').strpath
    ccd_data.write(filename, hdu_mask=None, hdu_uncertainty=None)
    ccd_after = CCDData.read(filename)
    assert ccd_after.uncertainty is None
    assert ccd_after.mask is None


def test_write_read_multiextensionfits_custom_ext_names(ccd_data, tmpdir):
    # Test writing mask, uncertainty in another extension than default
    ccd_data.mask = ccd_data.data > 10
    ccd_data.uncertainty = StdDevUncertainty(ccd_data.data * 10)
    filename = tmpdir.join('afile.fits').strpath
    ccd_data.write(filename, hdu_mask='Fun', hdu_uncertainty='NoFun')

    # Try reading with defaults extension names
    ccd_after = CCDData.read(filename)
    assert ccd_after.uncertainty is None
    assert ccd_after.mask is None

    # Try reading with custom extension names
    ccd_after = CCDData.read(filename, hdu_mask='Fun', hdu_uncertainty='NoFun')
    assert ccd_after.uncertainty is not None
    assert ccd_after.mask is not None
    np.testing.assert_array_equal(ccd_data.mask, ccd_after.mask)
    np.testing.assert_array_equal(ccd_data.uncertainty.array,
                                  ccd_after.uncertainty.array)


def test_wcs(ccd_data):
    ccd_data.wcs = 5
    assert ccd_data.wcs == 5


def test_recognized_fits_formats():
    from ..ccddata import _recognized_fits_file_extensions

    # These are the extensions that are supposed to be supported.
    supported_extensions = ['fit', 'fits', 'fts']

    # Make sure they are actually supported.
    assert len(set(_recognized_fits_file_extensions) -
               set(supported_extensions)) == 0


def test_recognized_fits_formats_for_read_write(ccd_data, tmpdir):
    # Test that incorporates astropy/ccdproc#355, which asked that .fts
    # be auto-identified as a FITS file extension.
    from ..ccddata import _recognized_fits_file_extensions

    for ext in _recognized_fits_file_extensions:
        path = tmpdir.join("test.{}".format(ext))
        ccd_data.write(path.strpath)
        from_disk = CCDData.read(path.strpath)
        assert (ccd_data.data == from_disk.data).all()
