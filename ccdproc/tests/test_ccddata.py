# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

import numpy as np
from astropy.io import fits

from astropy.tests.helper import pytest
from astropy.utils import NumpyRNGContext
from astropy.nddata import StdDevUncertainty
from astropy import units as u

from ..ccddata import CCDData, electron


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
    cd = CCDData.read(filename, unit=electron)
    assert cd.shape == (10, 10)
    assert cd.size == 100
    assert np.issubdtype(cd.data.dtype, np.float)
    for k, v in hdu.header.items():
        assert cd.meta[k] == v


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


def test_ccddata_meta_is_case_insensitive(ccd_data):
    key = 'SoMeKEY'
    ccd_data.meta[key] = 10
    assert key.lower() in ccd_data.meta
    assert key.upper() in ccd_data.meta


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
    cd = CCDData.read(filename, unit=electron)
    np.testing.assert_array_equal(cd.data, ccd_data.data)
    # but reading from the second should work too
    cd = CCDData.read(filename, hdu=1, unit=electron)
    np.testing.assert_array_equal(cd.data, 2 * ccd_data.data)


def test_metafromheader(ccd_data):
    hdr = fits.header.Header()
    hdr.set('observer', 'Edwin Hubble')
    hdr.set('exptime', '3600')

    d1 = CCDData(np.ones((5, 5)), meta=hdr, unit=electron)
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'
    assert d1.header['OBSERVER'] == 'Edwin Hubble'


def test_metafromdict():
    dic = {'OBSERVER': 'Edwin Hubble', 'EXPTIME': 3600}
    d1 = CCDData(np.ones((5, 5)), meta=dic, unit=electron)
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'


def test_header2meta():
    hdr = fits.header.Header()
    hdr.set('observer', 'Edwin Hubble')
    hdr.set('exptime', '3600')

    d1 = CCDData(np.ones((5, 5)), unit=electron)
    d1.header = hdr
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'
    assert d1.header['OBSERVER'] == 'Edwin Hubble'


def test_metafromstring_fail():
    hdr = 'this is not a valid header'
    with pytest.raises(TypeError):
        CCDData(np.ones((5, 5)), meta=hdr)


def test_setting_bad_uncertainty_raises_error(ccd_data):
    with pytest.raises(TypeError):
        # Uncertainty is supposed to be an instance of NDUncertainty
        ccd_data.uncertainty = 10


def test_to_hdu(ccd_data):
    ccd_data.meta = {'observer': 'Edwin Hubble'}
    fits_hdulist = ccd_data.to_hdu()
    assert isinstance(fits_hdulist, fits.HDUList)
    for k, v in ccd_data.meta.iteritems():
        assert fits_hdulist[0].header[k] == v
    np.testing.assert_array_equal(fits_hdulist[0].data, ccd_data.data)


def test_to_hdu_long_metadata_item(ccd_data):
    # There is no attempt to try to handle the general problem of
    # a long keyword (that requires HIERARCH) with a long string value
    # (that requires CONTINUE).
    # However, a long-ish keyword with a long value can happen because of
    # auto-logging, and we are supposed to handle that.

    # So, a nice long command:
    from ..ccdproc import subtract_dark, _short_names

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
