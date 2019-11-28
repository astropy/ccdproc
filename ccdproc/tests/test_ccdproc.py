# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.io import fits
from astropy.modeling import models
from astropy.units.quantity import Quantity
import astropy.units as u
from astropy.wcs import WCS
from astropy.tests.helper import catch_warnings
from astropy.utils.exceptions import AstropyUserWarning

from astropy.nddata import StdDevUncertainty, CCDData
import astropy

from numpy.testing import assert_array_equal
import pytest
import skimage

from ..core import (
    ccd_process, cosmicray_median, cosmicray_lacosmic, create_deviation,
    flat_correct, gain_correct, subtract_bias, subtract_dark, subtract_overscan,
    transform_image, trim_image, wcs_project, Keyword)
from ..core import _blkavg
from .pytest_fixtures import ccd_data as ccd_data_func

try:
    from ..core import block_reduce, block_average, block_replicate
    HAS_BLOCK_X_FUNCS = True
except ImportError:
    HAS_BLOCK_X_FUNCS = False


# test creating deviation
# success expected if u_image * u_gain = u_readnoise
@pytest.mark.parametrize('u_image,u_gain,u_readnoise,expect_success', [
                         (u.electron, None, u.electron, True),
                         (u.electron, u.electron, u.electron, False),
                         (u.adu, u.electron / u.adu, u.electron, True),
                         (u.electron, None, u.dimensionless_unscaled, False),
                         (u.electron, u.dimensionless_unscaled, u.electron, True),
                         (u.adu, u.dimensionless_unscaled, u.electron, False),
                         (u.adu, u.photon / u.adu, u.electron, False),
                         ])
def test_create_deviation(u_image, u_gain, u_readnoise,
                          expect_success):
    ccd_data = ccd_data_func(data_size=10, data_mean=100)
    ccd_data.unit = u_image
    if u_gain is not None:
        gain = 2.0 * u_gain
    else:
        gain = None
    readnoise = 5 * u_readnoise
    if expect_success:
        ccd_var = create_deviation(ccd_data, gain=gain, readnoise=readnoise)
        assert ccd_var.uncertainty.array.shape == (10, 10)
        assert ccd_var.uncertainty.array.size == 100
        assert ccd_var.uncertainty.array.dtype == np.dtype(float)
        if gain is not None:
            expected_var = np.sqrt(2 * ccd_data.data + 5 ** 2) / 2
        else:
            expected_var = np.sqrt(ccd_data.data + 5 ** 2)
        np.testing.assert_array_equal(ccd_var.uncertainty.array,
                                      expected_var)
        assert ccd_var.unit == ccd_data.unit
        # uncertainty should *not* have any units -- does it?
        with pytest.raises(AttributeError):
            ccd_var.uncertainty.array.unit
    else:
        with pytest.raises(u.UnitsError):
            ccd_var = create_deviation(ccd_data, gain=gain, readnoise=readnoise)


def test_create_deviation_from_negative():
    ccd_data = ccd_data_func(data_mean=0, data_scale=10)
    ccd_data.unit = u.electron
    readnoise = 5 * u.electron
    ccd_var = create_deviation(ccd_data, gain=None, readnoise=readnoise,
                               disregard_nan=False)
    np.testing.assert_array_equal(ccd_data.data < 0,
                                  np.isnan(ccd_var.uncertainty.array))


def test_create_deviation_from_negative():
    ccd_data = ccd_data_func(data_mean=0, data_scale=10)
    ccd_data.unit = u.electron
    readnoise = 5 * u.electron
    ccd_var = create_deviation(ccd_data, gain=None, readnoise=readnoise,
                               disregard_nan=True)
    mask = (ccd_data.data < 0)
    ccd_data.data[mask] = 0
    expected_var = np.sqrt(ccd_data.data + readnoise.value**2)
    np.testing.assert_array_equal(ccd_var.uncertainty.array,
                                  expected_var)


def test_create_deviation_keywords_must_have_unit():
    ccd_data = ccd_data_func()
    # gain must have units if provided
    with pytest.raises(TypeError):
        create_deviation(ccd_data, gain=3)
    # readnoise must have units
    with pytest.raises(TypeError):
        create_deviation(ccd_data, readnoise=5)
    # readnoise must be provided
    with pytest.raises(ValueError):
        create_deviation(ccd_data)


# tests for overscan
@pytest.mark.parametrize('data_rectangle', [False, True])
@pytest.mark.parametrize('median,transpose', [
                         (False, False),
                         (False, True),
                         (True, False), ])
def test_subtract_overscan(median, transpose, data_rectangle):
    ccd_data = ccd_data_func()
    # Make data non-square if desired
    if data_rectangle:
        ccd_data.data = ccd_data.data[:, :-30]

    # create the overscan region
    oscan = 300.
    oscan_region = (slice(None), slice(0, 10))  # indices 0 through 9
    fits_section = '[1:10, :]'
    science_region = (slice(None), slice(10, None))

    overscan_axis = 1
    if transpose:
        # Put overscan in first axis, not second, a test for #70
        oscan_region = oscan_region[::-1]
        fits_section = '[:, 1:10]'
        science_region = science_region[::-1]
        overscan_axis = 0

    ccd_data.data[oscan_region] = oscan
    # Add a fake sky background so the "science" part of the image has a
    # different average than the "overscan" part.
    sky = 10.
    original_mean = ccd_data.data[science_region].mean()
    ccd_data.data[science_region] += oscan + sky
    # Test once using the overscan argument to specify the overscan region
    ccd_data_overscan = subtract_overscan(ccd_data,
                                          overscan=ccd_data[oscan_region],
                                          overscan_axis=overscan_axis,
                                          median=median, model=None)
    # Is the mean of the "science" region the sum of sky and the mean the
    # "science" section had before backgrounds were added?
    np.testing.assert_almost_equal(
        ccd_data_overscan.data[science_region].mean(),
        sky + original_mean)
    # Is the overscan region zero?
    assert (ccd_data_overscan.data[oscan_region] == 0).all()

    # Now do what should be the same subtraction, with the overscan specified
    # with the fits_section
    ccd_data_fits_section = subtract_overscan(ccd_data,
                                              overscan_axis=overscan_axis,
                                              fits_section=fits_section,
                                              median=median, model=None)
    # Is the mean of the "science" region the sum of sky and the mean the
    # "science" section had before backgrounds were added?
    np.testing.assert_almost_equal(
        ccd_data_fits_section.data[science_region].mean(),
        sky + original_mean)
    # Is the overscan region zero?
    assert (ccd_data_fits_section.data[oscan_region] == 0).all()

    # Do both ways of subtracting overscan give exactly the same result?
    np.testing.assert_array_equal(ccd_data_overscan[science_region],
                                  ccd_data_fits_section[science_region])

    # Set overscan_axis to None, and let the routine figure out the axis.
    # This should lead to the same results as before.
    ccd_data_overscan_auto = subtract_overscan(
        ccd_data, overscan_axis=None, overscan=ccd_data[oscan_region],
        median=median, model=None)
    np.testing.assert_almost_equal(
        ccd_data_overscan_auto.data[science_region].mean(),
        sky + original_mean)
    # Use overscan_axis=None with a FITS section
    ccd_data_fits_section_overscan_auto = subtract_overscan(
        ccd_data, overscan_axis=None, fits_section=fits_section,
        median=median, model=None)
    np.testing.assert_almost_equal(
        ccd_data_fits_section_overscan_auto.data[science_region].mean(),
        sky + original_mean)
    # overscan_axis should be 1 for a square overscan region
    # This test only works for a non-square data region, but the
    # default has the wrong axis.
    if data_rectangle:
        ccd_data.data = ccd_data.data.T
        oscan_region = (slice(None), slice(0, -30))
        science_region = (slice(None), slice(-30, None))
        ccd_data_square_overscan_auto = subtract_overscan(
            ccd_data, overscan_axis=None, overscan=ccd_data[oscan_region],
            median=median, model=None)
        ccd_data_square = subtract_overscan(
            ccd_data, overscan_axis=1, overscan=ccd_data[oscan_region],
            median=median, model=None)
        np.testing.assert_allclose(ccd_data_square_overscan_auto,
                                   ccd_data_square)


# A more substantial test of overscan modeling
@pytest.mark.parametrize('transpose', [
                         True,
                         False])
def test_subtract_overscan_model(transpose):
    ccd_data = ccd_data_func()
    # create the overscan region
    size = ccd_data.shape[0]

    oscan_region = (slice(None), slice(0, 10))
    science_region = (slice(None), slice(10, None))

    yscan, xscan = np.mgrid[0:size, 0:size] / 10.0 + 300.0

    if transpose:
        oscan_region = oscan_region[::-1]
        science_region = science_region[::-1]
        scan = xscan
        overscan_axis = 0
    else:
        overscan_axis = 1
        scan = yscan

    original_mean = ccd_data.data[science_region].mean()

    ccd_data.data[oscan_region] = 0.  # only want overscan in that region
    ccd_data.data = ccd_data.data + scan

    ccd_data = subtract_overscan(ccd_data, overscan=ccd_data[oscan_region],
                                 overscan_axis=overscan_axis,
                                 median=False, model=models.Polynomial1D(2))
    np.testing.assert_almost_equal(ccd_data.data[science_region].mean(),
                                   original_mean)
    # Set the overscan_axis explicitly to None, and let the routine
    # figure it out.
    ccd_data = subtract_overscan(ccd_data, overscan=ccd_data[oscan_region],
                                 overscan_axis=None,
                                 median=False, model=models.Polynomial1D(2))
    np.testing.assert_almost_equal(ccd_data.data[science_region].mean(),
                                   original_mean)


def test_subtract_overscan_fails():
    ccd_data = ccd_data_func()
    # do we get an error if the *image* is neither CCDData nor an array?
    with pytest.raises(TypeError):
        subtract_overscan(3, np.zeros((5, 5)))
    # do we get an error if the *overscan* is not an image or an array?
    with pytest.raises(TypeError):
        subtract_overscan(np.zeros((10, 10)), 3, median=False, model=None)
    # Do we get an error if we specify both overscan and fits_section?
    with pytest.raises(TypeError):
        subtract_overscan(ccd_data, overscan=ccd_data[0:10],
                          fits_section='[1:10]')
    # do we raise an error if we specify neither overscan nor fits_section?
    with pytest.raises(TypeError):
        subtract_overscan(ccd_data)
    # Does a fits_section which is not a string raise an error?
    with pytest.raises(TypeError):
        subtract_overscan(ccd_data, fits_section=5)


def test_trim_image_fits_section_requires_string():
    ccd_data = ccd_data_func()
    with pytest.raises(TypeError):
        trim_image(ccd_data, fits_section=5)


@pytest.mark.parametrize('mask_data, uncertainty', [
                         (False, False),
                         (True, True)])
def test_trim_image_fits_section(mask_data, uncertainty):
    ccd_data = ccd_data_func(data_size=50)
    if mask_data:
        ccd_data.mask = np.zeros_like(ccd_data)
    if uncertainty:
        err = np.random.normal(size=ccd_data.shape)
        ccd_data.uncertainty = StdDevUncertainty(err)

    trimmed = trim_image(ccd_data, fits_section='[20:40,:]')
    # FITS reverse order, bounds are inclusive and starting index is 1-based
    assert trimmed.shape == (50, 21)
    np.testing.assert_array_equal(trimmed.data, ccd_data[:, 19:40])
    if mask_data:
        assert trimmed.shape == trimmed.mask.shape
    if uncertainty:
        assert trimmed.shape == trimmed.uncertainty.array.shape


def test_trim_image_no_section():
    ccd_data = ccd_data_func(data_size=50)
    trimmed = trim_image(ccd_data[:, 19:40])
    assert trimmed.shape == (50, 21)
    np.testing.assert_array_equal(trimmed.data, ccd_data[:, 19:40])


def test_trim_with_wcs_alters_wcs():
    ccd_data = ccd_data_func()
    # WCS construction example pulled form astropy.wcs docs
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = np.array(ccd_data.shape)/2
    wcs.wcs.cdelt = np.array([-0.066667, 0.066667])
    wcs.wcs.crval = [0, -90]
    wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    wcs.wcs.set_pv([(2, 1, 45.0)])
    ccd_wcs = CCDData(ccd_data, wcs=wcs)
    # The trim below should subtract 10 from the 2nd element of crpix.
    # (Second element because the FITS convention for index ordering is
    #  opposite that of python)
    trimmed = trim_image(ccd_wcs[10:, :])
    assert trimmed.wcs.wcs.crpix[1] == wcs.wcs.crpix[1] - 10


def test_subtract_bias():
    ccd_data = ccd_data_func()
    data_avg = ccd_data.data.mean()
    bias_level = 5.0
    ccd_data.data = ccd_data.data + bias_level
    ccd_data.header['key'] = 'value'
    master_bias_array = np.zeros_like(ccd_data.data) + bias_level
    master_bias = CCDData(master_bias_array, unit=ccd_data.unit)
    no_bias = subtract_bias(ccd_data, master_bias, add_keyword=None)
    # Does the data we are left with have the correct average?
    np.testing.assert_almost_equal(no_bias.data.mean(), data_avg)
    # With logging turned off, metadata should not change
    assert no_bias.header == ccd_data.header
    del no_bias.header['key']
    assert 'key' in ccd_data.header
    assert no_bias.header is not ccd_data.header


def test_subtract_bias_fails():
    ccd_data = ccd_data_func(data_size=50)
    # Should fail if shapes don't match
    bias = CCDData(np.array([200, 200]), unit=u.adu)
    with pytest.raises(ValueError):
        subtract_bias(ccd_data, bias)
    # should fail because units don't match
    bias = CCDData(np.zeros_like(ccd_data), unit=u.meter)
    with pytest.raises(u.UnitsError):
        subtract_bias(ccd_data, bias)


@pytest.mark.parametrize('exposure_keyword', [True, False])
@pytest.mark.parametrize('explicit_times', [True, False])
@pytest.mark.parametrize('scale', [True, False])
def test_subtract_dark(explicit_times, scale, exposure_keyword):
    ccd_data = ccd_data_func()
    exptime = 30.0
    exptime_key = 'exposure'
    exposure_unit = u.second
    dark_level = 1.7
    master_dark_data = np.zeros_like(ccd_data.data) + dark_level
    master_dark = CCDData(master_dark_data, unit=u.adu)
    master_dark.header[exptime_key] = 2 * exptime
    dark_exptime = master_dark.header[exptime_key]
    ccd_data.header[exptime_key] = exptime
    dark_exposure_unit = exposure_unit
    if explicit_times:
        # test case when units of dark and data exposures are different
        dark_exposure_unit = u.minute
        dark_sub = subtract_dark(ccd_data, master_dark,
                                 dark_exposure=dark_exptime * dark_exposure_unit,
                                 data_exposure=exptime * exposure_unit,
                                 scale=scale, add_keyword=None)
    elif exposure_keyword:
        key = Keyword(exptime_key, unit=u.second)
        dark_sub = subtract_dark(ccd_data, master_dark,
                                 exposure_time=key,
                                 scale=scale, add_keyword=None)
    else:
        dark_sub = subtract_dark(ccd_data, master_dark,
                                 exposure_time=exptime_key,
                                 exposure_unit=u.second,
                                 scale=scale, add_keyword=None)

    dark_scale = 1.0
    if scale:
        dark_scale = float((exptime / dark_exptime) *
                           (exposure_unit / dark_exposure_unit))

    np.testing.assert_array_equal(ccd_data.data - dark_scale * dark_level,
                                  dark_sub.data)
    # Headers should have the same content...do they?
    assert dark_sub.header == ccd_data.header
    # But the headers should not be the same object -- a copy was made
    assert dark_sub.header is not ccd_data.header


def test_subtract_dark_fails():
    ccd_data = ccd_data_func()
    # None of these tests check a result so the content of the master
    # can be anything.
    ccd_data.header['exptime'] = 30.0
    master = ccd_data.copy()

    # Do we fail if we give one of dark_exposure, data_exposure but not both?
    with pytest.raises(TypeError):
        subtract_dark(ccd_data, master, dark_exposure=30 * u.second)
    with pytest.raises(TypeError):
        subtract_dark(ccd_data, master, data_exposure=30 * u.second)

    # Do we fail if we supply dark_exposure and data_exposure and exposure_time
    with pytest.raises(TypeError):
        subtract_dark(ccd_data, master, dark_exposure=10 * u.second,
                      data_exposure=10 * u.second,
                      exposure_time='exptime')

    # Fail if we supply none of the exposure-related arguments?
    with pytest.raises(TypeError):
        subtract_dark(ccd_data, master)

    # Fail if we supply exposure time but not a unit?
    with pytest.raises(TypeError):
        subtract_dark(ccd_data, master, exposure_time='exptime')

    # Fail if ccd_data or master are not CCDData objects?
    with pytest.raises(TypeError):
        subtract_dark(ccd_data.data, master, exposure_time='exptime')
    with pytest.raises(TypeError):
        subtract_dark(ccd_data, master.data, exposure_time='exptime')

    # Fail if units do not match...

    # ...when there is no scaling?
    master = CCDData(ccd_data)
    master.unit = u.meter

    with pytest.raises(u.UnitsError) as e:
        subtract_dark(ccd_data, master, exposure_time='exptime',
                      exposure_unit=u.second)
    assert "uncalibrated image" in str(e.value)

    # fail when the arrays are not the same size
    with pytest.raises(ValueError):
        small_master = CCDData(ccd_data)
        small_master.data = np.zeros((1, 1))
        subtract_dark(ccd_data, small_master)


def test_unit_mismatch_behaves_as_expected():
    ccd_data = ccd_data_func()
    """
    Test to alert us to any changes in how errors are raised in astropy when units
    do not match.
    """
    bad_unit = ccd_data.copy()
    bad_unit.unit = u.meter

    if astropy.__version__.startswith('1.0'):
        expected_error = ValueError
        expected_message = 'operand units'
    else:
        expected_error = u.UnitConversionError
        # Make this an empty string, which always matches. In this case
        # we are really only checking by the type of error raised.
        expected_message = ''

    # Did we raise the right error?
    with pytest.raises(expected_error) as e:
        ccd_data.subtract(bad_unit)

    # Was the error message as expected?
    assert expected_message in str(e.value)


# test for flat correction
def test_flat_correct():
    ccd_data = ccd_data_func(data_scale=10)
    # add metadata to header for a test below...
    ccd_data.header['my_key'] = 42
    size = ccd_data.shape[0]
    # create the flat, with some scatter
    data = 2 * np.random.normal(loc=1.0, scale=0.05, size=(size, size))
    flat = CCDData(data, meta=fits.header.Header(), unit=ccd_data.unit)
    flat_data = flat_correct(ccd_data, flat, add_keyword=None)

    # check that the flat was normalized
    # Should be the case that flat * flat_data = ccd_data * flat.data.mean
    # if the normalization was done correctly.
    np.testing.assert_almost_equal((flat_data.data * flat.data).mean(),
                                   ccd_data.data.mean() * flat.data.mean())
    np.testing.assert_allclose(ccd_data.data / flat_data.data,
                               flat.data / flat.data.mean())

    # check that metadata is unchanged (since logging is turned off)
    assert flat_data.header == ccd_data.header


# test for flat correction with min_value
def test_flat_correct_min_value(data_scale=1, data_mean=5):
    ccd_data = ccd_data_func()
    size = ccd_data.shape[0]

    # create the flat
    data = 2 * np.random.normal(loc=1.0, scale=0.05, size=(size, size))
    flat = CCDData(data, meta=fits.header.Header(), unit=ccd_data.unit)
    flat_orig_data = flat.data.copy()
    min_value = 2.1  # should replace some, but not all, values
    flat_corrected_data = flat_correct(ccd_data, flat, min_value=min_value)
    flat_with_min = flat.copy()
    flat_with_min.data[flat_with_min.data < min_value] = min_value

    # Check that the flat was normalized. The asserts below, which look a
    # little odd, are correctly testing that
    #    flat_corrected_data = ccd_data / (flat_with_min / mean(flat_with_min))
    np.testing.assert_almost_equal(
        (flat_corrected_data.data * flat_with_min.data).mean(),
        (ccd_data.data * flat_with_min.data.mean()).mean()
    )
    np.testing.assert_allclose(ccd_data.data / flat_corrected_data.data,
                               flat_with_min.data / flat_with_min.data.mean())

    # Test that flat is not modified.
    assert (flat_orig_data == flat.data).all()
    assert flat_orig_data is not flat.data


def test_flat_correct_norm_value():
    ccd_data = ccd_data_func(data_scale=10)
    # Test flat correction with mean value that is different than
    # the mean of the flat frame.

    # create the flat, with some scatter
    # Note that mean value of flat is set below and is different than
    # the mean of the flat data.
    flat_mean = 5.0
    data = np.random.normal(loc=1.0, scale=0.05, size=ccd_data.shape)
    flat = CCDData(data, meta=fits.Header(), unit=ccd_data.unit)
    flat_data = flat_correct(ccd_data, flat, add_keyword=None,
                             norm_value=flat_mean)

    # check that the flat was normalized
    # Should be the case that flat * flat_data = ccd_data * flat_mean
    # if the normalization was done correctly.
    np.testing.assert_almost_equal((flat_data.data * flat.data).mean(),
                                   ccd_data.data.mean() * flat_mean)
    np.testing.assert_allclose(ccd_data.data / flat_data.data,
                               flat.data / flat_mean)


def test_flat_correct_norm_value_bad_value():
    ccd_data = ccd_data_func()
    # Test that flat_correct raises the appropriate error if
    # it is given a bad norm_value. Bad means <=0.

    # create the flat, with some scatter
    data = np.random.normal(loc=1.0, scale=0.05, size=ccd_data.shape)
    flat = CCDData(data, meta=fits.Header(), unit=ccd_data.unit)
    with pytest.raises(ValueError) as e:
        flat_correct(ccd_data, flat, add_keyword=None, norm_value=-7)
    assert "norm_value must be" in str(e.value)


# test for deviation and for flat correction
def test_flat_correct_deviation():
    ccd_data = ccd_data_func(data_scale=10, data_mean=300)
    size = ccd_data.shape[0]
    ccd_data.unit = u.electron
    ccd_data = create_deviation(ccd_data, readnoise=5 * u.electron)
    # create the flat
    data = 2 * np.ones((size, size))
    flat = CCDData(data, meta=fits.header.Header(), unit=ccd_data.unit)
    flat = create_deviation(flat, readnoise=0.5 * u.electron)
    ccd_data = flat_correct(ccd_data, flat)


# test the uncertainty on the data after flat correction
def test_flat_correct_data_uncertainty():
    # Regression test for #345
    dat = CCDData(np.ones([100, 100]), unit='adu',
                  uncertainty=np.ones([100, 100]))
    # Note flat is set to 10, error, if present, is set to one.
    flat = CCDData(10 * np.ones([100, 100]), unit='adu')
    res = flat_correct(dat, flat)
    assert (res.data == dat.data).all()
    assert (res.uncertainty.array == dat.uncertainty.array).all()


# tests for gain correction
def test_gain_correct():
    ccd_data = ccd_data_func()
    init_data = ccd_data.data
    gain_data = gain_correct(ccd_data, gain=3, add_keyword=None)
    assert_array_equal(gain_data.data, 3 * init_data)
    assert ccd_data.meta == gain_data.meta


def test_gain_correct_quantity():
    ccd_data = ccd_data_func()
    init_data = ccd_data.data
    g = Quantity(3, u.electron / u.adu)
    ccd_data = gain_correct(ccd_data, gain=g)

    assert_array_equal(ccd_data.data, 3 * init_data)
    assert ccd_data.unit == u.electron


# test transform is ccd
def test_transform_isccd():
    with pytest.raises(TypeError):
        transform_image(1, 1)


# test function is callable
def test_transform_isfunc():
    ccd_data = ccd_data_func()
    with pytest.raises(TypeError):
        transform_image(ccd_data, 1)


# test warning is issue if WCS information is available
def test_catch_transform_wcs_warning():
    ccd_data = ccd_data_func()

    def tran(arr):
        return 10 * arr

    with catch_warnings() as w:
        tran = transform_image(ccd_data, tran)


@pytest.mark.parametrize('mask_data, uncertainty', [
                         (False, False),
                         (True, True)])
def test_transform_image(mask_data, uncertainty):
    ccd_data = ccd_data_func(data_size=50)
    if mask_data:
        ccd_data.mask = np.zeros_like(ccd_data)
        ccd_data.mask[10, 10] = 1
    if uncertainty:
        err = np.random.normal(size=ccd_data.shape)
        ccd_data.uncertainty = StdDevUncertainty(err)

    def tran(arr):
        return 10 * arr

    tran = transform_image(ccd_data, tran)

    assert_array_equal(10 * ccd_data.data, tran.data)
    if mask_data:
        assert tran.shape == tran.mask.shape
        assert_array_equal(ccd_data.mask, tran.mask)
    if uncertainty:
        assert tran.shape == tran.uncertainty.array.shape
        assert_array_equal(10 * ccd_data.uncertainty.array,
                           tran.uncertainty.array)


# test block_reduce and block_replicate wrapper
@pytest.mark.skipif(not HAS_BLOCK_X_FUNCS, reason="needs astropy >= 1.1.x")
@pytest.mark.skipif((skimage.__version__ < '0.14.2') and
                    ('dev' in np.__version__),
                    reason="Incompatibility between scikit-image "
                           "and numpy 1.16")
def test_block_reduce():
    ccd = CCDData(np.ones((4, 4)), unit='adu', meta={'testkw': 1},
                  mask=np.zeros((4, 4), dtype=bool),
                  uncertainty=StdDevUncertainty(np.ones((4, 4)))
                  )
    with catch_warnings(AstropyUserWarning) as w:
        ccd_summed = block_reduce(ccd, (2, 2))
    assert len(w) == 1
    assert 'following attributes were set' in str(w[0].message)
    assert isinstance(ccd_summed, CCDData)
    assert np.all(ccd_summed.data == 4)
    assert ccd_summed.data.shape == (2, 2)
    assert ccd_summed.unit == u.adu
    # Other attributes are set to None. In case the function is modified to
    # work on these attributes correctly those tests need to be updated!
    assert ccd_summed.meta == {'testkw': 1}
    assert ccd_summed.mask is None
    assert ccd_summed.uncertainty is None

    # Make sure meta is copied
    ccd_summed.meta['testkw2'] = 10
    assert 'testkw2' not in ccd.meta


@pytest.mark.skipif(not HAS_BLOCK_X_FUNCS, reason="needs astropy >= 1.1.x")
@pytest.mark.skipif((skimage.__version__ < '0.14.2') and
                    ('dev' in np.__version__),
                    reason="Incompatibility between scikit-image "
                           "and numpy 1.16")
def test_block_average():
    ccd = CCDData(np.ones((4, 4)), unit='adu', meta={'testkw': 1},
                  mask=np.zeros((4, 4), dtype=bool),
                  uncertainty=StdDevUncertainty(np.ones((4, 4))))
    ccd.data[::2, ::2] = 2
    with catch_warnings(AstropyUserWarning) as w:
        ccd_avgd = block_average(ccd, (2, 2))
    assert len(w) == 1
    assert 'following attributes were set' in str(w[0].message)

    assert isinstance(ccd_avgd, CCDData)
    assert np.all(ccd_avgd.data == 1.25)
    assert ccd_avgd.data.shape == (2, 2)
    assert ccd_avgd.unit == u.adu
    # Other attributes are set to None. In case the function is modified to
    # work on these attributes correctly those tests need to be updated!
    assert ccd_avgd.meta == {'testkw': 1}
    assert ccd_avgd.mask is None
    assert ccd_avgd.wcs is None
    assert ccd_avgd.uncertainty is None

    # Make sure meta is copied
    ccd_avgd.meta['testkw2'] = 10
    assert 'testkw2' not in ccd.meta


@pytest.mark.skipif(not HAS_BLOCK_X_FUNCS, reason="needs astropy >= 1.1.x")
def test_block_replicate():
    ccd = CCDData(np.ones((4, 4)), unit='adu', meta={'testkw': 1},
                  mask=np.zeros((4, 4), dtype=bool),
                  uncertainty=StdDevUncertainty(np.ones((4, 4))))
    with catch_warnings(AstropyUserWarning) as w:
        ccd_repl = block_replicate(ccd, (2, 2))
    assert len(w) == 1
    assert 'following attributes were set' in str(w[0].message)

    assert isinstance(ccd_repl, CCDData)
    assert np.all(ccd_repl.data == 0.25)
    assert ccd_repl.data.shape == (8, 8)
    assert ccd_repl.unit == u.adu
    # Other attributes are set to None. In case the function is modified to
    # work on these attributes correctly those tests need to be updated!
    assert ccd_repl.meta == {'testkw': 1}
    assert ccd_repl.mask is None
    assert ccd_repl.wcs is None
    assert ccd_repl.uncertainty is None

    # Make sure meta is copied
    ccd_repl.meta['testkw2'] = 10
    assert 'testkw2' not in ccd.meta


# test blockaveraging ndarray
def test__blkavg_ndarray():
    with pytest.raises(TypeError):
        _blkavg(1, (5, 5))


# test rebinning dimensions
def test__blkavg_dimensions():
    ccd_data = ccd_data_func(data_size=10)
    with pytest.raises(ValueError):
        _blkavg(ccd_data.data, (5,))


# test blkavg works
def test__blkavg_larger():
    ccd_data = ccd_data_func(data_size=20)
    a = ccd_data.data
    b = _blkavg(a, (10, 10))

    assert b.shape == (10, 10)
    np.testing.assert_almost_equal(b.sum(), 0.25 * a.sum())


# test overscan changes
def test__overscan_schange():
    ccd_data = ccd_data_func()
    old_data = ccd_data.copy()
    new_data = subtract_overscan(ccd_data, overscan=ccd_data[:, 1], overscan_axis=0)
    assert not np.allclose(old_data.data, new_data.data)
    np.testing.assert_array_equal(old_data.data, ccd_data.data)


def test_create_deviation_does_not_change_input():
    ccd_data = ccd_data_func()
    original = ccd_data.copy()
    ccd = create_deviation(ccd_data, gain=5 * u.electron / u.adu, readnoise=10 * u.electron)
    np.testing.assert_array_equal(original.data, ccd_data.data)
    assert original.unit == ccd_data.unit


def test_cosmicray_median_does_not_change_input():
    ccd_data = ccd_data_func()
    original = ccd_data.copy()
    error = np.zeros_like(ccd_data)
    ccd = cosmicray_median(ccd_data, error_image=error, thresh=5, mbox=11, gbox=0, rbox=0)
    np.testing.assert_array_equal(original.data, ccd_data.data)
    assert original.unit == ccd_data.unit


def test_cosmicray_lacosmic_does_not_change_input():
    ccd_data = ccd_data_func()
    original = ccd_data.copy()
    error = np.zeros_like(ccd_data)
    ccd = cosmicray_lacosmic(ccd_data)
    np.testing.assert_array_equal(original.data, ccd_data.data)
    assert original.unit == ccd_data.unit


def test_flat_correct_does_not_change_input():
    ccd_data = ccd_data_func()
    original = ccd_data.copy()
    flat = CCDData(np.zeros_like(ccd_data), unit=ccd_data.unit)
    ccd = flat_correct(ccd_data, flat=flat)
    np.testing.assert_array_equal(original.data, ccd_data.data)
    assert original.unit == ccd_data.unit


def test_gain_correct_does_not_change_input():
    ccd_data = ccd_data_func()
    original = ccd_data.copy()
    ccd = gain_correct(ccd_data, gain=1, gain_unit=ccd_data.unit)
    np.testing.assert_array_equal(original.data, ccd_data.data)
    assert original.unit == ccd_data.unit


def test_subtract_bias_does_not_change_input():
    ccd_data = ccd_data_func()
    original = ccd_data.copy()
    master_frame = CCDData(np.zeros_like(ccd_data), unit=ccd_data.unit)
    ccd = subtract_bias(ccd_data, master=master_frame)
    np.testing.assert_array_equal(original.data, ccd_data.data)
    assert original.unit == ccd_data.unit


def test_trim_image_does_not_change_input():
    ccd_data = ccd_data_func()
    original = ccd_data.copy()
    ccd = trim_image(ccd_data, fits_section=None)
    np.testing.assert_array_equal(original.data, ccd_data.data)
    assert original.unit == ccd_data.unit


def test_transform_image_does_not_change_input():
    ccd_data = ccd_data_func()
    original = ccd_data.copy()
    ccd = transform_image(ccd_data, np.sqrt)
    np.testing.assert_array_equal(original.data, ccd_data)
    assert original.unit == ccd_data.unit


def wcs_for_testing(shape):
    # Set up a simply WCS, details are cut/pasted from astropy WCS docs,
    # mostly. CRPIX is set to the center of shape, rounded down.

    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = WCS(naxis=2)

    # Set up an "Airy's zenithal" projection
    # Vector properties may be set with Python lists, or Numpy arrays
    w.wcs.crpix = [shape[0] // 2, shape[1] // 2]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.crval = [0, -90]
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    w.wcs.set_pv([(2, 1, 45.0)])

    return w


def test_wcs_project_onto_same_wcs():
    ccd_data = ccd_data_func()
    # The trivial case, same WCS, no mask.
    target_wcs = wcs_for_testing(ccd_data.shape)
    ccd_data.wcs = wcs_for_testing(ccd_data.shape)

    new_ccd = wcs_project(ccd_data, target_wcs)

    # Make sure new image has correct WCS.
    assert new_ccd.wcs.wcs.compare(target_wcs.wcs)

    # Make sure data matches within some reasonable tolerance.
    np.testing.assert_allclose(ccd_data.data, new_ccd.data, rtol=1e-5)


def test_wcs_project_onto_same_wcs_remove_headers():
    ccd_data = ccd_data_func()
    # Remove an example WCS keyword from the header
    target_wcs = wcs_for_testing(ccd_data.shape)
    ccd_data.wcs = wcs_for_testing(ccd_data.shape)
    print(ccd_data.header)
    ccd_data.header = ccd_data.wcs.to_header()

    new_ccd = wcs_project(ccd_data, target_wcs)

    for k in ccd_data.wcs.to_header():
        assert k not in new_ccd.header


def test_wcs_project_onto_shifted_wcs():
    ccd_data = ccd_data_func()
    # Just make the target WCS the same as the initial with the center
    # pixel shifted by 1 in x and y.

    ccd_data.wcs = wcs_for_testing(ccd_data.shape)
    target_wcs = wcs_for_testing(ccd_data.shape)
    target_wcs.wcs.crpix += [1, 1]

    ccd_data.mask = np.random.choice([0, 1], size=ccd_data.shape)

    new_ccd = wcs_project(ccd_data, target_wcs)

    # Make sure new image has correct WCS.
    assert new_ccd.wcs.wcs.compare(target_wcs.wcs)

    # Make sure data matches within some reasonable tolerance, keeping in mind
    # that the pixels should all be shifted.
    masked_input = np.ma.array(ccd_data.data, mask=ccd_data.mask)
    masked_output = np.ma.array(new_ccd.data, mask=new_ccd.mask)
    np.testing.assert_allclose(masked_input[:-1, :-1],
                               masked_output[1:, 1:], rtol=1e-5)

    # The masks should all be shifted too.
    np.testing.assert_array_equal(ccd_data.mask[:-1, :-1],
                                  new_ccd.mask[1:, 1:])

    # We should have more values that are masked in the output array
    # than on input because some on output were not in the footprint
    # of the original array.

    # In the case of a shift, one row and one column should be nan, and they
    # will share one common nan where they intersect, so we know how many nan
    # there should be.
    assert np.isnan(new_ccd.data).sum() == np.sum(new_ccd.shape) - 1


# Use an odd number of pixels to make a well-defined center pixel
def test_wcs_project_onto_scale_wcs():
    # Make the target WCS with half the pixel scale and number of pixels
    # and the values should drop by a factor of 4.
    ccd_data = ccd_data_func(data_size=31)

    ccd_data.wcs = wcs_for_testing(ccd_data.shape)

    # Make sure wcs is centered at the center of the center pixel.
    ccd_data.wcs.wcs.crpix += 0.5

    # Use uniform input data value for simplicity.
    ccd_data.data = np.ones_like(ccd_data.data)

    # Make mask zero...
    ccd_data.mask = np.zeros_like(ccd_data.data)
    # ...except the center pixel, which is one.
    ccd_data.mask[int(ccd_data.wcs.wcs.crpix[0]),
                  int(ccd_data.wcs.wcs.crpix[1])] = 1

    target_wcs = wcs_for_testing(ccd_data.shape)
    target_wcs.wcs.cdelt /= 2

    # Choice below ensures we are really at the center pixel of an odd range.
    target_shape = 2 * np.array(ccd_data.shape) + 1
    target_wcs.wcs.crpix = 2 * target_wcs.wcs.crpix + 1 + 0.5

    # Explicitly set the interpolation method so we know what to
    # expect for the mass.
    new_ccd = wcs_project(ccd_data, target_wcs,
                          target_shape=target_shape,
                          order='nearest-neighbor')

    # Make sure new image has correct WCS.
    assert new_ccd.wcs.wcs.compare(target_wcs.wcs)

    # Define a cutout from the new array that should match the old.
    new_lower_bound = (np.array(new_ccd.shape) - np.array(ccd_data.shape)) // 2
    new_upper_bound = (np.array(new_ccd.shape) + np.array(ccd_data.shape)) // 2
    data_cutout = new_ccd.data[new_lower_bound[0]:new_upper_bound[0],
                               new_lower_bound[1]:new_upper_bound[1]]

    # Make sure data matches within some reasonable tolerance, keeping in mind
    # that the pixels have been scaled.
    np.testing.assert_allclose(ccd_data.data / 4,
                               data_cutout,
                               rtol=1e-5)

    # Mask should be true for four pixels (all nearest neighbors)
    # of the single pixel we masked initially.
    new_center = np.array(new_ccd.wcs.wcs.crpix, dtype=int, copy=False)
    assert np.all(new_ccd.mask[new_center[0]:new_center[0]+2,
                               new_center[1]:new_center[1]+2])

    # Those four, and any that reproject made nan because they draw on
    # pixels outside the footprint of the original image, are the only
    # pixels that should be masked.
    assert new_ccd.mask.sum() == 4 + np.isnan(new_ccd.data).sum()


def test_ccd_process_does_not_change_input():
    ccd_data = ccd_data_func()
    original = ccd_data.copy()
    ccd = ccd_process(ccd_data, gain=5 * u.electron / u.adu,
                      readnoise=10 * u.electron)
    np.testing.assert_array_equal(original.data, ccd_data.data)
    assert original.unit == ccd_data.unit


def test_ccd_process_parameters_are_appropriate():
    ccd_data = ccd_data_func()
    # oscan check
    with pytest.raises(TypeError):
        ccd_process(ccd_data, oscan=True)

    # trim section check
    with pytest.raises(TypeError):
        ccd_process(ccd_data, trim=True)

    # error frame check
    # gain and readnoise must be specified
    with pytest.raises(ValueError):
        ccd_process(ccd_data, error=True)

    # gain must be specified
    with pytest.raises(ValueError):
        ccd_process(ccd_data, error=True, gain=None, readnoise=5)

    # mask check
    with pytest.raises(TypeError):
        ccd_process(ccd_data, bad_pixel_mask=3)

    # master bias check
    with pytest.raises(TypeError):
        ccd_process(ccd_data, master_bias=3)

    # master flat check
    with pytest.raises(TypeError):
        ccd_process(ccd_data, master_flat=3)


def test_ccd_process():
    # test the through ccd_process
    ccd_data = CCDData(10.0 * np.ones((100, 100)), unit=u.adu)
    ccd_data.data[:, -10:] = 2
    ccd_data.meta['testkw'] = 100

    mask = np.zeros((100, 90))

    masterbias = CCDData(2.0 * np.ones((100, 90)), unit=u.electron)
    masterbias.uncertainty = StdDevUncertainty(np.zeros((100, 90)))

    dark_frame = CCDData(0.0 * np.ones((100, 90)), unit=u.electron)
    dark_frame.uncertainty = StdDevUncertainty(np.zeros((100, 90)))

    masterflat = CCDData(10.0 * np.ones((100, 90)), unit=u.electron)
    masterflat.uncertainty = StdDevUncertainty(np.zeros((100, 90)))

    occd = ccd_process(ccd_data, oscan=ccd_data[:, -10:], trim='[1:90,1:100]',
                       error=True, master_bias=masterbias,
                       master_flat=masterflat, dark_frame=dark_frame,
                       bad_pixel_mask=mask, gain=0.5 * u.electron/u.adu,
                       readnoise=5**0.5 * u.electron, oscan_median=True,
                       dark_scale=False, dark_exposure=1.*u.s,
                       data_exposure=1.*u.s)

    # final results should be (10 - 2) / 2.0 - 2 = 2
    # error should be (4 + 5)**0.5 / 0.5  = 3.0

    np.testing.assert_array_equal(2.0 * np.ones((100, 90)), occd.data)
    np.testing.assert_almost_equal(3.0 * np.ones((100, 90)),
                                   occd.uncertainty.array)
    np.testing.assert_array_equal(mask, occd.mask)
    assert(occd.unit == u.electron)
    # Make sure the original keyword is still present. Regression test for #401
    assert occd.meta['testkw'] == 100


def test_ccd_process_gain_corrected():
    # test the through ccd_process with gain_corrected as False
    ccd_data = CCDData(10.0 * np.ones((100, 100)), unit=u.adu)
    ccd_data.data[:, -10:] = 2
    ccd_data.meta['testkw'] = 100

    mask = np.zeros((100, 90))

    masterbias = CCDData(4.0 * np.ones((100, 90)), unit=u.adu)
    masterbias.uncertainty = StdDevUncertainty(np.zeros((100, 90)))

    dark_frame = CCDData(0.0 * np.ones((100, 90)), unit=u.adu)
    dark_frame.uncertainty = StdDevUncertainty(np.zeros((100, 90)))

    masterflat = CCDData(5.0 * np.ones((100, 90)), unit=u.adu)
    masterflat.uncertainty = StdDevUncertainty(np.zeros((100, 90)))

    occd = ccd_process(ccd_data, oscan=ccd_data[:, -10:], trim='[1:90,1:100]',
                       error=True, master_bias=masterbias,
                       master_flat=masterflat, dark_frame=dark_frame,
                       bad_pixel_mask=mask, gain=0.5 * u.electron/u.adu,
                       readnoise=5**0.5 * u.electron, oscan_median=True,
                       dark_scale=False, dark_exposure=1.*u.s,
                       data_exposure=1.*u.s, gain_corrected=False)

    # final results should be (10 - 2) / 2.0 - 2 = 2
    # error should be (4 + 5)**0.5 / 0.5  = 3.0

    np.testing.assert_array_equal(2.0 * np.ones((100, 90)), occd.data)
    np.testing.assert_almost_equal(3.0 * np.ones((100, 90)),
                                   occd.uncertainty.array)
    np.testing.assert_array_equal(mask, occd.mask)
    assert(occd.unit == u.electron)
    # Make sure the original keyword is still present. Regression test for #401
    assert occd.meta['testkw'] == 100
