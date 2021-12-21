# Licensed under a 3-clause BSD style license - see LICENSE.rst
from packaging.version import Version, parse

import numpy as np

import astropy.units as u
from astropy.stats import median_absolute_deviation as mad
import astropy

import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy.nddata import CCDData

from ccdproc.combiner import (Combiner, combine, _calculate_step_sizes,
                              _default_std, sigma_func)
from ccdproc.image_collection import ImageFileCollection
from ccdproc.tests.pytest_fixtures import ccd_data as ccd_data_func

SUPER_OLD_ASTROPY = parse(astropy.__version__) < Version('4.3.0')

# Several tests have many more NaNs in them than real data. numpy generates
# lots of warnings in those cases and it makes more sense to suppress them
# than to generate them.
pytestmark = pytest.mark.filterwarnings(
    'ignore:All-NaN slice encountered:RuntimeWarning'
)


# test that the Combiner raises error if empty
def test_combiner_empty():
    with pytest.raises(TypeError):
        Combiner()  # empty initializer should fail


# test that the Combiner raises error if empty if ccd_list is None
def test_combiner_init_with_none():
    with pytest.raises(TypeError):
        Combiner(None)  # empty initializer should fail


# test that Combiner throws an error if input
# objects are not ccddata objects
def test_ccddata_combiner_objects():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, None]
    with pytest.raises(TypeError):
        Combiner(ccd_list)  # different objects should fail


# test that Combiner throws an error if input
# objects do not have the same size
def test_ccddata_combiner_size():
    ccd_data = ccd_data_func()
    ccd_large = CCDData(np.zeros((200, 100)), unit=u.adu)
    ccd_list = [ccd_data, ccd_data, ccd_large]
    with pytest.raises(TypeError):
        Combiner(ccd_list)  # arrays of different sizes should fail


# test that Combiner throws an error if input
# objects do not have the same units
def test_ccddata_combiner_units():
    ccd_data = ccd_data_func()
    ccd_large = CCDData(np.zeros((100, 100)), unit=u.second)
    ccd_list = [ccd_data, ccd_data, ccd_large]
    with pytest.raises(TypeError):
        Combiner(ccd_list)


# test if mask and data array are created
def test_combiner_create():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    assert c.data_arr.shape == (3, 100, 100)
    assert c.data_arr.mask.shape == (3, 100, 100)


# test if dtype matches the value that is passed
def test_combiner_dtype():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list, dtype=np.float32)
    assert c.data_arr.dtype == np.float32
    avg = c.average_combine()
    # dtype of average should match input dtype
    assert avg.dtype == c.dtype
    med = c.median_combine()
    # dtype of median should match dtype of input
    assert med.dtype == c.dtype
    result_sum = c.sum_combine()
    # dtype of sum should match dtype of input
    assert result_sum.dtype == c.dtype


# test mask is created from ccd.data
def test_combiner_mask():
    data = np.zeros((10, 10))
    data[5, 5] = 1
    mask = (data == 0)
    ccd = CCDData(data, unit=u.adu, mask=mask)
    ccd_list = [ccd, ccd, ccd]
    c = Combiner(ccd_list)
    assert c.data_arr.shape == (3, 10, 10)
    assert c.data_arr.mask.shape == (3, 10, 10)
    assert not c.data_arr.mask[0, 5, 5]


def test_weights():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    with pytest.raises(TypeError):
        c.weights = 1


def test_weights_shape():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    with pytest.raises(ValueError):
        c.weights = ccd_data.data


def test_1Dweights():
    ccd_list = [CCDData(np.zeros((10, 10)), unit=u.adu),
                CCDData(np.zeros((10, 10)) - 1000, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 1000, unit=u.adu)]

    c = Combiner(ccd_list)
    c.weights = np.array([1, 5, 10])
    ccd = c.average_combine()
    np.testing.assert_almost_equal(ccd.data, 312.5)


def test_pixelwise_weights():
    ccd_list = [CCDData(np.zeros((10, 10)), unit=u.adu),
                CCDData(np.zeros((10, 10)) - 1000, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 1000, unit=u.adu)]
    c = Combiner(ccd_list)
    c.weights = np.ones_like(c.data_arr)
    c.weights[:, 5, 5] = [1, 5, 10]
    ccd = c.average_combine()
    np.testing.assert_almost_equal(ccd.data[5, 5], 312.5)
    np.testing.assert_almost_equal(ccd.data[0, 0], 0)


# test the min-max rejection
def test_combiner_minmax():
    ccd_list = [CCDData(np.zeros((10, 10)), unit=u.adu),
                CCDData(np.zeros((10, 10)) - 1000, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 1000, unit=u.adu)]

    c = Combiner(ccd_list)
    c.minmax_clipping(min_clip=-500, max_clip=500)
    ccd = c.median_combine()
    assert ccd.data.mean() == 0


def test_combiner_minmax_max():
    ccd_list = [CCDData(np.zeros((10, 10)), unit=u.adu),
                CCDData(np.zeros((10, 10)) - 1000, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 1000, unit=u.adu)]

    c = Combiner(ccd_list)
    c.minmax_clipping(min_clip=None, max_clip=500)
    assert c.data_arr[2].mask.all()


def test_combiner_minmax_min():
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
    # using mad for more robust statistics vs. std
    c.sigma_clipping(high_thresh=3, low_thresh=None, func=np.ma.median,
                     dev_func=mad)
    assert c.data_arr[5].mask.all()


def test_combiner_sigmaclip_single_pix():
    ccd_list = [CCDData(np.zeros((10, 10)), unit=u.adu),
                CCDData(np.zeros((10, 10)) - 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) - 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) - 10, unit=u.adu)]
    c = Combiner(ccd_list)
    # add a single pixel in another array to check that
    # that one gets rejected
    c.data_arr[0, 5, 5] = 0
    c.data_arr[1, 5, 5] = -5
    c.data_arr[2, 5, 5] = 5
    c.data_arr[3, 5, 5] = -5
    c.data_arr[4, 5, 5] = 25
    c.sigma_clipping(high_thresh=3, low_thresh=None, func=np.ma.median,
                     dev_func=mad)
    assert c.data_arr.mask[4, 5, 5]


def test_combiner_sigmaclip_low():
    ccd_list = [CCDData(np.zeros((10, 10)), unit=u.adu),
                CCDData(np.zeros((10, 10)) - 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) - 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) + 10, unit=u.adu),
                CCDData(np.zeros((10, 10)) - 1000, unit=u.adu)]

    c = Combiner(ccd_list)
    # using mad for more robust statistics vs. std
    c.sigma_clipping(high_thresh=None, low_thresh=3, func=np.ma.median,
                     dev_func=mad)
    assert c.data_arr[5].mask.all()


@pytest.mark.parametrize('threshold', [1, 10])
def test_combiner_sigma_clip_use_astropy_same_result(threshold):
    # If we turn on use_astropy and make no other changes we should get exactly
    # the same result as if we use ccdproc sigma_clipping
    ccd_list = [ccd_data_func(rng_seed=seed + 1) for seed in range(10)]
    c_ccdp = Combiner(ccd_list)
    c_apy = Combiner(ccd_list)

    c_ccdp.sigma_clipping(low_thresh=threshold, high_thresh=threshold)
    c_apy.sigma_clipping(low_thresh=threshold, high_thresh=threshold,
                         use_astropy=True)

    np.testing.assert_allclose(c_ccdp.data_arr.mask, c_apy.data_arr.mask)


# test that the median combination works and returns a ccddata object
def test_combiner_median():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.median_combine()
    assert isinstance(ccd, CCDData)
    assert ccd.shape == (100, 100)
    assert ccd.unit == u.adu
    assert ccd.meta['NCOMBINE'] == len(ccd_list)


# test that the average combination works and returns a ccddata object
def test_combiner_average():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.average_combine()
    assert isinstance(ccd, CCDData)
    assert ccd.shape == (100, 100)
    assert ccd.unit == u.adu
    assert ccd.meta['NCOMBINE'] == len(ccd_list)


# test that the sum combination works and returns a ccddata object
def test_combiner_sum():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.sum_combine()
    assert isinstance(ccd, CCDData)
    assert ccd.shape == (100, 100)
    assert ccd.unit == u.adu
    assert ccd.meta['NCOMBINE'] == len(ccd_list)


# test weighted sum
def test_combiner_sum_weighted():
    ccd_data = CCDData(data=[[0, 1], [2, 3]], unit='adu')
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    c.weights = np.array([1, 2, 3])
    ccd = c.sum_combine()
    expected_result = sum(w * d.data for w, d in
                          zip(c.weights, ccd_list))
    np.testing.assert_almost_equal(ccd,
                                   expected_result)


# test weighted sum
def test_combiner_sum_weighted_by_pixel():
    ccd_data = CCDData(data=[[1, 2], [4, 8]], unit='adu')
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    # Weights below are chosen so that every entry in
    weights_pixel = [
        [8, 4],
        [2, 1]
    ]
    c.weights = np.array([weights_pixel] * 3)
    ccd = c.sum_combine()
    expected_result = [
        [24, 24],
        [24, 24]
    ]
    np.testing.assert_almost_equal(ccd, expected_result)


# This warning is generated by numpy and is expected when
# many pixels are masked.
@pytest.mark.filterwarnings(
    'ignore:Mean of empty slice:RuntimeWarning',
    'ignore:Degrees of freedom <= 0:RuntimeWarning'
)
def test_combiner_mask_average():
    # test data combined with mask is created correctly
    data = np.zeros((10, 10))
    data[5, 5] = 1
    mask = (data == 0)
    ccd = CCDData(data, unit=u.adu, mask=mask)
    ccd_list = [ccd, ccd, ccd]
    c = Combiner(ccd_list)

    ccd = c.average_combine()

    # How can we assert anything about the data if all values
    # are masked?!
    # assert ccd.data[0, 0] == 0
    assert ccd.data[5, 5] == 1
    assert ccd.mask[0, 0]
    assert not ccd.mask[5, 5]


def test_combiner_with_scaling():
    ccd_data = ccd_data_func()
    # The factors below are not particularly important; just avoid anything
    # whose average is 1.
    ccd_data_lower = ccd_data.multiply(3)
    ccd_data_higher = ccd_data.multiply(0.9)
    combiner = Combiner([ccd_data, ccd_data_higher, ccd_data_lower])
    # scale each array to the mean of the first image
    scale_by_mean = lambda x: ccd_data.data.mean() / np.ma.average(x)
    combiner.scaling = scale_by_mean
    avg_ccd = combiner.average_combine()
    # Does the mean of the scaled arrays match the value to which it was
    # scaled?
    np.testing.assert_almost_equal(avg_ccd.data.mean(),
                                   ccd_data.data.mean())
    assert avg_ccd.shape == ccd_data.shape
    median_ccd = combiner.median_combine()
    # Does median also scale to the correct value?
    np.testing.assert_almost_equal(np.median(median_ccd.data),
                                   np.median(ccd_data.data))

    # Set the scaling manually...
    combiner.scaling = [scale_by_mean(combiner.data_arr[i]) for i in range(3)]
    avg_ccd = combiner.average_combine()
    np.testing.assert_almost_equal(avg_ccd.data.mean(),
                                   ccd_data.data.mean())
    assert avg_ccd.shape == ccd_data.shape


def test_combiner_scaling_fails():
    ccd_data = ccd_data_func()
    combiner = Combiner([ccd_data, ccd_data.copy()])
    # Should fail unless scaling is set to a function or list-like
    with pytest.raises(TypeError):
        combiner.scaling = 5


# test data combined with mask is created correctly
def test_combiner_mask_median():
    data = np.zeros((10, 10))
    data[5, 5] = 1
    mask = (data == 0)
    ccd = CCDData(data, unit=u.adu, mask=mask)
    ccd_list = [ccd, ccd, ccd]
    c = Combiner(ccd_list)
    ccd = c.median_combine()
    # We should not check the data value for masked entries.
    # Instead, just check that entries are masked appropriately.
    assert ccd.mask[0, 0]
    assert ccd.data[5, 5] == 1
    assert not ccd.mask[5, 5]


# Ignore warnings generated because most values are masked
@pytest.mark.filterwarnings(
    'ignore:Degrees of freedom <= 0:RuntimeWarning'
)
def test_combiner_mask_sum():
    # test data combined with mask is created correctly
    data = np.zeros((10, 10))
    data[5, 5] = 1
    mask = (data == 0)
    ccd = CCDData(data, unit=u.adu, mask=mask)
    ccd_list = [ccd, ccd, ccd]
    c = Combiner(ccd_list)
    ccd = c.sum_combine()
    assert ccd.data[0, 0] == 0
    assert ccd.data[5, 5] == 3
    assert ccd.mask[0, 0]
    assert not ccd.mask[5, 5]


# test combiner convenience function reads fits file and combine as expected
def test_combine_average_fitsimages():
    fitsfile = get_pkg_data_filename('data/a8280271.fits', package='ccdproc.tests')
    ccd = CCDData.read(fitsfile, unit=u.adu)
    ccd_list = [ccd] * 3
    c = Combiner(ccd_list)
    ccd_by_combiner = c.average_combine()

    fitsfilename_list = [fitsfile] * 3
    avgccd = combine(fitsfilename_list, output_file=None,
                     method='average', unit=u.adu)
    # averaging same fits images should give back same fits image
    np.testing.assert_array_almost_equal(avgccd.data, ccd_by_combiner.data)


def test_combine_numpyndarray():
    """ Test of numpy ndarray implementation: #493

    Test the average combine using ``Combiner`` and ``combine`` with input
    ``img_list`` in the format of ``numpy.ndarray``.
    """
    fitsfile = get_pkg_data_filename('data/a8280271.fits')
    ccd = CCDData.read(fitsfile, unit=u.adu)
    ccd_list = [ccd] * 3
    c = Combiner(ccd_list)
    ccd_by_combiner = c.average_combine()

    fitsfilename_list = np.array([fitsfile] * 3)
    avgccd = combine(fitsfilename_list, output_file=None,
                     method='average', unit=u.adu)
    # averaging same fits images should give back same fits image
    np.testing.assert_array_almost_equal(avgccd.data, ccd_by_combiner.data)


def test_combiner_result_dtype():
    """Regression test: #391

    The result should have the appropriate dtype not the dtype of the first
    input."""
    ccd = CCDData(np.ones((3, 3), dtype=np.uint16), unit='adu')
    res = combine([ccd, ccd.multiply(2)])
    # The default dtype of Combiner is float64
    assert res.data.dtype == np.float64
    ref = np.ones((3, 3)) * 1.5
    np.testing.assert_array_almost_equal(res.data, ref)

    res = combine([ccd, ccd.multiply(2), ccd.multiply(3)], dtype=int)
    # The result dtype should be integer:
    assert res.data.dtype == np.int_
    ref = np.ones((3, 3)) * 2
    np.testing.assert_array_almost_equal(res.data, ref)


def test_combiner_image_file_collection_input(tmp_path):
    # Regression check for #754
    ccd = ccd_data_func()
    for i in range(3):
        ccd.write(tmp_path / f'ccd-{i}.fits')

    ifc = ImageFileCollection(tmp_path)
    comb = Combiner(ifc.ccds())
    np.testing.assert_array_almost_equal(ccd.data,
                                         comb.average_combine().data)


def test_combine_image_file_collection_input(tmp_path):
    # Another regression check for #754 but this time with the
    # combine function instead of Combiner
    ccd = ccd_data_func()
    for i in range(3):
        ccd.write(tmp_path / f'ccd-{i}.fits')

    ifc = ImageFileCollection(tmp_path)

    comb_files = combine(ifc.files_filtered(include_path=True),
                         method='average')

    comb_ccds = combine(ifc.ccds(), method='average')

    np.testing.assert_array_almost_equal(ccd.data,
                                         comb_files.data)
    np.testing.assert_array_almost_equal(ccd.data,
                                         comb_ccds.data)

    with pytest.raises(FileNotFoundError):
        # This should fail because the test is not running in the
        # folder where the images are.
        _ = combine(ifc.files_filtered())


# test combiner convenience function works with list of ccddata objects
def test_combine_average_ccddata():
    fitsfile = get_pkg_data_filename('data/a8280271.fits')
    ccd = CCDData.read(fitsfile, unit=u.adu)
    ccd_list = [ccd] * 3
    c = Combiner(ccd_list)
    ccd_by_combiner = c.average_combine()

    avgccd = combine(ccd_list, output_file=None, method='average', unit=u.adu)
    # averaging same ccdData should give back same images
    np.testing.assert_array_almost_equal(avgccd.data, ccd_by_combiner.data)


# test combiner convenience function reads fits file and
# and combine as expected when asked to run in limited memory
def test_combine_limitedmem_fitsimages():
    fitsfile = get_pkg_data_filename('data/a8280271.fits')
    ccd = CCDData.read(fitsfile, unit=u.adu)
    ccd_list = [ccd] * 5
    c = Combiner(ccd_list)
    ccd_by_combiner = c.average_combine()

    fitsfilename_list = [fitsfile] * 5
    avgccd = combine(fitsfilename_list, output_file=None, method='average',
                     mem_limit=1e6, unit=u.adu)
    # averaging same ccdData should give back same images
    np.testing.assert_array_almost_equal(avgccd.data, ccd_by_combiner.data)


# test combiner convenience function reads fits file and
# and combine as expected when asked to run in limited memory with scaling
def test_combine_limitedmem_scale_fitsimages():
    fitsfile = get_pkg_data_filename('data/a8280271.fits')
    ccd = CCDData.read(fitsfile, unit=u.adu)
    ccd_list = [ccd] * 5
    c = Combiner(ccd_list)
    # scale each array to the mean of the first image
    scale_by_mean = lambda x: ccd.data.mean() / np.ma.average(x)
    c.scaling = scale_by_mean
    ccd_by_combiner = c.average_combine()

    fitsfilename_list = [fitsfile] * 5
    avgccd = combine(fitsfilename_list, output_file=None, method='average',
                     mem_limit=1e6, scale=scale_by_mean, unit=u.adu)

    np.testing.assert_array_almost_equal(
        avgccd.data, ccd_by_combiner.data, decimal=4)


# test the optional uncertainty function in average_combine
def test_average_combine_uncertainty():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.average_combine(uncertainty_func=np.sum)
    uncert_ref = np.sum(c.data_arr, 0) / np.sqrt(3)
    np.testing.assert_array_equal(ccd.uncertainty.array, uncert_ref)

    # Compare this also to the "combine" call
    ccd2 = combine(ccd_list, method='average',
                   combine_uncertainty_function=np.sum)
    np.testing.assert_array_equal(ccd.data, ccd2.data)
    np.testing.assert_array_equal(
        ccd.uncertainty.array, ccd2.uncertainty.array)


# test the optional uncertainty function in median_combine
def test_median_combine_uncertainty():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.median_combine(uncertainty_func=np.sum)
    uncert_ref = np.sum(c.data_arr, 0) / np.sqrt(3)
    np.testing.assert_array_equal(ccd.uncertainty.array, uncert_ref)

    # Compare this also to the "combine" call
    ccd2 = combine(ccd_list, method='median',
                   combine_uncertainty_function=np.sum)
    np.testing.assert_array_equal(ccd.data, ccd2.data)
    np.testing.assert_array_equal(
        ccd.uncertainty.array, ccd2.uncertainty.array)


# test the optional uncertainty function in sum_combine
def test_sum_combine_uncertainty():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.sum_combine(uncertainty_func=np.sum)
    uncert_ref = np.sum(c.data_arr, 0) * np.sqrt(3)
    np.testing.assert_almost_equal(ccd.uncertainty.array, uncert_ref)

    # Compare this also to the "combine" call
    ccd2 = combine(ccd_list, method='sum', combine_uncertainty_function=np.sum)
    np.testing.assert_array_equal(ccd.data, ccd2.data)
    np.testing.assert_array_equal(
        ccd.uncertainty.array, ccd2.uncertainty.array)


# Ignore warnings generated because most values are masked
@pytest.mark.filterwarnings(
    'ignore:Mean of empty slice:RuntimeWarning',
    'ignore:Degrees of freedom <= 0:RuntimeWarning'
)
@pytest.mark.parametrize('mask_point', [True, False])
@pytest.mark.parametrize('comb_func',
                         ['average_combine', 'median_combine', 'sum_combine'])
def test_combine_result_uncertainty_and_mask(comb_func, mask_point):
    # Regression test for #774
    # Turns out combine does not return an uncertainty or mask if the input
    # CCDData has no uncertainty or mask, which makes very little sense.
    ccd_data = ccd_data_func()

    # Make sure the initial ccd_data has no uncertainty, which was the condition that
    # led to no uncertainty being returned.
    assert ccd_data.uncertainty is None

    if mask_point:
        # Make one pixel really negative so we can clip it and guarantee a resulting
        # pixel is masked.
        ccd_data.data[0, 0] = -1000

    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)

    c.minmax_clipping(min_clip=-100)

    expected_result = getattr(c, comb_func)()

    # Just need the first part of the name for the combine function
    combine_method_name = comb_func.split('_')[0]

    ccd_comb = combine(ccd_list, method=combine_method_name,
                       minmax_clip=True, minmax_clip_min=-100)

    np.testing.assert_array_almost_equal(ccd_comb.uncertainty.array,
                                         expected_result.uncertainty.array)

    # Check that the right point is masked, and only one point is
    # masked
    assert expected_result.mask[0, 0] == mask_point
    assert expected_result.mask.sum() == mask_point
    assert ccd_comb.mask[0, 0] == mask_point
    assert ccd_comb.mask.sum() == mask_point


# test resulting uncertainty is corrected for the number of images
def test_combiner_uncertainty_average():
    ccd_list = [CCDData(np.ones((10, 10)), unit=u.adu),
                CCDData(np.ones((10, 10)) * 2, unit=u.adu)]
    c = Combiner(ccd_list)
    ccd = c.average_combine()
    # Just the standard deviation of ccd data.
    ref_uncertainty = np.ones((10, 10)) / 2
    # Correction because we combined two images.
    ref_uncertainty /= np.sqrt(2)
    np.testing.assert_array_almost_equal(ccd.uncertainty.array,
                                         ref_uncertainty)


# test resulting uncertainty is corrected for the number of images (with mask)
def test_combiner_uncertainty_average_mask():
    mask = np.zeros((10, 10), dtype=np.bool_)
    mask[5, 5] = True
    ccd_with_mask = CCDData(np.ones((10, 10)), unit=u.adu, mask=mask)
    ccd_list = [ccd_with_mask,
                CCDData(np.ones((10, 10)) * 2, unit=u.adu),
                CCDData(np.ones((10, 10)) * 3, unit=u.adu)]
    c = Combiner(ccd_list)
    ccd = c.average_combine()
    # Just the standard deviation of ccd data.
    ref_uncertainty = np.ones((10, 10)) * np.std([1, 2, 3])
    # Correction because we combined two images.
    ref_uncertainty /= np.sqrt(3)
    ref_uncertainty[5, 5] = np.std([2, 3]) / np.sqrt(2)
    np.testing.assert_array_almost_equal(ccd.uncertainty.array,
                                         ref_uncertainty)


# test resulting uncertainty is corrected for the number of images (with mask)
def test_combiner_uncertainty_median_mask():
    mad_to_sigma = 1.482602218505602
    mask = np.zeros((10, 10), dtype=np.bool_)
    mask[5, 5] = True
    ccd_with_mask = CCDData(np.ones((10, 10)), unit=u.adu, mask=mask)
    ccd_list = [ccd_with_mask,
                CCDData(np.ones((10, 10)) * 2, unit=u.adu),
                CCDData(np.ones((10, 10)) * 3, unit=u.adu)]
    c = Combiner(ccd_list)
    ccd = c.median_combine()
    # Just the standard deviation of ccd data.
    ref_uncertainty = np.ones((10, 10)) * mad_to_sigma * mad([1, 2, 3])
    # Correction because we combined two images.
    ref_uncertainty /= np.sqrt(3)  # 0.855980789955
    ref_uncertainty[5, 5] = mad_to_sigma * \
        mad([2, 3]) / np.sqrt(2)  # 0.524179041254
    np.testing.assert_array_almost_equal(ccd.uncertainty.array,
                                         ref_uncertainty)


# test resulting uncertainty is corrected for the number of images (with mask)
def test_combiner_uncertainty_sum_mask():
    mask = np.zeros((10, 10), dtype=np.bool_)
    mask[5, 5] = True
    ccd_with_mask = CCDData(np.ones((10, 10)), unit=u.adu, mask=mask)
    ccd_list = [ccd_with_mask,
                CCDData(np.ones((10, 10)) * 2, unit=u.adu),
                CCDData(np.ones((10, 10)) * 3, unit=u.adu)]
    c = Combiner(ccd_list)
    ccd = c.sum_combine()
    # Just the standard deviation of ccd data.
    ref_uncertainty = np.ones((10, 10)) * np.std([1, 2, 3])
    ref_uncertainty *= np.sqrt(3)
    ref_uncertainty[5, 5] = np.std([2, 3]) * np.sqrt(2)
    np.testing.assert_array_almost_equal(ccd.uncertainty.array,
                                         ref_uncertainty)


def test_combiner_3d():
    data1 = CCDData(3 * np.ones((5, 5, 5)), unit=u.adu)
    data2 = CCDData(2 * np.ones((5, 5, 5)), unit=u.adu)
    data3 = CCDData(4 * np.ones((5, 5, 5)), unit=u.adu)

    ccd_list = [data1, data2, data3]

    c = Combiner(ccd_list)
    assert c.data_arr.shape == (3, 5, 5, 5)
    assert c.data_arr.mask.shape == (3, 5, 5, 5)

    ccd = c.average_combine()
    assert ccd.shape == (5, 5, 5)
    np.testing.assert_array_almost_equal(ccd.data, data1, decimal=4)


def test_3d_combiner_with_scaling():
    ccd_data = ccd_data_func()
    # The factors below are not particularly important; just avoid anything
    # whose average is 1.
    ccd_data = CCDData(np.ones((5, 5, 5)), unit=u.adu)
    ccd_data_lower = CCDData(3 * np.ones((5, 5, 5)), unit=u.adu)
    ccd_data_higher = CCDData(0.9 * np.ones((5, 5, 5)), unit=u.adu)
    combiner = Combiner([ccd_data, ccd_data_higher, ccd_data_lower])
    # scale each array to the mean of the first image
    scale_by_mean = lambda x: ccd_data.data.mean() / np.ma.average(x)
    combiner.scaling = scale_by_mean
    avg_ccd = combiner.average_combine()
    # Does the mean of the scaled arrays match the value to which it was
    # scaled?
    np.testing.assert_almost_equal(avg_ccd.data.mean(),
                                   ccd_data.data.mean())
    assert avg_ccd.shape == ccd_data.shape
    median_ccd = combiner.median_combine()
    # Does median also scale to the correct value?
    np.testing.assert_almost_equal(np.median(median_ccd.data),
                                   np.median(ccd_data.data))

    # Set the scaling manually...
    combiner.scaling = [scale_by_mean(combiner.data_arr[i]) for i in range(3)]
    avg_ccd = combiner.average_combine()
    np.testing.assert_almost_equal(avg_ccd.data.mean(),
                                   ccd_data.data.mean())
    assert avg_ccd.shape == ccd_data.shape


def test_clip_extrema_3d():
    ccdlist = [CCDData(np.ones((3, 3, 3)) * 90., unit="adu"),
               CCDData(np.ones((3, 3, 3)) * 20., unit="adu"),
               CCDData(np.ones((3, 3, 3)) * 10., unit="adu"),
               CCDData(np.ones((3, 3, 3)) * 40., unit="adu"),
               CCDData(np.ones((3, 3, 3)) * 25., unit="adu"),
               CCDData(np.ones((3, 3, 3)) * 35., unit="adu"),
               ]
    c = Combiner(ccdlist)
    c.clip_extrema(nlow=1, nhigh=1)
    result = c.average_combine()
    expected = CCDData(np.ones((3, 3, 3)) * 30, unit="adu")
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('comb_func',
                         ['average_combine', 'median_combine', 'sum_combine'])
def test_writeable_after_combine(tmpdir, comb_func):
    ccd_data = ccd_data_func()
    tmp_file = tmpdir.join('tmp.fits')
    from ..combiner import Combiner
    combined = Combiner([ccd_data for _ in range(3)])
    ccd2 = getattr(combined, comb_func)()
    # This should not fail because the resulting uncertainty has a mask
    ccd2.write(tmp_file.strpath)


def test_clip_extrema():
    ccdlist = [CCDData(np.ones((3, 5)) * 90., unit="adu"),
               CCDData(np.ones((3, 5)) * 20., unit="adu"),
               CCDData(np.ones((3, 5)) * 10., unit="adu"),
               CCDData(np.ones((3, 5)) * 40., unit="adu"),
               CCDData(np.ones((3, 5)) * 25., unit="adu"),
               CCDData(np.ones((3, 5)) * 35., unit="adu"),
               ]
    ccdlist[0].data[0, 1] = 3.1
    ccdlist[1].data[1, 2] = 100.1
    ccdlist[1].data[2, 0] = 100.1
    c = Combiner(ccdlist)
    c.clip_extrema(nlow=1, nhigh=1)
    result = c.average_combine()
    expected = [[30.0, 22.5, 30.0, 30.0, 30.0],
                [30.0, 30.0, 47.5, 30.0, 30.0],
                [47.5, 30.0, 30.0, 30.0, 30.0]]
    np.testing.assert_array_equal(result, expected)


def test_clip_extrema_via_combine():
    ccdlist = [CCDData(np.ones((3, 5)) * 90., unit="adu"),
               CCDData(np.ones((3, 5)) * 20., unit="adu"),
               CCDData(np.ones((3, 5)) * 10., unit="adu"),
               CCDData(np.ones((3, 5)) * 40., unit="adu"),
               CCDData(np.ones((3, 5)) * 25., unit="adu"),
               CCDData(np.ones((3, 5)) * 35., unit="adu"),
               ]
    ccdlist[0].data[0, 1] = 3.1
    ccdlist[1].data[1, 2] = 100.1
    ccdlist[1].data[2, 0] = 100.1
    result = combine(ccdlist, clip_extrema=True, nlow=1, nhigh=1,)
    expected = [[30.0, 22.5, 30.0, 30.0, 30.0],
                [30.0, 30.0, 47.5, 30.0, 30.0],
                [47.5, 30.0, 30.0, 30.0, 30.0]]
    np.testing.assert_array_equal(result, expected)


def test_clip_extrema_with_other_rejection():
    ccdlist = [CCDData(np.ones((3, 5)) * 90., unit="adu"),
               CCDData(np.ones((3, 5)) * 20., unit="adu"),
               CCDData(np.ones((3, 5)) * 10., unit="adu"),
               CCDData(np.ones((3, 5)) * 40., unit="adu"),
               CCDData(np.ones((3, 5)) * 25., unit="adu"),
               CCDData(np.ones((3, 5)) * 35., unit="adu"),
               ]
    ccdlist[0].data[0, 1] = 3.1
    ccdlist[1].data[1, 2] = 100.1
    ccdlist[1].data[2, 0] = 100.1
    c = Combiner(ccdlist)
    # Reject ccdlist[1].data[1,2] by other means
    c.data_arr.mask[1, 1, 2] = True
    # Reject ccdlist[1].data[1,2] by other means
    c.data_arr.mask[3, 0, 0] = True

    c.clip_extrema(nlow=1, nhigh=1)
    result = c.average_combine()
    expected = [[80. / 3., 22.5, 30., 30., 30.],
                [30., 30., 47.5, 30., 30.],
                [47.5, 30., 30., 30., 30.]]
    np.testing.assert_array_equal(result, expected)


# The expected values below assume an image that is 2000x2000
@pytest.mark.parametrize('num_chunks, expected',
                         [(53, (37, 2000)),
                          (1500, (1, 2000)),
                          (2001, (1, 1000)),
                          (2999, (1, 1000)),
                          (10000, (1, 333))]
                         )
def test_ystep_calculation(num_chunks, expected):
    # Regression test for
    # https://github.com/astropy/ccdproc/issues/639
    # See that issue for the motivation for the choice of
    # image size and number of chunks in the test below.

    xstep, ystep = _calculate_step_sizes(2000, 2000, num_chunks)
    assert xstep == expected[0] and ystep == expected[1]

def test_combiner_gen():
    ccd_data = ccd_data_func()
    def create_gen():
        yield ccd_data
        yield ccd_data
        yield ccd_data
    c = Combiner(create_gen())
    assert c.data_arr.shape == (3, 100, 100)
    assert c.data_arr.mask.shape == (3, 100, 100)


@pytest.mark.parametrize('comb_func',
                         ['average_combine', 'median_combine', 'sum_combine'])
def test_combiner_with_scaling_uncertainty(comb_func):
    # A regression test for #719, in which it was pointed out that the
    # uncertainty was not properly calculated from scaled data in
    # median_combine

    ccd_data = ccd_data_func()
    # The factors below are not particularly important; just avoid anything
    # whose average is 1.
    ccd_data_lower = ccd_data.multiply(3)
    ccd_data_higher = ccd_data.multiply(0.9)

    combiner = Combiner([ccd_data, ccd_data_higher, ccd_data_lower])
    # scale each array to the mean of the first image
    scale_by_mean = lambda x: ccd_data.data.mean() / np.ma.average(x)
    combiner.scaling = scale_by_mean

    scaled_ccds = np.array([ccd_data.data * scale_by_mean(ccd_data.data),
                            ccd_data_lower.data * scale_by_mean(ccd_data_lower.data),
                            ccd_data_higher.data * scale_by_mean(ccd_data_higher.data)
                           ])

    avg_ccd = getattr(combiner, comb_func)()

    if comb_func != 'median_combine':
        uncertainty_func = _default_std()
    else:
        uncertainty_func = sigma_func

    expected_unc = uncertainty_func(scaled_ccds, axis=0)

    np.testing.assert_almost_equal(avg_ccd.uncertainty.array,
                                   expected_unc)


@pytest.mark.parametrize('comb_func',
                         ['average_combine', 'median_combine', 'sum_combine'])
def test_user_supplied_combine_func_that_relies_on_masks(comb_func):
    # Test to make sure that setting some values to NaN internally
    # does not affect results when the user supplies a function that
    # uses masks to screen out bad data.

    data = np.ones((10, 10))
    data[5, 5] = 2
    mask = (data == 2)
    ccd = CCDData(data, unit=u.adu, mask=mask)
    # Same, but no mask
    ccd2 = CCDData(data, unit=u.adu)

    ccd_list = [ccd, ccd, ccd2]
    c = Combiner(ccd_list)

    if comb_func == 'sum_combine':
        expected_result = 3 * data
        actual_result = c.sum_combine(sum_func=np.ma.sum)
    elif comb_func == 'average_combine':
        expected_result = data
        actual_result = c.average_combine(scale_func=np.ma.mean)
    elif comb_func == 'median_combine':
        expected_result = data
        actual_result = c.median_combine(median_func=np.ma.median)

    # Two of the three values are masked, so no matter what the combination
    # method is the result in this pixel should be 2.
    expected_result[5, 5] = 2

    np.testing.assert_almost_equal(expected_result,
                                   actual_result)
