# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import astropy.units as u
from astropy.stats import median_absolute_deviation as mad

import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy.nddata import CCDData

from ..combiner import Combiner, combine, _calculate_step_sizes
from .pytest_fixtures import ccd_data as ccd_data_func


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


# test data combined with mask is created correctly
def test_combiner_mask_average():
    data = np.zeros((10, 10))
    data[5, 5] = 1
    mask = (data == 0)
    ccd = CCDData(data, unit=u.adu, mask=mask)
    ccd_list = [ccd, ccd, ccd]
    c = Combiner(ccd_list)
    ccd = c.average_combine()
    assert ccd.data[0, 0] == 0
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
    assert ccd.data[0, 0] == 0
    assert ccd.data[5, 5] == 1
    assert ccd.mask[0, 0]
    assert not ccd.mask[5, 5]


# test data combined with mask is created correctly
def test_combiner_mask_sum():
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
    fitsfile = get_pkg_data_filename('data/a8280271.fits')
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
