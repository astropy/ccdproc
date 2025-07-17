# Licensed under a 3-clause BSD style license - see LICENSE.rst
import array_api_compat
import array_api_extra as xpx
import astropy.units as u
import numpy.ma as np_ma
import pytest
from astropy.nddata import CCDData
from astropy.stats import median_absolute_deviation as mad
from astropy.utils.data import get_pkg_data_filename
from numpy import median as np_median

from ccdproc.combiner import (
    Combiner,
    _calculate_step_sizes,
    _default_std,
    combine,
    sigma_func,
)

# Set up the array library to be used in tests
from ccdproc.conftest import testing_array_library as xp
from ccdproc.image_collection import ImageFileCollection
from ccdproc.tests.pytest_fixtures import ccd_data as ccd_data_func

# Several tests have many more NaNs in them than real data. numpy generates
# lots of warnings in those cases and it makes more sense to suppress them
# than to generate them.
pytestmark = pytest.mark.filterwarnings(
    "ignore:All-NaN slice encountered:RuntimeWarning"
)


def _make_mean_scaler(ccd_data):
    def scale_by_mean(x):
        # scale each array to the mean of the first image
        return ccd_data.data.mean() / np_ma.average(x)

    return scale_by_mean


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
    ccd_large = CCDData(xp.zeros((200, 100)), unit=u.adu)
    ccd_list = [ccd_data, ccd_data, ccd_large]
    with pytest.raises(TypeError):
        Combiner(ccd_list)  # arrays of different sizes should fail


# test that Combiner throws an error if input
# objects do not have the same units
def test_ccddata_combiner_units():
    ccd_data = ccd_data_func()
    ccd_large = CCDData(xp.zeros((100, 100)), unit=u.second)
    ccd_list = [ccd_data, ccd_data, ccd_large]
    with pytest.raises(TypeError):
        Combiner(ccd_list)


# test if mask and data array are created
def test_combiner_create():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    assert c._data_arr.shape == (3, 100, 100)
    assert c._data_arr_mask.shape == (3, 100, 100)


# test if dtype matches the value that is passed
def test_combiner_dtype():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list, dtype=xp.float32)
    assert c._data_arr.dtype == xp.float32
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
    data = xp.zeros((10, 10))
    data = xpx.at(data)[5, 5].set(1)
    mask = data == 0
    ccd = CCDData(data, unit=u.adu, mask=mask)
    ccd_list = [ccd, ccd, ccd]
    c = Combiner(ccd_list)
    assert c._data_arr.shape == (3, 10, 10)
    assert c._data_arr_mask.shape == (3, 10, 10)
    assert not c._data_arr_mask[0, 5, 5]


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
    ccd_list = [
        CCDData(xp.zeros((10, 10)), unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 1000, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 1000, unit=u.adu),
    ]

    combo = Combiner(ccd_list)
    combo.weights = xp.asarray([1, 5, 10])
    ccd = combo.average_combine()
    assert xp.all(xpx.isclose(ccd.data, 312.5))

    with pytest.raises(ValueError):
        combo.weights = xp.asarray([1, 5, 10, 20])


def test_pixelwise_weights():
    ccd_list = [
        CCDData(xp.zeros((10, 10)), unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 1000, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 1000, unit=u.adu),
    ]
    combo = Combiner(ccd_list)
    combo.weights = xp.ones_like(combo._data_arr)
    combo.weights = xpx.at(combo.weights)[:, 5, 5].set(xp.asarray([1, 5, 10]))
    ccd = combo.average_combine()
    assert xp.all(xpx.isclose(ccd.data[5, 5], 312.5))
    assert xp.all(xpx.isclose(ccd.data[0, 0], 0))


# test the min-max rejection
def test_combiner_minmax():
    ccd_list = [
        CCDData(xp.zeros((10, 10)), unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 1000, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 1000, unit=u.adu),
    ]

    c = Combiner(ccd_list)
    c.minmax_clipping(min_clip=-500, max_clip=500)
    ccd = c.median_combine()
    assert ccd.data.mean() == 0


def test_combiner_minmax_max():
    ccd_list = [
        CCDData(xp.zeros((10, 10)), unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 1000, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 1000, unit=u.adu),
    ]

    c = Combiner(ccd_list)
    c.minmax_clipping(min_clip=None, max_clip=500)
    assert c._data_arr_mask[2].all()


def test_combiner_minmax_min():
    ccd_list = [
        CCDData(xp.zeros((10, 10)), unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 1000, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 1000, unit=u.adu),
    ]

    c = Combiner(ccd_list)
    c.minmax_clipping(min_clip=-500, max_clip=None)
    assert c._data_arr_mask[1].all()


def test_combiner_sigmaclip_high():
    ccd_list = [
        CCDData(xp.zeros((10, 10)), unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 10, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 10, unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 10, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 10, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 1000, unit=u.adu),
    ]

    c = Combiner(ccd_list)
    # using mad for more robust statistics vs. std
    c.sigma_clipping(high_thresh=3, low_thresh=None, func="median", dev_func=mad)
    assert c._data_arr_mask[5].all()


def test_combiner_sigmaclip_single_pix():
    ccd_list = [
        CCDData(xp.zeros((10, 10)), unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 10, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 10, unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 10, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 10, unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 10, unit=u.adu),
    ]
    combo = Combiner(ccd_list)
    # add a single pixel in another array to check that
    # that one gets rejected
    combo._data_arr = xpx.at(combo._data_arr)[0, 5, 5].set(0)
    combo._data_arr = xpx.at(combo._data_arr)[1, 5, 5].set(-5)
    combo._data_arr = xpx.at(combo._data_arr)[2, 5, 5].set(5)
    combo._data_arr = xpx.at(combo._data_arr)[3, 5, 5].set(-5)
    combo._data_arr = xpx.at(combo._data_arr)[4, 5, 5].set(25)
    combo.sigma_clipping(high_thresh=3, low_thresh=None, func="median", dev_func=mad)
    assert combo._data_arr_mask[4, 5, 5]


def test_combiner_sigmaclip_low():
    ccd_list = [
        CCDData(xp.zeros((10, 10)), unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 10, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 10, unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 10, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 10, unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 1000, unit=u.adu),
    ]

    c = Combiner(ccd_list)
    # using mad for more robust statistics vs. std
    c.sigma_clipping(high_thresh=None, low_thresh=3, func="median", dev_func=mad)
    assert c._data_arr_mask[5].all()


# test that the median combination works and returns a ccddata object
def test_combiner_median():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.median_combine()
    assert isinstance(ccd, CCDData)
    assert ccd.shape == (100, 100)
    assert ccd.unit == u.adu
    assert ccd.meta["NCOMBINE"] == len(ccd_list)


# test that the average combination works and returns a ccddata object
def test_combiner_average():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.average_combine()
    assert isinstance(ccd, CCDData)
    assert ccd.shape == (100, 100)
    assert ccd.unit == u.adu
    assert ccd.meta["NCOMBINE"] == len(ccd_list)


# test that the sum combination works and returns a ccddata object
def test_combiner_sum():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.sum_combine()
    assert isinstance(ccd, CCDData)
    assert ccd.shape == (100, 100)
    assert ccd.unit == u.adu
    assert ccd.meta["NCOMBINE"] == len(ccd_list)


# test weighted sum
def test_combiner_sum_weighted():
    ccd_data = CCDData(data=xp.asarray([[0, 1], [2, 3]]), unit="adu")
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    c.weights = xp.asarray([1, 2, 3])
    ccd = c.sum_combine()
    expected_result = sum(w * d.data for w, d in zip(c.weights, ccd_list, strict=True))
    assert xp.all(xpx.isclose(ccd.data, expected_result))


# test weighted sum
def test_combiner_sum_weighted_by_pixel():
    ccd_data = CCDData(data=xp.asarray([[1, 2], [4, 8]]), unit="adu")
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    # Weights below are chosen so that every entry in
    weights_pixel = [[8, 4], [2, 1]]
    c.weights = xp.asarray([weights_pixel] * 3)
    ccd = c.sum_combine()
    expected_result = xp.asarray([[24, 24], [24, 24]])
    assert xp.all(xpx.isclose(ccd.data, expected_result))


# This warning is generated by numpy and is expected when
# many pixels are masked.
@pytest.mark.filterwarnings(
    "ignore:Mean of empty slice:RuntimeWarning",
    "ignore:Degrees of freedom <= 0:RuntimeWarning",
)
def test_combiner_mask_average():
    # test data combined with mask is created correctly
    data = xp.zeros((10, 10))
    data = xpx.at(data)[5, 5].set(1)
    mask = data == 0
    ccd = CCDData(data, unit=u.adu, mask=mask)
    ccd_list = [ccd, ccd, ccd]
    c = Combiner(ccd_list)

    ccd = c.average_combine()

    # How can we assert anything about the data if all values
    # are masked?!
    # assert ccd.data[0, 0] == 0
    assert ccd.data[5, 5] == 1
    # THE LINE BELOW IS CATCHING A REAL ERROR
    assert ccd.mask[0, 0]
    assert not ccd.mask[5, 5]


def test_combiner_with_scaling():
    ccd_data = ccd_data_func()
    # The factors below are not particularly important; just avoid anything
    # whose average is 1.
    ccd_data_lower = ccd_data.multiply(3)
    ccd_data_higher = ccd_data.multiply(0.9)
    combiner = Combiner([ccd_data, ccd_data_higher, ccd_data_lower])
    scale_by_mean = _make_mean_scaler(ccd_data)
    combiner.scaling = scale_by_mean
    avg_ccd = combiner.average_combine()
    # Does the mean of the scaled arrays match the value to which it was
    # scaled?
    assert xp.all(xpx.isclose(avg_ccd.data.mean(), ccd_data.data.mean()))
    assert avg_ccd.shape == ccd_data.shape
    median_ccd = combiner.median_combine()
    # Does median also scale to the correct value?
    # Some array libraries do not have a median, and median is not part of the
    # standard array API, so we use numpy's median here.
    # Odd; for dask, which does not have a full median, even falling back to numpy does
    # not work. For some reason the call to np_median fails. I suppose this is maybe
    # because dask just adds a median to its task list/compute graph thingy
    # and then tries to evaluate it itself?

    med_ccd = median_ccd.data
    med_inp_data = ccd_data.data
    # Try doing a compute on the data first, and if that fails it is no big deal
    try:
        med_ccd = med_ccd.compute()
        med_inp_data = med_inp_data.compute()
    except AttributeError:
        pass

    assert xp.all(xpx.isclose(np_median(med_ccd), np_median(med_inp_data)))

    # Set the scaling manually...
    combiner.scaling = [scale_by_mean(combiner._data_arr[i]) for i in range(3)]
    avg_ccd = combiner.average_combine()
    assert xp.all(xpx.isclose(avg_ccd.data.mean(), ccd_data.data.mean()))
    assert avg_ccd.shape == ccd_data.shape


def test_combiner_scaling_fails():
    ccd_data = ccd_data_func()
    combiner = Combiner([ccd_data, ccd_data.copy()])
    # Should fail unless scaling is set to a function or list-like
    with pytest.raises(TypeError):
        combiner.scaling = 5

    # Should calendar because the scaling function is not the right shape
    with pytest.raises(ValueError):
        combiner.scaling = [5, 5, 5]


# test data combined with mask is created correctly
def test_combiner_mask_median():
    data = xp.zeros((10, 10))
    data = xpx.at(data)[5, 5].set(1)
    mask = data == 0
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
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0:RuntimeWarning")
def test_combiner_mask_sum():
    # test data combined with mask is created correctly
    data = xp.zeros((10, 10))
    data = xpx.at(data)[5, 5].set(1)
    mask = data == 0
    ccd = CCDData(data, unit=u.adu, mask=mask)
    ccd_list = [ccd, ccd, ccd]
    c = Combiner(ccd_list)
    ccd = c.sum_combine()
    assert ccd.data[0, 0] == 0
    assert ccd.data[5, 5] == 3
    assert ccd.mask[0, 0]
    assert not ccd.mask[5, 5]


# Test that calling combine with a bad input raises an error
def test_combine_bad_input():
    with pytest.raises(ValueError, match="unrecognised input for list of images"):
        combine(1)

    with pytest.raises(ValueError, match="unrecognised combine method"):
        combine([1, 2, 3], method="bad_method")


# test combiner convenience function reads fits file and combine as expected
def test_combine_average_fitsimages():
    fitsfile = get_pkg_data_filename("data/a8280271.fits", package="ccdproc.tests")
    ccd = CCDData.read(fitsfile, unit=u.adu)
    ccd_list = [ccd] * 3
    c = Combiner(ccd_list)
    ccd_by_combiner = c.average_combine()

    fitsfilename_list = [fitsfile] * 3
    avgccd = combine(fitsfilename_list, output_file=None, method="average", unit=u.adu)
    # averaging same fits images should give back same fits image
    assert xp.all(xpx.isclose(avgccd.data, ccd_by_combiner.data))


def test_combine_numpyndarray():
    """Test of numpy ndarray implementation: #493

    Test the average combine using ``Combiner`` and ``combine`` with input
    ``img_list`` in the format of ``numpy.ndarray``.
    """
    fitsfile = get_pkg_data_filename("data/a8280271.fits")
    ccd = CCDData.read(fitsfile, unit=u.adu)
    ccd_list = [ccd] * 3
    c = Combiner(ccd_list)
    ccd_by_combiner = c.average_combine()

    fitsfilename_list = [fitsfile] * 3
    avgccd = combine(fitsfilename_list, output_file=None, method="average", unit=u.adu)
    # averaging same fits images should give back same fits image
    assert xp.all(xpx.isclose(avgccd.data, ccd_by_combiner.data))


def test_combiner_result_dtype():
    """Regression test: #391

    The result should have the appropriate dtype not the dtype of the first
    input."""
    ccd = CCDData(xp.ones((3, 3), dtype=xp.uint16), unit="adu")
    res = combine([ccd, ccd.multiply(2)])
    # The default dtype of Combiner is float64
    assert res.data.dtype == xp.float64
    ref = xp.ones((3, 3)) * 1.5
    assert xp.all(xpx.isclose(res.data, ref))
    res = combine([ccd, ccd.multiply(2), ccd.multiply(3)], dtype=int)
    # The result dtype should be integer:
    assert xp.isdtype(res.data.dtype, "integral")
    ref = xp.ones((3, 3)) * 2
    assert xp.all(xpx.isclose(res.data, ref))


def test_combiner_image_file_collection_input(tmp_path):
    # Regression check for #754
    ccd = ccd_data_func()
    for i in range(3):
        ccd.write(tmp_path / f"ccd-{i}.fits")

    ifc = ImageFileCollection(tmp_path)
    ccds = list(ifc.ccds())

    # Need to convert these to the array namespace.
    for a_ccd in ccds:
        a_ccd.data = xp.asarray(a_ccd.data, dtype=xp.float64)
        if a_ccd.mask is not None:
            a_ccd.mask = xp.asarray(a_ccd.mask, dtype=bool)
        if a_ccd.uncertainty is not None:
            a_ccd.uncertainty.array = xp.asarray(
                a_ccd.uncertainty.array, dtype=xp.float64
            )
    comb = Combiner(ccds)

    # Do this on a separate line from the assert to make debugging easier
    result = comb.average_combine()
    assert xp.all(xpx.isclose(ccd.data, result.data))


def test_combine_image_file_collection_input(tmp_path):
    # Another regression check for #754 but this time with the
    # combine function instead of Combiner
    ccd = ccd_data_func()
    xp = array_api_compat.array_namespace(ccd.data)
    for i in range(3):
        ccd.write(tmp_path / f"ccd-{i}.fits")

    ifc = ImageFileCollection(tmp_path, array_package=xp)

    comb_files = combine(
        ifc.files_filtered(include_path=True), method="average", array_package=xp
    )

    comb_ccds = combine(ifc.ccds(), method="average", array_package=xp)

    comb_string = combine(
        ",".join(ifc.files_filtered(include_path=True)),
        method="average",
        array_package=xp,
    )

    assert xp.all(xpx.isclose(ccd.data, comb_files.data))
    assert xp.all(xpx.isclose(ccd.data, comb_ccds.data))
    assert xp.all(xpx.isclose(ccd.data, comb_string.data))

    with pytest.raises(FileNotFoundError):
        # This should fail because the test is not running in the
        # folder where the images are.
        _ = combine(ifc.files_filtered())


# test combiner convenience function works with list of ccddata objects
def test_combine_average_ccddata():
    fitsfile = get_pkg_data_filename("data/a8280271.fits")
    ccd = CCDData.read(fitsfile, unit=u.adu)
    ccd_list = [ccd] * 3
    c = Combiner(ccd_list)
    ccd_by_combiner = c.average_combine()

    avgccd = combine(ccd_list, output_file=None, method="average", unit=u.adu)
    # averaging same ccdData should give back same images
    assert xp.all(xpx.isclose(avgccd.data, ccd_by_combiner.data))


# test combiner convenience function reads fits file and
# and combine as expected when asked to run in limited memory
def test_combine_limitedmem_fitsimages():
    fitsfile = get_pkg_data_filename("data/a8280271.fits")
    ccd = CCDData.read(fitsfile, unit=u.adu)
    ccd_list = [ccd] * 5
    c = Combiner(ccd_list)
    ccd_by_combiner = c.average_combine()

    fitsfilename_list = [fitsfile] * 5
    avgccd = combine(
        fitsfilename_list, output_file=None, method="average", mem_limit=1e6, unit=u.adu
    )
    # averaging same ccdData should give back same images
    assert xp.all(xpx.isclose(avgccd.data, ccd_by_combiner.data))


# test combiner convenience function reads fits file and
# and combine as expected when asked to run in limited memory with scaling
def test_combine_limitedmem_scale_fitsimages():
    fitsfile = get_pkg_data_filename("data/a8280271.fits")
    ccd = CCDData.read(fitsfile, unit=u.adu)
    ccd_list = [ccd] * 5
    c = Combiner(ccd_list)
    # scale each array to the mean of the first image
    scale_by_mean = _make_mean_scaler(ccd)
    c.scaling = scale_by_mean
    ccd_by_combiner = c.average_combine()

    fitsfilename_list = [fitsfile] * 5
    avgccd = combine(
        fitsfilename_list,
        output_file=None,
        method="average",
        mem_limit=1e6,
        scale=scale_by_mean,
        unit=u.adu,
    )

    assert xp.all(xpx.isclose(avgccd.data, ccd_by_combiner.data))


# test the optional uncertainty function in average_combine
def test_average_combine_uncertainty():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.average_combine(uncertainty_func=xp.sum)
    uncert_ref = xp.sum(c._data_arr, 0) / xp.sqrt(3)
    assert xp.all(xpx.isclose(ccd.uncertainty.array, uncert_ref))

    # Compare this also to the "combine" call
    ccd2 = combine(ccd_list, method="average", combine_uncertainty_function=xp.sum)
    assert xp.all(xpx.isclose(ccd.data, ccd2.data))
    assert xp.all(xpx.isclose(ccd.uncertainty.array, ccd2.uncertainty.array))


# test the optional uncertainty function in median_combine
def test_median_combine_uncertainty():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.median_combine(uncertainty_func=xp.sum)
    uncert_ref = xp.sum(c._data_arr, 0) / xp.sqrt(3)
    assert xp.all(xpx.isclose(ccd.uncertainty.array, uncert_ref))

    # Compare this also to the "combine" call
    ccd2 = combine(ccd_list, method="median", combine_uncertainty_function=xp.sum)
    assert xp.all(xpx.isclose(ccd.data, ccd2.data))
    assert xp.all(xpx.isclose(ccd.uncertainty.array, ccd2.uncertainty.array))


# test the optional uncertainty function in sum_combine
def test_sum_combine_uncertainty():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.sum_combine(uncertainty_func=xp.sum)
    uncert_ref = xp.sum(c._data_arr, 0) * xp.sqrt(3)
    assert xp.all(xpx.isclose(ccd.uncertainty.array, uncert_ref))

    # Compare this also to the "combine" call
    ccd2 = combine(ccd_list, method="sum", combine_uncertainty_function=xp.sum)
    assert xp.all(xpx.isclose(ccd.data, ccd2.data))
    assert xp.all(xpx.isclose(ccd.uncertainty.array, ccd2.uncertainty.array))


# Ignore warnings generated because most values are masked and we divide
# by zero in at least one place
@pytest.mark.filterwarnings(
    "ignore:Mean of empty slice:RuntimeWarning",
    "ignore:Degrees of freedom <= 0:RuntimeWarning",
    "ignore:invalid value encountered in divide:RuntimeWarning",
)
@pytest.mark.parametrize("mask_point", [True, False])
@pytest.mark.parametrize(
    "comb_func", ["average_combine", "median_combine", "sum_combine"]
)
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
        # Handle case where array is immutable by using array_api_extra,
        # which provides at for all array libraries.
        ccd_data.data = xpx.at(ccd_data.data)[0, 0].set(-1000)

    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)

    c.minmax_clipping(min_clip=-100)

    expected_result = getattr(c, comb_func)()

    # Just need the first part of the name for the combine function
    combine_method_name = comb_func.split("_")[0]

    ccd_comb = combine(
        ccd_list, method=combine_method_name, minmax_clip=True, minmax_clip_min=-100
    )

    assert xp.all(
        xpx.isclose(
            ccd_comb.uncertainty.array,
            expected_result.uncertainty.array,
            equal_nan=True,
        )
    )

    # Check that the right point is masked, and only one point is
    # masked
    assert expected_result.mask[0, 0] == mask_point
    assert expected_result.mask.sum() == mask_point
    assert ccd_comb.mask[0, 0] == mask_point
    assert ccd_comb.mask.sum() == mask_point


def test_combine_overwrite_output(tmp_path):
    """
    The combine function should *not* overwrite the result file
    unless the overwrite_output argument is True
    """
    output_file = tmp_path / "fake.fits"

    ccd = CCDData(xp.ones((3, 3)), unit="adu")

    # Make sure we have a file to overwrite
    ccd.write(output_file)
    # Test that overwrite does NOT happen by default
    with pytest.raises(OSError, match="fake.fits already exists"):
        res = combine([ccd, ccd.multiply(2)], output_file=str(output_file))

    # Should be no error here...
    # The default dtype of Combiner is float64
    res = combine(
        [ccd, ccd.multiply(2)], output_file=output_file, overwrite_output=True
    )

    # Need to convert this to the array namespace.
    res_from_disk = CCDData.read(output_file)
    res_from_disk.data = xp.asarray(
        res_from_disk.data, dtype=res_from_disk.data.dtype.type
    )

    # Data should be the same
    assert xp.all(xpx.isclose(res.data, res_from_disk.data))


# test resulting uncertainty is corrected for the number of images
def test_combiner_uncertainty_average():
    ccd_list = [
        CCDData(xp.ones((10, 10)), unit=u.adu),
        CCDData(xp.ones((10, 10)) * 2, unit=u.adu),
    ]
    c = Combiner(ccd_list)
    ccd = c.average_combine()
    # Just the standard deviation of ccd data.
    ref_uncertainty = xp.ones((10, 10)) / 2
    # Correction because we combined two images.
    ref_uncertainty /= xp.sqrt(2)
    assert xp.all(xpx.isclose(ccd.uncertainty.array, ref_uncertainty))


# test resulting uncertainty is corrected for the number of images (with mask)
def test_combiner_uncertainty_average_mask():
    mask = xp.zeros((10, 10), dtype=bool)
    mask = xpx.at(mask)[5, 5].set(True)
    ccd_with_mask = CCDData(xp.ones((10, 10)), unit=u.adu, mask=mask)
    ccd_list = [
        ccd_with_mask,
        CCDData(xp.ones((10, 10)) * 2, unit=u.adu),
        CCDData(xp.ones((10, 10)) * 3, unit=u.adu),
    ]
    c = Combiner(ccd_list)
    ccd = c.average_combine()
    # Just the standard deviation of ccd data.
    ref_uncertainty = xp.ones((10, 10)) * xp.std(xp.asarray([1, 2, 3]))
    # Correction because we combined two images.
    ref_uncertainty /= xp.sqrt(3)
    ref_uncertainty = xpx.at(ref_uncertainty)[5, 5].set(
        xp.std(xp.asarray([2, 3])) / xp.sqrt(2)
    )
    assert xp.all(xpx.isclose(ccd.uncertainty.array, ref_uncertainty))


# test resulting uncertainty is corrected for the number of images (with mask)
def test_combiner_uncertainty_median_mask():
    mad_to_sigma = 1.482602218505602
    mask = xp.zeros((10, 10), dtype=bool)
    mask = xpx.at(mask)[5, 5].set(True)
    ccd_with_mask = CCDData(xp.ones((10, 10)), unit=u.adu, mask=mask)
    ccd_list = [
        ccd_with_mask,
        CCDData(xp.ones((10, 10)) * 2, unit=u.adu),
        CCDData(xp.ones((10, 10)) * 3, unit=u.adu),
    ]
    c = Combiner(ccd_list)
    ccd = c.median_combine()
    # Just the standard deviation of ccd data.
    ref_uncertainty = xp.ones((10, 10)) * mad_to_sigma * mad([1, 2, 3])
    # Correction because we combined two images.
    ref_uncertainty /= xp.sqrt(3)  # 0.855980789955
    # It turns out that the expression below evaluates to a np.float64, which
    # introduces numpy into the array namespace, which raises an error
    # when arrat_api_compat tries to figure out the namespace. Casting
    # it to a regular float fixes that.
    med_value = float(mad_to_sigma * mad([2, 3]) / xp.sqrt(2))
    ref_uncertainty = xpx.at(ref_uncertainty)[5, 5].set(med_value)  # 0.524179041254
    assert xp.all(xpx.isclose(ccd.uncertainty.array, ref_uncertainty))


# test resulting uncertainty is corrected for the number of images (with mask)
def test_combiner_uncertainty_sum_mask():
    mask = xp.zeros((10, 10), dtype=bool)
    mask = xpx.at(mask)[5, 5].set(True)
    ccd_with_mask = CCDData(xp.ones((10, 10)), unit=u.adu, mask=mask)
    ccd_list = [
        ccd_with_mask,
        CCDData(xp.ones((10, 10)) * 2, unit=u.adu),
        CCDData(xp.ones((10, 10)) * 3, unit=u.adu),
    ]
    c = Combiner(ccd_list)
    ccd = c.sum_combine()
    # Just the standard deviation of ccd data.
    ref_uncertainty = xp.ones((10, 10)) * xp.std(xp.asarray([1, 2, 3]))
    ref_uncertainty *= xp.sqrt(3)
    ref_uncertainty = xpx.at(ref_uncertainty)[5, 5].set(
        xp.std(xp.asarray([2, 3])) * xp.sqrt(2)
    )
    assert xp.all(xpx.isclose(ccd.uncertainty.array, ref_uncertainty))


def test_combiner_3d():
    data1 = CCDData(3 * xp.ones((5, 5, 5)), unit=u.adu)
    data2 = CCDData(2 * xp.ones((5, 5, 5)), unit=u.adu)
    data3 = CCDData(4 * xp.ones((5, 5, 5)), unit=u.adu)

    ccd_list = [data1, data2, data3]

    c = Combiner(ccd_list)
    assert c._data_arr.shape == (3, 5, 5, 5)
    assert c._data_arr_mask.shape == (3, 5, 5, 5)

    ccd = c.average_combine()
    assert ccd.shape == (5, 5, 5)
    assert xp.all(xpx.isclose(ccd.data, data1.data))


def test_3d_combiner_with_scaling():
    ccd_data = ccd_data_func()
    # The factors below are not particularly important; just avoid anything
    # whose average is 1.
    ccd_data = CCDData(xp.ones((5, 5, 5)), unit=u.adu)
    ccd_data_lower = CCDData(3 * xp.ones((5, 5, 5)), unit=u.adu)
    ccd_data_higher = CCDData(0.9 * xp.ones((5, 5, 5)), unit=u.adu)
    combiner = Combiner([ccd_data, ccd_data_higher, ccd_data_lower])
    scale_by_mean = _make_mean_scaler(ccd_data)

    combiner.scaling = scale_by_mean
    avg_ccd = combiner.average_combine()
    # Does the mean of the scaled arrays match the value to which it was
    # scaled?
    assert xp.all(xpx.isclose(avg_ccd.data.mean(), ccd_data.data.mean()))
    assert avg_ccd.shape == ccd_data.shape
    median_ccd = combiner.median_combine()
    # Does median also scale to the correct value?
    # Once again, use numpy to find the median
    med_ccd = median_ccd.data
    med_inp_data = ccd_data.data
    try:
        med_ccd = med_ccd.compute()
        med_inp_data = med_inp_data.compute()
    except AttributeError:
        pass

    assert xp.all(xpx.isclose(np_median(med_ccd), np_median(med_inp_data)))

    # Set the scaling manually...
    combiner.scaling = [scale_by_mean(combiner._data_arr[i]) for i in range(3)]
    avg_ccd = combiner.average_combine()
    assert xp.all(xpx.isclose(avg_ccd.data.mean(), ccd_data.data.mean()))
    assert avg_ccd.shape == ccd_data.shape


def test_clip_extrema_3d():
    ccdlist = [
        CCDData(xp.ones((3, 3, 3)) * 90.0, unit="adu"),
        CCDData(xp.ones((3, 3, 3)) * 20.0, unit="adu"),
        CCDData(xp.ones((3, 3, 3)) * 10.0, unit="adu"),
        CCDData(xp.ones((3, 3, 3)) * 40.0, unit="adu"),
        CCDData(xp.ones((3, 3, 3)) * 25.0, unit="adu"),
        CCDData(xp.ones((3, 3, 3)) * 35.0, unit="adu"),
    ]
    c = Combiner(ccdlist)
    c.clip_extrema(nlow=1, nhigh=1)
    result = c.average_combine()
    expected = CCDData(xp.ones((3, 3, 3)) * 30, unit="adu")
    assert xp.all(xpx.isclose(result.data, expected.data))


@pytest.mark.parametrize(
    "comb_func", ["average_combine", "median_combine", "sum_combine"]
)
def test_writeable_after_combine(tmpdir, comb_func):
    ccd_data = ccd_data_func()
    tmp_file = tmpdir.join("tmp.fits")
    from ..combiner import Combiner

    combined = Combiner([ccd_data for _ in range(3)])
    ccd2 = getattr(combined, comb_func)()
    # This should not fail because the resulting uncertainty has a mask
    ccd2.write(tmp_file.strpath)


def test_clip_extrema_alone():
    ccdlist = [
        CCDData(xp.ones((3, 5)) * 90.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 20.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 10.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 40.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 25.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 35.0, unit="adu"),
    ]
    ccdlist[0].data = xpx.at(ccdlist[0].data)[0, 1].set(3.1)
    ccdlist[1].data = xpx.at(ccdlist[1].data)[1, 2].set(100.1)
    ccdlist[1].data = xpx.at(ccdlist[1].data)[2, 0].set(100.1)
    c = Combiner(ccdlist)
    c.clip_extrema(nlow=1, nhigh=1)
    result = c.average_combine()
    expected = xp.asarray(
        [
            [30.0, 22.5, 30.0, 30.0, 30.0],
            [30.0, 30.0, 47.5, 30.0, 30.0],
            [47.5, 30.0, 30.0, 30.0, 30.0],
        ]
    )
    assert xp.all(xpx.isclose(result.data, expected))


def test_clip_extrema_via_combine():
    ccdlist = [
        CCDData(xp.ones((3, 5)) * 90.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 20.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 10.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 40.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 25.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 35.0, unit="adu"),
    ]
    ccdlist[0].data = xpx.at(ccdlist[0].data)[0, 1].set(3.1)
    ccdlist[1].data = xpx.at(ccdlist[1].data)[1, 2].set(100.1)
    ccdlist[1].data = xpx.at(ccdlist[1].data)[2, 0].set(100.1)
    result = combine(
        ccdlist,
        clip_extrema=True,
        nlow=1,
        nhigh=1,
    )
    expected = xp.asarray(
        [
            [30.0, 22.5, 30.0, 30.0, 30.0],
            [30.0, 30.0, 47.5, 30.0, 30.0],
            [47.5, 30.0, 30.0, 30.0, 30.0],
        ]
    )
    assert xp.all(xpx.isclose(result.data, expected))


def test_clip_extrema_with_other_rejection():
    ccdlist = [
        CCDData(xp.ones((3, 5)) * 90.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 20.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 10.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 40.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 25.0, unit="adu"),
        CCDData(xp.ones((3, 5)) * 35.0, unit="adu"),
    ]
    ccdlist[0].data = xpx.at(ccdlist[0].data)[0, 1].set(3.1)
    ccdlist[1].data = xpx.at(ccdlist[1].data)[1, 2].set(100.1)
    ccdlist[1].data = xpx.at(ccdlist[1].data)[2, 0].set(100.1)
    c = Combiner(ccdlist)
    # Reject ccdlist[1].data[1,2] by other means
    c._data_arr_mask = xpx.at(c._data_arr_mask)[1, 1, 2].set(True)
    # Reject ccdlist[1].data[1,2] by other means
    c._data_arr_mask = xpx.at(c._data_arr_mask)[3, 0, 0].set(True)

    c.clip_extrema(nlow=1, nhigh=1)
    result = c.average_combine()
    expected = xp.asarray(
        [
            [80.0 / 3.0, 22.5, 30.0, 30.0, 30.0],
            [30.0, 30.0, 47.5, 30.0, 30.0],
            [47.5, 30.0, 30.0, 30.0, 30.0],
        ]
    )
    assert xp.all(xpx.isclose(result.data, expected))


# The expected values below assume an image that is 2000x2000
@pytest.mark.parametrize(
    "num_chunks, expected",
    [
        (53, (37, 2000)),
        (1500, (1, 2000)),
        (2001, (1, 1000)),
        (2999, (1, 1000)),
        (10000, (1, 333)),
    ],
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
    assert c._data_arr.shape == (3, 100, 100)
    assert c._data_arr_mask.shape == (3, 100, 100)


@pytest.mark.parametrize(
    "comb_func", ["average_combine", "median_combine", "sum_combine"]
)
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
    scale_by_mean = _make_mean_scaler(ccd_data)
    combiner.scaling = scale_by_mean

    scaled_ccds = xp.asarray(
        [
            ccd_data.data * scale_by_mean(ccd_data.data),
            ccd_data_lower.data * scale_by_mean(ccd_data_lower.data),
            ccd_data_higher.data * scale_by_mean(ccd_data_higher.data),
        ]
    )

    avg_ccd = getattr(combiner, comb_func)()

    if comb_func != "median_combine":
        uncertainty_func = _default_std(xp=xp)
    else:
        uncertainty_func = sigma_func

    expected_unc = uncertainty_func(scaled_ccds, axis=0)

    assert xp.all(xpx.isclose(avg_ccd.uncertainty.array, expected_unc, atol=1e-10))


@pytest.mark.parametrize(
    "comb_func", ["average_combine", "median_combine", "sum_combine"]
)
def test_user_supplied_combine_func_that_relies_on_masks(comb_func):
    # Test to make sure that setting some values to NaN internally
    # does not affect results when the user supplies a function that
    # uses masks to screen out bad data.

    data = xp.ones((10, 10))
    data = xpx.at(data)[5, 5].set(2)
    mask = data == 2
    ccd = CCDData(data, unit=u.adu, mask=mask)
    # Same, but no mask
    ccd2 = CCDData(data, unit=u.adu)

    ccd_list = [ccd, ccd, ccd2]
    c = Combiner(ccd_list)

    if comb_func == "sum_combine":

        def my_summer(data, mask, axis=None):
            xp = array_api_compat.array_namespace(data)
            new_data = []
            for i in range(data.shape[0]):
                if mask[i] is not None:
                    new_data.append(data[i] * ~mask[i])
                else:
                    new_data.append(xp.zeros_like(data[i]))

            new_data = xp.asarray(new_data)

            def sum_func(_, axis=axis):
                return xp.sum(new_data, axis=axis)

        expected_result = 3 * data
        actual_result = c.sum_combine(sum_func=my_summer(c._data_arr, c._data_arr_mask))
    elif comb_func == "average_combine":
        expected_result = data
        actual_result = c.average_combine(scale_func=xp.mean)
    elif comb_func == "median_combine":
        expected_result = data
        if not hasattr(xp, "median"):
            # If the array API does not have a median function, we
            # cannot test this.
            pytest.skip("The array library does not support median")
        actual_result = c.median_combine(median_func=xp.median)

    # Two of the three values are masked, so no matter what the combination
    # method is the result in this pixel should be 2.
    expected_result = xpx.at(expected_result)[5, 5].set(2)

    assert xp.all(xpx.isclose(expected_result, actual_result.data))
