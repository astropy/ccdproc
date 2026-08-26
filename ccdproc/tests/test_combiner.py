# Licensed under a 3-clause BSD style license - see LICENSE.rst
import math
import types
from functools import partial

import array_api_compat
import array_api_extra as xpx
import astropy.units as u
import numpy as np
import pytest
from astropy.nddata import CCDData
from astropy.stats import median_absolute_deviation as mad
from astropy.utils.data import get_pkg_data_filename
from numpy.testing import assert_allclose

from ccdproc import create_deviation
from ccdproc._nanfuncs import nanmean, nanmedian, nanstd, nansum
from ccdproc.combiner import (
    Combiner,
    _calculate_size_of_image,
    _calculate_step_sizes,
    _default_average,
    _default_median,
    _default_std,
    _default_sum,
    combine,
    sigma_func,
)

# Set up the array library to be used in tests
from ccdproc.conftest import testing_array_device as xp_device
from ccdproc.conftest import testing_array_library as xp
from ccdproc.core import _namespace_dtype, _native_numpy, _to_numpy
from ccdproc.image_collection import ImageFileCollection
from ccdproc.tests.pytest_fixtures import ccd_data as ccd_data_func
from ccdproc.tests.pytest_fixtures import numpy_ccddata

# Several tests have many more NaNs in them than real data. numpy generates
# lots of warnings in those cases and it makes more sense to suppress them
# than to generate them.
pytestmark = pytest.mark.filterwarnings(
    "ignore:All-NaN slice encountered:RuntimeWarning"
)


def _overall_median(arr):
    """NaN-aware median of every element of ``arr``, staying in ``xp``."""
    return nanmedian(xp.reshape(arr, (-1,)), axis=0, xp=xp)


def _make_mean_scaler(ccd_data):
    # Use the namespace of each argument: the reference image may be a
    # NumPy array read from FITS while ``x`` is a namespace array (or vice
    # versa), so neither side can assume the other's array library.
    ref_xp = array_api_compat.array_namespace(ccd_data.data)
    ref_mean = float(ref_xp.mean(ccd_data.data))

    def scale_by_mean(x):
        # scale each array to the mean of the first image
        xp_x = array_api_compat.array_namespace(x)
        return ref_mean / float(xp_x.mean(x))

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
    # Also test the public properties
    assert c.data.shape == c._data_arr.shape
    assert c.mask.shape == c._data_arr_mask.shape


# Each of the four Combiner defaults, paired with the name of the native
# function it prefers and the spec-only fallback it uses when the namespace
# has none. None of the four names are in the array API standard, so on a
# minimal namespace (array-api-strict) every one of them falls back.
_DEFAULT_FUNCS = [
    (_default_median, "nanmedian", nanmedian),
    (_default_average, "nanmean", nanmean),
    (_default_sum, "nansum", nansum),
    (_default_std, "nanstd", nanstd),
]


@pytest.mark.parametrize(("default_func", "function_name", "fallback"), _DEFAULT_FUNCS)
def test_bottleneck_defaults_respect_array_namespace(
    default_func, function_name, fallback
):
    default = default_func(xp=xp)

    if array_api_compat.is_numpy_namespace(xp):
        # Only this branch needs bottleneck; skipping the whole test on it
        # would leave the fallback branch below dead on the strict job,
        # whose env deliberately omits bottleneck.
        bottleneck = pytest.importorskip("bottleneck")
        expected = getattr(bottleneck, function_name)
    elif hasattr(xp, function_name):
        expected = getattr(xp, function_name)
    else:
        # The namespace has no such function -- array-api-strict has none of
        # the four -- so the spec-only fallback stands in, bound to it.
        assert isinstance(default, partial)
        assert default.func is fallback
        assert default.keywords == {"xp": xp}
        return
    assert default is expected


@pytest.mark.parametrize(("default_func", "_name", "fallback"), _DEFAULT_FUNCS)
def test_defaults_fall_back_without_native_nan_function(default_func, _name, fallback):
    # A namespace with none of the nan-aware reductions (not one of them is in
    # the array API standard) gets the spec-only fallback, bound to that
    # namespace. Use a stand-in namespace so this is exercised regardless of
    # which backend is under test; most backends in CI happen to provide all
    # four natively.
    fake_xp = types.ModuleType("not_a_real_array_namespace")
    default = default_func(xp=fake_xp)

    assert isinstance(default, partial)
    assert default.func is fallback
    assert default.keywords == {"xp": fake_xp}


@pytest.mark.parametrize(
    ("combine_method", "expected"),
    [("average_combine", 2.0), ("sum_combine", 6.0)],
)
def test_bottleneck_combination_defaults_respect_array_namespace(
    combine_method, expected
):
    pytest.importorskip("bottleneck")
    ccd_list = [
        CCDData(xp.full((2, 2), value), unit=u.adu) for value in (1.0, 2.0, 3.0)
    ]

    result = getattr(Combiner(ccd_list), combine_method)()

    assert array_api_compat.array_namespace(result.data) is xp
    assert xp.all(xpx.isclose(result.data, expected))


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


# A dtype that is not one of the namespace's own dtype objects (a builtin
# type, a string, a NumPy scalar type) is mapped to the namespace's dtype of
# the same name, so ``dtype=int`` works on every backend.
@pytest.mark.parametrize(
    "dtype,expected_name",
    [
        (int, "int64"),
        (float, "float64"),
        ("float32", "float32"),
        (np.float32, "float32"),
    ],
)
def test_combiner_dtype_mapped_to_namespace(dtype, expected_name):
    ccd_data = ccd_data_func()
    c = Combiner([ccd_data, ccd_data], dtype=dtype)
    assert c.dtype == getattr(xp, expected_name)
    assert c._data_arr.dtype == getattr(xp, expected_name)
    avg = c.average_combine()
    assert avg.dtype == getattr(xp, expected_name)


def test_namespace_dtype_passes_through_what_numpy_cannot_read():
    """
    A dtype that NumPy cannot interpret is returned unchanged for the
    namespace to deal with. On array-api-strict the namespace's own dtype
    objects take this path; on the other backends they are NumPy types, so
    an arbitrary object stands in for that case here.
    """

    class NotADtype:
        pass

    sentinel = NotADtype()
    assert _namespace_dtype(sentinel, xp) is sentinel
    # The namespace's own dtype objects always come back as themselves.
    assert _namespace_dtype(xp.float32, xp) == xp.float32


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


# Regression test for #965: the Combiner must stack the input images and
# masks instead of passing a nested list of arrays to xp.asarray, and the
# stacked arrays must live in the namespace and on the device of the inputs.
def test_combiner_stacks_arrays_on_input_device():
    data = xp.zeros((4, 4), dtype=xp.float32, device=xp_device)
    data = xpx.at(data)[1, 2].set(1)
    mask = xp.zeros((4, 4), dtype=xp.bool, device=xp_device)
    mask = xpx.at(mask)[3, 3].set(True)
    # astropy's CCDData.mask setter always coerces to numpy, which is not
    # possible for arrays on a non-default device, so assign the mask in the
    # data's namespace directly to emulate an array-API-aware CCDData.
    ccd_masked = CCDData(data, unit=u.adu)
    ccd_masked._mask = mask
    ccd_unmasked = CCDData(data, unit=u.adu)
    # Masks on CCDData may be plain numpy arrays even when the data is not.
    ccd_np_mask = CCDData(data, unit=u.adu, mask=np.ones((4, 4), dtype=bool))

    c = Combiner([ccd_masked, ccd_unmasked, ccd_np_mask], dtype=xp.float32)

    for arr in (c.data, c.mask):
        assert array_api_compat.array_namespace(
            arr
        ) is array_api_compat.array_namespace(data)
        assert array_api_compat.device(arr) == array_api_compat.device(data)
        assert arr.shape == (3, 4, 4)
    assert c.data.dtype == xp.float32
    assert c.mask.dtype == xp.bool
    assert c.data[0, 1, 2] == 1
    assert c.mask[0, 3, 3]
    assert not xp.any(c.mask[1, ...])
    assert xp.all(c.mask[2, ...])

    # Scaling values should end up on the same device as the data, and in
    # the dtype of the data: integer scaling must not rely on int/float type
    # promotion, which the array API does not guarantee.
    c.scaling = [1.0, 2.0, 3.0]
    assert array_api_compat.device(c.scaling) == array_api_compat.device(data)
    c.scaling = [1, 2, 3]
    assert c.scaling.dtype == c.data.dtype
    assert float(c.scaling[1, 0, 0]) == 2.0
    c.scaling = lambda arr: float(arr.shape[0])
    assert array_api_compat.device(c.scaling) == array_api_compat.device(data)
    c.scaling = lambda arr: arr.shape[0]
    assert c.scaling.dtype == c.data.dtype
    assert float(c.scaling[0, 0, 0]) == 4.0
    # A backend array, which need not implement __len__, is accepted as
    # scaling as long as its length matches the number of images.
    c.scaling = xp.asarray([1.0, 0.5, 2.0], device=xp_device)
    assert array_api_compat.device(c.scaling) == array_api_compat.device(data)
    assert float(c.scaling[2, 0, 0]) == 2.0
    with pytest.raises(ValueError, match="same length"):
        c.scaling = xp.asarray([1.0, 0.5], device=xp_device)
    with pytest.raises(TypeError, match="same length"):
        c.scaling = 2.0
    # A scaling callable that returns a 0-d array of the backend (rather than
    # a Python float) must also work; array-api-strict rejects a list of such
    # arrays passed to asarray.
    c.scaling = lambda arr: xp.mean(arr) + 1
    assert array_api_compat.device(c.scaling) == array_api_compat.device(data)
    assert c.scaling.shape == (3, 1, 1)
    # all three images share ``data``, whose mean is 1/16
    assert float(xp.max(xp.abs(c.scaling - (1.0 + 1 / 16)))) < 1e-6


def test_combine_scale_callable_returning_backend_scalar():
    # Added in #976: ``combine(scale=<callable>)`` must accept a callable that
    # returns a 0-d array of the backend. Only array-api-strict rejects the
    # previous ``xp.asarray([0-d array, ...])``, so this test guards the
    # scaling path on that backend.
    ccds = [
        CCDData(xp.full((4, 4), float(i), dtype=xp.float64), unit=u.adu)
        for i in (1, 2, 4)
    ]
    result = combine(ccds, method="average", scale=lambda arr: 1 / xp.mean(arr))
    assert_allclose(np.asarray(result.data), 1.0)


def test_combiner_explicit_namespace_differs_from_data():
    # Regression test for the review of #976: when the caller passes an ``xp``
    # that is not the namespace of the input data, the device of the inputs
    # must not be forced onto ``xp`` (numpy's 'cpu' means nothing to jax or
    # array-api-strict). The data is converted into ``xp`` on its default
    # device instead.
    #
    # Only array-api-strict actually rejects a foreign device: on numpy the
    # data namespace *is* ``xp`` so the device is legitimately reused, and
    # dask accepts ``device='cpu'`` regardless, so this test can fail only in
    # the array-api-strict job.
    np_ccds = [CCDData(np.ones((3, 3)) * i, unit=u.adu) for i in range(1, 3)]
    np_ccds[0].mask = np.zeros((3, 3), dtype=bool)
    c = Combiner(np_ccds, xp=xp)
    assert array_api_compat.array_namespace(c.data) is array_api_compat.array_namespace(
        xp.zeros(1)
    )
    assert c.data.shape == (2, 3, 3)
    assert c.data.dtype == xp.float64
    assert c.mask.dtype == xp.bool
    assert float(xp.sum(c.data)) == 27.0


def test_combiner_accepts_raw_module_as_namespace():
    # A plain module (numpy here) passed as ``xp`` is normalised to its
    # array-api-compat namespace, so array-API-only features such as
    # ``xp.bool`` and ``device=`` are available to the Combiner.
    np_ccds = [CCDData(np.ones((2, 2)) * i, unit=u.adu) for i in range(1, 3)]
    c = Combiner(np_ccds, xp=np)
    assert c._xp is array_api_compat.array_namespace(np.zeros(1))
    assert c.data.shape == (2, 2, 2)
    c.scaling = [1, 2]
    assert float(np.sum(c.average_combine().data)) == 10.0


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
    combo.weights = xpx.at(combo.weights)[:, 5, 5].set(xp.asarray([1.0, 5.0, 10.0]))
    ccd = combo.average_combine()
    assert xp.all(xpx.isclose(ccd.data[5, 5], 312.5))
    assert xp.all(xpx.isclose(ccd.data[0, 0], 0))


def test_combine_weighted_average_with_mask():
    mask = xp.asarray([[True, False], [False, False]])
    ccd_list = [
        CCDData(xp.asarray([[1, 2], [3, 4]]), unit=u.adu, mask=mask),
        CCDData(xp.asarray([[10, 20], [30, 40]]), unit=u.adu),
    ]

    combined = combine(
        ccd_list,
        method="average",
        weights=xp.asarray([1, 3]),
    )

    expected = xp.asarray([[10, 15.5], [23.25, 31]])
    assert xp.all(xpx.isclose(combined.data, expected))


def test_combiner_weighted_average_with_mask_by_pixel():
    mask = xp.asarray([[True, False], [False, False]])
    ccd_list = [
        CCDData(xp.asarray([[1, 2], [3, 4]]), unit=u.adu, mask=mask),
        CCDData(xp.asarray([[10, 20], [30, 40]]), unit=u.adu),
    ]
    combiner = Combiner(ccd_list)
    combiner.weights = xp.asarray(
        [
            [[4, 1], [1, 1]],
            [[1, 3], [2, 4]],
        ]
    )

    combined = combiner.average_combine()

    expected = xp.asarray([[10, 15.5], [21, 32.8]])
    assert xp.all(xpx.isclose(combined.data, expected))


def test_combiner_weighted_average_with_clipping():
    ccd_list = [
        CCDData(xp.ones((2, 2)), unit=u.adu),
        CCDData(xp.full((2, 2), 3), unit=u.adu),
        CCDData(xp.asarray([[100, 5], [5, 5]]), unit=u.adu),
    ]
    combiner = Combiner(ccd_list)
    combiner.weights = xp.asarray([1, 1, 10])
    combiner.minmax_clipping(max_clip=50)

    combined = combiner.average_combine()

    expected = xp.asarray([[2, 4.5], [4.5, 4.5]])
    assert xp.all(xpx.isclose(combined.data, expected))


def test_combiner_weighted_average_preserves_custom_scale_func():
    mask = xp.asarray([[True, False]])
    ccd_list = [
        CCDData(xp.asarray([[1, 2]]), unit=u.adu, mask=mask),
        CCDData(xp.asarray([[10, 20]]), unit=u.adu),
    ]
    combiner = Combiner(ccd_list)
    combiner.weights = xp.asarray([1, 3])

    combined = combiner.average_combine(scale_func=xp.mean)

    # A custom scale function historically opts out of internal NaN
    # substitution, and the weighted path does not call it.
    expected = xp.asarray([[7.75, 15.5]])
    assert xp.all(xpx.isclose(combined.data, expected))


@pytest.mark.filterwarnings(
    "ignore:Degrees of freedom <= 0 for slice.:RuntimeWarning",
    "ignore:invalid value encountered in divide:RuntimeWarning",
)
def test_combiner_weighted_average_fully_masked():
    mask = xp.ones((1, 1), dtype=xp.bool)
    ccd_list = [
        CCDData(xp.ones((1, 1)), unit=u.adu, mask=mask),
        CCDData(xp.ones((1, 1)), unit=u.adu, mask=mask),
    ]
    combiner = Combiner(ccd_list)
    combiner.weights = xp.asarray([1, 2])

    combined = combiner.average_combine()

    assert combined.mask[0, 0]


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
    assert xp.mean(ccd.data) == 0


def test_combiner_minmax_max():
    ccd_list = [
        CCDData(xp.zeros((10, 10)), unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 1000, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 1000, unit=u.adu),
    ]

    c = Combiner(ccd_list)
    c.minmax_clipping(min_clip=None, max_clip=500)
    assert xp.all(c._data_arr_mask[2, ...])


def test_combiner_minmax_min():
    ccd_list = [
        CCDData(xp.zeros((10, 10)), unit=u.adu),
        CCDData(xp.zeros((10, 10)) - 1000, unit=u.adu),
        CCDData(xp.zeros((10, 10)) + 1000, unit=u.adu),
    ]

    c = Combiner(ccd_list)
    c.minmax_clipping(min_clip=-500, max_clip=None)
    assert xp.all(c._data_arr_mask[1, ...])


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
    ccd_data = CCDData(data=xp.asarray([[0.0, 1.0], [2.0, 3.0]]), unit="adu")
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    c.weights = xp.asarray([1.0, 2.0, 3.0])
    ccd = c.sum_combine()
    expected_result = sum(w * d.data for w, d in zip(c.weights, ccd_list, strict=True))
    assert xp.all(xpx.isclose(ccd.data, expected_result))


# test weighted sum
def test_combiner_sum_weighted_by_pixel():
    ccd_data = CCDData(data=xp.asarray([[1, 2], [4, 8]]), unit="adu")
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    # Weights below are chosen so that every entry in
    weights_pixel = [[8.0, 4.0], [2.0, 1.0]]
    c.weights = xp.asarray([weights_pixel] * 3)
    ccd = c.sum_combine()
    expected_result = xp.asarray([[24.0, 24.0], [24.0, 24.0]])
    assert xp.all(xpx.isclose(ccd.data, expected_result))


def test_combiner_sum_weighted_with_mask():
    mask = xp.asarray([[True, False]])
    ccd_list = [
        CCDData(xp.asarray([[1, 2]]), unit=u.adu, mask=mask),
        CCDData(xp.asarray([[10, 20]]), unit=u.adu),
    ]
    combiner = Combiner(ccd_list)
    combiner.weights = xp.asarray([1.0, 3.0])

    combined = combiner.sum_combine()

    expected = xp.asarray([[30.0, 62.0]])
    assert xp.all(xpx.isclose(combined.data, expected))


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
    # Ensure that the mask is correctly applied to pixels that are fully masked
    assert ccd.mask[0, 0]
    assert not ccd.mask[5, 5]


def test_combiner_with_scaling():
    ccd_data = ccd_data_func()
    # The factors below are not particularly important; just avoid anything
    # whose average is 1.
    ccd_data_lower = CCDData(ccd_data.data * 3, unit=ccd_data.unit)
    ccd_data_higher = CCDData(ccd_data.data * 0.9, unit=ccd_data.unit)
    combiner = Combiner([ccd_data, ccd_data_higher, ccd_data_lower])
    scale_by_mean = _make_mean_scaler(ccd_data)
    combiner.scaling = scale_by_mean
    avg_ccd = combiner.average_combine()
    # Does the mean of the scaled arrays match the value to which it was
    # scaled?
    assert xp.all(xpx.isclose(xp.mean(avg_ccd.data), xp.mean(ccd_data.data)))
    assert avg_ccd.shape == ccd_data.shape
    median_ccd = combiner.median_combine()
    # Does median also scale to the correct value?
    assert xp.all(
        xpx.isclose(_overall_median(median_ccd.data), _overall_median(ccd_data.data))
    )

    # Set the scaling manually...
    combiner.scaling = [scale_by_mean(combiner._data_arr[i, ...]) for i in range(3)]
    avg_ccd = combiner.average_combine()
    assert xp.all(xpx.isclose(xp.mean(avg_ccd.data), xp.mean(ccd_data.data)))
    assert avg_ccd.shape == ccd_data.shape

    # Scale by a float
    avg_ccd = combiner.average_combine(scale_to=2.0)
    expected_avg = 2 * xp.mean(
        xp.asarray((ccd_data.data, ccd_data_lower.data, ccd_data_higher.data))
    )
    assert xp.all(xpx.isclose(xp.mean(avg_ccd.data), expected_avg))
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
    c = Combiner(ccd_list, xp=xp)
    ccd_by_combiner = c.average_combine()

    fitsfilename_list = [fitsfile] * 3
    avgccd = combine(
        fitsfilename_list,
        output_file=None,
        method="average",
        unit=u.adu,
        array_package=xp,
    )
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
    c = Combiner(ccd_list, xp=xp)
    ccd_by_combiner = c.average_combine()

    fitsfilename_list = [fitsfile] * 3
    avgccd = combine(
        fitsfilename_list,
        output_file=None,
        method="average",
        unit=u.adu,
        array_package=xp,
    )
    # averaging same fits images should give back same fits image
    assert xp.all(xpx.isclose(avgccd.data, ccd_by_combiner.data))


def test_combiner_result_dtype():
    """Regression test: #391

    The result should have the appropriate dtype not the dtype of the first
    input."""
    ccd = CCDData(xp.ones((3, 3), dtype=xp.uint16), unit="adu")
    ccd_times_2 = CCDData(ccd.data * 2, unit=ccd.unit)
    ccd_times_3 = CCDData(ccd.data * 3, unit=ccd.unit)
    res = combine([ccd, ccd_times_2])
    # The default dtype of Combiner is float64
    assert res.data.dtype == xp.float64
    ref = xp.ones((3, 3)) * 1.5
    assert xp.all(xpx.isclose(res.data, ref))
    res = combine([ccd, ccd_times_2, ccd_times_3], dtype=int)
    # The result dtype should be integer:
    assert xp.isdtype(res.data.dtype, "integral")
    # Compare with a Python int: an integer array and a float reference
    # cannot be promoted together under the array API standard.
    assert xp.all(res.data == 2)


def test_combiner_image_file_collection_input(tmp_path):
    # Regression check for #754
    ccd = ccd_data_func()
    for i in range(3):
        numpy_ccddata(ccd).write(tmp_path / f"ccd-{i}.fits")

    ifc = ImageFileCollection(tmp_path)
    ccds = list(ifc.ccds())

    # Need to convert these to the array namespace.
    for a_ccd in ccds:
        a_ccd.data = xp.asarray(a_ccd.data, dtype=xp.float64)
        if a_ccd.mask is not None:
            a_ccd.mask = xp.asarray(a_ccd.mask, dtype=xp.bool)
        if a_ccd.uncertainty is not None:
            a_ccd.uncertainty.array = xp.asarray(
                a_ccd.uncertainty.array, dtype=xp.float64
            )
    comb = Combiner(ccds)

    # Do this on a separate line from the assert to make debugging easier
    result = comb.average_combine()
    # ``result`` lands on the namespace's default device (the conversion
    # above did not request ``xp_device``), while ``ccd.data`` is on
    # ``xp_device``; compare via NumPy copies rather than requiring both
    # sides to share a device.
    assert_allclose(_to_numpy(ccd.data), _to_numpy(result.data))


def test_combine_image_file_collection_input(tmp_path):
    # Another regression check for #754 but this time with the
    # combine function instead of Combiner
    ccd = ccd_data_func()
    xp = array_api_compat.array_namespace(ccd.data)
    for i in range(3):
        numpy_ccddata(ccd).write(tmp_path / f"ccd-{i}.fits")

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

    # The combine() results land on the namespace's default device, while
    # ``ccd.data`` is on ``xp_device``; compare via NumPy copies rather than
    # requiring both sides to share a device.
    assert_allclose(_to_numpy(ccd.data), _to_numpy(comb_files.data))
    assert_allclose(_to_numpy(ccd.data), _to_numpy(comb_ccds.data))
    assert_allclose(_to_numpy(ccd.data), _to_numpy(comb_string.data))

    with pytest.raises(FileNotFoundError):
        # This should fail because the test is not running in the
        # folder where the images are.
        _ = combine(ifc.files_filtered())


# test combiner convenience function works with list of ccddata objects
def test_combine_average_ccddata():
    fitsfile = get_pkg_data_filename("data/a8280271.fits")
    ccd = CCDData.read(fitsfile, unit=u.adu)
    # ``combine`` ignores ``array_package`` for CCDData input, so convert
    # the data to the namespace ourselves.
    ccd.data = xp.asarray(_native_numpy(ccd.data))
    ccd_list = [ccd] * 3
    c = Combiner(ccd_list)
    ccd_by_combiner = c.average_combine()

    avgccd = combine(ccd_list, output_file=None, method="average", unit=u.adu)
    # averaging same ccdData should give back same images
    assert xp.all(xpx.isclose(avgccd.data, ccd_by_combiner.data))


# combine() sizes images without ``.nbytes``, which not every array library
# provides; check the count against the known element sizes.
@pytest.mark.parametrize(
    "dtype,element_size",
    [
        (xp.float32, 4),
        (xp.complex64, 8),
        (xp.complex128, 16),
    ],
)
def test_calculate_size_of_image(dtype, element_size):
    ccd = CCDData(
        xp.zeros((7, 5), dtype=dtype),
        unit=u.adu,
        mask=xp.zeros((7, 5), dtype=xp.bool),
    )
    # data (element_size bytes) plus bool mask (1 byte)
    assert _calculate_size_of_image(ccd) == 7 * 5 * (element_size + 1)


# test combiner convenience function reads fits file and
# and combine as expected when asked to run in limited memory
def test_combine_limitedmem_fitsimages():
    fitsfile = get_pkg_data_filename("data/a8280271.fits")
    ccd = CCDData.read(fitsfile, unit=u.adu)
    ccd_list = [ccd] * 5
    c = Combiner(ccd_list, xp=xp)
    ccd_by_combiner = c.average_combine()

    fitsfilename_list = [fitsfile] * 5
    avgccd = combine(
        fitsfilename_list,
        output_file=None,
        method="average",
        mem_limit=1e6,
        unit=u.adu,
        array_package=xp,
    )
    # averaging same ccdData should give back same images
    assert xp.all(xpx.isclose(avgccd.data, ccd_by_combiner.data))


# test combiner convenience function reads fits file and
# and combine as expected when asked to run in limited memory with scaling
def test_combine_limitedmem_scale_fitsimages():
    fitsfile = get_pkg_data_filename("data/a8280271.fits")
    ccd = CCDData.read(fitsfile, unit=u.adu)
    ccd_list = [ccd] * 5
    c = Combiner(ccd_list, xp=xp)
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
        array_package=xp,
    )

    assert xp.all(xpx.isclose(avgccd.data, ccd_by_combiner.data))


# test the optional uncertainty function in average_combine
def test_average_combine_uncertainty():
    ccd_data = ccd_data_func()
    ccd_list = [ccd_data, ccd_data, ccd_data]
    c = Combiner(ccd_list)
    ccd = c.average_combine(uncertainty_func=xp.sum)
    uncert_ref = xp.sum(c._data_arr, axis=0) / math.sqrt(3)
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
    uncert_ref = xp.sum(c._data_arr, axis=0) / math.sqrt(3)
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
    uncert_ref = xp.sum(c._data_arr, axis=0) * math.sqrt(3)
    assert xp.all(xpx.isclose(ccd.uncertainty.array, uncert_ref))

    # Compare this also to the "combine" call
    ccd2 = combine(ccd_list, method="sum", combine_uncertainty_function=xp.sum)
    assert xp.all(xpx.isclose(ccd.data, ccd2.data))
    assert xp.all(xpx.isclose(ccd.uncertainty.array, ccd2.uncertainty.array))


@pytest.mark.parametrize("scale", ["function", "mean"])
def test_combine_ccd_with_uncertainty_and_mask_from_fits(scale, tmp_path):
    # Test initializing a CCDData object with uncertainty and mask in the
    # combine function.
    fitsfile = get_pkg_data_filename("data/a8280271.fits", package="ccdproc.tests")
    ccd_data = CCDData.read(fitsfile, unit=u.adu)
    ccd_data.data = xp.asarray(ccd_data.data, dtype=xp.float64)
    # Set ._mask instead of .mask to avoid conversion to numpy array
    ccd_data._mask = xp.zeros_like(ccd_data.data, dtype=xp.bool)
    if scale == "function":
        scale_by_mean = _make_mean_scaler(ccd_data)
    else:
        scale_by_mean = [1.0, 1.0, 1.0]

    ccd_data = create_deviation(
        ccd_data, gain=1.0 * u.electron / u.adu, readnoise=5 * u.electron
    )
    fits_with_uncertainty = tmp_path / "test.fits"
    ccd_data.write(fits_with_uncertainty)

    ccd2 = combine(
        [fits_with_uncertainty] * 3,
        method="average",
        array_package=xp,
        scale=scale_by_mean,
    )
    assert xp.all(xpx.isclose(ccd2.data, ccd_data.data))


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
    assert int(xp.count_nonzero(expected_result.mask)) == int(mask_point)
    assert ccd_comb.mask[0, 0] == mask_point
    assert int(xp.count_nonzero(ccd_comb.mask)) == int(mask_point)


def test_combine_overwrite_output(tmp_path):
    """
    The combine function should *not* overwrite the result file
    unless the overwrite_output argument is True
    """
    output_file = tmp_path / "fake.fits"

    ccd = CCDData(xp.ones((3, 3)), unit="adu")
    ccd_times_2 = CCDData(ccd.data * 2, unit=ccd.unit)

    # Make sure we have a file to overwrite
    ccd.write(output_file)
    # Test that overwrite does NOT happen by default
    with pytest.raises(OSError, match="fake.fits already exists"):
        res = combine([ccd, ccd_times_2], output_file=str(output_file))

    # Should be no error here...
    # The default dtype of Combiner is float64
    res = combine([ccd, ccd_times_2], output_file=output_file, overwrite_output=True)

    # The returned result must still be in the array namespace, with its
    # uncertainty and mask; only the file is written from a NumPy copy.
    xp_compat = array_api_compat.array_namespace(xp.asarray(0))
    for arr in (res.data, res.uncertainty.array, res.mask):
        assert array_api_compat.array_namespace(arr) is xp_compat

    # Need to convert this to the array namespace.
    res_from_disk = CCDData.read(output_file)
    assert res_from_disk.uncertainty is not None
    assert res_from_disk.mask is not None
    res_from_disk.data = xp.asarray(_native_numpy(res_from_disk.data))

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
    ref_uncertainty /= xp.sqrt(xp.asarray(2.0))
    assert xp.all(xpx.isclose(ccd.uncertainty.array, ref_uncertainty))


# test resulting uncertainty is corrected for the number of images (with mask)
def test_combiner_uncertainty_average_mask():
    mask = xp.zeros((10, 10), dtype=xp.bool)
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
    ref_uncertainty = xp.ones((10, 10)) * xp.std(xp.asarray([1.0, 2.0, 3.0]))
    # Correction because we combined two images.
    ref_uncertainty /= xp.sqrt(xp.asarray(3.0))
    ref_uncertainty = xpx.at(ref_uncertainty)[5, 5].set(
        xp.std(xp.asarray([2.0, 3.0])) / xp.sqrt(xp.asarray(2.0))
    )
    assert xp.all(xpx.isclose(ccd.uncertainty.array, ref_uncertainty))


# test resulting uncertainty is corrected for the number of images (with mask)
def test_combiner_uncertainty_median_mask():
    mad_to_sigma = 1.482602218505602
    mask = xp.zeros((10, 10), dtype=xp.bool)
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
    # It turns out that the expression below evaluates to a np.float64, which
    # introduces numpy into the array namespace, which raises an error
    # when arrat_api_compat tries to figure out the namespace. Casting
    # it to a regular float fixes that.
    ref_uncertainty = xp.ones((10, 10)) * float(mad_to_sigma * mad([1, 2, 3]))
    # Correction because we combined two images.
    ref_uncertainty /= xp.sqrt(xp.asarray(3.0))  # 0.855980789955
    med_value = float(mad_to_sigma * mad([2, 3])) / float(xp.sqrt(xp.asarray(2.0)))
    ref_uncertainty = xpx.at(ref_uncertainty)[5, 5].set(med_value)  # 0.524179041254
    assert xp.all(xpx.isclose(ccd.uncertainty.array, ref_uncertainty))


# test resulting uncertainty is corrected for the number of images (with mask)
def test_combiner_uncertainty_sum_mask():
    mask = xp.zeros((10, 10), dtype=xp.bool)
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
    ref_uncertainty = xp.ones((10, 10)) * xp.std(xp.asarray([1.0, 2.0, 3.0]))
    ref_uncertainty *= xp.sqrt(xp.asarray(3.0))
    ref_uncertainty = xpx.at(ref_uncertainty)[5, 5].set(
        xp.std(xp.asarray([2.0, 3.0])) * xp.sqrt(xp.asarray(2.0))
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
    assert xp.all(xpx.isclose(xp.mean(avg_ccd.data), xp.mean(ccd_data.data)))
    assert avg_ccd.shape == ccd_data.shape
    median_ccd = combiner.median_combine()
    # Does median also scale to the correct value?
    assert xp.all(
        xpx.isclose(_overall_median(median_ccd.data), _overall_median(ccd_data.data))
    )

    # Set the scaling manually...
    combiner.scaling = [scale_by_mean(combiner._data_arr[i, ...]) for i in range(3)]
    avg_ccd = combiner.average_combine()
    assert xp.all(xpx.isclose(xp.mean(avg_ccd.data), xp.mean(ccd_data.data)))
    assert avg_ccd.shape == ccd_data.shape


def test_clip_extrema_stays_in_array_namespace_and_device():
    data = [
        [[9.0, 2.0, 7.0], [4.0, 8.0, 1.0]],
        [[3.0, 6.0, 5.0], [9.0, 2.0, 8.0]],
        [[6.0, 1.0, 4.0], [2.0, 7.0, 3.0]],
    ]

    c = Combiner(
        [CCDData(xp.asarray(image, device=xp_device), unit=u.adu) for image in data]
    )
    c.clip_extrema(nlow=1, nhigh=1)

    expected_mask = xp.asarray(
        [
            [[True, False, True], [False, True, True]],
            [[True, True, False], [True, True, True]],
            [[False, True, True], [True, False, False]],
        ],
        device=xp_device,
    )
    assert array_api_compat.array_namespace(c._data_arr_mask) is xp
    assert array_api_compat.device(c.mask) == array_api_compat.device(c.data)
    assert xp.all(c.mask == expected_mask)


def test_clip_extrema_masks_expected_indices():
    # pixel [0, 0] is a 3-way tie at 5.0; with the standard's stable=True
    # argsort the last tied image (index 3) is the one nhigh masks --
    # identical to the pre-rewrite behavior for this input.
    data = [[[5.0, 1.0]], [[5.0, 4.0]], [[3.0, 2.0]], [[5.0, 3.0]]]

    c = Combiner(
        [CCDData(xp.asarray(image, device=xp_device), unit=u.adu) for image in data]
    )
    c.clip_extrema(nlow=1, nhigh=1)

    expected_mask = xp.asarray(
        [
            [[False, True]],
            [[False, True]],
            [[True, False]],
            [[True, False]],
        ],
        device=xp_device,
    )
    assert xp.all(c.mask == expected_mask)


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
    numpy_ccddata(ccd2).write(tmp_file.strpath)


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
    ccd_data_lower = CCDData(ccd_data.data * 3, unit=ccd_data.unit)
    ccd_data_higher = CCDData(ccd_data.data * 0.9, unit=ccd_data.unit)

    combiner = Combiner([ccd_data, ccd_data_higher, ccd_data_lower])
    # scale each array to the mean of the first image
    scale_by_mean = _make_mean_scaler(ccd_data)
    combiner.scaling = scale_by_mean

    scaled_ccds = xp.stack(
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
                if mask[i, ...] is not None:
                    new_data.append(
                        xp.where(
                            mask[i, ...], xp.zeros_like(data[i, ...]), data[i, ...]
                        )
                    )
                else:
                    new_data.append(xp.zeros_like(data[i, ...]))

            new_data = xp.stack(new_data)

            def sum_func(_, axis=axis):
                return xp.sum(new_data, axis=axis)

        expected_result = 3 * data
        actual_result = c.sum_combine(sum_func=my_summer(c._data_arr, c._data_arr_mask))
    elif comb_func == "average_combine":
        expected_result = data
        actual_result = c.average_combine(scale_func=xp.mean)
    elif comb_func == "median_combine":
        expected_result = data
        actual_result = c.median_combine(median_func=partial(nanmedian, xp=xp))

    # Two of the three values are masked, so no matter what the combination
    # method is the result in this pixel should be 2.
    expected_result = xpx.at(expected_result)[5, 5].set(2)

    assert xp.all(xpx.isclose(expected_result, actual_result.data))


# Regression tests for #982: combine()'s ``array_package`` only normalised
# an already-instantiated array (via array_api_compat.array_namespace), so a
# raw array module (e.g. plain ``numpy`` or ``dask.array``, as opposed to
# ``array_api_compat.numpy``/``array_api_compat.dask.array``) passed through
# unnormalised into code that relies on array-API features the raw module
# does not provide.
def test_combine_array_package_raw_module(tmp_path):
    """A raw array module passed as ``array_package`` should be normalised
    to its array-api-compat namespace, the same way ``Combiner`` normalises
    its ``xp`` argument.
    """
    ccd = CCDData(np.arange(9, dtype=float).reshape(3, 3), unit=u.adu)
    files = []
    for i in range(3):
        path = tmp_path / f"raw-module-{i}.fits"
        ccd.write(path)
        files.append(str(path))

    result = combine(files, array_package=np, unit="adu")
    assert array_api_compat.is_numpy_array(result.data)

    # On strict and jax, the suite's own ``xp`` fixture is itself a raw
    # module (``array_api_strict``/``jax.numpy`` imported directly, not
    # through array_api_compat), so this also exercises the raw-module
    # path of #982 on those backends without a separate test case.
    result = combine(files, array_package=xp, unit="adu")
    expected_xp = array_api_compat.array_namespace(xp.asarray(0))
    assert array_api_compat.array_namespace(result.data) is expected_xp


def test_combine_array_package_dask_module(tmp_path):
    """Regression test for #982.

    Passing the raw ``dask.array`` module (rather than its
    array-api-compat wrapper, ``array_api_compat.dask.array``) as
    ``array_package`` used to reach ``dask.array.from_array`` with an
    unsupported ``device=`` keyword, raising ``TypeError: from_array() got
    an unexpected keyword argument 'device'``.
    """
    dask = pytest.importorskip("dask.array")

    ccd = CCDData(np.arange(9, dtype=float).reshape(3, 3), unit=u.adu)
    files = []
    for i in range(3):
        path = tmp_path / f"raw-dask-{i}.fits"
        ccd.write(path)
        files.append(str(path))

    result = combine(files, array_package=dask, unit="adu")
    assert array_api_compat.is_dask_array(result.data)
