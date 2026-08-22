# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

import array_api_compat
import array_api_extra as xpx
import pytest
from astropy import units as u
from astropy.nddata import (
    CCDData,
    InverseVariance,
    StdDevUncertainty,
    VarianceUncertainty,
)
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy import array as np_array
from numpy import zeros as np_zeros
from numpy.ma import array as np_ma_array
from numpy.ma import nomask as np_ma_nomask
from numpy.random import default_rng
from numpy.testing import assert_allclose

# Set up the array library to be used in tests
from ccdproc.conftest import testing_array_library as xp
from ccdproc.core import (
    background_deviation_box,
    background_deviation_filter,
    cosmicray_lacosmic,
    cosmicray_median,
)
from ccdproc.tests.pytest_fixtures import ccd_data as ccd_data_func

pytest.importorskip("astroscrappy", reason="astroscrappy not installed")

DATA_SCALE = 5.3
NCRAYS = 30


def add_cosmicrays(data, scale, threshold, ncrays=NCRAYS):
    from numpy import array as np_array

    size = data.shape[0]
    rng = default_rng(99)
    crrays = rng.integers(0, size, size=(ncrays, 2))
    # use (threshold + 15) below to make sure cosmic ray is well above the
    # threshold no matter what the random number generator returns
    # add_cosmicrays is highly sensitive to the seed
    # ideally threshold should be set so it is not sensitive to seed, but
    # this is not working right now
    crflux = np_array(10 * scale * rng.random(ncrays) + (threshold + 15) * scale)

    # Some array libraries (Jax) do not support setting individual elements,
    # so use NumPy.
    data_as_np = np_array(data.data)
    for i in range(ncrays):
        y, x = crrays[i]
        data_as_np[y, x] = crflux[i]
    data.data = xp.asarray(data_as_np)
    return crrays


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_lacosmic uses astroscrappy, which requires numpy "
    "and fails on a non-default device",
)
def test_cosmicray_lacosmic():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 10
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    _, crarr = cosmicray_lacosmic(ccd_data.data, sigclip=5.9)

    # check the number of cosmic rays detected
    # Note that to get this to succeed reliably meant tuning
    # both sigclip and the threshold
    assert crarr.sum() == NCRAYS


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_lacosmic uses astroscrappy, which requires numpy "
    "and fails on a non-default device",
)
def test_cosmicray_lacosmic_ccddata():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)

    ccd_data.uncertainty = StdDevUncertainty(noise)
    nccd_data = cosmicray_lacosmic(ccd_data, sigclip=5.9)

    # check the number of cosmic rays detected
    # Note that to get this to succeed reliably meant tuning
    # both sigclip and the threshold
    assert nccd_data.mask.sum() == NCRAYS


def test_cosmicray_lacosmic_check_data():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    with pytest.raises(TypeError):
        noise = DATA_SCALE * xp.ones_like(ccd_data.data)
        cosmicray_lacosmic(10, noise)


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_lacosmic uses astroscrappy, which requires numpy "
    "and fails on a non-default device",
)
@pytest.mark.parametrize("array_input", [True, False])
@pytest.mark.parametrize("gain_correct_data", [True, False])
def test_cosmicray_gain_correct(array_input, gain_correct_data):
    # Add regression check for #705 and for the new gain_correct
    # argument.
    # The issue is that cosmicray_lacosmic gain-corrects the
    # data and returns that gain corrected data. That is not the
    # intent...
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)

    ccd_data.uncertainty = StdDevUncertainty(noise)
    # No units here on purpose.
    gain = 2.0

    if array_input:
        new_data, cr_mask = cosmicray_lacosmic(
            ccd_data.data, gain=gain, gain_apply=gain_correct_data
        )
    else:
        new_ccd = cosmicray_lacosmic(ccd_data, gain=gain, gain_apply=gain_correct_data)
        new_data = new_ccd.data
        cr_mask = new_ccd.mask

    # Turn the mask into array API compatible thing
    cr_mask = xp.asarray(cr_mask)
    # Fill masked locations with 0 since there is no simple relationship
    # between the original value and the corrected value.
    orig_data = xpx.at(ccd_data.data)[cr_mask].set(0.0)
    new_data = xpx.at(new_data)[cr_mask].set(0.0)

    if gain_correct_data:
        gain_for_test = gain
    else:
        gain_for_test = 1.0

    assert_allclose(gain_for_test * orig_data, new_data)


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_lacosmic's gain and mask paths do not support "
    "array-api-strict's strict scalar and dtype rules",
)
@pytest.mark.parametrize(
    ("uncertainty_type", "gain_power", "gain_apply", "expected_uncertainty_unit"),
    [
        pytest.param(
            StdDevUncertainty,
            1,
            True,
            u.electron,
            id="stddev-gain-applied",
        ),
        pytest.param(
            VarianceUncertainty,
            2,
            True,
            u.electron**2,
            id="variance-gain-applied",
        ),
        pytest.param(
            InverseVariance,
            -2,
            True,
            u.electron**-2,
            id="inverse-variance-gain-applied",
        ),
        pytest.param(
            StdDevUncertainty,
            1,
            False,
            u.adu,
            id="stddev-gain-disabled",
        ),
        pytest.param(
            VarianceUncertainty,
            2,
            False,
            u.adu**2,
            id="variance-gain-disabled",
        ),
        pytest.param(
            InverseVariance,
            -2,
            False,
            u.adu**-2,
            id="inverse-variance-gain-disabled",
        ),
    ],
)
def test_cosmicray_gain_correct_uncertainty(
    monkeypatch, uncertainty_type, gain_power, gain_apply, expected_uncertainty_unit
):
    ccd_data, result, original_data, original_uncertainty, original_uncertainty_unit = (
        _run_cosmicray_gain_correct_uncertainty(
            monkeypatch, uncertainty_type, gain_power, gain_apply
        )
    )

    gain = 2.0
    gain_factor = gain**gain_power if gain_apply else 1.0
    expected_unit = u.electron if gain_apply else u.adu

    assert type(result.uncertainty) is uncertainty_type
    assert result.unit == expected_unit
    assert result.uncertainty.unit == expected_uncertainty_unit
    assert_allclose(result.data, (gain if gain_apply else 1.0) * original_data)
    assert_allclose(result.uncertainty.array, gain_factor * original_uncertainty)
    assert_allclose(ccd_data.data, original_data)
    assert_allclose(ccd_data.uncertainty.array, original_uncertainty)
    assert ccd_data.unit == u.adu
    assert ccd_data.uncertainty.unit == original_uncertainty_unit


def _run_cosmicray_gain_correct_uncertainty(
    monkeypatch, uncertainty_type, gain_power, gain_apply
):
    def no_cosmics(data, **_kwargs):
        return xp.zeros_like(data, dtype=xp.bool), data

    monkeypatch.setattr("astroscrappy.detect_cosmics", no_cosmics)

    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    uncertainty = DATA_SCALE**gain_power * xp.ones_like(ccd_data.data)
    ccd_data.uncertainty = uncertainty_type(uncertainty)
    original_data = xp.asarray(ccd_data.data, copy=True)
    original_uncertainty = xp.asarray(ccd_data.uncertainty.array, copy=True)
    original_uncertainty_unit = ccd_data.uncertainty.unit
    gain = 2.0

    result = cosmicray_lacosmic(ccd_data, gain=gain, gain_apply=gain_apply)

    return (
        ccd_data,
        result,
        original_data,
        original_uncertainty,
        original_uncertainty_unit,
    )


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_lacosmic's gain and mask paths do not support "
    "array-api-strict's strict scalar and dtype rules",
)
@pytest.mark.parametrize(
    ("uncertainty_type", "gain_power", "gain_apply"),
    [
        (StdDevUncertainty, 1, True),
        pytest.param(
            VarianceUncertainty,
            2,
            True,
            marks=pytest.mark.backend_xfail(
                "jax",
                reason="Astropy uncertainty propagation converts JAX variance "
                "arrays to NumPy",
            ),
        ),
        pytest.param(
            InverseVariance,
            -2,
            True,
            marks=pytest.mark.backend_xfail(
                "jax",
                reason="Astropy uncertainty propagation converts JAX inverse "
                "variance arrays to NumPy",
            ),
        ),
        (StdDevUncertainty, 1, False),
        (VarianceUncertainty, 2, False),
        (InverseVariance, -2, False),
    ],
)
def test_cosmicray_gain_correct_uncertainty_namespace(
    monkeypatch, uncertainty_type, gain_power, gain_apply
):
    _, result, _, _, _ = _run_cosmicray_gain_correct_uncertainty(
        monkeypatch, uncertainty_type, gain_power, gain_apply
    )

    assert array_api_compat.array_namespace(result.uncertainty.array) is xp


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_lacosmic uses astroscrappy, which requires numpy "
    "and fails on a non-default device",
)
def test_cosmicray_lacosmic_accepts_quantity_gain():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)

    ccd_data.uncertainty = StdDevUncertainty(noise)
    # The units below are the point of the test
    gain = 2.0 * u.electron / u.adu

    _ = cosmicray_lacosmic(ccd_data, gain=gain, gain_apply=True)


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_lacosmic uses astroscrappy, which requires numpy "
    "and fails on a non-default device",
)
def test_cosmicray_lacosmic_accepts_quantity_readnoise():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)

    ccd_data.uncertainty = StdDevUncertainty(noise)
    gain = 2.0 * u.electron / u.adu
    # The units below are the point of this test
    readnoise = 6.5 * u.electron
    _ = cosmicray_lacosmic(ccd_data, gain=gain, gain_apply=True, readnoise=readnoise)


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_lacosmic uses astroscrappy, which requires numpy "
    "and fails on a non-default device",
)
def test_cosmicray_lacosmic_detects_inconsistent_units():
    # This is intended to detect cases like a ccd with units
    # of adu, a readnoise in electrons and a gain in adu / electron.
    # That is not internally inconsistent.
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    ccd_data.unit = "adu"
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)

    ccd_data.uncertainty = StdDevUncertainty(noise)
    readnoise = 6.5 * u.electron

    # The units below are deliberately incorrect.
    gain = 2.0 * u.adu / u.electron
    with pytest.raises(ValueError) as e:
        cosmicray_lacosmic(ccd_data, gain=gain, gain_apply=True, readnoise=readnoise)
    assert "Inconsistent units" in str(e.value)


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_lacosmic uses astroscrappy, which requires numpy "
    "and fails on a non-default device",
)
def test_cosmicray_lacosmic_warns_on_ccd_in_electrons():
    # Check that an input ccd in electrons raises a warning.
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    # The unit below is important for the test; this unit on
    # input is supposed to raise an error.
    ccd_data.unit = u.electron
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)

    ccd_data.uncertainty = StdDevUncertainty(noise)
    # No units here on purpose.
    gain = 2.0
    # Don't really need to set this (6.5 is the default value) but want to
    # make lack of units explicit.
    readnoise = 6.5
    with pytest.warns(UserWarning, match="Image unit is electron"):
        cosmicray_lacosmic(ccd_data, gain=gain, gain_apply=True, readnoise=readnoise)


# The values for inbkg and invar are DELIBERATELY BAD. They are supposed to be
# arrays, so if detect_cosmics is called with these bad values a ValueError
# will be raised, which we can check for.
@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_lacosmic uses astroscrappy, which requires numpy "
    "and fails on a non-default device",
)
@pytest.mark.parametrize(
    "new_args", [dict(inbkg=5), dict(invar=5), dict(inbkg=5, invar=5)]
)
def test_cosmicray_lacosmic_invar_inbkg(new_args):
    # This IS NOT TESTING FUNCTIONALITY it is simply testing
    # that calling with the new keyword arguments to astroscrappy
    # 1.1.0 raises no error.
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)

    ccd_data.uncertainty = StdDevUncertainty(noise)

    with pytest.raises(TypeError):
        cosmicray_lacosmic(ccd_data, sigclip=5.9, **new_args)


def test_cosmicray_median_check_data():
    with pytest.raises(TypeError):
        ndata, crarr = cosmicray_median(10, thresh=5, mbox=11, error_image=DATA_SCALE)


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_median uses scipy.ndimage.median_filter, which "
    "requires numpy and fails on a non-default device",
)
def test_cosmicray_median():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    ndata, crarr = cosmicray_median(
        ccd_data.data, thresh=5, mbox=11, error_image=DATA_SCALE
    )

    # check the number of cosmic rays detected
    assert crarr.sum() == NCRAYS


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_median uses scipy.ndimage.median_filter, which "
    "requires numpy and fails on a non-default device",
)
def test_cosmicray_median_ccddata():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)

    ccd_data.uncertainty = StdDevUncertainty(ccd_data.data * 0.0 + DATA_SCALE)
    nccd = cosmicray_median(ccd_data, thresh=5, mbox=11, error_image=None)

    # check the number of cosmic rays detected
    assert nccd.mask.sum() == NCRAYS


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_median uses scipy.ndimage.median_filter, which "
    "requires numpy and fails on a non-default device",
)
def test_cosmicray_median_masked():
    # Regression test for #932: an input mask used to be silently ignored.
    # Mask a subset of the injected cosmic-ray pixels; those must NOT be
    # flagged, while the unmasked cosmic rays must still be detected.
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    crrays = add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    n_masked = NCRAYS // 3
    np_data = np_array(ccd_data.data)
    np_mask = np_zeros(np_data.shape, dtype=bool)
    for y, x in crrays[:n_masked]:
        np_mask[y, x] = True
    n_masked_unique = int(np_mask.sum())

    data = np_ma_array(np_data, mask=np_mask)
    ndata, crarr = cosmicray_median(data, thresh=5, mbox=11, error_image=DATA_SCALE)

    crarr = np_array(crarr)
    # no masked pixel is flagged as a cosmic ray...
    assert not crarr[np_mask].any()
    # ...and all of the unmasked cosmic rays are still detected
    for y, x in crrays[n_masked:]:
        assert crarr[y, x]
    assert crarr.sum() == NCRAYS - n_masked_unique
    # masked pixels are returned unchanged
    assert_allclose(np_array(ndata)[np_mask], np_data[np_mask])


def test_cosmicray_median_masked_region_does_not_bias_neighbors():
    # Regression test for #932: a masked region of very bright pixels must not
    # raise the local median of the pixels next to it. The region is wider
    # than the median box, so an adjacent pixel's box is almost half masked;
    # if the masked values leaked into the median, a marginal cosmic ray next
    # to the region would be missed.
    rng = default_rng(seed=1)
    sigma = 1.0
    np_data = rng.normal(loc=0, scale=sigma, size=(100, 100))
    np_mask = np_zeros(np_data.shape, dtype=bool)
    np_mask[40:61, 40:54] = True
    np_data[np_mask] = 1e4 * sigma
    threshold = 5
    cr_y, cr_x = 50, 54  # immediately to the right of the masked region
    np_data[cr_y, cr_x] = 1.1 * threshold * sigma

    masked = np_ma_array(np_data, mask=np_mask)
    ndata, crarr = cosmicray_median(
        masked, thresh=threshold, mbox=11, error_image=sigma
    )
    crarr = np_array(crarr)
    assert crarr[cr_y, cr_x]
    assert not crarr[np_mask].any()
    # the masked pixels come back untouched
    assert_allclose(np_array(ndata)[np_mask], np_data[np_mask])


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_median uses scipy.ndimage.median_filter, which "
    "requires numpy and fails on a non-default device",
)
def test_cosmicray_median_ccddata_masked_region_does_not_bias_neighbors():
    # Same as the test above, but with the mask carried by a CCDData. The
    # CCDData branch must forward its mask to the detection rather than only
    # OR-ing it into the output mask.
    rng = default_rng(seed=1)
    sigma = 1.0
    np_data = rng.normal(loc=0, scale=sigma, size=(100, 100))
    np_mask = np_zeros(np_data.shape, dtype=bool)
    np_mask[40:61, 40:54] = True
    np_data[np_mask] = 1e4 * sigma
    threshold = 5
    cr_y, cr_x = 50, 54  # immediately to the right of the masked region
    np_data[cr_y, cr_x] = 1.1 * threshold * sigma

    ccd = CCDData(
        xp.asarray(np_data),
        unit="adu",
        mask=xp.asarray(np_mask),
        uncertainty=StdDevUncertainty(xp.asarray(np_data * 0.0 + sigma)),
    )
    nccd = cosmicray_median(ccd, thresh=threshold, mbox=11)
    out_mask = np_array(nccd.mask)
    assert out_mask[cr_y, cr_x]
    assert out_mask[np_mask].all()
    expected = np_mask.copy()
    expected[cr_y, cr_x] = True
    assert (out_mask == expected).all()
    # the masked pixels come back untouched
    assert_allclose(np_array(nccd.data)[np_mask], np_data[np_mask])


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_median uses scipy.ndimage.median_filter, which "
    "requires numpy and fails on a non-default device",
)
def test_cosmicray_median_all_masked():
    # Every pixel masked: nothing is flagged, data are returned unchanged and
    # no warning (e.g. from a 0/0 in the fill value) is emitted.
    np_data = default_rng(seed=3).normal(size=(20, 20))
    data = np_ma_array(np_data, mask=np_zeros(np_data.shape, dtype=bool) | True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ndata, crarr = cosmicray_median(data, thresh=5, mbox=11, error_image=1.0)
    assert not np_array(crarr).any()
    assert_allclose(np_array(ndata), np_data)


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_median uses scipy.ndimage.median_filter, which "
    "requires numpy and fails on a non-default device",
)
def test_cosmicray_median_masked_nomask():
    # A masked array with no mask set behaves like a plain array.
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    data = np_ma_array(np_array(ccd_data.data))
    assert data.mask is np_ma_nomask
    ndata, crarr = cosmicray_median(data, thresh=5, mbox=11, error_image=DATA_SCALE)
    assert crarr.sum() == NCRAYS


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_median uses scipy.ndimage.median_filter, which "
    "requires numpy and fails on a non-default device",
)
def test_cosmicray_median_ccddata_masked():
    # A CCDData input with a mask: the output mask is the union of the input
    # mask and the detected cosmic rays.
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    crrays = add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    np_mask = np_zeros(ccd_data.shape, dtype=bool)
    # mask a few of the cosmic rays and a block of ordinary pixels
    for y, x in crrays[:5]:
        np_mask[y, x] = True
    np_mask[2:6, 3:9] = True
    ccd_data.mask = xp.asarray(np_mask)
    ccd_data.uncertainty = StdDevUncertainty(ccd_data.data * 0.0 + DATA_SCALE)

    nccd = cosmicray_median(ccd_data, thresh=5, mbox=11, error_image=None)

    out_mask = np_array(nccd.mask)
    # every input-masked pixel is still masked
    assert out_mask[np_mask].all()
    # every cosmic ray is masked
    for y, x in crrays:
        assert out_mask[y, x]
    # and nothing else is
    expected = np_mask.copy()
    for y, x in crrays:
        expected[y, x] = True
    assert (out_mask == expected).all()


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_median uses scipy.ndimage.median_filter, which "
    "requires numpy and fails on a non-default device",
)
def test_cosmicray_median_background_None():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    data, crarr = cosmicray_median(ccd_data.data, thresh=5, mbox=11, error_image=None)

    # check the number of cosmic rays detected
    assert crarr.sum() == NCRAYS


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_median uses scipy.ndimage.median_filter, which "
    "requires numpy and fails on a non-default device",
)
def test_cosmicray_median_gbox():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    scale = DATA_SCALE  # yuck. Maybe use pytest.parametrize?
    threshold = 5
    add_cosmicrays(ccd_data, scale, threshold, ncrays=NCRAYS)
    error = ccd_data.data * 0.0 + DATA_SCALE
    data, crarr = cosmicray_median(
        ccd_data.data, error_image=error, thresh=5, mbox=11, rbox=0, gbox=5
    )
    data = np_ma_array(data, mask=crarr)
    assert crarr.sum() > NCRAYS
    assert abs(data.std() - scale) < 0.1


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_median uses scipy.ndimage.median_filter, which "
    "requires numpy and fails on a non-default device",
)
def test_cosmicray_median_rbox():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    scale = DATA_SCALE  # yuck. Maybe use pytest.parametrize?
    threshold = 5
    add_cosmicrays(ccd_data, scale, threshold, ncrays=NCRAYS)
    error = ccd_data.data * 0.0 + DATA_SCALE
    data, crarr = cosmicray_median(
        ccd_data.data, error_image=error, thresh=5, mbox=11, rbox=21, gbox=5
    )
    assert data[crarr].mean() < ccd_data.data[crarr].mean()
    assert crarr.sum() > NCRAYS


def test_cosmicray_median_background_deviation():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    with pytest.raises(TypeError):
        cosmicray_median(ccd_data.data, thresh=5, mbox=11, error_image="blank")


def test_background_deviation_box():
    scale = 5.3
    cd = xp.asarray(default_rng(seed=123).normal(loc=0, size=(100, 100), scale=scale))
    bd = background_deviation_box(cd, 25)
    assert abs(bd.mean() - scale) < 0.10


def test_background_deviation_box_fail():
    scale = 5.3
    cd = xp.asarray(default_rng(seed=123).normal(loc=0, size=(100, 100), scale=scale))
    with pytest.raises(ValueError):
        background_deviation_box(cd, 0.5)


def test_background_deviation_filter():
    scale = 5.3
    cd = xp.asarray(default_rng(seed=123).normal(loc=0, size=(100, 100), scale=scale))
    bd = background_deviation_filter(cd, 25)
    assert abs(bd.mean() - scale) < 0.10


def test_background_deviation_filter_fail():
    scale = 5.3
    cd = xp.asarray(default_rng(seed=123).normal(loc=0, size=(100, 100), scale=scale))
    with pytest.raises(ValueError):
        background_deviation_filter(cd, 0.5)


# This test can be removed in ccdproc 3.0 when support for old
# astroscrappy is removed.
@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_lacosmic uses astroscrappy, which requires numpy "
    "and fails on a non-default device",
)
def test_cosmicray_lacosmic_pssl_deprecation_warning():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    with pytest.warns(AstropyDeprecationWarning):
        cosmicray_lacosmic(ccd_data, pssl=1.0)


def test_cosmicray_lacosmic_pssl_and_inbkg_fails():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    with pytest.raises(ValueError) as err:
        # An error should be raised if both pssl and inbkg are provided
        with pytest.warns(AstropyDeprecationWarning):
            # The deprecation warning is expected and should be captured
            cosmicray_lacosmic(ccd_data, pssl=3, inbkg=ccd_data.data)

    assert "pssl and inbkg" in str(err)


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_lacosmic uses astroscrappy, which requires numpy "
    "and fails on a non-default device",
)
def test_cosmicray_lacosmic_pssl_does_not_fail():
    # This test is a copy/paste of test_cosmicray_lacosmic_ccddata
    # except with pssl=0.0001 as an argument. Subtracting nearly zero from
    # the background should have no effect. The test is really
    # to make sure that passing in pssl does not lead to an error
    # since the new interface does not include pssl.
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)
    ccd_data.uncertainty = StdDevUncertainty(noise)
    with pytest.warns(AstropyDeprecationWarning):
        # The deprecation warning is expected and should be captured
        nccd_data = cosmicray_lacosmic(ccd_data, sigclip=5.9, pssl=0.0001)

    # check the number of cosmic rays detected
    # Note that to get this to succeed reliably meant tuning
    # both sigclip and the threshold
    assert nccd_data.mask.sum() == NCRAYS
