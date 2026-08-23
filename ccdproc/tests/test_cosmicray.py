# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

import array_api_compat
import array_api_extra as xpx
import numpy as np
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
from ccdproc.conftest import testing_array_device as xp_device
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
@pytest.mark.parametrize("masked", ["none", "some", "all"])
def test_cosmicray_median_masked(masked):
    # Regression test for #932: an input mask used to be silently ignored.
    # "none": a masked array with nomask behaves like a plain array.
    # "some": mask a subset of the injected cosmic rays; those must NOT be
    #         flagged while the rest are.
    # "all":  nothing is flagged, and no warning (e.g. 0/0) is emitted.
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    crrays = add_cosmicrays(ccd_data, DATA_SCALE, 5, ncrays=NCRAYS)
    np_data = np_array(ccd_data.data)
    np_mask = np_zeros(np_data.shape, dtype=bool)
    if masked == "some":
        for y, x in crrays[: NCRAYS // 3]:
            np_mask[y, x] = True
    elif masked == "all":
        np_mask[:] = True

    data = np_ma_array(np_data, mask=np_mask if masked != "none" else np_ma_nomask)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ndata, crarr = cosmicray_median(data, thresh=5, mbox=11, error_image=DATA_SCALE)

    crarr = np_array(crarr)
    # no masked pixel is flagged, every unmasked cosmic ray is, nothing else is
    assert not crarr[np_mask].any()
    for y, x in crrays:
        assert crarr[y, x] == (not np_mask[y, x])
    assert crarr.sum() == sum(not np_mask[y, x] for y, x in crrays)
    # masked pixels are returned unchanged
    assert_allclose(np_array(ndata)[np_mask], np_data[np_mask])


def _masked_column_image(seed=1, sigma=1.0):
    """
    A flat noise image with a masked, saturated column and two cosmic rays:
    one well away from the column and one immediately next to it.
    """
    rng = default_rng(seed=seed)
    np_data = rng.normal(loc=100.0, scale=sigma, size=(60, 60))
    np_mask = np_zeros(np_data.shape, dtype=bool)
    np_mask[:, 30] = True
    np_data[np_mask] = 65535.0
    crays = [(10, 10), (40, 31)]
    for y, x in crays:
        np_data[y, x] = 100.0 + 20 * sigma
    return np_data, np_mask, crays


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="cosmicray_median uses scipy.ndimage.median_filter, which "
    "requires numpy and fails on a non-default device",
)
# (The CCDData branch takes its error image from the uncertainty.)
@pytest.mark.parametrize(
    "kind,error_image",
    [("masked_array", 1.0), ("masked_array", None), ("ccddata", None)],
)
def test_cosmicray_median_masked_column(kind, error_image):
    # Masked pixels are never flagged, not even by the gbox growth step,
    # whether growth would start from a masked pixel (the saturated column
    # has an enormous residual) or spread into one from the cosmic ray next
    # to the column. With rbox > 0 the cosmic rays are replaced while masked
    # pixels are returned unchanged. With error_image=None on a masked array
    # the noise is estimated from the unmasked pixels only; if the saturated
    # column were included nothing would be detected. For a CCDData the
    # mask is forwarded to the detection and the output mask is the union of
    # the input mask and the (grown) cosmic rays.
    np_data, np_mask, crays = _masked_column_image()
    if kind == "masked_array":
        ccd = np_ma_array(np_data, mask=np_mask)
    else:
        ccd = CCDData(
            xp.asarray(np_data, device=xp_device),
            unit="adu",
            mask=np_mask,
            uncertainty=StdDevUncertainty(xp.ones_like(xp.asarray(np_data))),
        )
    result = cosmicray_median(
        ccd, thresh=5, mbox=11, gbox=3, rbox=5, error_image=error_image
    )
    if kind == "masked_array":
        ndata, crarr = result
    else:
        out_mask = np_array(result.mask)
        assert out_mask[np_mask].all()
        ndata, crarr = result.data, out_mask & ~np_mask
    crarr = np_array(crarr)
    ndata = np_array(ndata)

    assert not crarr[np_mask].any()
    for y, x in crays:
        assert crarr[y, x]
    # growth happened around the cosmic rays (3x3 minus the masked column)
    assert crarr.sum() == 9 + 6
    assert_allclose(ndata[np_mask], np_data[np_mask])
    for y, x in crays:
        assert abs(ndata[y, x] - 100.0) < 5


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
    assert abs(float(xp.mean(bd)) - scale) < 0.10


def test_background_deviation_box_per_box_values():
    # Regression test for #963: the per-box deviation must actually be
    # written into the result. A pure-mean check passes even if every box is
    # silently left at the global standard deviation.
    rng = default_rng(seed=123)
    left = rng.normal(loc=0, size=(100, 50), scale=1.0)
    right = rng.normal(loc=0, size=(100, 50), scale=10.0)
    cd = xp.asarray(np.hstack([left, right]))
    bd = background_deviation_box(cd, 50)
    global_std = float(xp.std(cd))
    left_val = float(bd[25, 25])
    right_val = float(bd[25, 75])
    # The sample standard deviation of a 50x50 box scatters by roughly
    # sigma / sqrt(2 * 2500) ~ 0.014 * sigma, so with a fixed seed these
    # bounds are comfortably above the noise while still far tighter than the
    # gap to the global standard deviation (~7).
    assert abs(left_val - 1.0) < 0.05
    assert abs(right_val - 10.0) < 0.5
    assert left_val != right_val
    assert abs(left_val - global_std) > 1.0
    assert abs(right_val - global_std) > 1.0
    # every pixel within a box shares that box's value. setbox clamps the
    # upper edge to len - 1, so the last row and column of the image are never
    # filled (they keep the global standard deviation) and are deliberately
    # excluded from the comparison.
    assert bool(xp.all(bd[:50, :50] == bd[0, 0]))
    assert bool(xp.all(bd[:50, 50:99] == bd[0, 50]))


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


def test_cosmicray_median_mask_shape_mismatch():
    # CCDData and MaskedArray validate the mask shape themselves, so exercise
    # the shared helper directly.
    from ccdproc.core import _cosmicray_median_array

    np_data = default_rng(seed=4).normal(size=(20, 20))
    # Use the compat namespace, as the public function does: plain numpy < 2
    # has no ``bool`` attribute.
    np_xp = array_api_compat.array_namespace(np_data)

    with pytest.raises(ValueError, match="mask is not the same shape"):
        _cosmicray_median_array(
            np_data, np_zeros((10, 20), dtype=bool), 1.0, 5, 11, 0, 0, np_xp
        )
