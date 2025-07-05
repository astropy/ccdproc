# Licensed under a 3-clause BSD style license - see LICENSE.rst

import array_api_compat
import pytest
from astropy import units as u
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy import array as np_array
from numpy.ma import array as np_ma_array
from numpy.random import default_rng
from numpy.testing import assert_allclose

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
    size = data.shape[0]
    rng = default_rng(99)
    xp = array_api_compat.array_namespace(data.data)
    crrays = xp.asarray(rng.integers(0, size, size=(ncrays, 2)))
    # use (threshold + 15) below to make sure cosmic ray is well above the
    # threshold no matter what the random number generator returns
    # add_cosmicrays is highly sensitive to the seed
    # ideally threshold should be set so it is not sensitive to seed, but
    # this is not working right now
    crflux = xp.asarray(10 * scale * rng.random(ncrays) + (threshold + 15) * scale)

    # Some array libraris (Dask) do not support setting individual elements,
    # so use nuympy.
    data_as_np = np_array(data.data)
    for i in range(ncrays):
        y, x = crrays[i]
        data_as_np[y, x] = crflux[i]
    data.data = xp.asarray(data_as_np)


def test_cosmicray_lacosmic():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 10
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    _, crarr = cosmicray_lacosmic(ccd_data.data, sigclip=5.9)

    # check the number of cosmic rays detected
    # Note that to get this to succeed reliably meant tuning
    # both sigclip and the threshold
    assert crarr.sum() == NCRAYS


def test_cosmicray_lacosmic_ccddata():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    xp = array_api_compat.array_namespace(ccd_data.data)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)
    # Workaround for the fact that upstream checks for numpy array
    # specifically.
    ccd_data.uncertainty = np_array(noise)
    nccd_data = cosmicray_lacosmic(ccd_data, sigclip=5.9)

    # check the number of cosmic rays detected
    # Note that to get this to succeed reliably meant tuning
    # both sigclip and the threshold
    assert nccd_data.mask.sum() == NCRAYS


def test_cosmicray_lacosmic_check_data():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    xp = array_api_compat.array_namespace(ccd_data.data)
    with pytest.raises(TypeError):
        noise = DATA_SCALE * xp.ones_like(ccd_data.data)
        cosmicray_lacosmic(10, noise)


@pytest.mark.parametrize("array_input", [True, False])
@pytest.mark.parametrize("gain_correct_data", [True, False])
def test_cosmicray_gain_correct(array_input, gain_correct_data):
    # Add regression check for #705 and for the new gain_correct
    # argument.
    # The issue is that cosmicray_lacosmic gain-corrects the
    # data and returns that gain corrected data. That is not the
    # intent...
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    xp = array_api_compat.array_namespace(ccd_data.data)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)
    # Workaround for the fact that upstream checks for numpy array
    # specifically.
    ccd_data.uncertainty = np_array(noise)
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
    # Fill masked locations with 0 since there is no simple relationship
    # between the original value and the corrected value.
    # Masking using numpy is a handy way to check the results here.
    orig_data = xp.array(np_ma_array(ccd_data.data, mask=cr_mask).filled(0))
    new_data = xp.array(np_ma_array(new_data.data, mask=cr_mask).filled(0))
    if gain_correct_data:
        gain_for_test = gain
    else:
        gain_for_test = 1.0

    assert_allclose(gain_for_test * orig_data, new_data)


def test_cosmicray_lacosmic_accepts_quantity_gain():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    xp = array_api_compat.array_namespace(ccd_data.data)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)
    # Workaround for the fact that upstream checks for numpy array
    # specifically.
    ccd_data.uncertainty = np_array(noise)
    # The units below are the point of the test
    gain = 2.0 * u.electron / u.adu

    _ = cosmicray_lacosmic(ccd_data, gain=gain, gain_apply=True)


def test_cosmicray_lacosmic_accepts_quantity_readnoise():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    xp = array_api_compat.array_namespace(ccd_data.data)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)
    # Workaround for the fact that upstream checks for numpy array
    # specifically.
    ccd_data.uncertainty = np_array(noise)
    gain = 2.0 * u.electron / u.adu
    # The units below are the point of this test
    readnoise = 6.5 * u.electron
    _ = cosmicray_lacosmic(ccd_data, gain=gain, gain_apply=True, readnoise=readnoise)


def test_cosmicray_lacosmic_detects_inconsistent_units():
    # This is intended to detect cases like a ccd with units
    # of adu, a readnoise in electrons and a gain in adu / electron.
    # That is not internally inconsistent.
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    xp = array_api_compat.array_namespace(ccd_data.data)
    ccd_data.unit = "adu"
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)
    # Workaround for the fact that upstream checks for numpy array
    # specifically.
    ccd_data.uncertainty = np_array(noise)
    readnoise = 6.5 * u.electron

    # The units below are deliberately incorrect.
    gain = 2.0 * u.adu / u.electron
    with pytest.raises(ValueError) as e:
        cosmicray_lacosmic(ccd_data, gain=gain, gain_apply=True, readnoise=readnoise)
    assert "Inconsistent units" in str(e.value)


def test_cosmicray_lacosmic_warns_on_ccd_in_electrons():
    # Check that an input ccd in electrons raises a warning.
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    xp = array_api_compat.array_namespace(ccd_data.data)
    # The unit below is important for the test; this unit on
    # input is supposed to raise an error.
    ccd_data.unit = u.electron
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)
    # Workaround for the fact that upstream checks for numpy array
    # specifically.
    ccd_data.uncertainty = np_array(noise)
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
@pytest.mark.parametrize(
    "new_args", [dict(inbkg=5), dict(invar=5), dict(inbkg=5, invar=5)]
)
def test_cosmicray_lacosmic_invar_inbkg(new_args):
    # This IS NOT TESTING FUNCTIONALITY it is simply testing
    # that calling with the new keyword arguments to astroscrappy
    # 1.1.0 raises no error.
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    xp = array_api_compat.array_namespace(ccd_data.data)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)
    # Workaround for the fact that upstream checks for numpy array
    # specifically.
    ccd_data.uncertainty = np_array(noise)

    with pytest.raises(TypeError):
        cosmicray_lacosmic(ccd_data, sigclip=5.9, **new_args)


def test_cosmicray_median_check_data():
    with pytest.raises(TypeError):
        ndata, crarr = cosmicray_median(10, thresh=5, mbox=11, error_image=DATA_SCALE)


def test_cosmicray_median():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    ndata, crarr = cosmicray_median(
        ccd_data.data, thresh=5, mbox=11, error_image=DATA_SCALE
    )

    # check the number of cosmic rays detected
    assert crarr.sum() == NCRAYS


def test_cosmicray_median_ccddata():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    # Workaround for the fact that upstream checks for numpy array
    # specifically.
    ccd_data.uncertainty = np_array(ccd_data.data * 0.0 + DATA_SCALE)
    nccd = cosmicray_median(ccd_data, thresh=5, mbox=11, error_image=None)

    # check the number of cosmic rays detected
    assert nccd.mask.sum() == NCRAYS


def test_cosmicray_median_masked():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    data = np_ma_array(ccd_data.data, mask=(ccd_data.data > -1e6))
    ndata, crarr = cosmicray_median(data, thresh=5, mbox=11, error_image=DATA_SCALE)

    # check the number of cosmic rays detected
    assert crarr.sum() == NCRAYS


def test_cosmicray_median_background_None():
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    data, crarr = cosmicray_median(ccd_data.data, thresh=5, mbox=11, error_image=None)

    # check the number of cosmic rays detected
    assert crarr.sum() == NCRAYS


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
    cd = default_rng(seed=123).normal(loc=0, size=(100, 100), scale=scale)
    bd = background_deviation_box(cd, 25)
    assert abs(bd.mean() - scale) < 0.10


def test_background_deviation_box_fail():
    scale = 5.3
    cd = default_rng(seed=123).normal(loc=0, size=(100, 100), scale=scale)
    with pytest.raises(ValueError):
        background_deviation_box(cd, 0.5)


def test_background_deviation_filter():
    scale = 5.3
    cd = default_rng(seed=123).normal(loc=0, size=(100, 100), scale=scale)
    bd = background_deviation_filter(cd, 25)
    assert abs(bd.mean() - scale) < 0.10


def test_background_deviation_filter_fail():
    scale = 5.3
    cd = default_rng(seed=123).normal(loc=0, size=(100, 100), scale=scale)
    with pytest.raises(ValueError):
        background_deviation_filter(cd, 0.5)


# This test can be removed in ccdproc 3.0 when support for old
# astroscrappy is removed.
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


def test_cosmicray_lacosmic_pssl_does_not_fail():
    # This test is a copy/paste of test_cosmicray_lacosmic_ccddata
    # except with pssl=0.0001 as an argument. Subtracting nearly zero from
    # the background should have no effect. The test is really
    # to make sure that passing in pssl does not lead to an error
    # since the new interface does not include pssl.
    ccd_data = ccd_data_func(data_scale=DATA_SCALE)
    xp = array_api_compat.array_namespace(ccd_data.data)
    threshold = 5
    add_cosmicrays(ccd_data, DATA_SCALE, threshold, ncrays=NCRAYS)
    noise = DATA_SCALE * xp.ones_like(ccd_data.data)
    ccd_data.uncertainty = np_array(noise)
    with pytest.warns(AstropyDeprecationWarning):
        # The deprecation warning is expected and should be captured
        nccd_data = cosmicray_lacosmic(ccd_data, sigclip=5.9, pssl=0.0001)

    # check the number of cosmic rays detected
    # Note that to get this to succeed reliably meant tuning
    # both sigclip and the threshold
    assert nccd_data.mask.sum() == NCRAYS
