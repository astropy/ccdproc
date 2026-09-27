# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Tests that a mask follows the array namespace and device of the data (#1024).

Astropy's ``CCDData.mask`` setter always makes a NumPy mask, so a mask read
from a file or built with NumPy stays NumPy even when the data is not. The
functions below move such a mask to the data's namespace and device through
``_set_mask`` or the wrapper's mask setter. A mask that cannot be moved there
raises instead.
"""

import array_api_compat
import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import CCDData, StdDevUncertainty

from ccdproc import (
    Combiner,
    ccd_process,
    combine,
    cosmicray_lacosmic,
    cosmicray_median,
    create_deviation,
    subtract_bias,
    subtract_overscan,
    trim_image,
)
from ccdproc.conftest import testing_array_device as xp_device
from ccdproc.conftest import testing_array_library as xp
from ccdproc.core import _to_numpy
from ccdproc.tests.pytest_fixtures import assert_same_namespace_and_device, to_xp

IS_NUMPY = array_api_compat.is_numpy_namespace(xp)

# True only on array-api-strict's non-default device, on which NumPy cannot
# read an array; that is the stand-in for a CuPy array on a GPU.
ON_NUMPY_UNREADABLE_DEVICE = array_api_compat.is_array_api_strict_namespace(
    xp
) and xp_device != xp.Device("CPU_DEVICE")

SIZE = 20

# A pixel the caller has flagged as bad, and a pixel hit by a cosmic ray.
BAD_PIXEL = (2, 3)
COSMIC_RAY = (10, 10)


def _image():
    """
    Sky-like noise with one bright cosmic ray, as a NumPy array.
    """
    data = np.random.default_rng(seed=123).normal(loc=100, scale=1, size=(SIZE, SIZE))
    data[COSMIC_RAY] = 500.0
    return data


def _bad_pixel_mask():
    """
    A NumPy bad-pixel mask, as it would be read from a FITS file.
    """
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[BAD_PIXEL] = True
    return mask


def _ccd_with_numpy_mask():
    """
    Data in the namespace under test with a NumPy mask, set through the
    public ``CCDData.mask`` setter as a user would.
    """
    ccd = CCDData(to_xp(_image()), unit=u.adu)
    ccd.uncertainty = StdDevUncertainty(to_xp(np.ones((SIZE, SIZE))))
    ccd.mask = _bad_pixel_mask()
    return ccd


def _call_subtract_bias(ccd):
    bias = CCDData(to_xp(np.zeros((SIZE, SIZE))), unit=u.adu)
    return subtract_bias(ccd, bias)


def _call_ccd_process(ccd):
    # Here the NumPy mask comes in as the bad-pixel mask rather than on ccd.
    ccd = CCDData(ccd.data, unit=ccd.unit)
    return ccd_process(ccd, bad_pixel_mask=_bad_pixel_mask())


def _call_cosmicray_median(ccd):
    return cosmicray_median(ccd, mbox=5)


def _call_cosmicray_lacosmic(ccd):
    return cosmicray_lacosmic(ccd)


def _call_combine(ccd):
    return combine([ccd, ccd.copy()])


def _call_combiner(ccd):
    return Combiner([ccd, ccd.copy()]).average_combine()


@pytest.mark.skipif(IS_NUMPY, reason="a NumPy mask is already in the data's namespace")
@pytest.mark.parametrize(
    ("call", "flags_cosmic_ray"),
    [
        (_call_subtract_bias, False),
        (_call_ccd_process, False),
        (_call_cosmicray_median, True),
        (_call_cosmicray_lacosmic, True),
        (_call_combine, False),
        (_call_combiner, False),
    ],
)
def test_numpy_mask_follows_the_data(call, flags_cosmic_ray):
    """
    A NumPy mask on non-NumPy data comes back in the data's namespace and on
    its device, with the caller's bad pixel still masked.

    Notes
    -----
    Each call reaches a different place that sets the result's mask: the
    arithmetic wrapper, ``ccd_process``'s bad-pixel mask, the cosmic-ray
    mask merges, and the template and outputs in ``combine`` and
    ``Combiner``. Only the ``subtract_bias`` and ``ccd_process`` cases
    failed before masks were set through ``_set_mask``: their masks landed
    on the default device rather than the data's. The other places already
    passed the data's device explicitly, so those cases guard against the
    sweep that replaced them losing it.
    """
    result = call(_ccd_with_numpy_mask())

    assert_same_namespace_and_device(result.mask, result.data)
    expected = _bad_pixel_mask()
    if flags_cosmic_ray:
        expected[COSMIC_RAY] = True
    np.testing.assert_array_equal(_to_numpy(result.mask), expected)


def _processed_image():
    """
    An image as ``ccd_process`` returns it, with a bad-pixel mask and an
    uncertainty; the mask is on the data's device.
    """
    ccd = CCDData(to_xp(_image()), unit=u.adu)
    return ccd_process(
        ccd,
        bad_pixel_mask=_bad_pixel_mask(),
        error=True,
        gain=1.0 * u.electron / u.adu,
        readnoise=5.0 * u.electron,
    )


def _unmasked_image():
    """
    An image in the units of ``_processed_image``, without a mask.
    """
    return CCDData(to_xp(_image()), unit=u.electron)


@pytest.mark.parametrize(
    ("call", "bad_pixel_masked"),
    [
        (lambda ccd: combine([ccd, _unmasked_image()]), False),
        (lambda ccd: Combiner([ccd, _unmasked_image()]).average_combine(), False),
        (lambda ccd: ccd_process(ccd), True),
        (
            lambda ccd: subtract_overscan(
                ccd, fits_section="[1:3, :]", overscan_axis=1
            ),
            True,
        ),
        (lambda ccd: trim_image(ccd, fits_section="[1:10, :]"), True),
        (lambda ccd: create_deviation(ccd, readnoise=5.0 * u.electron), True),
        (lambda ccd: cosmicray_median(ccd, mbox=5), True),
    ],
    ids=[
        "combine",
        "Combiner",
        "ccd_process",
        "subtract_overscan",
        "trim_image",
        "create_deviation",
        "cosmicray_median",
    ],
)
def test_processed_image_can_be_processed_again(call, bad_pixel_masked):
    """
    An image returned by ``ccd_process`` can be passed to ccdproc again,
    and the result's mask is on the data's device.

    Notes
    -----
    Now that the mask of a result is on the data's device, it is on a device
    NumPy cannot read whenever the data is, as on array-api-strict's
    non-default device, the stand-in for a GPU. ``combine``,
    ``ccd_process``, ``subtract_overscan`` and ``create_deviation`` copied
    or sliced their input as a plain ``CCDData``, whose mask setter sends
    the mask through NumPy, and so raised ``RuntimeError`` there for
    ccdproc's own output. ``Combiner``, ``trim_image`` and
    ``cosmicray_median`` did not, and are here as guards against the same
    mistake.

    The bad pixel is masked in the result except when combining, where the
    other image has no mask; a pixel is masked in a combined image only if
    it is masked in every input.
    """
    result = call(_processed_image())

    assert_same_namespace_and_device(result.mask, result.data)
    assert bool(_to_numpy(result.mask)[BAD_PIXEL]) is bad_pixel_masked


def _numpy_ccd_with_mask_on_device(image):
    """
    NumPy data with a mask on array-api-strict's non-default device.

    Neither ``CCDData(data, mask=...)`` nor ``ccd.mask = ...`` can build
    this, since astropy's setter cannot read the mask, so set the private
    attribute; this is the input described in #1024.
    """
    ccd = CCDData(image, unit=u.adu)
    ccd.uncertainty = StdDevUncertainty(np.ones((SIZE, SIZE)))
    ccd._mask = to_xp(_bad_pixel_mask())
    return ccd


@pytest.mark.skipif(
    not ON_NUMPY_UNREADABLE_DEVICE,
    reason="needs a mask on a device NumPy cannot read",
)
@pytest.mark.parametrize(
    "call",
    [
        _call_subtract_bias,
        _call_cosmicray_median,
        _call_combine,
    ],
)
def test_mask_on_unreadable_device_with_numpy_data_raises(call):
    """
    NumPy data with a mask that NumPy cannot read raises rather than
    returning a mask the data's namespace cannot use.

    Notes
    -----
    The mask follows the data, and here it cannot. This pins behaviour that
    was already agreed, and already true before masks were set through
    ``_set_mask``; it is not what that change fixed. Deleting this test would
    let a change to ``_set_mask`` that skips the conversion, or that falls
    back to leaving the mask where it is, go unnoticed; the result would
    then fail later, far from the cause.
    """
    ccd = _numpy_ccd_with_mask_on_device(_image())
    with pytest.raises(RuntimeError, match="Can't convert array"):
        call(ccd)


@pytest.mark.skipif(
    not ON_NUMPY_UNREADABLE_DEVICE,
    reason="needs a mask on a device NumPy cannot read",
)
def test_ccd_process_bad_pixel_mask_on_unreadable_device_raises():
    """
    ``ccd_process`` with NumPy data and a bad-pixel mask NumPy cannot read
    raises.

    Notes
    -----
    This is the one way to give NumPy data such a mask through the public
    API. This pins behaviour that was already agreed, and already true
    before masks were set through ``_set_mask``. Without the conversion the
    mask would be stored as it is, in a namespace the data's cannot combine
    with.
    """
    ccd = CCDData(_image(), unit=u.adu)
    with pytest.raises(RuntimeError, match="Can't convert array"):
        ccd_process(ccd, bad_pixel_mask=to_xp(_bad_pixel_mask()))


@pytest.mark.skipif(
    IS_NUMPY or ON_NUMPY_UNREADABLE_DEVICE,
    reason="needs a non-NumPy mask that NumPy can read",
)
def test_ccd_process_converts_bad_pixel_mask_to_data_namespace():
    """
    A bad-pixel mask from another namespace that can be converted is
    converted, silently, to the namespace of the data.

    Notes
    -----
    With NumPy data and, say, a jax mask, the result's mask is NumPy, as
    ``CCDData(numpy_data, mask=jax_mask)`` gives. This pins behaviour that
    was already agreed, and already true before masks were set through
    ``_set_mask``. Without the conversion the result would carry a jax mask
    next to NumPy data.
    """
    ccd = CCDData(_image(), unit=u.adu)

    result = ccd_process(ccd, bad_pixel_mask=to_xp(_bad_pixel_mask()))

    assert isinstance(result.mask, np.ndarray)
    np.testing.assert_array_equal(result.mask, _bad_pixel_mask())
