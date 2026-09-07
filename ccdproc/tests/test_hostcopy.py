# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Tests for the explicit host round-trip of the CPU-only operations (#935).

``wcs_project``, ``subtract_overscan(model=...)`` and ``cosmicray_lacosmic``
depend on numpy-only libraries (reproject, astropy.modeling, astroscrappy).
They copy their input to the host, run, copy every result back to the
caller's array namespace and device, and warn that they did so. The rest of
the test suite silences ``HostCopyWarning`` through the ``filterwarnings``
list in ``pyproject.toml``, so the warning contract is pinned here, once,
instead of at every call site.
"""

import warnings

import array_api_compat
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import models
from astropy.nddata import CCDData
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS

import ccdproc
from ccdproc import (
    HostCopyWarning,
    combine,
    cosmicray_lacosmic,
    gain_correct,
    subtract_overscan,
    wcs_project,
)

# Set up the array library to be used in tests
from ccdproc.conftest import testing_array_device as xp_device
from ccdproc.conftest import testing_array_library as xp
from ccdproc.core import _from_numpy
from ccdproc.tests.pytest_fixtures import ccd_data as ccd_data_func

IS_NUMPY = array_api_compat.is_numpy_namespace(xp)

DATA_SIZE = 20


def _wcs_for_testing(shape):
    """A celestial WCS centred on the middle of an image of ``shape``."""
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[0] // 2, shape[1] // 2]
    # These are plain WCS metadata, not image data, so they stay NumPy.
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.crval = [0, -90]
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    w.wcs.set_pv([(2, 1, 45.0)])
    return w


def _run_wcs_project():
    """Call ``wcs_project`` on a masked image; return the input and result."""
    ccd = ccd_data_func(data_size=DATA_SIZE)
    ccd.wcs = _wcs_for_testing(ccd.shape)
    mask = np.zeros(ccd.shape, dtype=bool)
    mask[2, 3] = True
    # TODO: change back to .mask when CCDData is array-api compliant
    ccd._mask = xp.asarray(mask, device=xp_device)
    target_wcs = _wcs_for_testing(ccd.shape)
    target_wcs.wcs.crpix += [1, 1]
    return ccd, wcs_project(ccd, target_wcs)


def _run_subtract_overscan():
    """Call the model path of ``subtract_overscan``; return input and result."""
    ccd = ccd_data_func(data_size=DATA_SIZE)
    return ccd, subtract_overscan(
        ccd, overscan=ccd[:, :5], overscan_axis=1, model=models.Polynomial1D(1)
    )


def _run_cosmicray_lacosmic():
    """Call ``cosmicray_lacosmic`` on a CCDData; return the input and result."""
    ccd = ccd_data_func(data_size=DATA_SIZE)
    return ccd, cosmicray_lacosmic(ccd)


CPU_ONLY_CALLS = [
    pytest.param(_run_wcs_project, "wcs_project", id="wcs_project"),
    pytest.param(_run_subtract_overscan, "subtract_overscan", id="subtract_overscan"),
    pytest.param(
        _run_cosmicray_lacosmic, "cosmicray_lacosmic", id="cosmicray_lacosmic"
    ),
]


def test_host_copy_warning_is_public_and_an_astropy_warning():
    """
    ``HostCopyWarning`` is reachable as ``ccdproc.HostCopyWarning`` and is an
    ``AstropyUserWarning``, so that the filter documented in
    ``docs/array_api.rst`` and the one in ``pyproject.toml`` can name it.
    """
    assert ccdproc.HostCopyWarning is HostCopyWarning
    assert issubclass(HostCopyWarning, AstropyUserWarning)


@pytest.mark.parametrize("dtype", ["float64", "bool"])
def test_from_numpy_restores_namespace_device_and_dtype(dtype):
    """
    ``_from_numpy`` is the inverse of ``_to_numpy``: it puts a NumPy result
    back in the namespace *and* on the device of the array it came from,
    keeping its dtype. The device half only bites on a backend with more
    than one device (array-api-strict on its non-default device), which is
    exactly the case that motivated the helper.
    """
    like = xp.asarray(np.zeros((3, 3)), device=xp_device)
    np_array = np.ones((3, 3), dtype=dtype)

    result = _from_numpy(np_array, like=like)

    assert array_api_compat.array_namespace(result) is array_api_compat.array_namespace(
        like
    )
    assert array_api_compat.device(result) == array_api_compat.device(like)
    assert result.dtype == getattr(xp, dtype)


def test_from_numpy_passes_none_through():
    """
    ``_from_numpy`` returns `None` unchanged so that call sites do not need
    to special-case an absent mask.
    """
    like = xp.asarray(np.zeros((3, 3)), device=xp_device)
    assert _from_numpy(None, like=like) is None


@pytest.mark.skipif(IS_NUMPY, reason="NumPy input is never copied to the host")
@pytest.mark.parametrize(("call", "function_name"), CPU_ONLY_CALLS)
def test_cpu_only_function_warns_once(call, function_name):
    """
    Each CPU-only operation emits exactly one ``HostCopyWarning``, naming
    itself, on a non-NumPy backend -- one per call, not one per array that
    crosses to the host.
    """
    with pytest.warns(HostCopyWarning) as record:
        call()

    host_copies = [w for w in record if issubclass(w.category, HostCopyWarning)]
    assert len(host_copies) == 1
    assert function_name in str(host_copies[0].message)


@pytest.mark.skipif(not IS_NUMPY, reason="only the NumPy path must stay silent")
@pytest.mark.parametrize(("call", "_function_name"), CPU_ONLY_CALLS)
def test_numpy_input_does_not_warn(call, _function_name):
    """
    NumPy input is not copied anywhere, so nothing warns. Recording with
    ``simplefilter("always")`` rather than ``pytest.warns`` is what makes
    this a real check: it would catch a regression in which the NumPy path
    started going through the conversion helpers.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call()

    assert not [w for w in record if issubclass(w.category, HostCopyWarning)]


@pytest.mark.parametrize(("call", "_function_name"), CPU_ONLY_CALLS)
def test_cpu_only_function_returns_caller_namespace(call, _function_name):
    """
    The caller never gets a NumPy array back in place of what it passed in:
    the data, and the mask when there is one, come back in the namespace and
    on the device of the input. This is the #930/#933 regression check.
    """
    ccd, result = call()

    input_namespace = array_api_compat.array_namespace(ccd.data)
    input_device = array_api_compat.device(ccd.data)

    assert array_api_compat.array_namespace(result.data) is input_namespace
    assert array_api_compat.device(result.data) == input_device

    if result.mask is not None:
        assert array_api_compat.array_namespace(result.mask) is input_namespace
        assert array_api_compat.device(result.mask) == input_device


def test_cosmicray_lacosmic_bare_array_returns_caller_namespace():
    """
    The bare-array branch of ``cosmicray_lacosmic`` returns both the cleaned
    data and the cosmic-ray mask in the caller's namespace and on its
    device, not the NumPy arrays astroscrappy handed back.
    """
    ccd = ccd_data_func(data_size=DATA_SIZE)
    cleaned, crmask = cosmicray_lacosmic(ccd.data)

    input_namespace = array_api_compat.array_namespace(ccd.data)
    input_device = array_api_compat.device(ccd.data)

    for result in (cleaned, crmask):
        assert array_api_compat.array_namespace(result) is input_namespace
        assert array_api_compat.device(result) == input_device


@pytest.mark.skipif(IS_NUMPY, reason="NumPy input is never copied to the host")
def test_combine_output_file_does_not_warn(tmp_path):
    """
    Writing a FITS file through ``combine(output_file=...)`` copies to the
    host too, but nothing comes back into the pipeline, so by policy it does
    not warn.
    """
    ccd = CCDData(xp.asarray(np.ones((3, 3)), device=xp_device), unit=u.adu)
    ccd_times_2 = CCDData(ccd.data * 2, unit=ccd.unit)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        combine([ccd, ccd_times_2], output_file=tmp_path / "combined.fits")

    assert not [w for w in record if issubclass(w.category, HostCopyWarning)]


def test_scalar_quantity_gain_keeps_namespace_and_device():
    """
    Closes #936: a scalar ``Quantity`` gain does not drag the data out of its
    namespace or off its device. ``gain_correct`` and ``cosmicray_lacosmic``
    are the two public functions that build arrays from ``gain.value``, so
    they are the ones pinned here.
    """
    gain = 2.0 * u.electron / u.adu

    input_namespace = array_api_compat.array_namespace(
        ccd_data_func(data_size=DATA_SIZE).data
    )

    ccd = ccd_data_func(data_size=DATA_SIZE)
    gain_corrected = gain_correct(ccd, gain=gain)
    assert array_api_compat.array_namespace(gain_corrected.data) is input_namespace
    assert array_api_compat.device(gain_corrected.data) == array_api_compat.device(
        ccd.data
    )

    ccd = ccd_data_func(data_size=DATA_SIZE)
    cleaned = cosmicray_lacosmic(ccd, gain=gain, gain_apply=True)
    assert array_api_compat.array_namespace(cleaned.data) is input_namespace
    assert array_api_compat.device(cleaned.data) == array_api_compat.device(ccd.data)
