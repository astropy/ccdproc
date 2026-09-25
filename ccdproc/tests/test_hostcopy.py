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

import inspect
import warnings

import array_api_compat
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import models
from astropy.nddata import CCDData
from astropy.utils.exceptions import AstropyUserWarning

import ccdproc
from ccdproc import (
    HostCopyWarning,
    ccd_process,
    combine,
    cosmicray_lacosmic,
    gain_correct,
    subtract_overscan,
    wcs_project,
)

# Set up the array library to be used in tests
from ccdproc.conftest import testing_array_device as xp_device
from ccdproc.conftest import testing_array_library as xp
from ccdproc.core import _from_numpy, _to_numpy
from ccdproc.tests.pytest_fixtures import ccd_data as ccd_data_func
from ccdproc.tests.pytest_fixtures import numpy_ccddata, wcs_for_testing

IS_NUMPY = array_api_compat.is_numpy_namespace(xp)

DATA_SIZE = 20


def _run_wcs_project():
    """
    Call ``wcs_project`` on a masked image.

    Returns the input, the result, and the line number of the call to
    ``wcs_project`` below, so that callers can check exactly where a
    ``HostCopyWarning`` was attributed.
    """
    ccd = ccd_data_func(data_size=DATA_SIZE)
    ccd.wcs = wcs_for_testing(ccd.shape)
    mask = np.zeros(ccd.shape, dtype=bool)
    mask[2, 3] = True
    # TODO: change back to .mask when CCDData is array-api compliant
    ccd._mask = xp.asarray(mask, device=xp_device)
    target_wcs = wcs_for_testing(ccd.shape)
    target_wcs.wcs.crpix += [1, 1]
    call_line = inspect.currentframe().f_lineno + 1
    result = wcs_project(ccd, target_wcs)
    return ccd, result, call_line


def _run_subtract_overscan():
    """
    Call the model path of ``subtract_overscan``.

    Returns the input, the result, and the line number of the call to
    ``subtract_overscan`` below.
    """
    ccd = ccd_data_func(data_size=DATA_SIZE)
    call_line = inspect.currentframe().f_lineno + 1
    result = subtract_overscan(
        ccd, overscan=ccd[:, :5], overscan_axis=1, model=models.Polynomial1D(1)
    )
    return ccd, result, call_line


def _run_cosmicray_lacosmic():
    """
    Call ``cosmicray_lacosmic`` on a CCDData.

    Returns the input, the result, and the line number of the call to
    ``cosmicray_lacosmic`` below.
    """
    ccd = ccd_data_func(data_size=DATA_SIZE)
    call_line = inspect.currentframe().f_lineno + 1
    result = cosmicray_lacosmic(ccd)
    return ccd, result, call_line


def _run_ccd_process():
    """
    Call ``ccd_process``'s overscan-model path.

    Returns the input, the result, and the line number of the call to
    ``ccd_process`` below.

    Notes
    -----
    ``ccd_process`` calls ``subtract_overscan`` itself, through its own
    ``@log_to_metadata`` wrapper, so this is the case that exposed the bug
    in the old, fixed ``stacklevel=4``: the warning used to be attributed
    to ``ccd_process``'s own call to ``subtract_overscan`` inside
    ``ccdproc/core.py`` rather than to the line below.
    """
    ccd = ccd_data_func(data_size=DATA_SIZE)
    call_line = inspect.currentframe().f_lineno + 1
    result = ccd_process(ccd, oscan=ccd[:, :5], oscan_model=models.Polynomial1D(1))
    return ccd, result, call_line


CPU_ONLY_CALLS = [
    pytest.param(_run_wcs_project, "wcs_project", id="wcs_project"),
    pytest.param(_run_subtract_overscan, "subtract_overscan", id="subtract_overscan"),
    pytest.param(
        _run_cosmicray_lacosmic, "cosmicray_lacosmic", id="cosmicray_lacosmic"
    ),
    # subtract_overscan's own function_name: ccd_process calls it, so that
    # is the name the HostCopyWarning message carries, not "ccd_process".
    pytest.param(_run_ccd_process, "subtract_overscan", id="ccd_process"),
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
    crosses to the host -- and attributes it to this file's own call to the
    public function.

    Notes
    -----
    The whole message is pinned: it only states what always happens, that
    the data was copied to NumPy. An earlier wording also claimed that the
    result was copied back, which is false when a NumPy ``ccd`` comes with
    a non-NumPy ``inbkg``.

    ``filename == __file__`` alone would not catch a one-frame drift in the
    ``stacklevel`` computation: a drift of exactly one frame still lands
    inside this file, on ``test_cpu_only_function_warns_once`` itself
    rather than on the ``_run_*`` helper's call site. Pinning ``lineno``
    too is what makes this test catch that case; it is exactly the bug the
    ``ccd_process`` case (which calls ``subtract_overscan`` through another
    ccdproc frame) used to trigger with a fixed ``stacklevel``.
    """
    with pytest.warns(HostCopyWarning) as record:
        ccd, _, call_line = call()

    host_copies = [w for w in record if issubclass(w.category, HostCopyWarning)]
    assert len(host_copies) == 1
    namespace = array_api_compat.array_namespace(ccd.data)
    assert str(host_copies[0].message) == (
        f"{function_name} runs on the host CPU, so {namespace.__name__} array "
        "data was copied to numpy."
    )
    assert host_copies[0].filename == __file__
    assert host_copies[0].lineno == call_line


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
    ccd, result, _ = call()

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


def test_cosmicray_lacosmic_merges_existing_mask():
    """
    A ``CCDData`` that already carries a mask gets back the union of that
    mask and the cosmic-ray mask, in its own namespace and on its own
    device. The union is computed after the round trip, so it is the one
    place where a host-copied mask meets a native one; the bare-array call
    on the same data supplies the cosmic-ray mask to compare against.
    """
    ccd = ccd_data_func(data_size=DATA_SIZE)
    device = array_api_compat.device(ccd.data)

    # Plant one unmistakable cosmic ray so that the detected mask is not
    # empty. Edit on the host, as in test_cosmicray.add_cosmicrays, because
    # not every library supports item assignment.
    data_as_np = np.array(_to_numpy(ccd.data))
    data_as_np[10, 10] = 500.0
    ccd.data = xp.asarray(data_as_np, device=device)

    existing = np.zeros(ccd.shape, dtype=bool)
    existing[2, 3] = True
    # TODO: change back to .mask when CCDData is array-api compliant
    ccd._mask = xp.asarray(existing, device=device)

    _, crmask = cosmicray_lacosmic(ccd.data)
    result = cosmicray_lacosmic(ccd)

    assert bool(crmask[10, 10])
    assert not bool(crmask[2, 3])
    assert bool(result.mask[2, 3])
    assert bool(xp.all(result.mask == xp.logical_or(ccd.mask, crmask)))
    assert array_api_compat.array_namespace(result.mask) is xp
    assert array_api_compat.device(result.mask) == device


@pytest.mark.skipif(
    IS_NUMPY, reason="ccd and inbkg cannot be in different namespaces on numpy"
)
def test_cosmicray_lacosmic_warns_once_for_non_numpy_inbkg():
    """
    A NumPy ``ccd`` with a non-NumPy, array-valued ``inbkg`` still emits
    exactly one ``HostCopyWarning``, and still returns a NumPy result.

    ``inbkg`` and ``invar`` are copied to the host along with ``ccd``, so
    they count towards the warning too. If the decision looked only at
    ``ccd``'s namespace, a NumPy ``ccd`` with a non-NumPy ``inbkg`` or
    ``invar`` would be copied to the host with no warning at all.
    """
    ccd = numpy_ccddata(ccd_data_func(data_size=DATA_SIZE))
    inbkg = xp.asarray(np.zeros(ccd.shape), device=xp_device)

    with pytest.warns(HostCopyWarning) as record:
        result = cosmicray_lacosmic(ccd, inbkg=inbkg)

    host_copies = [w for w in record if issubclass(w.category, HostCopyWarning)]
    assert len(host_copies) == 1
    assert array_api_compat.is_numpy_namespace(
        array_api_compat.array_namespace(result.data)
    )


def _wcs_project_with_numpy_xp(ccd):
    """Call ``wcs_project`` on ``ccd`` with an explicit ``xp=np``."""
    target_wcs = wcs_for_testing(ccd.shape)
    target_wcs.wcs.crpix += [1, 1]
    return wcs_project(ccd, target_wcs, xp=np)


def _subtract_overscan_with_numpy_xp(ccd):
    """Call ``subtract_overscan``'s model path with an explicit ``xp=np``."""
    return subtract_overscan(
        ccd,
        overscan=ccd[:, :5],
        overscan_axis=1,
        model=models.Polynomial1D(1),
        xp=np,
    )


@pytest.mark.skipif(
    not array_api_compat.is_dask_namespace(xp),
    reason="only dask data is handled by numpy functions without an error",
)
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(_wcs_project_with_numpy_xp, id="wcs_project"),
        pytest.param(_subtract_overscan_with_numpy_xp, id="subtract_overscan"),
    ],
)
def test_numpy_xp_with_dask_data_warns_once(call):
    """
    Dask data with an explicit ``xp=np`` still emits exactly one
    ``HostCopyWarning``.

    Notes
    -----
    The dask array is computed and copied to the host whatever ``xp`` says,
    so the warning is decided from the data rather than from ``xp``, which
    used to silence it. Whether an ``xp`` that disagrees with the data
    should be accepted at all is a separate question, for every public
    function; only dask is tested because the other backends fail in
    their own ways when handed to NumPy functions.
    """
    ccd = ccd_data_func(data_size=DATA_SIZE)
    ccd.wcs = wcs_for_testing(ccd.shape)

    with pytest.warns(HostCopyWarning) as record:
        call(ccd)

    host_copies = [w for w in record if issubclass(w.category, HostCopyWarning)]
    assert len(host_copies) == 1


def test_cosmicray_lacosmic_unit_mismatch_does_not_convert_inbkg(monkeypatch):
    """
    A unit-mismatch ``ValueError`` is raised before ``inbkg`` is converted.

    ``cosmicray_lacosmic`` validates its input, then warns, then copies
    ``inbkg`` and ``invar`` to the host. Copying them first would waste a
    possibly large device-to-host transfer on a call that is about to be
    rejected, and would break ``_warn_host_copy``'s promise that the warning
    comes before any conversion. Monkeypatching ``ccdproc.core._to_numpy``
    to record its arguments makes the order observable.
    """
    ccd = ccd_data_func(data_size=DATA_SIZE)
    seen = []
    original = ccdproc.core._to_numpy

    def spy(arr):
        seen.append(arr)
        return original(arr)

    monkeypatch.setattr(ccdproc.core, "_to_numpy", spy)

    inbkg = np.zeros(ccd.shape)

    with pytest.raises(ValueError, match="Inconsistent units"):
        cosmicray_lacosmic(
            ccd, gain=2.0 * u.electron / u.adu, readnoise=6.5 * u.adu, inbkg=inbkg
        )

    assert not any(arr is inbkg for arr in seen)


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
