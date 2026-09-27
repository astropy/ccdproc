# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Tests for the check that the array inputs of one call agree (#1025).

The functions that take more than one image -- ``subtract_overscan``,
``subtract_bias``, ``subtract_dark``, ``flat_correct``, ``ccd_process``,
``cosmicray_median`` with an array ``error_image``, ``Combiner`` and
``combine`` -- take their array namespace from the data, so data from two
array libraries, or on two devices, cannot both be honoured. They raise
instead, naming the arguments that disagree.
"""

import re

import array_api_compat
import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import CCDData

from ccdproc import (
    Combiner,
    ccd_process,
    combine,
    cosmicray_median,
    flat_correct,
    subtract_bias,
    subtract_dark,
)
from ccdproc.conftest import testing_array_library as xp
from ccdproc.core import _namespace_of
from ccdproc.tests.pytest_fixtures import to_xp

_IMAGE = np.arange(1.0, 101.0).reshape(10, 10)


def _foreign_namespace():
    """
    An array namespace other than the one under test: dask when the tests
    run on numpy, numpy otherwise.
    """
    if array_api_compat.is_numpy_namespace(xp):
        dask_array = pytest.importorskip("dask.array")
        return array_api_compat.array_namespace(dask_array.zeros(1))
    return array_api_compat.numpy


def _call_subtract_bias(ccd, other):
    return subtract_bias(ccd, other)


def _call_subtract_dark(ccd, other):
    return subtract_dark(ccd, other, dark_exposure=1 * u.s, data_exposure=1 * u.s)


def _call_flat_correct(ccd, other):
    return flat_correct(ccd, other)


def _call_cosmicray_median_ccddata(ccd, other):
    return cosmicray_median(ccd, error_image=other.data, mbox=3)


def _call_cosmicray_median_array(ccd, other):
    return cosmicray_median(ccd.data, error_image=other.data, mbox=3)


def _call_combiner(ccd, other):
    return Combiner([ccd, other])


def _call_combine(ccd, other):
    return combine([ccd, other])


def _call_ccd_process_master_bias(ccd, other):
    return ccd_process(ccd, master_bias=other)


def _call_ccd_process_dark_frame(ccd, other):
    return ccd_process(
        ccd, dark_frame=other, dark_exposure=1 * u.s, data_exposure=1 * u.s
    )


def _call_ccd_process_master_flat(ccd, other):
    return ccd_process(ccd, master_flat=other)


# Each call site, with the names its error message gives the second input
# and the first. subtract_overscan and ccd_process's oscan are checked too,
# but not tested here: an overscan is a slice of its image, so the two
# cannot come from different array libraries or devices in real use.
_CALL_SITES = [
    pytest.param(_call_subtract_bias, "master", "ccd", id="subtract_bias"),
    pytest.param(_call_subtract_dark, "master", "ccd", id="subtract_dark"),
    pytest.param(_call_flat_correct, "flat", "ccd", id="flat_correct"),
    pytest.param(
        _call_cosmicray_median_ccddata,
        "error_image",
        "ccd",
        id="cosmicray_median-ccddata",
    ),
    pytest.param(
        _call_cosmicray_median_array, "error_image", "ccd", id="cosmicray_median-array"
    ),
    pytest.param(_call_combiner, "ccd_iter[1]", "ccd_iter[0]", id="Combiner"),
    pytest.param(_call_combine, "img_list[1]", "img_list[0]", id="combine"),
    pytest.param(
        _call_ccd_process_master_bias,
        "master_bias",
        "ccd",
        id="ccd_process-master_bias",
    ),
    pytest.param(
        _call_ccd_process_dark_frame, "dark_frame", "ccd", id="ccd_process-dark_frame"
    ),
    pytest.param(
        _call_ccd_process_master_flat,
        "master_flat",
        "ccd",
        id="ccd_process-master_flat",
    ),
]


def test_namespace_of_returns_the_shared_namespace_skipping_none_and_scalars():
    """
    Agreeing arrays give their namespace; `None` and the scalars -- a Python
    float, a NumPy scalar, a 0-d NumPy array and a scalar Quantity -- are
    skipped rather than compared.

    The scalars matter: a single value computed with NumPy, such as the
    standard deviation of a NumPy copy of the data, comes back as one of the
    NumPy kinds, and it broadcasts against an array of any library. If
    scalars were compared, every non-NumPy call passing one would raise
    ``TypeError``; if `None` were, every call leaving an optional array
    argument out would.
    """
    data = to_xp(_IMAGE)
    result = _namespace_of(
        a=data,
        b=None,
        c=1.5,
        d=np.float64(2.0),
        e=np.asarray(3.0),
        f=4.0 * u.adu,
        g=to_xp(_IMAGE + 1),
    )
    assert result is array_api_compat.array_namespace(data)


def test_namespace_of_raises_when_no_argument_is_an_array():
    """
    With no array among the arguments ``_namespace_of`` raises
    ``TypeError`` naming them.

    There is then no namespace to return; the error says so instead of
    failing with an ``IndexError`` from inside the helper.
    """
    with pytest.raises(TypeError, match="none of first, second is an array"):
        _namespace_of(first=1.0, second=None)


def test_namespace_of_names_both_arguments_when_namespaces_differ():
    """
    Arrays from two array libraries raise ``TypeError`` naming both
    arguments and both libraries.

    The names are what make the error actionable for a caller of a
    function with several image arguments; without the check the
    arithmetic would convert one of them silently or fail somewhere deeper
    with an unrelated message.
    """
    foreign = _foreign_namespace()
    library = foreign.__name__.removeprefix("array_api_compat.")
    with pytest.raises(
        TypeError, match=rf"second comes from {re.escape(library)} but first comes"
    ):
        _namespace_of(first=to_xp(_IMAGE), second=foreign.asarray(_IMAGE))


@pytest.mark.parametrize(("call", "other_name", "first_name"), _CALL_SITES)
def test_mixed_namespaces_raise(call, other_name, first_name):
    """
    Every function that takes more than one image raises ``TypeError``,
    naming the arguments, when they come from different array libraries.

    Before the check, some of these combinations silently converted one
    image into the other's library, possibly through a host copy, and
    others failed with an error that named neither argument. Deleting a
    call to ``_namespace_of`` brings that back for its function, which this
    test would catch.
    """
    ccd = CCDData(to_xp(_IMAGE), unit=u.adu)
    other = CCDData(_foreign_namespace().asarray(_IMAGE), unit=u.adu)
    with pytest.raises(
        TypeError,
        match=rf"{re.escape(other_name)} comes from .* but "
        rf"{re.escape(first_name)} comes from",
    ):
        call(ccd, other)


@pytest.mark.parametrize(("call", "other_name", "first_name"), _CALL_SITES)
def test_mixed_devices_raise(call, other_name, first_name):
    """
    Every function that takes more than one image raises ``ValueError``,
    naming the arguments, when they are on different devices.

    This is the CuPy multi-GPU case, stood in for by array-api-strict's
    two devices; its arrays are built directly so this runs whenever it is
    installed. Before the check, ``Combiner`` (and so ``combine``) silently
    moved every image onto the first image's device, and the other
    functions failed inside array-api-strict (or CuPy) with a message that
    names neither argument.
    """
    strict = pytest.importorskip("array_api_strict")
    ccd = CCDData(
        strict.asarray(_IMAGE, device=strict.Device("CPU_DEVICE")), unit=u.adu
    )
    other = CCDData(strict.asarray(_IMAGE, device=strict.Device("device1")), unit=u.adu)
    with pytest.raises(
        ValueError,
        match=rf"{re.escape(other_name)} is on device .* but "
        rf"{re.escape(first_name)} is on device",
    ):
        call(ccd, other)


@pytest.mark.backend_skip(
    "array-api-strict",
    reason="array-api-strict rejects NumPy scalars in arithmetic whatever the check",
)
@pytest.mark.parametrize(
    "error_image", [np.float64(1.0), np.asarray(1.0)], ids=["scalar", "0-d-array"]
)
def test_cosmicray_median_accepts_a_numpy_scalar_error_image(error_image):
    """
    A single NumPy value as ``error_image``, whether a NumPy scalar or a 0-d
    array, is not rejected as a mixed input.

    ``np.std`` of a NumPy copy of the data is a natural way to make one,
    and it broadcasts against a jax, dask or CuPy array; the mixed-input
    check must treat it as a scalar, not as a NumPy array, or this call
    would raise ``TypeError`` on every backend but NumPy, although it worked
    before the check was added. array-api-strict is skipped because its own
    arithmetic accepts only Python scalars.
    """
    data = to_xp(_IMAGE)
    cleaned, crmask = cosmicray_median(data, error_image=error_image, mbox=3)
    assert array_api_compat.array_namespace(
        cleaned
    ) is array_api_compat.array_namespace(data)
    assert crmask.shape == data.shape


def test_subtract_bias_bare_array_master_reports_missing_unit():
    """
    A bare array passed as ``master`` to ``subtract_bias`` raises the
    ``ValueError`` about its missing unit.

    That is the error it gave before the mixed-input check; the check must
    run on the wrapped inputs, or it would fail first with an
    ``AttributeError`` for the array's missing ``data``.
    """
    ccd = CCDData(to_xp(_IMAGE), unit=u.adu)
    with pytest.raises(ValueError, match="a unit for CCDData must be specified"):
        subtract_bias(ccd, to_xp(_IMAGE))


def test_ccd_process_rejects_a_bare_array():
    """
    ``ccd_process`` raises ``TypeError`` saying ``ccd`` is not a CCDData
    when given a bare array.

    Notes
    -----
    Without the check the mixed-input check reports that none of ``ccd``,
    ``oscan`` and the masters is an array, which misleads when ``ccd`` is
    one. Before #1025 a bare array failed too, with an unrelated
    ``TypeError`` from deeper in the reduction.
    """
    with pytest.raises(TypeError, match="ccd is not a CCDData object"):
        ccd_process(to_xp(_IMAGE))
