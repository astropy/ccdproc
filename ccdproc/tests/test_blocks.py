# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the array-API-native block functions in `ccdproc._blocks` and for
the namespace dispatch in `ccdproc.core` that selects them.

Everything here runs on whichever backend ``CCDPROC_ARRAY_LIBRARY`` selects.
The reference values always come from ``astropy.nddata`` applied to the
*numpy* source array, so each test is a differential test against the
implementation the numpy code path still uses.
"""

import array_api_compat
import array_api_extra as xpx
import numpy as np
import pytest
from astropy import nddata
from astropy import units as u
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.utils.exceptions import AstropyUserWarning

from ccdproc import _blocks, core
from ccdproc.conftest import testing_array_device as xp_device
from ccdproc.conftest import testing_array_library as xp

_rng = np.random.default_rng(3141)

_1D = _rng.normal(size=(10,))
_2D = _rng.normal(size=(6, 8))
# Neither axis is divisible by the block sizes used below, so these cases
# exercise the trimming astropy does before blocking.
_2D_RAGGED = _rng.normal(size=(7, 9))
_3D = _rng.normal(size=(4, 6, 8))
_INT = _rng.integers(0, 100, size=(6, 8))

_CASES = [
    pytest.param(_2D, (2, 2), id="2d-square"),
    pytest.param(_2D, (2, 4), id="2d-rectangular"),
    pytest.param(_2D, 2, id="2d-scalar"),
    pytest.param(_2D, 2.0, id="2d-integral-float"),
    pytest.param(_2D_RAGGED, (2, 3), id="2d-non-divisible"),
    pytest.param(_2D_RAGGED, 4, id="2d-non-divisible-scalar"),
    pytest.param(_1D, 2, id="1d-scalar"),
    pytest.param(_1D, (3,), id="1d-non-divisible"),
    pytest.param(_3D, (2, 3, 4), id="3d-rectangular"),
    pytest.param(_3D, 2, id="3d-scalar"),
]

# ``block_size`` values astropy rejects, with the message it rejects them
# with. The order of the three checks matters: (0, 2.5) is caught by the
# positivity check, not the integrality one.
_INVALID_BLOCK_SIZES = [
    pytest.param(0, id="zero"),
    pytest.param(-2, id="negative"),
    pytest.param((2, 0), id="zero-entry"),
    pytest.param(2.5, id="non-integral"),
    pytest.param((2, 2.5), id="non-integral-entry"),
    pytest.param((2, 2, 2), id="too-long"),
    pytest.param((0, 2.5), id="positivity-beats-integrality"),
]


def _to_xp(data):
    """Convert a numpy reference array to the backend under test."""
    return xp.asarray(data, device=xp_device)


def _assert_matches(result, expected_np):
    """
    Assert that ``result`` equals the numpy reference ``expected_np`` in
    shape, dtype and value, comparing entirely inside the backend's
    namespace so that nothing has to leave a non-default device.
    """
    expected = _to_xp(expected_np)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    if xp.isdtype(result.dtype, "real floating"):
        assert bool(xp.all(xpx.isclose(result, expected)))
    else:
        assert bool(xp.all(result == expected))


@pytest.mark.parametrize(
    ("func", "reference_func"),
    [
        pytest.param(None, np.sum, id="default-sum"),
        pytest.param(xp.mean, np.mean, id="mean"),
    ],
)
@pytest.mark.parametrize(("data", "block_size"), _CASES)
def test_block_reduce_matches_astropy(data, block_size, func, reference_func):
    """
    The native ``block_reduce`` reproduces `astropy.nddata.block_reduce` for
    square, rectangular and scalar block sizes, 1-d/2-d/3-d input, and
    shapes the block size does not divide evenly (which astropy trims).
    Parity with astropy is the whole contract of the native path: numpy
    input still goes to astropy, so any divergence would make a result
    depend on the array library.
    """
    reference = nddata.block_reduce(data, block_size, reference_func)
    result = _blocks.block_reduce(_to_xp(data), block_size, func)
    _assert_matches(result, reference)


@pytest.mark.parametrize("conserve_sum", [True, False])
@pytest.mark.parametrize(("data", "block_size"), _CASES)
def test_block_replicate_matches_astropy(data, block_size, conserve_sum):
    """
    The native ``block_replicate`` reproduces
    `astropy.nddata.block_replicate` for the same block-size and
    dimensionality spread, with ``conserve_sum`` both on and off. Same
    reason as for ``block_reduce``: the two code paths must not disagree.
    """
    reference = nddata.block_replicate(data, block_size, conserve_sum)
    result = _blocks.block_replicate(_to_xp(data), block_size, conserve_sum)
    _assert_matches(result, reference)


def test_block_reduce_integer_input_matches_astropy():
    """
    Integer input is summed as an integer, exactly as astropy does: the
    default ``xp.sum`` must not quietly promote it. Pinned separately from
    the float cases because the result dtype, not just the value, is part
    of the parity claim.
    """
    reference = nddata.block_reduce(_INT, 2)
    result = _blocks.block_reduce(_to_xp(_INT), 2)
    _assert_matches(result, reference)
    assert xp.isdtype(result.dtype, "integral")


@pytest.mark.parametrize("conserve_sum", [True, False])
def test_block_replicate_integer_input_matches_astropy(conserve_sum):
    """
    ``conserve_sum=True`` on integer input promotes to the namespace's
    default real floating dtype and ``conserve_sum=False`` leaves the
    integer dtype alone.

    This is the one place the native path needs an explicit cast:
    array-api-strict raises on integer true division instead of promoting
    the way numpy does, so without the promotion this call would fail
    there. Promoting to the *default* real dtype rather than float64 keeps
    a backend that has no float64 (jax without ``JAX_ENABLE_X64``) working,
    and matches astropy's float64 result everywhere else.
    """
    data = _to_xp(_INT)
    reference = nddata.block_replicate(_INT, 2, conserve_sum)
    result = _blocks.block_replicate(data, 2, conserve_sum)
    _assert_matches(result, reference)

    if conserve_sum:
        default_real = xp.__array_namespace_info__().default_dtypes(device=xp_device)[
            "real floating"
        ]
        assert result.dtype == default_real
    else:
        assert result.dtype == data.dtype


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(_blocks.block_reduce, id="block_reduce"),
        pytest.param(_blocks.block_replicate, id="block_replicate"),
    ],
)
@pytest.mark.parametrize("block_size", _INVALID_BLOCK_SIZES)
def test_invalid_block_size_matches_astropy_message(function, block_size):
    """
    An invalid ``block_size`` raises `ValueError` with astropy's message,
    verbatim, on every backend.

    The messages are asserted against astropy's own live output rather than
    hard-coded strings so that they cannot silently drift apart, and so the
    order of astropy's three checks (positivity, then length, then
    integrality) is pinned too -- ``(0, 2.5)`` is invalid twice over and
    must report the same one of the two as astropy.
    """
    astropy_function = getattr(nddata, function.__name__)
    with pytest.raises(ValueError) as astropy_error:
        astropy_function(_2D, block_size)
    with pytest.raises(ValueError) as native_error:
        function(_to_xp(_2D), block_size)
    assert str(native_error.value) == str(astropy_error.value)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(_blocks.block_reduce, id="block_reduce"),
        pytest.param(_blocks.block_replicate, id="block_replicate"),
    ],
)
def test_native_functions_preserve_namespace_and_device(function):
    """
    The result stays in the caller's namespace and on the caller's device.

    This is the bug the module exists to fix: astropy's versions coerce
    with ``numpy.asanyarray``, which returns numpy for dask and jax and
    raises for data on a device numpy cannot reach.
    """
    data = _to_xp(_2D)
    result = function(data, 2)
    assert array_api_compat.array_namespace(result) is array_api_compat.array_namespace(
        data
    )
    assert array_api_compat.device(result) == array_api_compat.device(data)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(core.block_reduce, id="block_reduce"),
        pytest.param(core.block_average, id="block_average"),
        pytest.param(core.block_replicate, id="block_replicate"),
    ],
)
def test_core_wrappers_preserve_namespace_and_device(function):
    """
    The public ``ccdproc`` wrappers dispatch to the native implementation
    for a non-numpy namespace, so a bare array in, say, dask comes back as
    a dask array on the same device rather than as numpy. On the numpy
    backend this pins that the astropy path still returns numpy.
    """
    data = _to_xp(_2D)
    result = function(data, 2)
    assert array_api_compat.array_namespace(result) is array_api_compat.array_namespace(
        data
    )
    assert array_api_compat.device(result) == array_api_compat.device(data)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(_blocks.block_reduce, id="block_reduce"),
        pytest.param(_blocks.block_replicate, id="block_replicate"),
    ],
)
def test_ccddata_input_warns_once_about_ignored_attributes(function):
    """
    A `~astropy.nddata.CCDData` argument is unpacked to its ``.data`` and
    produces exactly one "following attributes were set" warning, as
    astropy's own decorated functions do.

    That behaviour comes from `astropy.nddata.support_nddata`, which the
    native functions are decorated with precisely so the warning the
    existing ``ccdproc`` tests assert on is emitted identically on every
    backend rather than only on numpy.
    """
    ccd = CCDData(
        _to_xp(_2D),
        unit=u.adu,
        meta={"testkw": 1},
        uncertainty=StdDevUncertainty(_to_xp(_2D)),
    )
    with pytest.warns(AstropyUserWarning) as warning_list:
        result = function(ccd, 2)
    assert len(warning_list) == 1
    assert "following attributes were set" in str(warning_list[0].message)
    # The unpacked data, not the CCDData, is what the function works on.
    assert array_api_compat.array_namespace(result) is array_api_compat.array_namespace(
        ccd.data
    )
