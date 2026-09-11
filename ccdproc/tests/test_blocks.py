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
from astropy.nddata import CCDData, NDData, StdDevUncertainty
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
_COMPLEX = _2D + 1j * _rng.normal(size=(6, 8))

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

# ``block_size`` values astropy rejects, paired with a substring of the
# `ValueError` astropy raises for each. The order of the three checks
# matters: (0, 2.5) is caught by the positivity check, not the integrality
# one, so its expected substring is the positivity one. The substrings are
# asserted with ``pytest.raises(..., match=...)`` rather than compared
# byte-for-byte against astropy's live message: astropy's exact wording is a
# private implementation detail this suite should not be coupled to, only
# which of the three checks fires.
_INVALID_BLOCK_SIZES = [
    pytest.param(0, "strictly positive", id="zero"),
    pytest.param(-2, "strictly positive", id="negative"),
    pytest.param((2, 0), "strictly positive", id="zero-entry"),
    pytest.param(2.5, "must be integers", id="non-integral"),
    pytest.param((2, 2.5), "must be integers", id="non-integral-entry"),
    pytest.param(
        (2, 2, 2), "same length as the number of data dimensions", id="too-long"
    ),
    pytest.param((0, 2.5), "strictly positive", id="positivity-beats-integrality"),
]

# The public ``ccdproc.core`` wrappers, each paired with the plain
# ``astropy.nddata`` call it must agree with. Shared by every test that
# needs both halves of that pairing, so the pairing itself is written once.
_CORE_WRAPPERS_AND_REFERENCES = [
    pytest.param(core.block_reduce, nddata.block_reduce, id="block_reduce"),
    pytest.param(
        core.block_average,
        lambda data, block_size: nddata.block_reduce(data, block_size, np.mean),
        id="block_average",
    ),
    pytest.param(core.block_replicate, nddata.block_replicate, id="block_replicate"),
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
    if xp.isdtype(result.dtype, ("real floating", "complex floating")):
        assert bool(xp.all(xpx.isclose(result, expected)))
    else:
        assert bool(xp.all(result == expected))


def _assert_same_namespace_and_device(result, data):
    """
    Assert that ``result`` is in the same array-API namespace and on the
    same device as ``data``.
    """
    assert array_api_compat.array_namespace(result) is array_api_compat.array_namespace(
        data
    )
    assert array_api_compat.device(result) == array_api_compat.device(data)


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


def test_block_reduce_bool_input_matches_astropy():
    """
    A boolean mask is summed as an integer count, matching astropy's
    bool-sum (which promotes to int64 through ``numpy.sum``) instead of
    diverging by backend.

    Without promoting boolean input to the
    namespace's default integral dtype before the default ``xp.sum``,
    array-api-strict rejects a boolean sum outright and jax without X64
    would give a narrower int32 than astropy's int64, so block-summing a
    mask (counting flagged pixels per block) would depend on the backend.
    """
    mask = _rng.integers(0, 2, size=(4, 4)).astype(bool)
    reference = nddata.block_reduce(mask, 2)
    result = _blocks.block_reduce(_to_xp(mask), 2)
    _assert_matches(result, reference)
    assert xp.isdtype(result.dtype, "integral")


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(_blocks.block_average, id="native"),
        pytest.param(core.block_average, id="core-wrapper"),
    ],
)
@pytest.mark.parametrize(
    "data",
    [pytest.param(_2D, id="float-input"), pytest.param(_INT, id="integer-input")],
)
def test_block_average_matches_astropy(function, data):
    """
    ``block_average`` reproduces ``astropy.nddata.block_reduce(..., np.mean)``
    for float and for integer input, and always returns a float.

    numpy's ``mean`` promotes an integer array on its own, and jax and dask
    follow it, but array-api-strict refuses a non-floating ``mean``
    outright. Without the explicit promotion the native path does before
    averaging, an integer image would average fine on three backends and
    raise on the fourth; this pins that all four behave the way numpy
    already did.
    """
    reference = nddata.block_reduce(data, 2, np.mean)
    result = function(_to_xp(data), 2)
    _assert_matches(result, reference)
    assert xp.isdtype(result.dtype, "real floating")


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
    ("function", "reference_function"),
    [
        pytest.param(
            _blocks.block_average,
            lambda data, block_size: nddata.block_reduce(data, block_size, np.mean),
            id="block_average",
        ),
        pytest.param(
            lambda data, block_size: _blocks.block_replicate(data, block_size, True),
            lambda data, block_size: nddata.block_replicate(data, block_size, True),
            id="block_replicate-conserve_sum",
        ),
    ],
)
def test_complex_input_matches_astropy(function, reference_function):
    """
    Complex input to ``block_average`` and to ``block_replicate`` with
    ``conserve_sum=True`` keeps its imaginary part and its complex128
    dtype, matching astropy.

    Regression test: both functions promote integer and boolean input
    before dividing, and an earlier version did so with a helper that
    treats anything not real floating as needing promotion, so a complex
    array was cast to the default *real* dtype and silently lost its
    imaginary part. Complex is already floating and must pass through.
    """
    reference = reference_function(_COMPLEX, 2)
    result = function(_to_xp(_COMPLEX), 2)
    _assert_matches(result, reference)
    assert xp.isdtype(result.dtype, "complex floating")


def test_block_replicate_float32_input_keeps_float32():
    """
    Float32 input keeps its dtype through ``block_replicate`` with
    ``conserve_sum=True``, where `astropy.nddata.block_replicate` promotes
    it to float64.

    This is a documented difference: astropy divides by ``numpy.prod(block_size)``,
    an ``int64`` scalar that NEP 50 promotes against, while the native
    version divides by a Python int and so keeps the input's floating
    dtype -- deliberately, since FITS images are overwhelmingly float32 and
    doubling their size on upsampling is the wrong default for the GPU and
    lazy backends this module exists for. Values are compared after casting
    astropy's float64 reference down to float32, with a relaxed ``rtol``
    (rather than through ``_assert_matches``, which requires exact dtype
    equality), because the two float32 divisions round slightly
    differently: max relative difference ~5e-8 for ``block_size=3`` on a
    shape astropy has to trim. The assertion on the reference's dtype is a
    canary for astropy/astropy#20360, so that the documented difference is
    dropped once astropy stops upcasting.
    """
    data32 = _2D.astype(np.float32)
    reference = nddata.block_replicate(data32, 3, True)
    # Canary for astropy/astropy#20360: when astropy stops upcasting, this
    # fails on the devdeps job, and the "second difference" in
    # docs/array_api.rst and CHANGES.rst should be dropped.
    assert reference.dtype == np.float64
    result = _blocks.block_replicate(_to_xp(data32), 3, True)
    assert result.dtype == _to_xp(data32).dtype
    expected = _to_xp(reference.astype(np.float32))
    assert bool(xp.all(xpx.isclose(result, expected, rtol=1e-6)))


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(_blocks.block_reduce, id="block_reduce"),
        pytest.param(_blocks.block_replicate, id="block_replicate"),
    ],
)
@pytest.mark.parametrize(("block_size", "match"), _INVALID_BLOCK_SIZES)
def test_invalid_block_size_matches_astropy_message(function, block_size, match):
    """
    An invalid ``block_size`` raises `ValueError` on both the astropy and
    native paths, and the same one of astropy's three checks (positivity,
    then length, then integrality) fires first on each -- ``(0, 2.5)`` is
    invalid twice over and must report the same one of the two as astropy.

    The substrings, not astropy's exact wording, are what is pinned:
    the parity was already leaky (e.g. a 2-d block size raises a different
    message than astropy for some invalid shapes, and non-finite sizes
    diverge further, see ``test_non_finite_block_size_is_rejected_as_non_integral``
    below), and no other test in this suite compares error text to another
    library byte for byte.
    """
    astropy_function = getattr(nddata, function.__name__)
    with pytest.raises(ValueError, match=match):
        astropy_function(_2D, block_size)
    with pytest.raises(ValueError, match=match):
        function(_to_xp(_2D), block_size)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(_blocks.block_reduce, id="native-block_reduce"),
        pytest.param(_blocks.block_average, id="native-block_average"),
        pytest.param(_blocks.block_replicate, id="native-block_replicate"),
        pytest.param(core.block_reduce, id="core-block_reduce"),
        pytest.param(core.block_average, id="core-block_average"),
        pytest.param(core.block_replicate, id="core-block_replicate"),
    ],
)
def test_functions_preserve_namespace_and_device(function):
    """
    The result stays in the caller's namespace and on the caller's device,
    both for the native ``_blocks`` functions directly and through the
    public ``ccdproc.core`` wrappers.

    This is the bug the module exists to fix: astropy's versions coerce
    with ``numpy.asanyarray``, which returns numpy for dask and jax and
    raises for data on a device numpy cannot reach. On the numpy backend
    this also pins that the ``core`` wrappers' astropy path still returns
    numpy.
    """
    data = _to_xp(_2D)
    result = function(data, 2)
    _assert_same_namespace_and_device(result, data)


def test_ccddata_input_preserves_device():
    """
    A `~astropy.nddata.CCDData` argument is unpacked to its ``.data`` and
    the result stays on that data's device, including a non-default device
    such as array-api-strict's ``device1``.

    The warning about the ignored ``unit``/``meta``/``uncertainty``
    attributes is asserted by ``test_ccdproc.py::test_block_{reduce,average,
    replicate}`` through the ``ccdproc.core`` wrappers on every backend, so
    the only thing unique to this test is the device claim -- constructing the
    `~astropy.nddata.CCDData` directly against `_blocks` is not itself a
    production path, since `ccdproc.core` never passes one to `_blocks`
    without unwrapping it first, but the device it carries is real.
    """
    ccd = CCDData(
        _to_xp(_2D),
        unit=u.adu,
        meta={"testkw": 1},
        uncertainty=StdDevUncertainty(_to_xp(_2D)),
    )
    with pytest.warns(AstropyUserWarning, match="following attributes were set"):
        result = _blocks.block_reduce(ccd, 2)
    _assert_same_namespace_and_device(result, ccd.data)


@pytest.mark.parametrize(
    ("function", "reference_function"), _CORE_WRAPPERS_AND_REFERENCES
)
def test_core_wrappers_honour_an_explicit_xp(function, reference_function):
    """
    An explicit ``xp`` is used as given instead of being inferred from the
    data, and the result is unchanged by passing it.

    The ``xp`` keyword is part of the wrappers' public signature so a
    caller can name the namespace up front; this pins that the keyword is
    honoured on every backend and still selects the astropy path on numpy.
    """
    data = _to_xp(_2D)
    result = function(data, 2, xp=xp)
    _assert_matches(result, reference_function(_2D, 2))
    _assert_same_namespace_and_device(result, data)


@pytest.mark.parametrize(
    ("function", "reference_function"), _CORE_WRAPPERS_AND_REFERENCES
)
@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(NDData, id="nddata"),
        pytest.param(lambda data: data.tolist(), id="list"),
    ],
)
def test_core_wrappers_accept_nddata_and_list_input(function, reference_function, wrap):
    """
    A plain `~astropy.nddata.NDData` or a nested list reaches the astropy
    path and comes back as a plain `numpy.ndarray`, matching astropy, on
    every backend the suite runs against.

    Regression test: an earlier version of ``_namespace_for`` only
    unwrapped `~astropy.nddata.CCDData` and let
    ``array_api_compat.array_namespace`` reject everything else, so a plain
    `~astropy.nddata.NDData` or nested list raised `TypeError` where
    `astropy.nddata` always accepted them -- a regression from ``main``.
    Neither input form is converted by ``_to_xp``, so both are plain numpy
    regardless of ``CCDPROC_ARRAY_LIBRARY`` and always exercise the
    numpy/astropy fallback, which is exactly the path this pins.
    """
    result = function(wrap(_2D), 2)
    assert isinstance(result, np.ndarray)
    assert not isinstance(result, CCDData)
    np.testing.assert_array_equal(result, reference_function(_2D, 2))


@pytest.mark.parametrize(
    "block_size",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param((2, float("inf")), id="inf-entry"),
    ],
)
def test_non_finite_block_size_is_rejected_as_non_integral(block_size):
    """
    A NaN or infinite ``block_size`` raises the "must be integers" error
    rather than escaping as a `ValueError` or `OverflowError` from
    ``int()``.

    NaN passes the positivity check (every comparison with it is false),
    so it reaches the integrality check, and ``float(x).is_integer()`` is
    false for both NaN and infinity, so neither ever reaches ``int()``.
    The message is hard-coded rather than compared against astropy's live
    output because astropy's own check first emits a numpy cast warning
    for these values.
    """
    with pytest.raises(ValueError, match="block_size elements must be integers"):
        _blocks.block_reduce(_to_xp(_2D), block_size)
