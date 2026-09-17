# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the array-API-native block functions in `ccdproc._blocks` and for
the namespace dispatch in `ccdproc.core` that selects them.

Everything here runs on whichever backend ``CCDPROC_ARRAY_LIBRARY`` selects.
The reference values always come from ``astropy.nddata`` applied to the
*numpy* source array, so each test is a differential test against the
implementation the numpy code path still uses.
"""

import importlib
from functools import partial

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

# The array library itself rather than the array-api-compat wrapper the rest
# of the suite uses. For numpy and dask the two differ; for jax and
# array-api-strict, which are their own compat namespace, they coincide.
_RAW_MODULE = importlib.import_module(xp.__name__.removeprefix("array_api_compat."))

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

# Block sizes for which no axis pair has to be merged, so the reshape at the
# end of ``block_replicate`` can return a view of its input rather than a
# fresh array. Paired with ``conserve_sum`` because that decides whether the
# view is of the caller's own data or of a freshly divided copy.
_DEGENERATE_REPLICATE_CASES = [
    pytest.param(_2D, 1, True, id="all-ones-conserve_sum"),
    pytest.param(_2D, 1, False, id="all-ones-no-conserve_sum"),
    pytest.param(_2D[:3, :1], (1, 4), False, id="singleton-axis-no-conserve_sum"),
]

# ``block_size`` values astropy rejects, paired with a substring of the
# `ValueError` raised for each. The order of the three checks matters:
# (0, 2.5) is caught by the positivity check, not the integrality one, so its
# expected substring is the positivity one. NaN passes the positivity check,
# since every comparison with it is false, and so reaches the integrality
# one. The substrings are asserted with ``pytest.raises(..., match=...)``
# rather than compared byte-for-byte against astropy's live message:
# astropy's exact wording is a private implementation detail this suite
# should not be coupled to, only which of the three checks fires.
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
    pytest.param(float("nan"), "must be integers", id="nan"),
    pytest.param(float("inf"), "must be integers", id="inf"),
    pytest.param((2, float("inf")), "must be integers", id="inf-entry"),
]


def _astropy_block_average(data, block_size):
    """astropy's spelling of ``block_average``: ``block_reduce`` with ``np.mean``."""
    return nddata.block_reduce(data, block_size, np.mean)


# The native ``_blocks`` functions, each paired with the plain
# ``astropy.nddata`` call it must agree with value for value.
_NATIVE_AND_REFERENCES = [
    pytest.param(_blocks.block_reduce, nddata.block_reduce, id="block_reduce"),
    pytest.param(
        partial(_blocks.block_reduce, func=xp.mean),
        _astropy_block_average,
        id="block_reduce-explicit-mean",
    ),
    pytest.param(_blocks.block_average, _astropy_block_average, id="block_average"),
    pytest.param(
        partial(_blocks.block_replicate, conserve_sum=True),
        partial(nddata.block_replicate, conserve_sum=True),
        id="block_replicate-conserve_sum",
    ),
    pytest.param(
        partial(_blocks.block_replicate, conserve_sum=False),
        partial(nddata.block_replicate, conserve_sum=False),
        id="block_replicate-no-conserve_sum",
    ),
]

# The public ``ccdproc.core`` wrappers, each paired with the plain
# ``astropy.nddata`` call it must agree with. Shared by every test that
# needs both halves of that pairing, so the pairing itself is written once.
# The last two rows cover the wrappers' own argument forwarding: the
# ``args = (block_size,) if func is None else ...`` branch in
# ``core.block_reduce`` and ``conserve_sum`` in ``core.block_replicate``.
_CORE_WRAPPERS_AND_REFERENCES = [
    pytest.param(core.block_reduce, nddata.block_reduce, id="block_reduce"),
    pytest.param(core.block_average, _astropy_block_average, id="block_average"),
    pytest.param(core.block_replicate, nddata.block_replicate, id="block_replicate"),
    pytest.param(
        partial(core.block_reduce, func=xp.mean),
        _astropy_block_average,
        id="block_reduce-explicit-func",
    ),
    pytest.param(
        partial(core.block_replicate, conserve_sum=False),
        partial(nddata.block_replicate, conserve_sum=False),
        id="block_replicate-no-conserve",
    ),
]

# The same pairings minus the explicit-``func`` row, for the tests whose
# input is always numpy and so always takes the astropy path: there astropy
# would call the backend's ``mean`` -- ``dask.array.mean``, say -- on a numpy
# array and hand back something that is not a `numpy.ndarray`, which says
# nothing about ccdproc's dispatch.
_CORE_WRAPPERS_ON_NUMPY_INPUT = [
    case
    for case in _CORE_WRAPPERS_AND_REFERENCES
    if case.id != "block_reduce-explicit-func"
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


@pytest.mark.parametrize(("function", "reference_function"), _NATIVE_AND_REFERENCES)
@pytest.mark.parametrize(("data", "block_size"), _CASES)
def test_native_block_functions_match_astropy(
    data, block_size, function, reference_function
):
    """
    Each native block function reproduces its `astropy.nddata` counterpart,
    value for value and dtype for dtype, over square, rectangular and scalar
    block sizes, 1-d/2-d/3-d input, and shapes the block size does not divide
    evenly (which astropy trims).

    Parity with the numpy path is the whole contract of the native path:
    numpy input still goes to astropy, so any divergence would make a result
    depend on which array library the caller happens to use. The dtype
    differences that *are* deliberate are floating-point-only here and are
    pinned by the integer, boolean, complex and float32 tests below.
    """
    reference = reference_function(data, block_size)
    result = function(_to_xp(data), block_size)
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


def test_block_reduce_explicit_sum_matches_the_default_on_a_boolean_mask():
    """
    Spelling out the default reduction, ``func=xp.sum``, gives the same
    result on a boolean mask as omitting ``func`` altogether.

    The boolean-to-integer promotion above used to be gated on ``func is
    None``, so a caller who wrote the documented default out, or factored it
    into a variable, got a ``TypeError`` from array-api-strict where the
    plain call worked. The gate now also accepts ``xp.sum`` itself.
    """
    mask = _to_xp(_rng.integers(0, 2, size=(4, 4)).astype(bool))
    default = core.block_reduce(mask, 2)
    explicit = core.block_reduce(mask, 2, func=xp.sum)
    assert explicit.dtype == default.dtype
    assert bool(xp.all(explicit == default))


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(_blocks.block_average, id="native"),
        pytest.param(core.block_average, id="core-wrapper"),
    ],
)
def test_block_average_integer_input_matches_astropy(function):
    """
    ``block_average`` reproduces ``astropy.nddata.block_reduce(..., np.mean)``
    for integer input, returning a float.

    numpy's ``mean`` promotes an integer array on its own, and jax and dask
    follow it, but array-api-strict refuses a non-floating ``mean``
    outright. Without the explicit promotion the native path does before
    averaging, an integer image would average fine on three backends and
    raise on the fourth; this pins that all four behave the way numpy
    already did.
    """
    reference = _astropy_block_average(_INT, 2)
    result = function(_to_xp(_INT), 2)
    _assert_matches(result, reference)


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

    if not conserve_sum:
        assert result.dtype == data.dtype


@pytest.mark.parametrize(
    ("function", "reference_function"),
    [
        pytest.param(
            _blocks.block_average,
            _astropy_block_average,
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


def test_block_replicate_float32_input_keeps_float32():
    """
    Float32 input keeps its dtype through ``block_replicate`` with
    ``conserve_sum=True``, on any astropy version.

    `astropy.nddata.block_replicate` promoted such input to float64 before
    the fix for astropy/astropy#20360 (astropy PR #20364, in 7.2.3 and
    8.0.2): it divided by ``numpy.prod(block_size)``, an ``int64`` scalar
    that NEP 50 promotes against, while the native version divides by a
    Python int and so keeps the input's floating dtype -- deliberately,
    since FITS images are overwhelmingly float32 and doubling their size on
    upsampling is the wrong default for the GPU and lazy backends this
    module exists for. The test does not depend on which astropy it runs
    against: it compares values after casting the reference to float32, with
    a relaxed ``rtol`` rather than through ``_assert_matches``, which
    requires exact dtype equality, because a float64 division rounded to
    float32 can differ from a float32 division: max relative difference
    ~5e-8 for ``block_size=3`` on a shape astropy has to trim.
    """
    data32 = _2D.astype(np.float32)
    reference = nddata.block_replicate(data32, 3, True)
    result = _blocks.block_replicate(_to_xp(data32), 3, True)
    assert result.dtype == xp.float32
    expected = _to_xp(reference.astype(np.float32))
    assert bool(xp.all(xpx.isclose(result, expected, rtol=1e-6)))


@pytest.mark.parametrize(
    ("data", "block_size", "conserve_sum"), _DEGENERATE_REPLICATE_CASES
)
def test_block_replicate_degenerate_block_size_matches_astropy(
    data, block_size, conserve_sum
):
    """
    A block size that replicates nothing -- all ones, or a size greater than
    one only on a length-one axis -- still gives astropy's shape, values and
    dtype.

    These are the cases in which the final ``reshape`` has no axis pair to
    merge and so can hand back a view; the copy that fixes that (pinned by
    the test below) must not change the result itself, and nothing else in
    the suite exercises a block size of one.
    """
    reference = nddata.block_replicate(data, block_size, conserve_sum)
    result = _blocks.block_replicate(_to_xp(data), block_size, conserve_sum)
    _assert_matches(result, reference)


@pytest.mark.backend_skip(
    "jax",
    reason="jax arrays are immutable, so neither the input nor the result of "
    "any function can be assigned into",
)
@pytest.mark.parametrize(
    ("data", "block_size", "conserve_sum"), _DEGENERATE_REPLICATE_CASES
)
def test_block_replicate_degenerate_block_size_returns_a_fresh_writable_array(
    data, block_size, conserve_sum
):
    """
    For a block size that replicates nothing, the result is a fresh writable
    array rather than a view of the input.

    When every axis has ``length == 1`` or ``size == 1`` the broadcast is a
    no-op and the final ``reshape`` returns a view, which with
    ``conserve_sum=False`` is the caller's own data: mutating the input
    changed the result, and the result was read-only, so assigning into it
    raised and a `~astropy.nddata.CCDData` built from it had read-only
    ``.data``. astropy's ``numpy.repeat`` always returns a fresh writable
    array, so this pinned parity was quietly missing.
    """
    # A private copy: on the numpy backend ``_to_xp`` hands back the module
    # level array itself, and this test writes into its input.
    source = _to_xp(data.copy())
    result = _blocks.block_replicate(source, block_size, conserve_sum)
    before = xp.asarray(result, copy=True)

    source[0, 0] = xp.asarray(-999.0, device=xp_device)
    assert bool(xp.all(result == before))

    result[0, 0] = xp.asarray(1.0, device=xp_device)
    assert float(result[0, 0]) == 1.0


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(_blocks.block_reduce, id="block_reduce"),
        pytest.param(_blocks.block_replicate, id="block_replicate"),
    ],
)
@pytest.mark.parametrize(("block_size", "match"), _INVALID_BLOCK_SIZES)
def test_invalid_block_size_is_rejected(function, block_size, match):
    """
    An invalid ``block_size`` raises `ValueError`, and the expected one of
    the three checks (positivity, then length, then integrality) fires
    first -- ``(0, 2.5)`` is invalid twice over and must report the
    positivity failure, and a NaN or infinite size must report the
    integrality one rather than escaping as a `ValueError` or
    `OverflowError` from ``int()``.

    Only ccdproc's own messages are pinned, by substring: astropy's live
    error text is not something this suite should assert on, since a
    rewording upstream would fail the devdeps job for a reason ccdproc does
    not control, and ``_block_size`` explicitly does not undertake to track
    that wording.
    """
    with pytest.raises(ValueError, match=match):
        function(_to_xp(_2D), block_size)


@pytest.mark.backend_skip(
    "numpy",
    "jax",
    "array-api-strict",
    "cupy",
    reason="only dask produces arrays whose shape is not fully known",
)
@pytest.mark.parametrize(
    "function",
    [
        pytest.param(core.block_reduce, id="block_reduce"),
        pytest.param(core.block_average, id="block_average"),
        pytest.param(core.block_replicate, id="block_replicate"),
    ],
)
def test_unknown_shape_is_rejected_with_a_clear_message(function):
    """
    A dask array with unknown chunk sizes is rejected with a message that
    names the fix, rather than with whatever the Python arithmetic on the
    shape happens to raise.

    The native path computes block counts from ``data.shape`` in plain
    Python, and dask reports an unknown chunk size as a ``nan`` float, so
    without this check the call died with ``cannot convert float NaN to
    integer``, which names neither dask nor chunks.
    """
    unknown = _to_xp(_2D)
    unknown = unknown[unknown[:, 0] > -1]
    assert any(not isinstance(length, int) for length in unknown.shape)
    with pytest.raises(ValueError, match="fully known shape"):
        function(unknown, 2)


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
@pytest.mark.parametrize(
    "namespace",
    [pytest.param(xp, id="compat"), pytest.param(_RAW_MODULE, id="raw-module")],
)
def test_core_wrappers_honour_an_explicit_xp(namespace, function, reference_function):
    """
    An explicit ``xp`` is used as given instead of being inferred from the
    data, and the result is unchanged by passing it, whether the caller
    names the array-api-compat namespace or the plain array library module.

    The ``xp`` keyword is part of the wrappers' public signature so a caller
    can name the namespace up front. Regression test for the raw-module
    half: a plain module such as ``dask.array`` used to be forwarded
    verbatim to ``_blocks``, which needs the array-API spellings
    ``isdtype``, ``permute_dims`` and ``__array_namespace_info__`` that such
    a module may not have, so all three wrappers raised `AttributeError`.
    """
    data = _to_xp(_2D)
    result = function(data, 2, xp=namespace)
    _assert_matches(result, reference_function(_2D, 2))
    _assert_same_namespace_and_device(result, data)


@pytest.mark.parametrize(
    ("function", "reference_function"), _CORE_WRAPPERS_ON_NUMPY_INPUT
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

    Regression test: an earlier version of the namespace resolution only
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
