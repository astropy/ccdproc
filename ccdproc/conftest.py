# Licensed under a 3-clause BSD style license - see LICENSE.rst

# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.
import logging
import os
import traceback

import array_api_compat
import array_api_compat.numpy
import numpy as np
import pytest

try:
    # When the pytest_astropy_header package is installed
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

    def pytest_configure(config):
        config.option.astropy_header = True

except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}


from .tests._escape_triage import (
    _env_truthy,
    pytest_runtest_makereport,  # noqa: F401 pytest hook, used via attribute lookup
    pytest_sessionfinish,  # noqa: F401 pytest hook, used via attribute lookup
    pytest_sessionstart,  # noqa: F401 pytest hook, used via attribute lookup
    pytest_terminal_summary,  # noqa: F401 pytest hook, used via attribute lookup
    record_escape_log,
)
from .tests.pytest_fixtures import (
    triage_setup,  # noqa: F401 this is used in tests
)

# This is to figure out ccdproc version, rather than using Astropy's
try:
    from ccdproc import __version__ as version
except ImportError:
    version = "dev"

TESTED_VERSIONS["ccdproc"] = version

# Add astropy to test header information and remove unused packages.
PYTEST_HEADER_MODULES["Astropy"] = "astropy"
PYTEST_HEADER_MODULES["astroscrappy"] = "astroscrappy"
PYTEST_HEADER_MODULES["reproject"] = "reproject"
PYTEST_HEADER_MODULES.pop("h5py", None)

# Set up the array library to be used in tests
# What happens here is controlled by an environmental variable
array_library = os.environ.get("CCDPROC_ARRAY_LIBRARY", "numpy").lower()

# Device to create test arrays on. This is only meaningful for backends that
# support multiple devices (currently array-api-strict, as a CPU-only proxy
# for testing non-default-device behavior like CuPy's GPU device). Leaving
# CCDPROC_ARRAY_DEVICE unset selects the backend's testing default: the
# non-default "device1" for array-api-strict, None (the library's usual
# device) for everything else. Setting it to "default" selects the library's
# normal default device; any other value is passed to the library's Device
# constructor.
testing_array_device = None

match array_library:
    case "numpy":
        import array_api_compat.numpy as testing_array_library  # noqa: F401

    case "jax":
        import jax.numpy as testing_array_library  # noqa: F401

        PYTEST_HEADER_MODULES["jax"] = "jax"

    case "dask":
        import array_api_compat.dask.array as testing_array_library  # noqa: F401

        PYTEST_HEADER_MODULES["dask"] = "dask"

    case "cupy":
        import array_api_compat.cupy as testing_array_library  # noqa: F401

        PYTEST_HEADER_MODULES["cupy"] = "cupy"

    case "array-api-strict" | "array_api_strict":
        import array_api_strict as testing_array_library  # noqa: F401

        PYTEST_HEADER_MODULES["array_api_strict"] = "array_api_strict"

        # array-api-strict exposes a couple of extra fake devices in addition
        # to its default CPU_DEVICE. Using one of those non-default devices
        # here makes np.asarray() raise on the resulting arrays, the same way
        # it would for a CuPy array living on a GPU. That makes
        # array-api-strict a convenient CPU-only proxy for catching the
        # "silent conversion to numpy" bugs that would otherwise only show up
        # on CuPy.
        device_name = os.environ.get("CCDPROC_ARRAY_DEVICE", "device1")
        if device_name.lower() == "default":
            # The library's normal CPU device, on which np.asarray() succeeds.
            testing_array_device = testing_array_library.Device("CPU_DEVICE")
        else:
            testing_array_device = testing_array_library.Device(device_name)

    case _:
        raise ValueError(
            f"Unsupported array library: {array_library}. "
            "Supported libraries are listed at https://ccdproc.readthedocs.io/en/latest/array_api.html."
        )


# ---------------------------------------------------------------------------
# Per-backend xfail/skip markers
#
# @pytest.mark.backend_xfail("cupy", "array-api-strict", reason="...")
# @pytest.mark.backend_skip("cupy", reason="...")
#
# These let individual tests be marked as expected-failures or skips only
# when run against specific array backends (as set by CCDPROC_ARRAY_LIBRARY),
# without affecting the default numpy-backed test run.
# ---------------------------------------------------------------------------


def _normalize_backend_name(name):
    return name.lower().replace("_", "-")


_ACTIVE_BACKEND = _normalize_backend_name(array_library)


def pytest_collection_modifyitems(items):
    """
    Apply the ``backend_skip`` / ``backend_xfail`` markers at collection
    time. For each collected test, the backends named in a marker's
    positional args (normalized, so ``array_api_strict`` and
    ``array-api-strict`` match) are compared against the active backend from
    ``CCDPROC_ARRAY_LIBRARY``; on a match the corresponding standard pytest
    marker is attached -- a plain skip, or a *non-strict* xfail so a test
    that starts passing XPASSes instead of failing the run (see the "prune
    on XPASS" note in docs/array_api.rst). Tests without a matching marker,
    including everything in the default numpy run, are untouched.
    """
    for item in items:
        for marker in item.iter_markers(name="backend_skip"):
            backends = {_normalize_backend_name(b) for b in marker.args}
            if _ACTIVE_BACKEND in backends:
                reason = marker.kwargs.get(
                    "reason", f"skipped for array backend {_ACTIVE_BACKEND!r}"
                )
                item.add_marker(pytest.mark.skip(reason=reason))
                # One matching marker decides the outcome; skip the rest.
                break

        for marker in item.iter_markers(name="backend_xfail"):
            backends = {_normalize_backend_name(b) for b in marker.args}
            if _ACTIVE_BACKEND in backends:
                reason = marker.kwargs.get(
                    "reason", f"expected failure for array backend {_ACTIVE_BACKEND!r}"
                )
                item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
                break


# ---------------------------------------------------------------------------
# Escape logger: catch silent conversion of non-numpy array-API arrays back
# to numpy. On CuPy such a conversion typically raises immediately (because
# the array lives on a GPU), which is how those bugs are usually found. On
# backends like dask and jax the conversion often succeeds silently, so the
# bug is easy to miss. This monkeypatches np.asarray/np.asanyarray (and
# np.ma.asanyarray) to log a warning, when active, any time they are handed
# an array whose array-API namespace (as resolved by
# array_api_compat.array_namespace) is not numpy.
#
# Only Python-level calls through the module attributes are visible:
# C-level coercions inside compiled dependencies (scipy, astroscrappy,
# reproject) and references bound before the patch ("from numpy import
# asarray") bypass the wrappers entirely, so absence of a warning is not
# proof there was no conversion.
#
# Activated by setting the environment variable CCDPROC_LOG_ARRAY_ESCAPES to
# a truthy value.
# ---------------------------------------------------------------------------

_escape_logger = logging.getLogger("ccdproc.array_escape")

_LOG_ARRAY_ESCAPES = _env_truthy(os.environ.get("CCDPROC_LOG_ARRAY_ESCAPES", ""))

# array_api_compat.numpy wraps plain numpy; arrays built through it report
# __array_namespace__() as one of these two modules. Neither counts as an
# "escape" -- we only care about arrays from a genuinely different library
# (jax, dask, cupy, array_api_strict, ...) ending up in a numpy-only call.
_NUMPY_LIKE_NAMESPACES = {np, array_api_compat.numpy}


def _foreign_namespace(obj):
    """
    Return obj's array-API namespace if it is foreign, else None.

    numpy arrays (including np.ma masked arrays) are never foreign.
    Detection goes through array_api_compat.array_namespace() rather than
    the raw __array_namespace__ dunder because some backends (notably dask)
    do not define the dunder on their array objects even though
    array-api-compat can resolve a namespace for them. The namespace is
    foreign unless it is numpy or array_api_compat's numpy wrapper -- i.e.
    exactly the arrays whose conversion to numpy would fail (CuPy on GPU)
    or silently densify/transfer (dask, jax). The try/except guards against
    non-arrays and misbehaving objects (including unhashable namespaces);
    those are treated as not foreign so the logger never breaks the call
    it wraps.
    """
    if isinstance(obj, np.ndarray):
        return None
    try:
        namespace = array_api_compat.array_namespace(obj)
        if namespace in _NUMPY_LIKE_NAMESPACES:
            return None
    except Exception:
        return None
    return namespace


def _escape_site_frame():
    """
    Return the FrameSummary for the innermost stack frame that is inside
    ccdproc but not inside ccdproc's own test suite -- the frame blamed for
    an escape -- or None if there is no such frame. Frames in this module are
    classified as test frames and are never chosen, so the exact call depth
    here does not affect the result.
    """
    from .tests._escape_triage import locate_escape_site

    return locate_escape_site(traceback.extract_stack())


def _describe_escape_site(frame):
    """Short "file:line function" string for an escape log message."""
    if frame is None:
        return "<unknown location>"
    return f"{frame.filename}:{frame.lineno} {frame.name}"


class _ReentrancyGuard:
    """Small helper to keep our wrappers from recursing into themselves."""

    def __init__(self):
        self.active = False


def _make_escape_logging_wrapper(original, funcname, guard):
    """
    Build the replacement for one numpy coercion entry point (``original`` is
    the real ``np.asarray``, ``np.asanyarray`` or ``np.ma.asanyarray``).
    The wrapper logs and tallies an "array-API escape" whenever its first
    positional argument is an array from a non-numpy array-API library, then
    always finishes by calling ``original`` -- behavior is unchanged, escapes
    are only observed. ``funcname`` is the dotted numpy name (e.g.
    ``numpy.asarray``) recorded with each escape. ``guard`` breaks recursion:
    the namespace detection and stack extraction below can themselves end up
    calling the patched functions.
    """

    def wrapper(*args, **kwargs):
        # Do nothing extra on re-entrant calls (guard held) or when there is
        # no positional argument to inspect.
        if not guard.active and args:
            guard.active = True
            try:
                obj = args[0]
                namespace = _foreign_namespace(obj)
                if namespace is not None:
                    # A foreign array is about to be coerced to numpy: warn
                    # with the innermost non-test ccdproc frame to blame, and
                    # tally the site for the end-of-session summary and the
                    # baseline ratchet.
                    frame = _escape_site_frame()
                    site = _describe_escape_site(frame)
                    _escape_logger.warning(
                        "array-API escape: %s() called on a %r array "
                        "(namespace=%r) at %s",
                        funcname,
                        type(obj),
                        namespace,
                        site,
                    )
                    record_escape_log(frame, funcname)
            finally:
                guard.active = False
        return original(*args, **kwargs)

    wrapper.__name__ = getattr(original, "__name__", funcname)
    wrapper.__doc__ = getattr(original, "__doc__", None)
    return wrapper


@pytest.fixture(autouse=True, scope="session")
def _log_array_escapes():
    """
    Session-scoped autouse fixture that, when CCDPROC_LOG_ARRAY_ESCAPES is
    set, monkeypatches numpy's array-coercion entry points for the duration
    of the test session so that any silent conversion of a non-numpy
    array-API array is logged instead of passing unnoticed.
    """
    if not _LOG_ARRAY_ESCAPES:
        yield
        return

    guard = _ReentrancyGuard()

    originals = {
        "np.asarray": np.asarray,
        "np.asanyarray": np.asanyarray,
        "np.ma.asanyarray": np.ma.asanyarray,
    }

    np.asarray = _make_escape_logging_wrapper(
        originals["np.asarray"], "numpy.asarray", guard
    )
    np.asanyarray = _make_escape_logging_wrapper(
        originals["np.asanyarray"], "numpy.asanyarray", guard
    )
    np.ma.asanyarray = _make_escape_logging_wrapper(
        originals["np.ma.asanyarray"], "numpy.ma.asanyarray", guard
    )

    try:
        yield
    finally:
        np.asarray = originals["np.asarray"]
        np.asanyarray = originals["np.asanyarray"]
        np.ma.asanyarray = originals["np.ma.asanyarray"]
