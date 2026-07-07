# Licensed under a 3-clause BSD style license - see LICENSE.rst

# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.
import logging
import os
import traceback

import array_api_compat  # noqa: F401
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
    pytest_terminal_summary,  # noqa: F401 pytest hook, used via attribute lookup
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
    for item in items:
        for marker in item.iter_markers(name="backend_skip"):
            backends = {_normalize_backend_name(b) for b in marker.args}
            if _ACTIVE_BACKEND in backends:
                reason = marker.kwargs.get(
                    "reason", f"skipped for array backend {_ACTIVE_BACKEND!r}"
                )
                item.add_marker(pytest.mark.skip(reason=reason))
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
# an array that reports its own `__array_namespace__` and that namespace is
# not numpy.
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


def _is_foreign_array(obj):
    """
    Return True if obj is an array from a non-numpy array-API library.

    numpy arrays (including np.ma masked arrays) are never foreign. Anything
    else that implements the array-API protocol marker __array_namespace__()
    is foreign unless its namespace is numpy or array_api_compat's numpy
    wrapper -- i.e. exactly the arrays whose conversion to numpy would fail
    (CuPy on GPU) or silently densify/transfer (dask, jax). The try/except
    guards against objects whose __array_namespace__() raises; those are
    treated as not foreign so the logger never breaks the call it wraps.
    """
    if isinstance(obj, np.ndarray):
        return False

    get_namespace = getattr(obj, "__array_namespace__", None)
    if get_namespace is None:
        return False

    try:
        namespace = get_namespace()
    except Exception:
        return False

    return namespace not in _NUMPY_LIKE_NAMESPACES


def _describe_escape_site():
    """
    Return a short "file:line function" string for the innermost stack
    frame that is inside ccdproc but not inside ccdproc's own test suite,
    for use in escape log messages.
    """
    from .tests._escape_triage import locate_escape_site

    # Drop this function's own frame before searching.
    frames = traceback.extract_stack()[:-1]
    site = locate_escape_site(frames)
    if site is None:
        return "<unknown location>"
    return f"{site.filename}:{site.lineno} {site.name}"


class _ReentrancyGuard:
    """Small helper to keep our wrappers from recursing into themselves."""

    def __init__(self):
        self.active = False


def _make_escape_logging_wrapper(original, funcname, guard):
    def wrapper(*args, **kwargs):
        if not guard.active and args:
            guard.active = True
            try:
                obj = args[0]
                if _is_foreign_array(obj):
                    site = _describe_escape_site()
                    _escape_logger.warning(
                        "array-API escape: %s() called on a %r array "
                        "(namespace=%r) at %s",
                        funcname,
                        type(obj),
                        obj.__array_namespace__(),
                        site,
                    )
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
