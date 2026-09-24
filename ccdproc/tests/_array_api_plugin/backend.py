# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Backend selection: one array-API namespace per test run.

`select_backend` turns the ``<PREFIX>_ARRAY_LIBRARY`` and
``<PREFIX>_ARRAY_DEVICE`` environment variables into a
``(namespace_module, device)`` pair, which the `xp` and `xp_device` fixtures
hand to tests.

Notes
-----
One backend per run, chosen by an environment variable, rather than a
fixture that parametrizes every backend in a single session. That keeps a
test run's output attributable to one library and lets CI give each backend
its own job; a parametrized mode would be a compatible later addition.

The selection is memoized on the requested library and device so that a
package's ``conftest.py`` can call this at import time -- to expose the
namespace as a module attribute -- and the plugin can call it again in
``pytest_configure`` without importing anything twice or registering the
same header module twice.
"""

import os

#: Backend names accepted for ``<PREFIX>_ARRAY_LIBRARY``, normalized.
SUPPORTED_BACKENDS = ("numpy", "jax", "dask", "cupy", "array-api-strict")

_SELECTION_CACHE = {}


def normalize_backend_name(name):
    """Fold underscores and case so ``array_api_strict`` matches its spelling."""
    return str(name).lower().replace("_", "-")


def _add_header_module(label, module_name):
    """
    Add a module to the pytest-astropy-header banner, if it is installed.

    Notes
    -----
    The dependency is optional: without ``pytest_astropy_header`` installed
    there is simply no banner to add to.
    """
    try:
        from pytest_astropy_header.display import PYTEST_HEADER_MODULES
    except ImportError:
        return
    PYTEST_HEADER_MODULES[label] = module_name


def select_backend(env_prefix, environ=None, docs_url=None):
    """
    Resolve the array namespace and device for this test run.

    Parameters
    ----------
    env_prefix : str
        Prefix of the environment variables to read, e.g. ``"CCDPROC"``.
    environ : mapping, optional
        Environment to read, defaulting to ``os.environ``.
    docs_url : str, optional
        Documentation link appended to the error raised for an unsupported
        backend name.

    Returns
    -------
    namespace : module
        The array-API namespace to build test arrays with.
    device : object or None
        Device to pass as ``device=`` when creating arrays, or None for the
        library's usual device.

    Notes
    -----
    Leaving ``<PREFIX>_ARRAY_DEVICE`` unset selects the backend's testing
    default: the non-default ``"device1"`` for ``array-api-strict``, and None
    for every other library. Setting it to ``"default"`` selects the
    library's normal default device; any other value is passed to the
    library's ``Device`` constructor. ``array-api-strict`` on a non-default
    device makes ``numpy.asarray()`` raise, exactly as it would for a CuPy
    array resident on a GPU, which makes it a CPU-only proxy for catching
    silent conversions to NumPy.
    """
    environ = os.environ if environ is None else environ
    library = environ.get(f"{env_prefix}_ARRAY_LIBRARY", "numpy").lower()
    device_name = environ.get(f"{env_prefix}_ARRAY_DEVICE")

    cache_key = (library, device_name)
    if cache_key in _SELECTION_CACHE:
        return _SELECTION_CACHE[cache_key]

    device = None

    match normalize_backend_name(library):
        case "numpy":
            import array_api_compat.numpy as namespace

        case "jax":
            import jax.numpy as namespace

            _add_header_module("jax", "jax")

        case "dask":
            import array_api_compat.dask.array as namespace

            _add_header_module("dask", "dask")

        case "cupy":
            import array_api_compat.cupy as namespace

            _add_header_module("cupy", "cupy")

        case "array-api-strict":
            import array_api_strict as namespace

            _add_header_module("array_api_strict", "array_api_strict")

            name = device_name if device_name is not None else "device1"
            if name.lower() == "default":
                # The library's normal CPU device, on which numpy.asarray()
                # succeeds.
                device = namespace.Device("CPU_DEVICE")
            else:
                device = namespace.Device(name)

        case _:
            supported = ", ".join(SUPPORTED_BACKENDS)
            message = (
                f"Unsupported array library: {library}. "
                f"Set {env_prefix}_ARRAY_LIBRARY to one of: {supported}."
            )
            if docs_url:
                message += f" See {docs_url}."
            raise ValueError(message)

    _SELECTION_CACHE[cache_key] = (namespace, device)
    return namespace, device
