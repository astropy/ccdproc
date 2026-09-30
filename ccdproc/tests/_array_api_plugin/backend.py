# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Backend selection: one array-API namespace per test run.

`select_backend` turns the ``<PREFIX>_ARRAY_LIBRARY`` and
``<PREFIX>_ARRAY_DEVICE`` environment variables into a
``(name, namespace_module, device)`` triple, from which the `xp` and
`xp_device` fixtures hand the namespace and device to tests.

Notes
-----
One backend per run, chosen by an environment variable, rather than a
fixture that parametrizes every backend in a single session. That keeps a
test run's output attributable to one library and lets CI give each backend
its own job; a parametrized mode would be a compatible later addition.

The plugin selects the backend once, when it builds its per-session state in
``pytest_configure``. Anything else that needs the selection -- such as a
package ``conftest.py`` exposing the namespace as a module attribute --
reads it from the plugin rather than selecting again, so there is a single
place where the environment variables are parsed.
"""

import pytest

from .config import ENV_ARRAY_DEVICE, ENV_ARRAY_LIBRARY

#: Backend names accepted for ``<PREFIX>_ARRAY_LIBRARY``, normalized.
SUPPORTED_BACKENDS = ("numpy", "jax", "dask", "cupy", "array-api-strict")


def normalize_backend_name(name):
    """
    Fold underscores and case so ``array_api_strict`` matches its spelling.

    Parameters
    ----------
    name : str
        A backend name as written in an environment variable or a marker.

    Returns
    -------
    str
        The name lower-cased, with underscores replaced by hyphens.
    """
    return str(name).lower().replace("_", "-")


def _add_header_module(label, module_name):
    """
    Add a module to the pytest-astropy-header banner, if it is installed.

    Parameters
    ----------
    label : str
        The name shown in the banner.
    module_name : str
        The import name whose version the banner reports.

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


def select_backend(settings):
    """
    Resolve the array library, namespace and device for this test run.

    Parameters
    ----------
    settings : `.config.Settings`
        Supplies the environment to read, the variable prefix, and the
        documentation link quoted in the error for an unsupported library.

    Returns
    -------
    name : str
        Normalized name of the selected library, one of `SUPPORTED_BACKENDS`.
    namespace : module
        The array-API namespace to build test arrays with.
    device : object or None
        Device to pass as ``device=`` when creating arrays, or None for the
        library's usual device.

    Raises
    ------
    pytest.UsageError
        If ``<PREFIX>_ARRAY_LIBRARY`` names an unsupported library.

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
    library = settings.env(ENV_ARRAY_LIBRARY, "numpy")
    device_name = settings.env(ENV_ARRAY_DEVICE, None)
    name = normalize_backend_name(library)

    device = None

    match name:
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

            requested = device_name if device_name is not None else "device1"
            if requested.lower() == "default":
                # The library's normal CPU device, on which numpy.asarray()
                # succeeds.
                device = namespace.Device("CPU_DEVICE")
            else:
                device = namespace.Device(requested)

        case _:
            supported = ", ".join(SUPPORTED_BACKENDS)
            message = (
                f"Unsupported array library: {library}. "
                f"Set {settings.env_name(ENV_ARRAY_LIBRARY)} to one of: "
                f"{supported}."
            )
            if settings.docs_url:
                message += f" See {settings.docs_url}."
            raise pytest.UsageError(message)

    return name, namespace, device
