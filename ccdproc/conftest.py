# Licensed under a 3-clause BSD style license - see LICENSE.rst

# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from .tests.pytest_fixtures import (
    triage_setup,  # noqa: F401 this is used in tests
)

#: Plugins loaded for every ccdproc test session.
#:
#: ``ccdproc.tests._array_api_plugin`` is the array-API test tooling: backend
#: selection, the ``backend_skip``/``backend_xfail`` markers, the escape
#: logger, failure triage and the escape-baseline ratchet. It is configured
#: entirely through the ``array_api_escapes_*`` ini options in
#: ``pyproject.toml`` and the ``CCDPROC_*`` environment variables those name.
#:
#: ``pytester`` supplies the fixture of the same name, used by
#: ``ccdproc/tests/test_array_api_plugin.py`` to run the plugin end to end
#: against a throwaway package.
#:
#: Nothing in this module may import ``ccdproc.tests._array_api_plugin`` at
#: import time: pytest rewrites assertions in the plugins named here, and a
#: plugin already in ``sys.modules`` when it is registered raises
#: ``PytestAssertRewriteWarning``. Hence the lazy ``__getattr__`` below.
pytest_plugins = [
    "pytester",
    "ccdproc.tests._array_api_plugin",
]

try:
    # When the pytest_astropy_header package is installed
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

    def pytest_configure(config):
        config.option.astropy_header = True

except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}


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


#: Documentation for the supported array libraries, quoted in the error
#: raised for an unknown ``CCDPROC_ARRAY_LIBRARY``.
_ARRAY_API_DOCS = "https://ccdproc.readthedocs.io/en/latest/array_api.html"

_ARRAY_BACKEND = None


def _array_backend():
    """
    Return the ``(namespace, device)`` pair selected for this test run.

    Notes
    -----
    The selection is memoized inside the plugin, so this call and the
    plugin's own call in ``pytest_configure`` resolve to the same objects no
    matter which of them runs first.
    """
    global _ARRAY_BACKEND
    if _ARRAY_BACKEND is None:
        from .tests._array_api_plugin.backend import select_backend

        _ARRAY_BACKEND = select_backend("CCDPROC", docs_url=_ARRAY_API_DOCS)
    return _ARRAY_BACKEND


def __getattr__(name):
    """
    Provide the ``testing_array_library`` / ``testing_array_device`` attributes.

    Notes
    -----
    The array library and device for this run are also available as the
    session-scoped ``xp`` and ``xp_device`` fixtures the plugin provides,
    which is the preferred way to reach them in new tests. They stay
    available as module attributes because most of ccdproc's test modules do
    ``from ccdproc.conftest import testing_array_library as xp`` at import
    time; migrating those to the fixtures is a separate change.

    Resolving them lazily, through the module ``__getattr__`` of :pep:`562`,
    keeps the plugin package out of ``sys.modules`` until pytest has
    registered it -- see the note on ``pytest_plugins`` above.
    """
    if name == "testing_array_library":
        return _array_backend()[0]
    if name == "testing_array_device":
        return _array_backend()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
