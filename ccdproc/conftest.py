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
#: by `pytest_array_api_escapes_config` below and driven by the ``CCDPROC_*``
#: environment variables that configuration names.
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

    _HAVE_ASTROPY_HEADER = True
except ImportError:
    _HAVE_ASTROPY_HEADER = False
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

#: The config of the running session, kept for `__getattr__`.
_CONFIG = None


def pytest_configure(config):
    """
    Remember the session config and switch on the astropy test header.

    Parameters
    ----------
    config : pytest.Config
        The pytest config object of this session.
    """
    global _CONFIG
    _CONFIG = config
    if _HAVE_ASTROPY_HEADER:
        config.option.astropy_header = True


def pytest_array_api_escapes_config():
    """
    Configure the array-API escape plugin for ccdproc.

    Returns
    -------
    dict
        ccdproc's settings for the plugin.

    Notes
    -----
    These live here rather than in ``pyproject.toml`` because this file
    ships with the tests: the settings then apply however the tests are run,
    including ``pytest --pyargs ccdproc`` against an installed copy from a
    directory with no ini file. The baseline path is relative to the
    directory of the ini file, or to the rootdir when there is none, so an
    installed copy of the baseline is never the one the ratchet enforces.

    The hookspec passes ``config``; this implementation does not need it,
    and pluggy lets an implementation accept a subset of the arguments.
    """
    return {
        "package": "ccdproc",
        "test_paths": ["ccdproc.tests"],
        "baseline": "ccdproc/tests/array_escape_baseline.txt",
        "env_prefix": "CCDPROC",
        "logger": "ccdproc.array_escape",
        "docs_url": "https://ccdproc.readthedocs.io/en/latest/array_api.html",
    }


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

    Both are read from the plugin's own selection, so the environment
    variables are parsed in one place. Resolving them lazily, through the
    module ``__getattr__`` of :pep:`562`, keeps the plugin package out of
    ``sys.modules`` until pytest has registered it -- see the note on
    ``pytest_plugins`` above -- and works whichever of this module's and the
    plugin's ``pytest_configure`` ran first: in the ``test`` tox factor this
    conftest is registered during collection, when pytest configures it
    before the plugins it names.
    """
    if name in ("testing_array_library", "testing_array_device"):
        from .tests._array_api_plugin import get_plugin

        plugin = get_plugin(_CONFIG) if _CONFIG is not None else None
        if plugin is None:
            raise RuntimeError(
                f"ccdproc.conftest.{name} is only available in a pytest "
                "session that has loaded the array-API escape plugin."
            )
        return plugin.namespace if name == "testing_array_library" else plugin.device
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
