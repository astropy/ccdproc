# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A pytest plugin for packages adopting the Python array API standard.

The plugin gives a test suite five things:

* one array backend per run, selected with ``<PREFIX>_ARRAY_LIBRARY``, and
  exposed through the session-scoped ``xp`` and ``xp_device`` fixtures;
* ``backend_skip`` / ``backend_xfail`` markers, applied only when the run
  uses one of the named backends;
* an escape logger (``<PREFIX>_LOG_ARRAY_ESCAPES``) that reports silent
  coercions of foreign array-API arrays back to NumPy;
* failure triage (``<PREFIX>_TRIAGE_ESCAPES``) grouping test failures by the
  in-package call site that caused them;
* a checked-in baseline of known coercion sites that may only shrink
  (``<PREFIX>_ENFORCE_ESCAPE_BASELINE`` /
  ``<PREFIX>_WRITE_ESCAPE_BASELINE``).

Nothing here is specific to the package being tested. Everything the plugin
needs comes from the ini options declared in `.config` -- which package's
frames are library frames, which are test frames, where the baseline lives,
what the environment-variable prefix is, and which logger to write to -- so
this directory can be lifted into a stand-alone distribution (working name
``pytest-array-api-escapes``, provisional) without edits.

Load it from the package's own ``conftest.py``::

    pytest_plugins = ["mypackage.tests._array_api_plugin"]

and configure it in ``pyproject.toml``::

    [tool.pytest.ini_options]
    array_api_escapes_package = "mypackage"
    array_api_escapes_test_paths = ["mypackage.tests"]
    array_api_escapes_baseline = "mypackage/tests/array_escape_baseline.txt"
    array_api_escapes_env_prefix = "MYPACKAGE"
    array_api_escapes_logger = "mypackage.array_escape"

Notes
-----
A session that sets none of those options still runs: the markers and the
``xp`` fixtures work (defaulting to NumPy), and the features that need to
classify stack frames refuse to start, with a message naming the missing
option, rather than blaming the wrong frames.
"""

import pytest

from . import baseline as _baseline
from .backend import normalize_backend_name, select_backend
from .config import ENV_ARRAY_LIBRARY, add_ini_options, build_settings, env_truthy
from .escape_logger import EscapeLog, foreign_namespace
from .markers import apply_backend_markers, register_markers
from .triage import FailureTriage, FrameClassifier

__all__ = [
    "ArrayApiEscapePlugin",
    "EscapeLog",
    "FailureTriage",
    "FrameClassifier",
    "env_truthy",
    "foreign_namespace",
    "get_plugin",
    "normalize_backend_name",
    "select_backend",
]

#: Where the per-session runtime lives on ``config``.
PLUGIN_KEY = pytest.StashKey()


class ArrayApiEscapePlugin:
    """
    Everything this plugin needs for one test session.

    Parameters
    ----------
    settings : `.config.Settings`
        The resolved configuration.

    Notes
    -----
    Holding the state on an instance rather than in module globals is what
    lets the same process run the plugin twice -- which is exactly what the
    plugin's own ``pytester`` tests do -- and keeps the frame roots a
    configured value instead of something derived from ``__file__``.
    """

    def __init__(self, settings):
        self.settings = settings
        self.classifier = FrameClassifier(settings)
        self.escape_log = EscapeLog(settings, self.classifier)
        self.triage = FailureTriage(settings, self.classifier)
        self.baseline = _baseline.Baseline(settings, self.classifier, self.escape_log)
        namespace, device = select_backend(settings.env_prefix)
        self.namespace = namespace
        self.device = device
        self.active_backend = normalize_backend_name(
            settings.env(ENV_ARRAY_LIBRARY, "numpy")
        )


def get_plugin(config):
    """
    Return the `ArrayApiEscapePlugin` built for ``config``, or None.

    Notes
    -----
    None only when ``pytest_configure`` has not run or raised, which pytest
    can still follow with a terminal summary; the hooks below then do
    nothing rather than masking the original error with a `KeyError`.
    """
    return config.stash.get(PLUGIN_KEY, None)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    """Declare the plugin's ini options."""
    add_ini_options(parser)


def pytest_configure(config):
    """Build the session runtime and register the two backend markers."""
    settings = build_settings(config)
    config.stash[PLUGIN_KEY] = ArrayApiEscapePlugin(settings)
    register_markers(config, settings)


def pytest_sessionstart(session):
    """Reject unusable combinations of environment variables and ini options."""
    plugin = get_plugin(session.config)
    if plugin is not None:
        _baseline.check_usage(session.config, plugin.settings, plugin.classifier)


def pytest_collection_modifyitems(config, items):
    """Apply ``backend_skip`` / ``backend_xfail`` for the active backend."""
    plugin = get_plugin(config)
    if plugin is not None:
        apply_backend_markers(items, plugin.active_backend)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Record the escape site of every failing test.

    Notes
    -----
    Runs as a wrapper around report generation for each test phase. With a
    new-style wrapper, ``yield`` hands back the ``TestReport`` itself (not a
    ``pluggy.Result``), and the wrapper must return it.
    """
    report = yield
    plugin = get_plugin(item.config)
    if plugin is not None:
        plugin.triage.record_report(item, call, report)
    return report


def pytest_terminal_summary(terminalreporter):
    """Print the triage, escape-log and baseline summaries."""
    plugin = get_plugin(terminalreporter.config)
    if plugin is None:
        return
    plugin.triage.report(terminalreporter)
    plugin.escape_log.report(terminalreporter)
    plugin.baseline.report(terminalreporter)


def pytest_sessionfinish(session, exitstatus):
    """
    Regenerate or enforce the baseline ratchet.

    Notes
    -----
    In write mode the baseline file is rewritten from the escapes observed
    this run, refusing to truncate it if nothing was observed. In enforce
    mode the session exit status is forced nonzero when a new library escape
    appeared, so CI fails on a regression; a pre-existing nonzero status
    (real test failures) is left untouched.
    """
    plugin = get_plugin(session.config)
    if plugin is None:
        return
    if plugin.settings.write_baseline:
        plugin.baseline.write()
        return
    if (
        plugin.settings.enforce_baseline
        and exitstatus == 0
        and plugin.baseline.new_escapes()
    ):
        session.exitstatus = 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def xp(pytestconfig):
    """The array-API namespace selected for this test run."""
    return get_plugin(pytestconfig).namespace


@pytest.fixture(scope="session")
def xp_device(pytestconfig):
    """The device to create test arrays on, or None for the library default."""
    return get_plugin(pytestconfig).device


@pytest.fixture(autouse=True, scope="session")
def _log_array_escapes(pytestconfig):
    """
    Monkeypatch NumPy's coercion entry points for the whole session.

    Notes
    -----
    Active only when ``<PREFIX>_LOG_ARRAY_ESCAPES`` is truthy. While active,
    any silent conversion of a foreign array-API array to NumPy is logged
    and tallied instead of passing unnoticed.
    """
    escape_log = get_plugin(pytestconfig).escape_log
    if not escape_log.active:
        yield
        return
    restore = escape_log.patch()
    try:
        yield
    finally:
        restore()
