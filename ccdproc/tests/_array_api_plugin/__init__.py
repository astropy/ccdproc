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
needs comes from the package's configuration -- which package's frames are
library frames, which are test frames, where the baseline lives, what the
environment-variable prefix is, and which logger to write to -- so this
directory can be lifted into a stand-alone distribution (working name
``pytest-array-api-escapes``, provisional) without edits.

Load it from the package's own ``conftest.py`` and configure it there, by
implementing the `~.hooks.pytest_array_api_escapes_config` hook::

    pytest_plugins = ["mypackage.tests._array_api_plugin"]


    def pytest_array_api_escapes_config(config):
        return {
            "package": "mypackage",
            "test_paths": ["mypackage.tests"],
            "baseline": "mypackage/tests/array_escape_baseline.txt",
            "env_prefix": "MYPACKAGE",
            "logger": "mypackage.array_escape",
            "docs_url": "https://mypackage.example.org/array_api.html",
        }

Each of the first five values can also be set, or overridden, by the ini
option of the same name with an ``array_api_escapes_`` prefix, e.g.
``array_api_escapes_package`` in ``[tool.pytest.ini_options]``.

Notes
-----
The hook is the primary route because the ``conftest.py`` travels with the
tests: it configures the plugin even when the tests run against an installed
copy of the package from a directory with no ini file.

A session that configures nothing still runs: the markers and the ``xp``
fixtures work (defaulting to NumPy), and the features that need to classify
stack frames refuse to start, with a message naming the missing setting,
rather than blaming the wrong frames.
"""

import pytest

from . import baseline as _baseline
from .backend import normalize_backend_name, select_backend
from .config import add_ini_options, build_settings, env_truthy
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
        """Build the per-session state from ``settings``."""
        self.settings = settings
        self.classifier = FrameClassifier(settings)
        self.escape_log = EscapeLog(settings, self.classifier)
        self.triage = FailureTriage(settings, self.classifier)
        self.baseline = _baseline.Baseline(settings, self.classifier, self.escape_log)
        self.active_backend, self.namespace, self.device = select_backend(settings)


def get_plugin(config):
    """
    Return the `ArrayApiEscapePlugin` built for ``config``, or None.

    Parameters
    ----------
    config : pytest.Config
        The pytest config object of this session.

    Returns
    -------
    ArrayApiEscapePlugin or None
        The session's plugin state, or None if it was never built.

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


def pytest_addhooks(pluginmanager):
    """
    Register the plugin's configuration hook.

    Parameters
    ----------
    pluginmanager : pytest.PytestPluginManager
        The plugin manager to add `~.hooks.pytest_array_api_escapes_config`
        to.
    """
    from . import hooks

    pluginmanager.add_hookspecs(hooks)


def pytest_addoption(parser):
    """
    Declare the plugin's ini options.

    Parameters
    ----------
    parser : pytest.Parser
        The parser to add the options to.
    """
    add_ini_options(parser)


def pytest_configure(config):
    """
    Build the session runtime, check its usage and register the markers.

    Parameters
    ----------
    config : pytest.Config
        The pytest config object of this session.

    Raises
    ------
    pytest.UsageError
        If the environment variables and settings cannot work together (see
        `.baseline.check_usage`).

    Notes
    -----
    The usage checks run here rather than in ``pytest_sessionstart`` because
    ``pytest_configure`` is a historic hook: pytest also calls it for a plugin
    registered after the session has started. That happens when the conftest
    that loads this plugin is found only during collection, as in a
    ``--pyargs`` run from outside the source tree, and the checks must not be
    skipped there.
    """
    settings = build_settings(config)
    plugin = ArrayApiEscapePlugin(settings)
    _baseline.check_usage(config, settings, plugin.classifier)
    config.stash[PLUGIN_KEY] = plugin
    register_markers(config, settings)


def pytest_collection_modifyitems(config, items):
    """
    Apply ``backend_skip`` / ``backend_xfail`` for the active backend.

    Parameters
    ----------
    config : pytest.Config
        The pytest config object of this session.
    items : list of pytest.Item
        The collected test items, modified in place.
    """
    plugin = get_plugin(config)
    if plugin is not None:
        apply_backend_markers(items, plugin.active_backend)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Record the escape site of every failing test.

    Parameters
    ----------
    item : pytest.Item
        The test the report is for.
    call : pytest.CallInfo
        The result of the test phase, including any exception.

    Yields
    ------
    None
        Control passes to the inner implementations of the hook, and the
        ``TestReport`` they build is sent back in.

    Returns
    -------
    pytest.TestReport
        The report, unchanged.

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
    """
    Print the triage, escape-log and baseline summaries.

    Parameters
    ----------
    terminalreporter : _pytest.terminal.TerminalReporter
        The reporter to write the summary sections to.
    """
    plugin = get_plugin(terminalreporter.config)
    if plugin is None:
        return
    plugin.triage.report(terminalreporter)
    plugin.escape_log.report(terminalreporter)
    plugin.baseline.report(terminalreporter)


def pytest_sessionfinish(session, exitstatus):
    """
    Regenerate or enforce the baseline ratchet.

    Parameters
    ----------
    session : pytest.Session
        The finished session; its ``exitstatus`` may be set to 1.
    exitstatus : int or pytest.ExitCode
        The exit status the session would otherwise end with.

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
    """
    Provide the array-API namespace selected for this test run.

    Parameters
    ----------
    pytestconfig : pytest.Config
        The pytest config object of this session.

    Returns
    -------
    module
        The namespace to build test arrays with.
    """
    return get_plugin(pytestconfig).namespace


@pytest.fixture(scope="session")
def xp_device(pytestconfig):
    """
    Provide the device to create test arrays on.

    Parameters
    ----------
    pytestconfig : pytest.Config
        The pytest config object of this session.

    Returns
    -------
    object or None
        The device to pass as ``device=``, or None for the library default.
    """
    return get_plugin(pytestconfig).device


@pytest.fixture(autouse=True, scope="session")
def _log_array_escapes(pytestconfig):
    """
    Monkeypatch NumPy's coercion entry points for the whole session.

    Parameters
    ----------
    pytestconfig : pytest.Config
        The pytest config object of this session.

    Yields
    ------
    None
        The session runs while the wrappers are installed.

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
