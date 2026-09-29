# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the array-API escape plugin in ``ccdproc.tests._array_api_plugin``."""

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, get_ident
from traceback import FrameSummary

import array_api_compat.numpy
import pytest

import ccdproc
from ccdproc.tests._array_api_plugin import escape_logger
from ccdproc.tests._array_api_plugin.backend import select_backend
from ccdproc.tests._array_api_plugin.baseline import check_usage
from ccdproc.tests._array_api_plugin.config import (
    INI_BASELINE,
    INI_ENV_PREFIX,
    INI_LOGGER,
    INI_PACKAGE,
    INI_TEST_PATHS,
    Settings,
    build_settings,
)
from ccdproc.tests._array_api_plugin.escape_logger import EscapeLog, ReentrancyGuard
from ccdproc.tests._array_api_plugin.triage import FrameClassifier

CCDPROC_ROOT = os.path.dirname(os.path.abspath(ccdproc.__file__))


def use_this_ccdproc_in_subprocess(monkeypatch):
    """
    Make a ``pytester`` subprocess import the ccdproc this session is testing.

    Notes
    -----
    The end-to-end tests load the plugin in a fresh pytest process by its
    import name, so that process has to find the same copy of ccdproc. In a
    source-tree run the installed ccdproc may be an editable install pointing
    somewhere else entirely, so put this copy's parent directory first on
    ``PYTHONPATH``.
    """
    existing = os.environ.get("PYTHONPATH", "")
    parent = os.path.dirname(CCDPROC_ROOT)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([parent, existing]).rstrip(os.pathsep)
    )


class StubHook:
    """
    Stand-in for ``config.hook``, answering the plugin's configuration hook.

    Parameters
    ----------
    result : dict or None
        What ``pytest_array_api_escapes_config`` returns.
    """

    def __init__(self, result):
        self.result = result
        self.config = None

    def pytest_array_api_escapes_config(self, config):
        self.config = config
        return self.result


class StubConfig:
    """
    Minimal stand-in for pytest's ``Config`` for `build_settings`.

    Notes
    -----
    `build_settings` only needs ``getini``, ``hook``, ``inipath`` and
    ``rootpath``. Using a stub instead of a real config keeps these tests
    independent of how the ini values and the hook result reach pytest; the
    end-to-end ``pytester`` tests below cover the real ini-file and
    conftest-hook paths.
    """

    def __init__(self, rootpath, hook_result=None, inipath=None, **ini):
        self.rootpath = rootpath
        self.inipath = inipath
        self.hook = StubHook(hook_result)
        self._ini = ini

    def getini(self, name):
        if name == INI_TEST_PATHS:
            return self._ini.get(name, [])
        return self._ini.get(name, "")


def unconfigured_settings():
    """Settings for a session that set none of the plugin's ini options."""
    return Settings(
        package_name="",
        package_root=None,
        test_roots=(),
        baseline_path=None,
        env_prefix="TESTPKG",
        logger_name="testpkg.array_escape",
        environ={},
    )


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------


def test_package_root_is_resolved_by_importing_the_package(tmp_path):
    """
    Pin that the library root comes from importing the configured package.

    Notes
    -----
    The predecessor of this plugin anchored frame classification on its own
    ``__file__``. That breaks when the tests run against an installed copy of
    the package (``pytest --pyargs``, which is what the ``test`` tox factor
    does): the frames to classify then live in ``site-packages`` while the
    anchor points wherever the plugin happened to be imported from.
    """
    settings = build_settings(StubConfig(tmp_path, **{INI_PACKAGE: "ccdproc"}))
    assert settings.package_root == CCDPROC_ROOT
    assert FrameClassifier(settings).is_library_frame(
        os.path.join(CCDPROC_ROOT, "core.py")
    )


def test_test_paths_default_to_the_packages_tests_subpackage(tmp_path):
    """Pin the default of ``array_api_escapes_test_paths``, which ccdproc relies on."""
    settings = build_settings(StubConfig(tmp_path, **{INI_PACKAGE: "ccdproc"}))
    classifier = FrameClassifier(settings)
    assert classifier.is_test_frame(os.path.join(CCDPROC_ROOT, "tests", "test_gain.py"))
    assert not classifier.is_library_frame(
        os.path.join(CCDPROC_ROOT, "tests", "test_gain.py")
    )


@pytest.mark.parametrize(
    "entry",
    [
        "ccdproc.utils",
        "utils",
        "ccdproc/utils",
        "utils/sample_directory.py",
        "ccdproc.utils.sample_directory",
        "utils.sample_directory",
    ],
)
def test_custom_test_paths_are_honoured(tmp_path, entry):
    """
    Pin that a configured test path, dotted or package-relative, is honoured.

    Notes
    -----
    Both spellings must work because a stand-alone plugin cannot know
    whether a project thinks of its test helpers as importable modules or as
    directories, and a dotted name must work for a single module as well as
    for a package. ``ccdproc.utils`` is used here only as a directory that is
    definitely not the default ``ccdproc.tests``.
    """
    settings = build_settings(
        StubConfig(tmp_path, **{INI_PACKAGE: "ccdproc", INI_TEST_PATHS: [entry]})
    )
    classifier = FrameClassifier(settings)
    helper = os.path.join(CCDPROC_ROOT, "utils", "sample_directory.py")
    assert classifier.is_test_frame(helper)
    assert not classifier.is_library_frame(helper)
    # The configured list replaces the default, it does not extend it.
    assert not classifier.is_test_frame(
        os.path.join(CCDPROC_ROOT, "tests", "test_gain.py")
    )


def test_package_conftest_is_always_a_test_frame(tmp_path):
    """
    Pin that ``<package>/conftest.py`` counts as a test frame unconditionally.

    Notes
    -----
    The conftest hosts the test infrastructure that loads this plugin, so it
    is never an actionable escape site. It must not depend on a project
    remembering to list it in ``array_api_escapes_test_paths``, so the check
    here configures a test path that excludes it.
    """
    settings = build_settings(
        StubConfig(
            tmp_path, **{INI_PACKAGE: "ccdproc", INI_TEST_PATHS: ["ccdproc.utils"]}
        )
    )
    classifier = FrameClassifier(settings)
    conftest = os.path.join(CCDPROC_ROOT, "conftest.py")
    assert classifier.is_test_frame(conftest)
    assert not classifier.is_library_site("conftest.py")


def test_env_prefix_defaults_to_the_package_name(tmp_path):
    """Pin that an unset ``array_api_escapes_env_prefix`` follows the package name."""
    settings = build_settings(StubConfig(tmp_path, **{INI_PACKAGE: "ccdproc"}))
    assert settings.env_name("ARRAY_LIBRARY") == "CCDPROC_ARRAY_LIBRARY"
    assert settings.logger_name == "ccdproc.array_escape"


def test_env_prefix_override_renames_every_environment_variable(tmp_path):
    """
    Pin that the ini prefix, not the package name, names the environment variables.

    Notes
    -----
    This is what lets a project keep historical variable names that do not
    match its import name, and it is the knob the end-to-end test below uses
    to prove the plugin is not wired to ccdproc.
    """
    environ = {"OTHER_LOG_ARRAY_ESCAPES": "yes", "CCDPROC_LOG_ARRAY_ESCAPES": "1"}
    settings = build_settings(
        StubConfig(
            tmp_path,
            **{
                INI_PACKAGE: "ccdproc",
                INI_ENV_PREFIX: "OTHER",
                INI_LOGGER: "other.escapes",
            },
        ),
        environ=environ,
    )
    assert settings.env_name("TRIAGE_ESCAPES") == "OTHER_TRIAGE_ESCAPES"
    assert settings.log_escapes
    assert settings.logger_name == "other.escapes"

    del environ["OTHER_LOG_ARRAY_ESCAPES"]
    settings = build_settings(
        StubConfig(tmp_path, **{INI_PACKAGE: "ccdproc", INI_ENV_PREFIX: "OTHER"}),
        environ=environ,
    )
    assert not settings.log_escapes


def test_baseline_path_is_resolved_against_the_ini_file_directory(tmp_path):
    """
    Pin that the baseline path is relative to the ini file's directory.

    Notes
    -----
    Anchoring on the plugin's own location pointed the ratchet at whichever
    copy of the package was imported; relative to the ini file it always
    names the checked-in file, including when the tests run against an
    installed copy from a temporary directory. The anchor is the ini file's
    directory rather than the rootdir because ``--rootdir`` moves the latter
    but not the former, and the documentation (like pytest's own
    path-valued ini options) promises the ini file's directory. Only when
    there is no ini file at all, as for a package configured solely by its
    conftest hook, does the rootdir stand in.
    """
    project = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"
    ini = {INI_PACKAGE: "ccdproc", INI_BASELINE: "ccdproc/tests/baseline.txt"}

    settings = build_settings(
        StubConfig(elsewhere, inipath=project / "pyproject.toml", **ini)
    )
    assert settings.baseline_path == str(project / "ccdproc" / "tests" / "baseline.txt")

    settings = build_settings(StubConfig(elsewhere, **ini))
    assert settings.baseline_path == str(
        elsewhere / "ccdproc" / "tests" / "baseline.txt"
    )


def test_hook_configures_the_plugin_and_ini_options_override_it(tmp_path):
    """
    Pin the precedence of the two configuration sources: ini, then hook.

    Notes
    -----
    A package configures the plugin through the conftest hook, because its
    ``conftest.py`` ships with the tests and its ini file does not. The ini
    options must still win, value by value, so a project (or a one-off
    ``-o`` on the command line) can override a single setting without
    restating the rest.
    """
    hook_result = {
        "package": "ccdproc",
        "env_prefix": "HOOKED",
        "logger": "hooked.escapes",
        "docs_url": "https://example.org/docs",
    }
    config = StubConfig(tmp_path, hook_result=hook_result)
    settings = build_settings(config)
    assert config.hook.config is config
    assert settings.package_root == CCDPROC_ROOT
    assert settings.env_prefix == "HOOKED"
    assert settings.logger_name == "hooked.escapes"
    assert settings.docs_url == "https://example.org/docs"

    settings = build_settings(
        StubConfig(tmp_path, hook_result=hook_result, **{INI_ENV_PREFIX: "INI"})
    )
    assert settings.env_prefix == "INI"
    assert settings.logger_name == "hooked.escapes"


def test_hook_result_with_an_unknown_key_is_rejected(tmp_path):
    """
    Pin that a misspelled key in the hook's result is an error, not ignored.

    Notes
    -----
    An ignored key would silently fall back to the default, e.g. a
    ``"basline"`` typo would leave the ratchet with no baseline file and a
    ``"prefix"`` typo would switch every environment variable's name.
    """
    config = StubConfig(tmp_path, hook_result={"package": "ccdproc", "prefix": "X"})
    with pytest.raises(pytest.UsageError, match="prefix"):
        build_settings(config)


def test_backend_selection_reads_the_settings_environment():
    """
    Pin that the backend is chosen from the settings' environment and prefix.

    Notes
    -----
    The markers use the backend name and the fixtures use the namespace, so
    both must come from one reading of the same variable. Reading
    ``os.environ`` or a hard-coded prefix for either one splits them: the
    markers then act for one backend while the tests run on another.
    """
    settings = Settings(
        "", None, (), None, "X", "x", environ={"X_ARRAY_LIBRARY": "NumPy"}
    )
    name, namespace, device = select_backend(settings)
    assert name == "numpy"
    assert namespace is array_api_compat.numpy
    assert device is None


def test_unsupported_backend_is_a_usage_error_naming_the_docs():
    """
    Pin that an unknown array library ends the session with a clean error.

    Notes
    -----
    The selection runs in ``pytest_configure``, where any exception other
    than `pytest.UsageError` is reported as an ``INTERNALERROR`` traceback.
    The message must name the variable to fix and link the documentation
    that lists the supported libraries.
    """
    settings = Settings(
        "",
        None,
        (),
        None,
        "X",
        "x",
        environ={"X_ARRAY_LIBRARY": "foo"},
        docs_url="https://example.org/docs",
    )
    with pytest.raises(pytest.UsageError) as excinfo:
        select_backend(settings)
    message = str(excinfo.value)
    assert "X_ARRAY_LIBRARY" in message
    assert "https://example.org/docs" in message


def test_features_needing_frame_classification_refuse_to_run_unconfigured(tmp_path):
    """
    Pin the error when a feature is switched on without a configured package.

    Notes
    -----
    Without a package the plugin cannot tell library frames from test frames,
    so it would blame the wrong code and, in enforce mode, pass vacuously. A
    session that switches nothing on is still allowed to run; that is covered
    end to end by `test_session_without_ini_options_still_runs`.
    """
    settings = build_settings(
        StubConfig(tmp_path), environ={"ARRAY_API_TRIAGE_ESCAPES": "1"}
    )
    classifier = FrameClassifier(settings)
    assert not classifier.configured
    # No baseline mode is requested, so check_usage never touches the config.
    with pytest.raises(pytest.UsageError, match=INI_PACKAGE):
        check_usage(None, settings, classifier)


# ---------------------------------------------------------------------------
# Escape logger internals
# ---------------------------------------------------------------------------


def test_escape_site_never_blames_the_plugin_itself(tmp_path):
    """
    Pin that the plugin's own frames are skipped when blaming an escape.

    Notes
    -----
    The escape logger locates the site from inside its NumPy wrapper, so the
    innermost frames of the stack are always the plugin's. When no library
    frame is on the stack (e.g. astropy coercing a foreign mask called
    straight from a test body) the innermost *package* frame wins, and with
    the plugin inside ``ccdproc/tests`` that was the plugin's
    ``locate_site`` rather than the test. Once the plugin lives outside the
    package the same stack would fall through to the innermost frame
    overall, again the plugin's. Both layouts must blame the test.
    """
    plugin_dir = os.path.dirname(os.path.abspath(escape_logger.__file__))
    test_file = os.path.join(CCDPROC_ROOT, "tests", "test_ccdproc.py")
    test_frame = FrameSummary(test_file, 1103, "test_block_average")
    stack = [
        FrameSummary("/elsewhere/_pytest/python.py", 10, "pytest_pyfunc_call"),
        test_frame,
        FrameSummary("/elsewhere/astropy/nddata/compat.py", 171, "mask"),
        FrameSummary(os.path.join(plugin_dir, "escape_logger.py"), 190, "wrapper"),
        FrameSummary(os.path.join(plugin_dir, "escape_logger.py"), 153, "locate_site"),
    ]

    settings = build_settings(StubConfig(tmp_path, **{INI_PACKAGE: "ccdproc"}))
    assert FrameClassifier(settings).locate_escape_site(stack) is test_frame

    # Unconfigured, nothing is a package frame and step 3 picks the innermost
    # frame overall, which must not be the plugin's.
    unconfigured = FrameClassifier(unconfigured_settings())
    assert unconfigured.locate_escape_site(stack) is stack[2]

    assert unconfigured.locate_escape_site(stack[3:]) is None


def test_escape_wrapper_guard_is_thread_local(monkeypatch):
    """
    Pin that the re-entrancy guard blocks recursion per thread, not globally.

    Notes
    -----
    The guard stops the wrapper's own namespace detection and stack
    extraction, which call back into the patched NumPy entry points, from
    recursing. A process-wide flag would also silence a *concurrent* escape
    in another thread, so the guard is thread-local and this test holds one
    thread inside the wrapper while a second thread goes through it.
    """
    escape_log = EscapeLog(
        unconfigured_settings(), FrameClassifier(unconfigured_settings())
    )

    first_entered = Event()
    release_first = Event()
    inspected = []
    logged = []
    original_calls = []

    def original(value):
        original_calls.append(value)
        return value

    guard = ReentrancyGuard()
    wrapper = escape_log.make_wrapper(original, "numpy.asarray", guard)
    nested_wrapper = escape_log.make_wrapper(original, "numpy.asanyarray", guard)

    def foreign_namespace(value):
        if not isinstance(value, str) or value not in ("first", "second"):
            return None
        inspected.append(value)
        if value == "first":
            assert nested_wrapper("nested") == "nested"
            first_entered.set()
            assert release_first.wait(timeout=5)
        return "foreign"

    monkeypatch.setattr(escape_logger, "foreign_namespace", foreign_namespace)
    monkeypatch.setattr(escape_log, "locate_site", lambda: None)
    monkeypatch.setattr(
        escape_log, "record", lambda _frame, funcname: logged.append(funcname)
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(wrapper, "first")
        try:
            assert first_entered.wait(timeout=5)
            unrelated = object()
            assert wrapper(unrelated) is unrelated
            second = executor.submit(wrapper, "second")
            assert second.result(timeout=5) == "second"
        finally:
            release_first.set()
        assert first.result(timeout=5) == "first"

    assert inspected == ["first", "second"]
    assert logged == ["numpy.asarray", "numpy.asarray"]
    assert original_calls == ["nested", unrelated, "second", "first"]


def test_escape_log_tally_increment_is_serialized(monkeypatch):
    """
    Pin that every read and write of the escape tally holds the tally lock.

    Notes
    -----
    Escapes are recorded from whichever thread made the coercion, so the
    read-modify-write of the per-site counter has to be serialized or counts
    are silently lost.
    """

    class TrackingLock:
        """Track which thread owns the lock used by the tally."""

        def __init__(self):
            self.lock = Lock()
            self.owner = None

        def __enter__(self):
            self.lock.acquire()
            self.owner = get_ident()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.owner = None
            self.lock.release()

        def held_by_current_thread(self):
            return self.owner == get_ident()

    class GuardedCounts(dict):
        """Assert every tally read and write happens while holding the lock."""

        def __init__(self, lock):
            super().__init__()
            self.lock = lock

        def __getitem__(self, key):
            assert self.lock.held_by_current_thread()
            return self.get(key, 0)

        def __setitem__(self, key, value):
            assert self.lock.held_by_current_thread()
            super().__setitem__(key, value)

    settings = unconfigured_settings()
    escape_log = EscapeLog(settings, FrameClassifier(settings))
    lock = TrackingLock()
    counts = GuardedCounts(lock)
    monkeypatch.setattr(escape_log, "counts", counts)
    monkeypatch.setattr(escape_log, "counts_lock", lock)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(escape_log.record, None, "numpy.asarray") for _ in range(20)
        ]
        for future in futures:
            future.result(timeout=5)

    assert counts.get(("<unknown location>", 0, "", "numpy.asarray")) == 20


# ---------------------------------------------------------------------------
# End-to-end: a different package, a different environment prefix
# ---------------------------------------------------------------------------

FAKE_PACKAGE_FILES = {
    "otherpkg/__init__.py": "",
    "otherpkg/lib.py": (
        "import numpy as np\n"
        "\n"
        "\n"
        "def coerce(value):\n"
        "    return np.asarray(value)\n"
    ),
    "otherpkg/conftest.py": (
        '"""Loads the plugin under test exactly as ccdproc/conftest.py does."""\n'
        "\n"
        'pytest_plugins = ["ccdproc.tests._array_api_plugin"]\n'
    ),
    "otherpkg/checks/__init__.py": "",
    "otherpkg/checks/test_escape.py": (
        "import types\n"
        "\n"
        "from otherpkg.lib import coerce\n"
        "\n"
        "FAKE_NAMESPACE = types.ModuleType('fakexp')\n"
        "\n"
        "\n"
        "class ForeignArray:\n"
        "    def __array_namespace__(self, api_version=None):\n"
        "        return FAKE_NAMESPACE\n"
        "\n"
        "    def __array__(self, dtype=None, copy=None):\n"
        "        import numpy as np\n"
        "\n"
        "        return np.zeros(3)\n"
        "\n"
        "\n"
        "def test_coercion_is_logged():\n"
        "    assert coerce(ForeignArray()).shape == (3,)\n"
    ),
}


def test_plugin_is_not_wired_to_ccdproc(pytester, monkeypatch):
    """
    Pin end to end that another package can drive the plugin under its own name.

    Notes
    -----
    A throwaway package configures the plugin with its own import name, its
    own test directory and its own environment-variable prefix, and switches
    on the escape logger and the baseline writer through *that* prefix. The
    written baseline must blame ``otherpkg/lib.py`` and the terminal section
    must be titled after ``otherpkg``, which is only possible if nothing in
    the plugin is hard-wired to ccdproc. Setting the ccdproc variables to
    conflicting values in the same environment pins that they are ignored.
    """
    use_this_ccdproc_in_subprocess(monkeypatch)
    for path, content in FAKE_PACKAGE_FILES.items():
        target = pytester.path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    pytester.makeini("""
        [pytest]
        addopts = --strict-config --strict-markers
        array_api_escapes_package = otherpkg
        array_api_escapes_test_paths = otherpkg.checks
        array_api_escapes_baseline = otherpkg/escapes.txt
        array_api_escapes_env_prefix = OTHERPKG
        array_api_escapes_logger = otherpkg.escapes
        """)
    monkeypatch.setenv("OTHERPKG_LOG_ARRAY_ESCAPES", "1")
    monkeypatch.setenv("OTHERPKG_WRITE_ESCAPE_BASELINE", "1")
    # ccdproc's own variables must have no effect on another package's run.
    monkeypatch.setenv("CCDPROC_LOG_ARRAY_ESCAPES", "0")
    monkeypatch.setenv("CCDPROC_ENFORCE_ESCAPE_BASELINE", "1")

    result = pytester.runpytest_subprocess("otherpkg", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*otherpkg array-API escape log summary*"])

    written = (pytester.path / "otherpkg" / "escapes.txt").read_text(encoding="utf-8")
    assert "lib.py  coerce  numpy.asarray" in written
    # The escape is blamed on library code, not on the test that triggered it.
    assert "test_escape.py" not in written


def test_conftest_hook_configures_a_package_without_an_ini_file(pytester, monkeypatch):
    """
    Pin end to end that a package's conftest hook alone configures the plugin.

    Notes
    -----
    This is the case the hook exists for: the tests of an installed package
    run from a directory with no ini file, so only the package's
    ``conftest.py`` can say which frames are library frames and what the
    environment-variable prefix is. The throwaway package here has no ini
    file anywhere above it, sets its prefix only through the hook, and
    switches on the escape logger through that prefix; the summary must be
    titled after it and blame its library module.
    """
    use_this_ccdproc_in_subprocess(monkeypatch)
    files = dict(FAKE_PACKAGE_FILES)
    files["otherpkg/conftest.py"] = (
        'pytest_plugins = ["ccdproc.tests._array_api_plugin"]\n'
        "\n"
        "\n"
        "def pytest_array_api_escapes_config(config):\n"
        '    return {"package": "otherpkg", "test_paths": ["otherpkg.checks"],\n'
        '            "env_prefix": "HOOKPKG"}\n'
    )
    for path, content in files.items():
        target = pytester.path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    monkeypatch.setenv("HOOKPKG_LOG_ARRAY_ESCAPES", "1")

    result = pytester.runpytest_subprocess("otherpkg", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
    # pytest names the ini file it found in the header; there must be none.
    assert "configfile:" not in result.stdout.str()
    result.stdout.fnmatch_lines(
        [
            "*otherpkg array-API escape log summary*",
            "*numpy.asarray()  lib.py:* coerce",
        ]
    )


def test_session_without_ini_options_still_runs(pytester, monkeypatch):
    """
    Pin that loading the plugin without configuring it does not break a session.

    Notes
    -----
    A stand-alone plugin is installed before it is configured, so a bare
    session has to keep working: the two markers and the ``xp``/``xp_device``
    fixtures stay available, defaulting to NumPy, and only the features that
    must classify stack frames refuse to start.
    """
    use_this_ccdproc_in_subprocess(monkeypatch)
    pytester.makeini("[pytest]\naddopts = --strict-markers\n")
    pytester.makeconftest('pytest_plugins = ["ccdproc.tests._array_api_plugin"]\n')
    pytester.makepyfile(test_bare="""
        import pytest


        @pytest.mark.backend_skip("cupy", reason="never selected here")
        def test_fixtures_are_available(xp, xp_device):
            assert xp.asarray([1, 2, 3]).shape == (3,)
            assert xp_device is None
        """)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
