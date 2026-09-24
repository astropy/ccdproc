# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Configuration surface of the array-API escape plugin.

Everything the rest of the plugin needs to know about the package under test
is gathered here into a single `Settings` object, built once in
``pytest_configure`` from pytest ini options (declared by
`add_ini_options`) with environment-variable overrides.

Notes
-----
The plugin deliberately contains no hard-coded package name: section titles,
log messages, error text, the baseline path and the environment-variable
names are all derived from the ini options below, so this package can be
lifted out of ``ccdproc`` into a stand-alone distribution without changes.
"""

import importlib
import os

import pytest

#: Values (case-insensitively) accepted as "true" in the plugin's own
#: environment variables.
TRUTHY = {"1", "true", "yes", "on"}

#: Prefix shared by every ini option this plugin declares.
INI_PREFIX = "array_api_escapes"

INI_PACKAGE = f"{INI_PREFIX}_package"
INI_TEST_PATHS = f"{INI_PREFIX}_test_paths"
INI_BASELINE = f"{INI_PREFIX}_baseline"
INI_ENV_PREFIX = f"{INI_PREFIX}_env_prefix"
INI_LOGGER = f"{INI_PREFIX}_logger"

#: Suffixes of the environment variables the plugin reads. Each one is
#: joined to the configured environment prefix with an underscore, so a
#: prefix of ``CCDPROC`` yields ``CCDPROC_ARRAY_LIBRARY`` and friends.
ENV_ARRAY_LIBRARY = "ARRAY_LIBRARY"
ENV_ARRAY_DEVICE = "ARRAY_DEVICE"
ENV_LOG_ESCAPES = "LOG_ARRAY_ESCAPES"
ENV_TRIAGE = "TRIAGE_ESCAPES"
ENV_ENFORCE = "ENFORCE_ESCAPE_BASELINE"
ENV_WRITE = "WRITE_ESCAPE_BASELINE"

#: Environment prefix used when neither the ini option nor a package name is
#: configured.
FALLBACK_ENV_PREFIX = "ARRAY_API"

#: Appended to the package name to build the default logger name.
DEFAULT_LOGGER_SUFFIX = "array_escape"

#: Logger name used when no package name is configured.
FALLBACK_LOGGER = "array_api_escapes"


def env_truthy(value):
    """True for the strings the plugin accepts as "on"."""
    return str(value).strip().lower() in TRUTHY


def add_ini_options(parser):
    """
    Declare the plugin's ini options on ``parser``.

    Notes
    -----
    Called from ``pytest_addoption``. Because ``pytest_addoption`` is a
    historic hook, this still runs when the plugin is registered late (from
    a conftest deep inside a package), which is what lets ``--strict-config``
    accept the options below when they are set in ``pyproject.toml``.
    """
    parser.addini(
        INI_PACKAGE,
        "Dotted import name of the package whose stack frames count as "
        "library frames for array-API escape triage. Its directory is found "
        "by importing it, so an installed copy is classified correctly.",
        default="",
    )
    parser.addini(
        INI_TEST_PATHS,
        "Modules or directories, dotted or package-relative, whose frames "
        "count as test frames and are therefore never blamed for an escape. "
        f"Defaults to '<{INI_PACKAGE}>.tests'. The package's own conftest.py "
        "always counts as a test frame.",
        type="args",
        default=[],
    )
    parser.addini(
        INI_BASELINE,
        "Path to the checked-in escape baseline file, relative to the "
        "directory holding the ini file (the pytest rootdir).",
        default="",
    )
    parser.addini(
        INI_ENV_PREFIX,
        "Prefix of the environment variables that drive the plugin, e.g. "
        "'CCDPROC' for CCDPROC_ARRAY_LIBRARY. Defaults to the configured "
        f"package name upper-cased, or '{FALLBACK_ENV_PREFIX}'.",
        default="",
    )
    parser.addini(
        INI_LOGGER,
        "Name of the logger the escape logger writes to. Defaults to "
        f"'<{INI_PACKAGE}>.{DEFAULT_LOGGER_SUFFIX}'.",
        default="",
    )


def _as_str(value):
    """Collapse an ini value that pytest may hand back as a list."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    return "" if value is None else str(value)


def _default_env_prefix(package_name):
    """Environment prefix implied by ``package_name`` (``ccdproc`` -> CCDPROC)."""
    if not package_name:
        return FALLBACK_ENV_PREFIX
    return "".join(c if c.isalnum() else "_" for c in package_name).upper()


def _resolve_package_root(package_name):
    """
    Absolute directory of ``package_name``, found by importing it.

    Notes
    -----
    Importing is what makes frame classification work when the tests run
    against an installed copy of the package (``pytest --pyargs``): an anchor
    derived from this file's own ``__file__`` would point at whichever copy
    of the plugin happened to be imported, which is not necessarily the copy
    of the package being tested.
    """
    try:
        module = importlib.import_module(package_name)
    except ImportError as exc:
        raise pytest.UsageError(
            f"{INI_PACKAGE} = {package_name!r} could not be imported: {exc}"
        ) from exc
    filename = getattr(module, "__file__", None)
    if not filename:
        raise pytest.UsageError(
            f"{INI_PACKAGE} = {package_name!r} has no __file__ (a namespace "
            "package?), so its directory cannot be used to classify stack "
            "frames."
        )
    return os.path.dirname(os.path.abspath(filename))


def _resolve_test_root(entry, package_name, package_root):
    """
    Turn one ``array_api_escapes_test_paths`` entry into an absolute path.

    Notes
    -----
    Entries may be dotted module names (``mypkg.tests``, or just ``tests``)
    or package-relative paths (``tests``, ``tests/helpers.py``). Both are
    resolved against the package directory rather than imported, so listing a
    test package costs nothing and has no import side effects.
    """
    entry = entry.strip()
    if os.path.isabs(entry):
        return os.path.abspath(entry)
    if package_name:
        if entry == package_name:
            return package_root
        for prefix in (f"{package_name}.", f"{package_name}/"):
            if entry.startswith(prefix):
                entry = entry[len(prefix) :]
                break
    if "/" not in entry and not entry.endswith(".py"):
        entry = entry.replace(".", "/")
    return os.path.abspath(os.path.join(package_root, *entry.split("/")))


class Settings:
    """
    Frozen view of the plugin's configuration for one test session.

    Parameters
    ----------
    package_name : str
        Dotted import name of the package under test, or ``""`` when the
        plugin was loaded without configuration.
    package_root : str or None
        Absolute directory of that package, or None.
    test_roots : sequence of str
        Absolute paths (directories or files) whose frames are test frames.
    baseline_path : str or None
        Absolute path of the escape baseline file, or None if unconfigured.
    env_prefix : str
        Prefix of the plugin's environment variables.
    logger_name : str
        Logger the escape logger writes to.
    environ : mapping, optional
        Environment to read, defaulting to ``os.environ``.
    """

    def __init__(
        self,
        package_name,
        package_root,
        test_roots,
        baseline_path,
        env_prefix,
        logger_name,
        environ=None,
    ):
        self.package_name = package_name
        self.package_root = package_root
        self.test_roots = tuple(test_roots)
        self.baseline_path = baseline_path
        self.env_prefix = env_prefix
        self.logger_name = logger_name
        self._environ = os.environ if environ is None else environ

        # Snapshot the switches once so that code mutating the environment
        # mid-session cannot half-enable the escape logger.
        self.log_escapes = self.env_flag(ENV_LOG_ESCAPES)
        self.triage = self.env_flag(ENV_TRIAGE)
        self.enforce_baseline = self.env_flag(ENV_ENFORCE)
        self.write_baseline = self.env_flag(ENV_WRITE)

    # -- environment -----------------------------------------------------
    def env_name(self, suffix):
        """Full name of the environment variable with ``suffix``."""
        return f"{self.env_prefix}_{suffix}"

    def env(self, suffix, default=""):
        """Value of the environment variable with ``suffix``."""
        return self._environ.get(self.env_name(suffix), default)

    def env_flag(self, suffix):
        """True when the environment variable with ``suffix`` is truthy."""
        return env_truthy(self.env(suffix, ""))

    # -- presentation ----------------------------------------------------
    @property
    def baseline_modes_active(self):
        """True when either baseline mode was requested."""
        return self.enforce_baseline or self.write_baseline

    @property
    def library_label(self):
        """Human-readable name of the package under test."""
        return self.package_name or "the package under test"

    def section(self, title):
        """Terminal-summary section title, qualified by the package name."""
        return f"{self.package_name} {title}".strip()


def build_settings(config, environ=None):
    """
    Build the `Settings` for this session from ``config``.

    Notes
    -----
    A session that sets none of the ini options still starts: the settings
    simply carry no package, and the features that need to classify stack
    frames refuse to run (see `.baseline.check_usage`) rather than silently
    blaming the wrong frames.
    """
    package_name = _as_str(config.getini(INI_PACKAGE)).strip()
    package_root = _resolve_package_root(package_name) if package_name else None

    entries = [str(entry) for entry in config.getini(INI_TEST_PATHS)]
    if not entries and package_name:
        entries = [f"{package_name}.tests"]
    test_roots = []
    if package_root is not None:
        test_roots = [
            _resolve_test_root(entry, package_name, package_root) for entry in entries
        ]
        # The package's own conftest.py hosts test infrastructure (this
        # plugin is loaded from it), so it must never be reported as an
        # escape site.
        test_roots.append(os.path.join(package_root, "conftest.py"))

    baseline = _as_str(config.getini(INI_BASELINE)).strip()
    baseline_path = (
        os.path.abspath(os.path.join(str(config.rootpath), baseline))
        if baseline
        else None
    )

    env_prefix = _as_str(config.getini(INI_ENV_PREFIX)).strip()
    if not env_prefix:
        env_prefix = _default_env_prefix(package_name)

    logger_name = _as_str(config.getini(INI_LOGGER)).strip()
    if not logger_name:
        if package_name:
            logger_name = f"{package_name}.{DEFAULT_LOGGER_SUFFIX}"
        else:
            logger_name = FALLBACK_LOGGER

    return Settings(
        package_name=package_name,
        package_root=package_root,
        test_roots=test_roots,
        baseline_path=baseline_path,
        env_prefix=env_prefix,
        logger_name=logger_name,
        environ=environ,
    )
