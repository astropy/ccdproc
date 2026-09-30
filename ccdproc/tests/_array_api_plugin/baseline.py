# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The escape-baseline ratchet.

A checked-in text file lists the library call sites where a foreign
array-API array is still silently coerced to NumPy. In enforce mode
(``<PREFIX>_ENFORCE_ESCAPE_BASELINE``) any observed library escape that is
not in that file fails the session, so new escapes cannot creep in while the
known ones are migrated; in write mode
(``<PREFIX>_WRITE_ESCAPE_BASELINE``) the escapes observed during the run
that are not yet listed are added to the file.

Notes
-----
The file format is one entry per line: three whitespace-separated tokens
(``<file> <function> <coercion>``) followed by an optional free-text
reason/tag that the ratchet ignores. Blank lines and ``#`` comments are
skipped. Entries are only ever deleted by hand, as call sites are migrated:
write mode keeps every existing entry and its reason, because an entry a run
did not hit may only mean its tests did not run (a subset run, a skipped
test). The terminal summary lists those entries as candidates for deletion.
"""

import os

import pytest

from .config import (
    ENV_ENFORCE,
    ENV_LOG_ESCAPES,
    ENV_TRIAGE,
    ENV_WRITE,
    INI_BASELINE,
    INI_PACKAGE,
)


class Baseline:
    """
    Load, compare and rewrite the checked-in escape baseline.

    Parameters
    ----------
    settings : `.config.Settings`
        Supplies the baseline path, the mode flags and the env-var names
        used in messages.
    classifier : `.triage.FrameClassifier`
        Used to turn the baseline path into a readable relative path.
    escape_log : `.escape_logger.EscapeLog`
        Source of the escapes observed during this session.
    """

    def __init__(self, settings, classifier, escape_log):
        """Hold the collaborators; nothing is read until asked for."""
        self.settings = settings
        self.classifier = classifier
        self.escape_log = escape_log
        #: Entries added by the last `write` call, for the terminal summary.
        self.added = []

    @property
    def path(self):
        """
        Absolute path of the baseline file, or None if unconfigured.

        Returns
        -------
        str or None
            The configured path.
        """
        return self.settings.baseline_path

    def load(self):
        """
        Parse the baseline file into ``{(relfile, function, coercion): reason}``.

        Returns
        -------
        dict
            Maps each ``(relfile, function, coercion)`` entry to its free-text
            reason, ``""`` if it has none. Empty when no baseline is
            configured or the file does not exist.

        Notes
        -----
        Blank lines and ``#`` comments are ignored. The first three
        whitespace-separated tokens are the key (none of them contains a
        space); anything after is a free-text reason for humans.
        """
        baseline = {}
        if self.path is None:
            return baseline
        try:
            with open(self.path, encoding="utf-8") as f:
                raw_lines = f.readlines()
        except FileNotFoundError:
            return baseline
        for raw in raw_lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 3)
            if len(parts) < 3:
                continue
            baseline[(parts[0], parts[1], parts[2])] = (
                parts[3] if len(parts) > 3 else ""
            )
        return baseline

    def new_escapes(self):
        """
        Library escapes observed this run that are absent from the baseline.

        Returns
        -------
        list of tuple
            Sorted ``(relfile, function, coercion)`` entries.
        """
        return sorted(self.escape_log.observed_library_sites() - set(self.load()))

    def stale_entries(self):
        """
        Baseline entries not hit this run (candidates for deletion).

        Returns
        -------
        list of tuple
            Sorted ``(relfile, function, coercion)`` entries.
        """
        return sorted(set(self.load()) - self.escape_log.observed_library_sites())

    def _header(self):
        """
        Comment lines written at the top of a regenerated baseline file.

        Returns
        -------
        list of str
            The lines, without newlines, naming the configured environment
            variables.
        """
        env = self.settings.env_name
        return [
            "# Array-API escape baseline for non-numpy backends (dask/jax).",
            "# Columns: <file> <function> <coercion>  <reason/tag>",
            "#",
            f"# The ratchet ({env(ENV_ENFORCE)}=1) fails the session",
            "# if a library escape appears that is not listed here. Delete an",
            "# entry by hand as you migrate that call site.",
            f"# Add newly observed escapes with {env(ENV_WRITE)}=1, which",
            f"# requires {env(ENV_LOG_ESCAPES)}=1 and a non-numpy",
            f"# {env('ARRAY_LIBRARY')} (e.g. dask); existing entries and tags",
            "# are kept, even for sites the run did not hit.",
            "#",
            "# Tags are for humans, not the ratchet: TODO = still to migrate,",
            "# BOUNDARY = a numpy-only dependency (scipy/astroscrappy/reproject)",
            "# that will never leave. Verify/adjust the seeded tags by hand.",
            "#",
        ]

    def write(self):
        """
        Add the library escapes observed this run to the baseline file.

        Raises
        ------
        pytest.UsageError
            If no baseline file is configured.

        Notes
        -----
        Every existing entry is kept with its reason; new sites are tagged
        ``TODO``. An entry this run did not hit is not dropped, because that
        may only mean its tests did not run. The file is left untouched when
        nothing new was observed.
        """
        env = self.settings.env_name
        if self.path is None:
            raise pytest.UsageError(
                f"{env(ENV_WRITE)}=1: no baseline file is configured; return "
                "'baseline' from pytest_array_api_escapes_config or set the "
                f"'{INI_BASELINE}' ini option."
            )
        existing = self.load()
        self.added[:] = self.new_escapes()
        if not self.added:
            return
        sites = sorted(set(existing) | set(self.added))

        # Per-column widths for aligned output; sites holds at least the
        # added entries, so max() cannot see an empty sequence.
        w_file, w_func, w_co = (
            max(len(s) for s in col) for col in zip(*sites, strict=True)
        )
        body = []
        for key in sites:
            relfile, function, coercion = key
            reason = existing.get(key, "TODO")
            body.append(
                f"{relfile:<{w_file}}  {function:<{w_func}}  "
                f"{coercion:<{w_co}}  {reason}".rstrip()
            )
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._header() + body) + "\n")

    def _display_path(self):
        """
        Baseline path relative to the package, for messages.

        Returns
        -------
        str
            The relative path, or ``"<unset>"`` when none is configured.
        """
        return self.classifier.relpath(self.path) if self.path else "<unset>"

    def report(self, terminalreporter):
        """
        Print the ratchet result and, in write mode, what was added.

        Parameters
        ----------
        terminalreporter : _pytest.terminal.TerminalReporter
            The reporter to write to.
        """
        self._report_enforcement(terminalreporter)
        self._report_written(terminalreporter)

    def _report_enforcement(self, terminalreporter):
        """
        Print any new library escapes and any baseline entries not hit.

        Parameters
        ----------
        terminalreporter : _pytest.terminal.TerminalReporter
            The reporter to write to.

        Notes
        -----
        Only shown when enforcement is active. If no foreign array was seen
        at all the baseline is not checked and every entry would look stale,
        so say so instead of tempting someone to delete the file's contents.
        """
        if not self.settings.enforce_baseline:
            return

        section = self.settings.section("array-API escape baseline")

        if not self.escape_log.observed_library_sites():
            terminalreporter.section(section)
            terminalreporter.write_line(
                "Escape logger was active but observed no foreign-array "
                "escapes (numpy backend, or a subset run exercising no escape "
                "site); baseline not checked."
            )
            return

        new = self.new_escapes()
        stale = self.stale_entries()

        terminalreporter.section(section)
        if not new:
            terminalreporter.write_line("OK: no library escapes outside the baseline.")
        else:
            terminalreporter.write_line(
                f"NEW escapes not in baseline ({len(new)}) -- these fail the "
                "session:"
            )
            for relfile, function, coercion in new:
                terminalreporter.write_line(f"    + {relfile}  {function}  {coercion}")

        if stale:
            terminalreporter.write_line("")
            terminalreporter.write_line(
                f"Baseline entries not hit this run ({len(stale)}) -- delete "
                "them if the migration removed the escape:"
            )
            for relfile, function, coercion in stale:
                terminalreporter.write_line(f"    - {relfile}  {function}  {coercion}")

    def _report_written(self, terminalreporter):
        """
        Say what write mode added and which entries it did not hit.

        Parameters
        ----------
        terminalreporter : _pytest.terminal.TerminalReporter
            The reporter to write to.

        Notes
        -----
        The entries not hit are kept in the file; listing them lets someone
        who ran the full suite delete the ones whose escape is gone.
        """
        if not self.settings.write_baseline:
            return

        terminalreporter.section(
            self.settings.section("array-API escape baseline (write mode)")
        )
        if not self.escape_log.observed_library_sites():
            terminalreporter.write_line(
                "Observed no library escapes (numpy backend, or a subset run "
                f"exercising no escape site); {self._display_path()} unchanged."
            )
            return
        if self.added:
            terminalreporter.write_line(
                f"Added {len(self.added)} new entr"
                f"{'y' if len(self.added) == 1 else 'ies'} to "
                f"{self._display_path()} (tagged TODO):"
            )
            for relfile, function, coercion in self.added:
                terminalreporter.write_line(f"    + {relfile}  {function}  {coercion}")
        else:
            terminalreporter.write_line(
                f"No new escapes; {self._display_path()} unchanged."
            )
        stale = self.stale_entries()
        if stale:
            terminalreporter.write_line("")
            terminalreporter.write_line(
                f"Entries kept but not hit this run ({len(stale)}) -- after a "
                "full-suite run, delete the ones the migration removed:"
            )
            for relfile, function, coercion in stale:
                terminalreporter.write_line(f"    - {relfile}  {function}  {coercion}")


def xdist_active(config):
    """
    True when pytest-xdist is about to run the tests in worker processes.

    Parameters
    ----------
    config : pytest.Config
        The pytest config object of this session.

    Returns
    -------
    bool
        Whether ``-n`` asked for worker processes.

    Notes
    -----
    The escape tally is a per-process object: under ``pytest -n`` the workers
    do the tallying and the controller process, where ``pytest_sessionfinish``
    runs, sees an empty tally. Write mode would then add nothing and enforce
    mode would pass without checking anything.
    """
    if not config.pluginmanager.hasplugin("xdist"):
        return False
    try:
        numprocesses = config.getoption("numprocesses", None)
    except (KeyError, ValueError):
        return False
    return bool(numprocesses)


def check_usage(config, settings, classifier):
    """
    Fail fast on unusable configurations, before any test runs.

    Parameters
    ----------
    config : pytest.Config
        The pytest config object of this session; only consulted in the
        baseline modes.
    settings : `.config.Settings`
        The resolved configuration.
    classifier : `.triage.FrameClassifier`
        The session's classifier, which says whether frames can be
        classified at all.

    Raises
    ------
    pytest.UsageError
        If the requested features cannot work with this configuration.

    Notes
    -----
    Both baseline modes read the sites tallied by the live escape logger, so
    neither can do anything meaningful without it: write mode would observe
    nothing and add nothing, and enforce mode would pass vacuously.
    The same failure modes occur under pytest-xdist (see `xdist_active`), so
    both modes are rejected there too. Anything that has to classify stack
    frames also needs a configured package.
    """
    env = settings.env_name

    if settings.baseline_modes_active and xdist_active(config):
        raise pytest.UsageError(
            f"{env(ENV_WRITE)} / {env(ENV_ENFORCE)} cannot run under "
            "pytest-xdist: escape tallies live in each worker process and "
            "never reach the controller, so write mode would add nothing "
            "and enforcement would pass vacuously. Re-run without -n (or with "
            "-n0)."
        )
    if settings.write_baseline and not settings.log_escapes:
        raise pytest.UsageError(
            f"{env(ENV_WRITE)}=1 requires the escape logger: without "
            f"{env(ENV_LOG_ESCAPES)}=1 no escapes are observed and nothing "
            f"would be added. Set {env(ENV_LOG_ESCAPES)}=1 and a non-numpy "
            f"{env('ARRAY_LIBRARY')} (e.g. dask)."
        )
    if settings.enforce_baseline and not settings.log_escapes:
        raise pytest.UsageError(
            f"{env(ENV_ENFORCE)}=1 requires the escape logger: without "
            f"{env(ENV_LOG_ESCAPES)}=1 no escapes are observed and enforcement "
            f"would pass without checking anything. Set "
            f"{env(ENV_LOG_ESCAPES)}=1 (and a non-numpy "
            f"{env('ARRAY_LIBRARY')}, e.g. dask)."
        )

    needs_frames = [
        (settings.log_escapes, env(ENV_LOG_ESCAPES)),
        (settings.triage, env(ENV_TRIAGE)),
        (settings.enforce_baseline, env(ENV_ENFORCE)),
        (settings.write_baseline, env(ENV_WRITE)),
    ]
    if not classifier.configured:
        for requested, name in needs_frames:
            if requested:
                raise pytest.UsageError(
                    f"{name}=1 needs to know which stack frames belong to the "
                    "package under test, but no package is configured: return "
                    "'package' from pytest_array_api_escapes_config in the "
                    f"package's conftest.py, or set the ini option "
                    f"'{INI_PACKAGE}'."
                )

    if not settings.baseline_modes_active:
        return

    if settings.baseline_path is None:
        raise pytest.UsageError(
            f"{env(ENV_WRITE)} / {env(ENV_ENFORCE)} need a baseline file; "
            "return 'baseline' from pytest_array_api_escapes_config or set "
            f"the ini option '{INI_BASELINE}' to its path, relative to the "
            "directory holding the ini file (or the rootdir without one)."
        )

    # Only checked in the baseline modes: outside them the path is never
    # read, and a session run from an unusual rootdir must not be aborted
    # just because the configured path does not resolve there.
    if settings.write_baseline and not os.path.isdir(
        os.path.dirname(settings.baseline_path)
    ):
        raise pytest.UsageError(
            f"{env(ENV_WRITE)}=1: the directory for the '{INI_BASELINE}' file "
            f"({settings.baseline_path!r}) does not exist."
        )
    if settings.enforce_baseline and not os.path.isfile(settings.baseline_path):
        raise pytest.UsageError(
            f"{env(ENV_ENFORCE)}=1: no baseline file at "
            f"{settings.baseline_path!r}. The baseline path is resolved "
            "against the directory holding the ini file (or the rootdir "
            "without one), so check that pytest picked the ini file you "
            "expect."
        )
