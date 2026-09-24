# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The escape-baseline ratchet.

A checked-in text file lists the library call sites where a foreign
array-API array is still silently coerced to NumPy. In enforce mode
(``<PREFIX>_ENFORCE_ESCAPE_BASELINE``) any observed library escape that is
not in that file fails the session, so new escapes cannot creep in while the
known ones are migrated; in write mode
(``<PREFIX>_WRITE_ESCAPE_BASELINE``) the file is regenerated from the
escapes observed during the run, preserving the hand-written tags of sites
that are still seen.

Notes
-----
The file format is one entry per line: three whitespace-separated tokens
(``<file> <function> <coercion>``) followed by an optional free-text
reason/tag that the ratchet ignores. Blank lines and ``#`` comments are
skipped. The ratchet is one-directional by convention: entries are deleted by
hand as call sites are migrated, and write mode refuses to run when nothing
was observed, so a numpy-backend or subset run cannot truncate the file.
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
        self.settings = settings
        self.classifier = classifier
        self.escape_log = escape_log
        #: Entries dropped by the last `write` call: present in the old file
        #: but not observed this run. Stashed so the terminal summary can
        #: make the shrink visible.
        self.dropped = []

    @property
    def path(self):
        """Absolute path of the baseline file, or None if unconfigured."""
        return self.settings.baseline_path

    def load(self):
        """
        Parse the baseline file into ``{(relfile, function, coercion): reason}``.

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
        """Library escapes observed this run that are absent from the baseline."""
        return sorted(self.escape_log.observed_library_sites() - set(self.load()))

    def stale_entries(self):
        """Baseline entries not hit this run (candidates for deletion)."""
        return sorted(set(self.load()) - self.escape_log.observed_library_sites())

    def _header(self):
        env = self.settings.env_name
        return [
            "# Array-API escape baseline for non-numpy backends (dask/jax).",
            "# Columns: <file> <function> <coercion>  <reason/tag>",
            "#",
            f"# The ratchet ({env(ENV_ENFORCE)}=1) fails the session",
            "# if a library escape appears that is not listed here. Delete an",
            "# entry as you migrate that call site; this file only shrinks.",
            f"# Regenerate with {env(ENV_WRITE)}=1 (preserves tags);",
            f"# that requires {env(ENV_LOG_ESCAPES)}=1, a non-numpy",
            f"# {env('ARRAY_LIBRARY')} (e.g. dask), and a *full* test-suite run --",
            "# a subset run drops the entries its tests never exercise.",
            "#",
            "# Tags are for humans, not the ratchet: TODO = still to migrate,",
            "# BOUNDARY = a numpy-only dependency (scipy/astroscrappy/reproject)",
            "# that will never leave. Verify/adjust the seeded tags by hand.",
            "#",
        ]

    def write(self):
        """
        Rewrite the baseline file from the escapes observed this run.

        Notes
        -----
        Reasons and tags already in the file are preserved for sites that are
        still observed, so hand annotations survive a refresh. Writing is
        refused, with a `pytest.UsageError`, when no library escape was
        observed at all: that means the run could not have exercised the
        escapes (a NumPy backend, or a subset run that hits none) and writing
        would truncate the baseline.
        """
        env = self.settings.env_name
        if self.path is None:
            raise pytest.UsageError(
                f"{env(ENV_WRITE)}=1: no baseline file is configured; set the "
                f"'{INI_BASELINE}' ini option."
            )
        sites = sorted(self.escape_log.observed_library_sites())
        if not sites:
            raise pytest.UsageError(
                f"{env(ENV_WRITE)}=1: no library escapes were observed this "
                f"run, refusing to truncate {self._display_path()}. Regenerate "
                f"the baseline with {env(ENV_LOG_ESCAPES)}=1, a non-numpy "
                f"{env('ARRAY_LIBRARY')} (e.g. dask), and a full-suite run."
            )
        existing = self.load()
        self.dropped[:] = sorted(set(existing) - set(sites))

        # Per-column widths for aligned output; sites is non-empty here (the
        # empty-observation case raised above), so max() cannot see an empty
        # sequence.
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
        """Baseline path relative to the package, for messages."""
        return self.classifier.relpath(self.path) if self.path else "<unset>"

    def report(self, terminalreporter):
        """Print the ratchet result and, after a rewrite, what it dropped."""
        self._report_enforcement(terminalreporter)
        self._report_dropped(terminalreporter)

    def _report_enforcement(self, terminalreporter):
        """
        Print any new library escapes and any baseline entries not hit.

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

    def _report_dropped(self, terminalreporter):
        """
        Warn loudly about entries a rewrite dropped.

        Notes
        -----
        On a full-suite run dropping stale entries is the point of a refresh,
        but on a subset run the drop only means those tests never ran --
        either way it must be visible, not a silent shrink of the ratchet.
        """
        if not self.settings.write_baseline or not self.dropped:
            return

        terminalreporter.section(
            self.settings.section("array-API escape baseline (rewritten)")
        )
        plural = "y" if len(self.dropped) == 1 else "ies"
        terminalreporter.write_line(
            f"WARNING: rewrite DROPPED {len(self.dropped)} entr{plural} present "
            "in the old baseline but not observed this run:",
            red=True,
            bold=True,
        )
        for relfile, function, coercion in self.dropped:
            terminalreporter.write_line(f"    - {relfile}  {function}  {coercion}")
        terminalreporter.write_line(
            "If this was not a full-suite run these entries were dropped only "
            "because their tests never ran -- restore the file (git checkout) "
            "and regenerate from a full run."
        )


def xdist_active(config):
    """
    True when pytest-xdist is about to run the tests in worker processes.

    Notes
    -----
    The escape tally is a per-process object: under ``pytest -n`` the workers
    do the tallying and the controller process, where ``pytest_sessionfinish``
    runs, sees an empty tally. Write mode would then truncate the baseline and
    enforce mode would pass without checking anything.
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

    Notes
    -----
    Both baseline modes read the sites tallied by the live escape logger, so
    neither can do anything meaningful without it: write mode would observe
    nothing and truncate the baseline, and enforce mode would pass vacuously.
    The same failure modes occur under pytest-xdist (see `xdist_active`), so
    both modes are rejected there too. Anything that has to classify stack
    frames also needs a configured package.
    """
    env = settings.env_name

    if settings.baseline_modes_active and xdist_active(config):
        raise pytest.UsageError(
            f"{env(ENV_WRITE)} / {env(ENV_ENFORCE)} cannot run under "
            "pytest-xdist: escape tallies live in each worker process and "
            "never reach the controller, so the baseline would be truncated "
            "or enforcement would pass vacuously. Re-run without -n (or with "
            "-n0)."
        )
    if settings.write_baseline and not settings.log_escapes:
        raise pytest.UsageError(
            f"{env(ENV_WRITE)}=1 requires the escape logger: without "
            f"{env(ENV_LOG_ESCAPES)}=1 no escapes are observed and the "
            f"baseline would be wiped. Set {env(ENV_LOG_ESCAPES)}=1 and a "
            f"non-numpy {env('ARRAY_LIBRARY')} (e.g. dask) and run the full "
            "test suite."
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
                    f"package under test, but the ini option '{INI_PACKAGE}' "
                    "is not set."
                )

    if not settings.baseline_modes_active:
        return

    if settings.baseline_path is None:
        raise pytest.UsageError(
            f"{env(ENV_WRITE)} / {env(ENV_ENFORCE)} need a baseline file; set "
            f"the ini option '{INI_BASELINE}' to its path, relative to the "
            "directory holding the ini file."
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
            f"{settings.baseline_path!r}. The '{INI_BASELINE}' ini option is "
            "resolved against the directory holding the ini file, so check "
            "that pytest picked the rootdir you expect."
        )
