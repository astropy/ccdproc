# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Stack-frame classification and failure triage.

`FrameClassifier` answers "which frame is to blame?" for a stack or a
traceback, using only the roots carried by the `.config.Settings` object, and
`FailureTriage` groups test failures by that frame so a batch of
backend-specific failures collapses into a short list of root-cause call
sites.

Notes
-----
Triage is activated by the ``<PREFIX>_TRIAGE_ESCAPES`` environment variable.
Only the ``call`` phase of a failing test is recorded: an escape that raises
during fixture setup is not triaged. That is a deliberate trade-off, since
fixtures are usually plain array construction and escapes surface in the test
bodies.
"""

import os
import traceback
from collections import defaultdict

import pytest

from .config import INI_PACKAGE

#: Key used when no frame at all could be identified.
UNKNOWN_LOCATION = "<unknown location>"


class FrameClassifier:
    """
    Classify stack frames as library, test or foreign.

    Parameters
    ----------
    settings : `.config.Settings`
        Supplies the package directory and the test roots.

    Notes
    -----
    Classification is anchored on the package directory resolved by importing
    the configured package, never on a substring test for the package name: a
    checkout directory is often named after the package it contains, so a
    substring test would classify every frame, including ones in
    ``site-packages``, as a package frame.
    """

    def __init__(self, settings):
        self.settings = settings
        root = settings.package_root
        self.package_root = (root.rstrip(os.sep) + os.sep) if root else None
        self.test_roots = tuple(os.path.abspath(path) for path in settings.test_roots)

    @property
    def configured(self):
        """True when a package directory is known and frames can be classified."""
        return self.package_root is not None

    def is_package_frame(self, filename):
        """True if ``filename`` lives inside the package under test."""
        if not self.configured:
            return False
        return os.path.abspath(filename).startswith(self.package_root)

    def is_test_frame(self, filename):
        """True if ``filename`` is part of the package's test infrastructure."""
        abspath = os.path.abspath(filename)
        for root in self.test_roots:
            if abspath == root or abspath.startswith(root.rstrip(os.sep) + os.sep):
                return True
        return False

    def is_library_frame(self, filename):
        """True for real library code: in the package, outside the tests."""
        return self.is_package_frame(filename) and not self.is_test_frame(filename)

    def relpath(self, filename):
        """Package-relative, forward-slash path, for stable baseline keys."""
        if not self.configured:
            return os.path.abspath(filename).replace(os.sep, "/")
        try:
            rel = os.path.relpath(os.path.abspath(filename), self.package_root)
        except ValueError:  # e.g. a different drive on Windows
            rel = os.path.abspath(filename)
        return rel.replace(os.sep, "/")

    def is_library_site(self, relfile):
        """
        True for escapes blamed on real library code.

        Notes
        -----
        Only library sites go into the baseline ratchet: an escape blamed on
        a test frame, or on an unknown location, is not an actionable
        migration target.
        """
        if relfile == UNKNOWN_LOCATION or not self.configured:
            return False
        abspath = os.path.join(self.package_root, relfile.replace("/", os.sep))
        return self.is_library_frame(abspath)

    def locate_escape_site(self, frames):
        """
        Find the frame most likely responsible for an array-API "escape".

        Parameters
        ----------
        frames : iterable
            Frame-summary-like objects, as returned by
            ``traceback.extract_tb`` or ``traceback.extract_stack``: anything
            with ``.filename``, ``.lineno`` and ``.name``.

        Returns
        -------
        frame or None
            None if ``frames`` is empty.

        Notes
        -----
        Preference order:

        1. The innermost frame inside the package that is not part of its
           test suite.
        2. Failing that, the innermost frame inside the package at all (this
           will typically be a test-suite frame).
        3. Failing that, the innermost frame overall.

        Frames belonging to this plugin are test frames (the plugin lives
        under one of the configured test roots), so the depth at which this
        is called never affects the result.
        """
        frames = list(frames)
        if not frames:
            return None

        library = [f for f in frames if self.is_library_frame(f.filename)]
        if library:
            return library[-1]

        package = [f for f in frames if self.is_package_frame(f.filename)]
        if package:
            return package[-1]

        return frames[-1]

    def describe(self, frame):
        """Short "file:line function" string for a log message."""
        if frame is None:
            return UNKNOWN_LOCATION
        return f"{frame.filename}:{frame.lineno} {frame.name}"


class FailureTriage:
    """
    Group failing tests by the call site that is blamed for the failure.

    Parameters
    ----------
    settings : `.config.Settings`
        Supplies the activation flag and the terminal-section title.
    classifier : `FrameClassifier`
        Used to reduce each failure's traceback to one frame.
    """

    #: How many example test ids to print per escape site.
    example_limit = 5

    def __init__(self, settings, classifier):
        self.settings = settings
        self.classifier = classifier
        #: Maps a (filename, lineno, function) escape site to the node ids
        #: that failed with that site as their innermost library frame.
        self.sites = defaultdict(list)

    @property
    def active(self):
        """True when failure triage was requested for this session."""
        return self.settings.triage

    def record_report(self, item, call, report):
        """
        Record the escape site of one failing test report.

        Notes
        -----
        Called from a ``pytest_runtest_makereport`` wrapper for every phase
        of every test; only failures in the ``call`` phase are recorded.
        """
        if not self.active or report.when != "call" or not report.failed:
            return
        if call.excinfo is None:
            return
        site = self.classifier.locate_escape_site(traceback.extract_tb(call.excinfo.tb))
        if site is not None:
            self.sites[(site.filename, site.lineno, site.name)].append(item.nodeid)

    def report(self, terminalreporter):
        """
        Print the failure-triage summary.

        Notes
        -----
        One section listing each escape site with its failure count (most
        common first) and a few example test ids, so a large batch of backend
        failures collapses to a short list of root-cause call sites.
        """
        if not self.active or not self.sites:
            return

        terminalreporter.section(self.settings.section("array-API escape triage"))
        terminalreporter.write_line(
            "Failures grouped by innermost non-test "
            f"{self.settings.library_label} frame (file:line function), most "
            "common first:"
        )

        ordered = sorted(self.sites.items(), key=lambda kv: len(kv[1]), reverse=True)
        for (filename, lineno, function), test_ids in ordered:
            terminalreporter.write_line("")
            terminalreporter.write_line(
                f"{filename}:{lineno} {function}  ({len(test_ids)} failures)"
            )
            for test_id in test_ids[: self.example_limit]:
                terminalreporter.write_line(f"    - {test_id}")
            if len(test_ids) > self.example_limit:
                terminalreporter.write_line(
                    f"    ... and {len(test_ids) - self.example_limit} more"
                )


def require_classifier(classifier, reason):
    """
    Raise a `pytest.UsageError` if frames cannot be classified.

    Parameters
    ----------
    classifier : `FrameClassifier`
        The classifier built for this session.
    reason : str
        What the caller was trying to do, used in the error message.
    """
    if classifier.configured:
        return
    raise pytest.UsageError(
        f"{reason} needs to know which stack frames belong to the package "
        f"under test, but the ini option '{INI_PACKAGE}' is not set."
    )
