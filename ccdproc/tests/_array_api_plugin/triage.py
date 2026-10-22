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

#: Key used when no frame at all could be identified.
UNKNOWN_LOCATION = "<unknown location>"

#: This plugin's own directory; its frames are never blamed for an escape.
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _is_plugin_frame(filename):
    """
    True if ``filename`` belongs to this plugin.

    Parameters
    ----------
    filename : str
        Path of a frame's source file.

    Returns
    -------
    bool
        Whether the file lives in this plugin's directory.
    """
    return os.path.abspath(filename).startswith(_PLUGIN_DIR)


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
        """Normalize the package and test roots from ``settings``."""
        self.settings = settings
        root = settings.package_root
        self.package_root = (root.rstrip(os.sep) + os.sep) if root else None
        self.test_roots = tuple(os.path.abspath(path) for path in settings.test_roots)

    @property
    def configured(self):
        """
        True when a package directory is known and frames can be classified.

        Returns
        -------
        bool
            Whether a package root is configured.
        """
        return self.package_root is not None

    def is_package_frame(self, filename):
        """
        True if ``filename`` lives inside the package under test.

        Parameters
        ----------
        filename : str
            Path of the frame's source file.

        Returns
        -------
        bool
            Whether the file is under the package root; always False when
            unconfigured.
        """
        if not self.configured:
            return False
        return os.path.abspath(filename).startswith(self.package_root)

    def is_test_frame(self, filename):
        """
        True if ``filename`` is part of the package's test infrastructure.

        Parameters
        ----------
        filename : str
            Path of the frame's source file.

        Returns
        -------
        bool
            Whether the file is, or is inside, one of the test roots.
        """
        abspath = os.path.abspath(filename)
        for root in self.test_roots:
            if abspath == root or abspath.startswith(root.rstrip(os.sep) + os.sep):
                return True
        return False

    def is_library_frame(self, filename):
        """
        True for real library code: in the package, outside the tests.

        Parameters
        ----------
        filename : str
            Path of the frame's source file.

        Returns
        -------
        bool
            Whether the file is a package frame but not a test frame.
        """
        return self.is_package_frame(filename) and not self.is_test_frame(filename)

    def relpath(self, filename):
        """
        Package-relative, forward-slash path, for stable baseline keys.

        Parameters
        ----------
        filename : str
            Path of the frame's source file.

        Returns
        -------
        str
            The path relative to the package root, or the absolute path when
            unconfigured or on another drive, with ``/`` separators.
        """
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

        Parameters
        ----------
        relfile : str
            A tally key's file, as returned by `relpath`.

        Returns
        -------
        bool
            Whether the file is a library frame.

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
            None if ``frames`` is empty or holds only this plugin's own
            frames.

        Notes
        -----
        This plugin's own frames are dropped first, then the preference
        order is:

        1. The innermost frame inside the package that is not part of its
           test suite.
        2. Failing that, the innermost frame inside the package at all (this
           will typically be a test-suite frame).
        3. Failing that, the innermost frame overall.

        Dropping the plugin's frames is what makes the result independent of
        the depth at which this is called. The escape logger calls it from
        inside its NumPy wrapper, so the innermost frames of the live stack
        are the plugin's own; while the plugin lives inside the package
        they would win step 2, and once it lives outside it they would win
        step 3.
        """
        frames = [f for f in frames if not _is_plugin_frame(f.filename)]
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
        """
        Short "file:line function" string for a log message.

        Parameters
        ----------
        frame : traceback.FrameSummary or None
            The frame to describe.

        Returns
        -------
        str
            The description, or `UNKNOWN_LOCATION` for None.
        """
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
        """Start with no failures recorded."""
        self.settings = settings
        self.classifier = classifier
        #: Maps a (filename, lineno, function) escape site to the node ids
        #: that failed with that site as their innermost library frame.
        self.sites = defaultdict(list)

    @property
    def active(self):
        """
        True when failure triage was requested for this session.

        Returns
        -------
        bool
            Whether ``<PREFIX>_TRIAGE_ESCAPES`` is truthy.
        """
        return self.settings.triage

    def record_report(self, item, call, report):
        """
        Record the escape site of one failing test report.

        Parameters
        ----------
        item : pytest.Item
            The test the report is for.
        call : pytest.CallInfo
            The result of the test phase, including any exception.
        report : pytest.TestReport
            The report built for that phase.

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

        Parameters
        ----------
        terminalreporter : _pytest.terminal.TerminalReporter
            The reporter to write to.

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
