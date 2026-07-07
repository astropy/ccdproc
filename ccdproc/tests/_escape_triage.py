# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Pytest hooks used to triage array-API "escape" failures: cases where a
non-numpy array unexpectedly gets silently converted to (or otherwise
touches) plain numpy deep inside ccdproc. Such conversions are usually
harmless on numpy itself, but will raise loudly on backends like CuPy
(because the array lives on a GPU) and are the kind of bug this tooling is
meant to help find on other backends too.

Activated by setting the environment variable ``CCDPROC_TRIAGE_ESCAPES`` to
a truthy value ("1", "true", "yes", "on", case-insensitive). When active,
every test failure's traceback is inspected to find the "escape site": the
innermost frame that is inside the ``ccdproc`` package but *not* inside
``ccdproc``'s own test suite (``ccdproc/tests``). If no such frame is found,
falls back to the innermost ``ccdproc`` frame, and finally to the innermost
frame overall. Failures are grouped by that site and a summary is printed at
the end of the test session (most frequent site first). This is meant to
replace manually eyeballing ``--tb=long`` output to find patterns across a
batch of failures.
"""

import os
import traceback
from collections import defaultdict

import pytest

_TRUTHY = {"1", "true", "yes", "on"}


def _env_truthy(value):
    return str(value).strip().lower() in _TRUTHY


TRIAGE_ACTIVE = _env_truthy(os.environ.get("CCDPROC_TRIAGE_ESCAPES", ""))

#: Maps a (filename, lineno, function) escape site to a list of test node
#: ids that failed with that site as their innermost non-test ccdproc frame.
_ESCAPE_SITES = defaultdict(list)


def _is_ccdproc_frame(filename):
    return f"{os.sep}ccdproc{os.sep}" in filename


def _is_ccdproc_test_frame(filename):
    return f"{os.sep}ccdproc{os.sep}tests{os.sep}" in filename


def locate_escape_site(frames):
    """
    Given an iterable of frame-summary-like objects (as returned by
    ``traceback.extract_tb`` or ``traceback.extract_stack``, i.e. anything
    with ``.filename``, ``.lineno`` and ``.name`` attributes), find the
    frame most likely responsible for an array-API "escape".

    Preference order:

    1. The innermost frame inside the ``ccdproc`` package that is not part
       of ``ccdproc``'s own test suite.
    2. If there is no such frame, the innermost frame inside ``ccdproc``
       at all (this will typically be a test-suite frame).
    3. If there is no ``ccdproc`` frame at all, the innermost frame overall.

    Returns ``None`` if given no frames at all.
    """
    frames = list(frames)
    if not frames:
        return None

    non_test_ccdproc = [
        f
        for f in frames
        if _is_ccdproc_frame(f.filename) and not _is_ccdproc_test_frame(f.filename)
    ]
    if non_test_ccdproc:
        return non_test_ccdproc[-1]

    ccdproc_frames = [f for f in frames if _is_ccdproc_frame(f.filename)]
    if ccdproc_frames:
        return ccdproc_frames[-1]

    return frames[-1]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield

    if not TRIAGE_ACTIVE:
        return

    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return

    excinfo = call.excinfo
    if excinfo is None:
        return

    site = locate_escape_site(traceback.extract_tb(excinfo.tb))
    if site is None:
        return

    key = (site.filename, site.lineno, site.name)
    _ESCAPE_SITES[key].append(item.nodeid)


def pytest_terminal_summary(terminalreporter):
    if not TRIAGE_ACTIVE or not _ESCAPE_SITES:
        return

    terminalreporter.section("ccdproc array-API escape triage")
    terminalreporter.write_line(
        "Failures grouped by innermost non-test ccdproc frame "
        "(file:line function), most common first:"
    )

    ordered = sorted(_ESCAPE_SITES.items(), key=lambda kv: len(kv[1]), reverse=True)
    example_limit = 5
    for (filename, lineno, function), test_ids in ordered:
        terminalreporter.write_line("")
        terminalreporter.write_line(
            f"{filename}:{lineno} {function}  ({len(test_ids)} failures)"
        )
        for test_id in test_ids[:example_limit]:
            terminalreporter.write_line(f"    - {test_id}")
        if len(test_ids) > example_limit:
            terminalreporter.write_line(
                f"    ... and {len(test_ids) - example_limit} more"
            )
