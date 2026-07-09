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

#: When set, compare the live-logged library escape sites against the
#: checked-in baseline and fail the session if any *new* site appears.
ENFORCE_BASELINE = _env_truthy(os.environ.get("CCDPROC_ENFORCE_ESCAPE_BASELINE", ""))

#: When set, (re)write the baseline file from the escapes observed this run
#: instead of enforcing it. Use to seed or refresh the baseline.
WRITE_BASELINE = _env_truthy(os.environ.get("CCDPROC_WRITE_ESCAPE_BASELINE", ""))

#: Maps a (filename, lineno, function) escape site to a list of test node
#: ids that failed with that site as their innermost non-test ccdproc frame.
_ESCAPE_SITES = defaultdict(list)

#: Maps a (relfile, lineno, function, funcname) escape key -- the
#: package-relative file, line and function of the innermost frame the live
#: logger blamed, plus the numpy entry point (e.g. "numpy.asarray") that
#: performed the coercion -- to the number of times that conversion was
#: logged during the session. Populated by the live escape logger in
#: ``conftest.py`` (only when ``CCDPROC_LOG_ARRAY_ESCAPES`` is active).
#: Unlike ``_ESCAPE_SITES`` these escapes do not fail the test -- on dask/jax
#: the conversion succeeds silently -- so the tally is the only way to
#: collapse the streamed warnings into a summary, and it is the observed-set
#: input to the baseline ratchet.
_ESCAPE_LOG_COUNTS = defaultdict(int)


def record_escape_log(frame, funcname):
    """
    Tally one live-logged escape for the end-of-session summary and the
    baseline ratchet. ``frame`` is the FrameSummary chosen by
    ``locate_escape_site()`` (or None if no frame could be found).
    """
    if frame is None:
        key = ("<unknown location>", 0, "", funcname)
    else:
        key = (_relpath(frame.filename), frame.lineno, frame.name, funcname)
    _ESCAPE_LOG_COUNTS[key] += 1


# Anchor frame classification on this file's actual location rather than
# looking for "ccdproc" anywhere in the path: on CI the repository checkout
# directory is itself named ccdproc, so a substring test would classify every
# frame (including tox's site-packages) as a ccdproc frame.
_TESTS_ROOT = os.path.dirname(os.path.abspath(__file__)) + os.sep
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
_CONFTEST_PATH = os.path.join(_PACKAGE_ROOT, "conftest.py")


def _is_ccdproc_frame(filename):
    return os.path.abspath(filename).startswith(_PACKAGE_ROOT)


def _is_ccdproc_test_frame(filename):
    # ccdproc/conftest.py hosts the escape-logger wrapper and other test
    # infrastructure, so it counts as a test frame: it must never be
    # reported as the escape site.
    abspath = os.path.abspath(filename)
    return abspath.startswith(_TESTS_ROOT) or abspath == _CONFTEST_PATH


#: Checked-in list of known library escape sites (the ratchet baseline).
_BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "array_escape_baseline.txt"
)


def _relpath(filename):
    """Package-relative, forward-slash path, for stable baseline keys."""
    try:
        rel = os.path.relpath(os.path.abspath(filename), _PACKAGE_ROOT)
    except ValueError:  # e.g. a different drive on Windows
        rel = os.path.abspath(filename)
    return rel.replace(os.sep, "/")


def _is_library_site(relfile):
    """
    True for escapes blamed on real ccdproc library code -- i.e. not the test
    suite, conftest, or an unknown location. Only library sites go into the
    baseline ratchet: an escape blamed on a test frame is not an actionable
    migration target.
    """
    if relfile == "<unknown location>":
        return False
    abspath = os.path.join(_PACKAGE_ROOT, relfile.replace("/", os.sep))
    return _is_ccdproc_frame(abspath) and not _is_ccdproc_test_frame(abspath)


def _observed_library_sites():
    """Set of (relfile, function, coercion) for the library escapes seen."""
    return {
        (relfile, function, funcname)
        for (relfile, lineno, function, funcname) in _ESCAPE_LOG_COUNTS
        if _is_library_site(relfile)
    }


def _load_escape_baseline():
    """
    Parse the baseline file into ``{(relfile, function, coercion): reason}``.

    Blank lines and ``#`` comments are ignored. Each entry is whitespace
    separated: the first three tokens are the file, function and coercion
    (none of which contain spaces); anything after is a free-text reason/tag
    for humans and is ignored by the ratchet.
    """
    baseline = {}
    try:
        with open(_BASELINE_PATH, encoding="utf-8") as f:
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
        key = (parts[0], parts[1], parts[2])
        baseline[key] = parts[3] if len(parts) > 3 else ""
    return baseline


def _new_escapes():
    """Library escapes observed this run that are absent from the baseline."""
    return sorted(_observed_library_sites() - set(_load_escape_baseline()))


def _stale_baseline_entries():
    """Baseline entries not hit this run (candidates for deletion)."""
    return sorted(set(_load_escape_baseline()) - _observed_library_sites())


def _write_escape_baseline():
    """
    (Re)write the baseline file from the library escapes observed this run.
    Reasons/tags already present in the file are preserved for sites that are
    still observed, so hand-annotations survive a refresh.
    """
    sites = sorted(_observed_library_sites())
    existing = _load_escape_baseline()
    header = [
        "# Array-API escape baseline for non-numpy backends (dask/jax).",
        "# Columns: <file> <function> <coercion>  <reason/tag>",
        "#",
        "# The ratchet (CCDPROC_ENFORCE_ESCAPE_BASELINE=1) fails the session",
        "# if a library escape appears that is not listed here. Delete an",
        "# entry as you migrate that call site; this file only shrinks.",
        "# Regenerate with CCDPROC_WRITE_ESCAPE_BASELINE=1 (preserves tags).",
        "#",
        "# Tags are for humans, not the ratchet: TODO = still to migrate,",
        "# BOUNDARY = a numpy-only dependency (scipy/astroscrappy/reproject)",
        "# that will never leave. Verify/adjust the seeded tags by hand.",
        "#",
    ]
    if not sites:
        _write_lines(header + ["# (no library escapes observed this run)"])
        return
    w_file = max(len(f) for f, _, _ in sites)
    w_func = max(len(fn) for _, fn, _ in sites)
    w_co = max(len(c) for _, _, c in sites)
    body = []
    for key in sites:
        relfile, function, coercion = key
        reason = existing.get(key, "TODO")
        body.append(
            f"{relfile:<{w_file}}  {function:<{w_func}}  "
            f"{coercion:<{w_co}}  {reason}".rstrip()
        )
    _write_lines(header + body)


def _write_lines(lines):
    with open(_BASELINE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Record the escape site of every failing test.

    Runs as a wrapper around report generation for each test phase; only the
    'call' phase (the test body itself, not setup/teardown) of failing tests
    is recorded, so an escape that raises during fixture setup is not
    triaged. That is a deliberate trade-off: fixtures in this test suite are
    mostly plain array/CCDData construction, so escapes surface in the test
    bodies. The failure's traceback is reduced to its most informative frame
    by locate_escape_site() and the test id is filed under that
    (file, line, function) key for the end-of-session summary.
    """
    # With a new-style wrapper, yield hands back the TestReport itself
    # (not a pluggy.Result), and the wrapper must return it.
    report = yield

    if (
        TRIAGE_ACTIVE
        and report.when == "call"
        and report.failed
        and call.excinfo is not None
    ):
        site = locate_escape_site(traceback.extract_tb(call.excinfo.tb))
        if site is not None:
            key = (site.filename, site.lineno, site.name)
            _ESCAPE_SITES[key].append(item.nodeid)

    return report


def pytest_terminal_summary(terminalreporter):
    """
    Print the array-API escape summaries at the end of the test session.

    Two independent sections, either of which may be empty:

    * the failure-triage summary (from ``_ESCAPE_SITES``), grouping test
      failures by root-cause call site, and
    * the live escape-log summary (from ``_ESCAPE_LOG_COUNTS``), collapsing
      the streamed "array-API escape" warnings into per-site counts.
    """
    _report_escape_failures(terminalreporter)
    _report_escape_log_counts(terminalreporter)
    _report_escape_baseline(terminalreporter)


def _report_escape_failures(terminalreporter):
    """
    Print the failure-triage summary.

    One section listing each escape site with its failure count (most
    common first) and up to five example test ids, so a large batch of
    backend failures collapses to a short list of root-cause call sites.
    """
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


def _report_escape_log_counts(terminalreporter):
    """
    Print the live escape-log summary.

    When ``CCDPROC_LOG_ARRAY_ESCAPES`` is active the logger in ``conftest.py``
    streams one warning per silent numpy coercion of a foreign array and
    tallies each call site in ``_ESCAPE_LOG_COUNTS``. These escapes do not
    fail the test -- on dask/jax the conversion succeeds silently -- so this
    collapses the streamed warnings into one deduplicated, most-frequent-first
    list of call sites still to migrate.
    """
    if not _ESCAPE_LOG_COUNTS:
        return

    terminalreporter.section("ccdproc array-API escape log summary")
    total = sum(_ESCAPE_LOG_COUNTS.values())
    n_sites = len(_ESCAPE_LOG_COUNTS)
    terminalreporter.write_line(
        f"{total} silent numpy coercion(s) of foreign arrays across "
        f"{n_sites} call site(s), most frequent first:"
    )

    ordered = sorted(_ESCAPE_LOG_COUNTS.items(), key=lambda kv: kv[1], reverse=True)
    for (relfile, lineno, function, funcname), count in ordered:
        terminalreporter.write_line(
            f"    {count:>5}x  {funcname}()  {relfile}:{lineno} {function}"
        )


def _report_escape_baseline(terminalreporter):
    """
    Print the baseline ratchet result: any new library escapes (which fail
    the session) and any baseline entries no longer hit (safe to delete).
    Only shown when enforcement is active.
    """
    if not ENFORCE_BASELINE:
        return

    # No foreign arrays seen at all means the logger ran on a numpy backend
    # (or was never installed). The baseline is about non-numpy backends, so
    # don't report every entry as "stale" and tempt someone to delete it.
    if not _observed_library_sites():
        terminalreporter.section("ccdproc array-API escape baseline")
        terminalreporter.write_line(
            "No foreign-array escapes observed (numpy backend?); "
            "baseline not checked."
        )
        return

    new = _new_escapes()
    stale = _stale_baseline_entries()

    terminalreporter.section("ccdproc array-API escape baseline")
    if not new:
        terminalreporter.write_line("OK: no library escapes outside the baseline.")
    else:
        terminalreporter.write_line(
            f"NEW escapes not in baseline ({len(new)}) -- these fail the session:"
        )
        for relfile, function, coercion in new:
            terminalreporter.write_line(f"    + {relfile}  {function}  {coercion}")

    if stale:
        terminalreporter.write_line("")
        terminalreporter.write_line(
            f"Baseline entries not hit this run ({len(stale)}) -- delete them "
            "if the migration removed the escape:"
        )
        for relfile, function, coercion in stale:
            terminalreporter.write_line(f"    - {relfile}  {function}  {coercion}")


def pytest_sessionfinish(session, exitstatus):
    """
    Baseline ratchet regeneration / enforcement.

    In write mode (CCDPROC_WRITE_ESCAPE_BASELINE) the baseline file is
    rewritten from the escapes observed this run. In enforce mode
    (CCDPROC_ENFORCE_ESCAPE_BASELINE) the session exit status is forced
    nonzero when a new library escape appeared, so CI fails on a regression.
    A pre-existing nonzero status (real test failures) is left untouched.
    """
    if WRITE_BASELINE:
        _write_escape_baseline()
        return
    if ENFORCE_BASELINE and exitstatus == 0 and _new_escapes():
        session.exitstatus = 1
