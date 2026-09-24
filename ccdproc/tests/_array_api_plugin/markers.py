# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Per-backend skip/xfail markers.

::

    @pytest.mark.backend_xfail("cupy", "array-api-strict", reason="...")
    @pytest.mark.backend_skip("cupy", reason="...")

These mark a test as an expected failure or a skip only when it runs against
particular array backends, without affecting the default NumPy run.

Notes
-----
The xfail added here is deliberately *non-strict*, even in suites that set
``xfail_strict = true``: a backend bug that is later fixed then shows up as
an XPASS instead of failing the run. The cost is that stale markers linger,
so check for XPASSes (``-rX``) occasionally and prune.
"""

import pytest

from .backend import normalize_backend_name

SKIP_MARKER = "backend_skip"
XFAIL_MARKER = "backend_xfail"


def register_markers(config, settings):
    """
    Declare the two markers so ``--strict-markers`` accepts them.

    Notes
    -----
    The descriptions name the environment variable that selects the backend,
    which depends on the configured prefix, so they are built here rather
    than written into a project's ini file.
    """
    env_var = settings.env_name("ARRAY_LIBRARY")
    config.addinivalue_line(
        "markers",
        f"{XFAIL_MARKER}(*backends, reason=...): mark test as an expected "
        f"(non-strict) failure when the active {env_var} matches one of the "
        "named backends",
    )
    config.addinivalue_line(
        "markers",
        f"{SKIP_MARKER}(*backends, reason=...): skip test when the active "
        f"{env_var} matches one of the named backends",
    )


def apply_backend_markers(items, active_backend):
    """
    Turn the two markers into plain ``skip``/``xfail`` markers at collection.

    Parameters
    ----------
    items : list
        The collected test items, modified in place.
    active_backend : str
        Normalized name of the backend this run uses.

    Notes
    -----
    Backend names in a marker's positional args are normalized before
    comparison, so ``array_api_strict`` and ``array-api-strict`` both match.
    The first matching marker of each kind decides the outcome. Tests without
    a matching marker, including everything in a default NumPy run, are
    untouched.
    """
    for item in items:
        for marker in item.iter_markers(name=SKIP_MARKER):
            backends = {normalize_backend_name(b) for b in marker.args}
            if active_backend in backends:
                reason = marker.kwargs.get(
                    "reason", f"skipped for array backend {active_backend!r}"
                )
                item.add_marker(pytest.mark.skip(reason=reason))
                break

        for marker in item.iter_markers(name=XFAIL_MARKER):
            backends = {normalize_backend_name(b) for b in marker.args}
            if active_backend in backends:
                reason = marker.kwargs.get(
                    "reason", f"expected failure for array backend {active_backend!r}"
                )
                item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
                break
