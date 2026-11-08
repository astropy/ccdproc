# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Hook specifications of the array-API escape plugin.

A package configures the plugin by implementing
`pytest_array_api_escapes_config` in the ``conftest.py`` that loads it. The
ini options declared in `.config` remain available, and override the hook
value by value, for projects that would rather configure the plugin there.

Notes
-----
The hook is the primary route because a ``conftest.py`` ships with the tests
it configures: it is found whenever those tests run, including against an
installed copy of the package from a directory with no ini file, which is
exactly the case an ini-only configuration misses.
"""

import pytest

#: Keys a `pytest_array_api_escapes_config` implementation may return.
CONFIG_KEYS = frozenset(
    {"package", "test_paths", "baseline", "env_prefix", "logger", "docs_url"}
)


@pytest.hookspec(firstresult=True)
def pytest_array_api_escapes_config(config):
    """
    Return the package's settings for the array-API escape plugin.

    Parameters
    ----------
    config : pytest.Config
        The pytest config object of this session.

    Returns
    -------
    dict or None
        Settings keyed by any of ``"package"``, ``"test_paths"``,
        ``"baseline"``, ``"env_prefix"`` and ``"logger"``, which have the
        same meaning as the ``array_api_escapes_*`` ini options of the same
        name, plus ``"docs_url"``, a documentation link quoted in the error
        for an unsupported array library. Missing keys take the plugin's
        defaults. Return None to leave the plugin unconfigured.

    Notes
    -----
    This is a ``firstresult`` hook: the first implementation returning
    something other than None wins, which is the one in the innermost
    ``conftest.py``. An ini option that is set takes precedence over the
    value returned here.
    """
