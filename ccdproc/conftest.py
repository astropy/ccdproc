# Licensed under a 3-clause BSD style license - see LICENSE.rst

# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

try:
    # When the pytest_astropy_header package is installed
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

    def pytest_configure(config):
        config.option.astropy_header = True

except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}


from .tests.pytest_fixtures import (
    triage_setup,  # noqa: F401 this is used in tests
)

# This is to figure out ccdproc version, rather than using Astropy's
try:
    from ccdproc import __version__ as version
except ImportError:
    version = "dev"

TESTED_VERSIONS["ccdproc"] = version

# Add astropy to test header information and remove unused packages.
PYTEST_HEADER_MODULES["Astropy"] = "astropy"
PYTEST_HEADER_MODULES["astroscrappy"] = "astroscrappy"
PYTEST_HEADER_MODULES["reproject"] = "reproject"
PYTEST_HEADER_MODULES.pop("h5py", None)
