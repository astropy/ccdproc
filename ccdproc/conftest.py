# Licensed under a 3-clause BSD style license - see LICENSE.rst

# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

import os

try:
    # When the pytest_astropy_header package is installed
    from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)

    def pytest_configure(config):
        config.option.astropy_header = True
except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}


from .tests.pytest_fixtures import *

# This is to figure out ccdproc version, rather than using Astropy's
try:
    from .version import version
except ImportError:
    version = 'dev'

packagename = os.path.basename(os.path.dirname(__file__))
TESTED_VERSIONS[packagename] = version

# Add astropy to test header information and remove unused packages.

try:
    PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
    PYTEST_HEADER_MODULES['astroscrappy'] = 'astroscrappy'
    PYTEST_HEADER_MODULES['reproject'] = 'reproject'
    del PYTEST_HEADER_MODULES['h5py']
except KeyError:
    pass
