# Licensed under a 3-clause BSD style license - see LICENSE.rst

# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

import os

try:
    from astropy.tests.plugins.display import (pytest_report_header,
                                               PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)
except ImportError:
    # When using astropy 2.0
    from astropy.tests.pytest_plugins import (pytest_report_header,
                                              PYTEST_HEADER_MODULES,
                                              TESTED_VERSIONS)

try:
    # This is the way to get plugins in astropy 2.x
    from astropy.tests.pytest_plugins import *
except ImportError:
    # Otherwise they are installed as separate packages that pytest
    # automagically finds.
    pass

from .tests.pytest_fixtures import *

# This is to figure out ccdproc version, rather than using Astropy's
try:
    from .version import version
except ImportError:
    version = 'dev'

packagename = os.path.basename(os.path.dirname(__file__))
TESTED_VERSIONS[packagename] = version

# Uncomment the following line to treat all DeprecationWarnings as
# exceptions
# enable_deprecations_as_exceptions()

# Add astropy to test header information and remove unused packages.

try:
    PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
    PYTEST_HEADER_MODULES['astroscrappy'] = 'astroscrappy'
    PYTEST_HEADER_MODULES['reproject'] = 'reproject'
    del PYTEST_HEADER_MODULES['h5py']
except KeyError:
    pass
