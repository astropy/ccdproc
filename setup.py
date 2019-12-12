#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from distutils.version import LooseVersion
import sys

try:
    import setuptools
    assert LooseVersion(setuptools.__version__) >= LooseVersion('30.3')
except (ImportError, AssertionError):
    sys.stderr.write('ERROR: setuptools 30.3 or later is required\n')
    sys.exit(1)

from setuptools import setup

from setup_commands import cmdclass
from setup_extensions import ext_modules

setup(use_scm_version=True, ext_modules=ext_modules, cmdclass=cmdclass)
