# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The ccdproc package is a collection of code that will be helpful in basic CCD
processing. These steps will allow reduction of basic CCD data as either a
stand-alone processing or as part of a pipeline.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# set up namespace, unless we are in setup...
if not _ASTROPY_SETUP_:
    from .core import *
    from .ccddata import *
    from .combiner import *
    from .image_collection import *
