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

# set up the version
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = 'unknown'

# set up namespace, unless we are in setup...
if not _ASTROPY_SETUP_:
    from .core import *
    from .ccddata import *
    from .combiner import *
    from .image_collection import *
    from astropy import config as _config

    class Conf(_config.ConfigNamespace):
        """
        Configuration parameters for ccdproc.
        """
        auto_logging = _config.ConfigItem(
            True,
            'Whether to automatically log operations to metadata'
            'If set to False, there is no need to specify add_keyword=False'
            'when calling processing operations.'
            )
    conf = Conf()
