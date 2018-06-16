# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements the base CCDData class."""

from astropy.nddata import fits_ccddata_reader, fits_ccddata_writer, CCDData


__all__ = ['CCDData', 'fits_ccddata_reader', 'fits_ccddata_writer']


# This should be be a tuple to ensure it isn't inadvertently changed
# elsewhere.
_recognized_fits_file_extensions = ('fit', 'fits', 'fts')
