# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import numbers

import numpy as np

from astropy.nddata import NDDataArray
from astropy.nddata.nduncertainty import StdDevUncertainty, NDUncertainty
from astropy.io import fits, registry
from astropy import units as u
from astropy import log
from astropy.wcs import WCS

from .utils.collections import CaseInsensitiveOrderedDict

__all__ = ['CCDData']


class CCDData(NDDataArray):

    """A class describing basic CCD data

    The CCDData class is based on the NDData object and includes a data array,
    uncertainty frame, mask frame, meta data, units, and WCS information for a
    single CCD image.

    Parameters
    -----------
    data : `~numpy.ndarray` or :class:`~astropy.nddata.NDData`
        The actual data contained in this `~astropy.nddata.NDData` object.
        Note that this will always be copies by *reference* , so you should
        make copy the `data` before passing it in if that's the  desired
        behavior.

    uncertainty : `~astropy.nddata.StdDevUncertainty` or `~numpy.ndarray`,
        optional Uncertainties on the data.

    mask : `~numpy.ndarray`, optional
        Mask for the data, given as a boolean Numpy array with a shape
        matching that of the data. The values must be ``False`` where
        the data is *valid* and ``True`` when it is not (like Numpy
        masked arrays). If `data` is a numpy masked array, providing
        `mask` here will causes the mask from the masked array to be
        ignored.

    flags : `~numpy.ndarray` or `~astropy.nddata.FlagCollection`, optional
        Flags giving information about each pixel. These can be specified
        either as a Numpy array of any type with a shape matching that of the
        data, or as a `~astropy.nddata.FlagCollection` instance which has a
        shape matching that of the data.

    wcs : `~astropy.wcs.WCS` object, optional
        WCS-object containing the world coordinate system for the data.

    meta : `dict`-like object, optional
        Metadata for this object.  "Metadata" here means all information that
        is included with this object but not part of any other attribute
        of this particular object.  e.g., creation date, unique identifier,
        simulation parameters, exposure time, telescope name, etc.

    unit : `~astropy.units.Unit` instance or str, optional
        The units of the data.

    Raises
    ------
    ValueError
        If the `uncertainty` or `.mask` inputs cannot be broadcast (e.g., match
        shape) onto `data`.

    Notes
    -----
    `NDData` objects can be easily converted to a regular Numpy array
    using `numpy.asarray`

    For example::

        >>> from astropy.nddata import NDData
        >>> import numpy as np
        >>> x = NDData([1,2,3])
        >>> np.asarray(x)
        array([1, 2, 3])

    If the `~astropy.nddata.NDData` object has a `mask`, `numpy.asarray` will
    return a Numpy masked array.

    This is useful, for example, when plotting a 2D image using
    matplotlib::

        >>> from astropy.nddata import NDData
        >>> from matplotlib import pyplot as plt
        >>> x = NDData([[1,2,3], [4,5,6]])
        >>> plt.imshow(x)

    """
    def __init__(self, *args, **kwd):
        if 'meta' not in kwd:
            kwd['meta'] = kwd.pop('header', None)
        if 'header' in kwd:
            raise ValueError("Can't have both header and meta")
        
        super(CCDData, self).__init__(*args, **kwd)
        if self.unit is None:
            raise ValueError("Unit for CCDData must be specified")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        if value is not None:
            self._unit = u.Unit(value)
        else:
            self._unit = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def header(self):
        return self._meta

    @header.setter
    def header(self, value):
        self.meta = value

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        if value is None:
            self._meta = {}
        else:
            if hasattr(value, 'keys'):
                self._meta = value
            else:
                raise TypeError('NDData meta attribute must be dict-like')

    @property
    def uncertainty(self):
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        if value is not None:
            if isinstance(value, NDUncertainty):
                self._uncertainty = value
            elif isinstance(value, np.ndarray):
                if value.shape != self.shape:
                    raise ValueError("Uncertainty must have same shape as "
                                     "data")
                self._uncertainty = StdDevUncertainty(value)
                log.info("Array provided for uncertainty; assuming it is a "
                         "StdDevUncertainty.")
            else:
                raise TypeError("Uncertainty must be an instance of a "
                                "NDUncertainty object or a numpy array.")
            self._uncertainty._parent_nddata = self
        else:
            self._uncertainty = value

    def to_hdu(self):
        """Creates an HDUList object from a CCDData object.

           Raises
           -------
           ValueError
              Multi-Exenstion FITS files are not supported

           Returns
           -------
           hdulist : astropy.io.fits.HDUList object

        """
        if isinstance(self.header, fits.Header):
            # Copy here so that we can modify the HDU header by adding WCS
            # information without changing the header of the CCDData object.
            header = self.header.copy()
        else:
            # Because _insert_in_metadata_fits_safe is written as a method
            # we need to create a dummy CCDData instance to hold the FITS
            # header we are constructing. This probably indicates that
            # _insert_in_metadata_fits_safe should be rewritten in a more
            # sensible way...
            dummy_ccd = CCDData(data=[1], meta=fits.Header(), unit="adu")
            for k, v in self.header.items():
                dummy_ccd._insert_in_metadata_fits_safe(k, v)
            header = dummy_ccd.header
        if self.unit is not u.dimensionless_unscaled:
            header['bunit'] = self.unit.to_string()
        if self.wcs:
            # Simply extending the FITS header with the WCS can lead to
            # duplicates of the WCS keywords; iterating over the WCS
            # header should be safer.
            #
            # Turns out if I had read the io.fits.Header.extend docs more
            # carefully, I would have realized that the keywords exist to
            # avoid duplicates and preserve, as much as possible, the
            # structure of the commentary cards.
            #
            # Note that until astropy/astropy#3967 is closed, the extend
            # will fail if there are comment cards in the WCS header but
            # not header.
            wcs_header = self.wcs.to_header()
            header.extend(wcs_header, useblanks=False, unique=True)
        hdu = fits.PrimaryHDU(self.data, header)
        hdulist = fits.HDUList([hdu])
        return hdulist

    def copy(self):
        """
        Return a copy of the CCDData object.
        """
        return copy.deepcopy(self)

    def _ccddata_arithmetic(self, other, operation, scale_uncertainty=False):
        """
        Perform the common parts of arithmetic operations on CCDData objects

        This should only be called when `other` is a Quantity or a number
        """
        # THE "1 *" IS NECESSARY to get the right result, at least in
        # astropy-0.4dev. Using the np.multiply, etc, methods with a Unit
        # and a Quantity is currently broken, but it works with two Quantity
        # arguments.
        if isinstance(other, u.Quantity):
            other_value = other.value
        elif isinstance(other, numbers.Number):
            other_value = other
        else:
            raise TypeError("Cannot do arithmetic with type '{0}' "
                            "and 'CCDData'".format(type(other)))

        result_unit = operation(1 * self.unit, other).unit
        result_data = operation(self.data, other_value)

        if self.uncertainty:
            result_uncertainty = self.uncertainty.array
            if scale_uncertainty:
                result_uncertainty = operation(result_uncertainty, other_value)
            result_uncertainty = StdDevUncertainty(result_uncertainty)
        else:
            result_uncertainty = None

        result = CCDData(data=result_data, unit=result_unit,
                         uncertainty=result_uncertainty,
                         meta=self.meta)
        return result

    def multiply(self, other):
        if isinstance(other, CCDData):
            return super(CCDData, self).multiply(other)

        return self._ccddata_arithmetic(other, np.multiply,
                                        scale_uncertainty=True)

    def divide(self, other):
        if isinstance(other, CCDData):
            return super(CCDData, self).divide(other)

        return self._ccddata_arithmetic(other, np.divide,
                                        scale_uncertainty=True)

    def add(self, other):
        if isinstance(other, CCDData):
            return super(CCDData, self).add(other)

        return self._ccddata_arithmetic(other, np.add,
                                        scale_uncertainty=False)

    def subtract(self, other):
        if isinstance(other, CCDData):
            return super(CCDData, self).subtract(other)

        return self._ccddata_arithmetic(other, np.subtract,
                                        scale_uncertainty=False)

    def _insert_in_metadata_fits_safe(self, key, value):
        """
        Insert key/value pair into metadata in a way that FITS can serialize.

        Parameters
        ----------

        key : str
            Key to be inserted in dictionary.

        value : str or None
            Value to be inserted.

        Notes
        -----

        This addresses a shortcoming of the FITS standard. There are length
        restrictions on both the ``key`` (8 characters) and ``value`` (72
        characters) in the FITS standard. There is a convention for handline
        long keywords and a convention for handling long values, but the
        two conventions cannot be used at the same time.

        Autologging in `ccdproc` frequently creates keywords/values with this
        combination. The workaround is to use a shortened name for the keyword.
        """
        from .core import _short_names

        if key in _short_names and isinstance(self.meta, fits.Header):
            # This keyword was (hopefully) added by autologging but the
            # combination of it and its value not FITS-compliant in two
            # ways: the keyword name may be more than 8 characters and
            # the value may be too long. FITS cannot handle both of
            # those problems at once, so this fixes one of those
            # problems...
            # Shorten, sort of...
            short_name = _short_names[key]
            self.meta[key] = (short_name, "Shortened name for ccdproc command")
            self.meta[short_name] = value
        else:
            self.meta[key] = value


def fits_ccddata_reader(filename, hdu=0, unit=None, **kwd):
    """
    Generate a CCDData object from a FITS file.

    Parameters
    ----------

    filename : str
        Name of fits file.

    hdu : int, optional
        FITS extension from which CCDData should be initialized.  If zero and
        and no data in the primary extention, it will search for the first
        extension with data.

    unit : astropy.units.Unit, optional
        Units of the image data. If this argument is provided and there is a
        unit for the image in the FITS header (the keyword ``BUNIT`` is used
        as the unit, if present), this argument is used for the unit.

    kwd :
        Any additional keyword parameters are passed through to the FITS reader
        in :mod:`astropy.io.fits`; see Notes for additional discussion.

    Notes
    -----

    FITS files that contained scaled data (e.g. unsigned integer images) will
    be scaled and the keywords used to manage scaled data in
    :mod:`astropy.io.fits` are disabled.
    """
    unsupport_open_keywords = {
        'do_not_scale_image_data': ('Image data must be scaled to perform '
                                    'ccdproc operations.'),
        'scale_back': 'Scale information is not preserved.'
    }
    for key, msg in unsupport_open_keywords.items():
        if key in kwd:
            prefix = 'Unsupported keyword: {0}.'.format(key)
            raise TypeError(' '.join([prefix, msg]))
    hdus = fits.open(filename, **kwd)
    hdr = hdus[hdu].header

    # search for the first instance with data if the primary header is empty
    if hdu == 0 and hdus[hdu].data is None:
        for i in range(len(hdus)):
            if hdus.fileinfo(i)['datSpan'] > 0:
                hdu = i
                log.info("First HDU with data is exention {0}".format(hdu))
                break

    try:
        fits_unit_string = hdr['bunit']
        # patch to handle FITS files using ADU for the unit instead of the
        # standard version of 'adu'
        if fits_unit_string.strip().lower() == 'adu':
            fits_unit_string = fits_unit_string.lower()
    except KeyError:
        fits_unit_string = None

    if unit is not None and fits_unit_string:
        log.info("Using the unit {0} passed to the FITS reader instead of "
                 "the unit {1} in the FITS file.".format(unit,
                                                         fits_unit_string))

    use_unit = unit or fits_unit_string
    # Try constructing a WCS object. This may generate a warning, but never
    # an error.
    wcs = WCS(hdr)
    # Test for success by checking to see if the wcs ctype has a non-empty
    # value.
    wcs = wcs if wcs.wcs.ctype[0] else None
    ccd_data = CCDData(hdus[hdu].data, meta=hdus[hdu].header, unit=use_unit,
                       wcs=wcs)
    hdus.close()
    return ccd_data


def fits_ccddata_writer(ccd_data, filename, **kwd):
    """
    Write CCDData object to FITS file.

    Parameters
    ----------

    filename : str
        Name of file

    kwd :
        All additional keywords are passed to :py:mod:`astropy.io.fits`
    """
    hdu = ccd_data.to_hdu()
    hdu.writeto(filename, **kwd)


registry.register_reader('fits', CCDData, fits_ccddata_reader)
registry.register_writer('fits', CCDData, fits_ccddata_writer)
registry.register_identifier('fits', CCDData, fits.connect.is_fits)
