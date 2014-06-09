# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

import copy
import numbers

import numpy as np

from astropy.nddata import NDData
from astropy.nddata.nduncertainty import StdDevUncertainty, NDUncertainty
from astropy.io import fits, registry
from astropy.utils.compat.odict import OrderedDict
from astropy import units as u
import astropy

from .utils.collections import CaseInsensitiveOrderedDict

adu = u.adu
electron = u.def_unit('electron', doc="Electron count")
u.add_enabled_units([electron])
photon = u.photon

__all__ = ['CCDData', 'electron']


class CCDData(NDData):

    """A class describing basic CCD data

    The CCDData class is based on the NDData object and includes a data array,
    variance frame, mask frame, meta data, units, and WCS information for a
    single CCD image.

    Parameters
    -----------
    data : `~numpy.ndarray` or :class:`~astropy.nddata.NDData`
        The actual data contained in this `~astropy.nddata.NDData` object.
        Note that this will always be copies by *reference* , so you should
        make copy the `data` before passing it in if that's the  desired
        behavior.

    uncertainty : `~astropy.nddata.StdDevUncertainty`, optional
        Uncertainties on the data.

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

    wcs : undefined, optional
        WCS-object containing the world coordinate system for the data.

        .. warning::
            This is not yet defind because the discussion of how best to
            represent this class's WCS system generically is still under
            consideration. For now just leave it as None

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
        super(CCDData, self).__init__(*args, **kwd)
        if self.unit is None:
            raise ValueError("Unit for CCDData must be specified")

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
            self._meta = CaseInsensitiveOrderedDict()
        else:
            h = CaseInsensitiveOrderedDict()
            try:
                for k, v in value.items():
                    h[k] = v
            except (ValueError, AttributeError):
                raise TypeError('NDData meta attribute must be dict-like')
            self._meta = h

    @property
    def uncertainty(self):
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        if value is not None:
            if isinstance(value, NDUncertainty):
                self._uncertainty = value
                self._uncertainty._parent_nddata = self
            else:
                raise TypeError("Uncertainty must be an instance of a "
                                "NDUncertainty object")
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
        from .core import _short_names

        header = fits.Header()
        for k, v in self.header.items():
            if k in _short_names:
                # This keyword was (hopefully) added by autologging but the
                # combination of it and its value FITS-compliant in two ways.
                # Shorten, sort of...
                short_name = _short_names[k]
                header[k] = (short_name, "Shortened name for ccdproc command")
                header[short_name] = v
            else:
                header[k] = v
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


def fits_ccddata_reader(filename, hdu=0, unit=None, **kwd):
    """
    Generate a CCDData object from a FITS file.

    Parameters
    ----------

    filename : str
        Name of fits file.

    hdu : int, optional
        FITS extension from which CCDData should be initialized.

    unit : astropy.units.Unit
        Units of the image data

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
    ccd_data = CCDData(hdus[hdu].data, meta=hdus[hdu].header, unit=unit)
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
