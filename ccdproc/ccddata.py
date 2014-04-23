# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

import copy

from astropy.nddata import NDData
from astropy.nddata.nduncertainty import StdDevUncertainty, NDUncertainty
from astropy.io import fits, registry
from astropy.utils.compat.odict import OrderedDict
from astropy import units as u
import astropy

adu = u.adu
electron = u.def_unit('electron')
photon = u.photon


class CCDData(NDData):

    """A class describing basic CCD data

    The CCDData class is based on the NDData object and includes a data array,
    variance frame, mask frame, meta data, units, and WCS information for a
    single CCD image.

    Parameters
    -----------
    data : `~numpy.ndarray` or `~astropy.nddata.NDData`
        The actual data contained in this `NDData` object. Not that this
        will always be copies by *reference* , so you should make copy
        the `data` before passing it in if that's the  desired behavior.

    uncertainty : `~astropy.nddata.NDUncertainty`, optional
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

    unit : `astropy.units.UnitBase` instance or str, optional
        The units of the data.

    Raises
    ------
    ValueError
        If the `uncertainty` or `mask` inputs cannot be broadcast (e.g., match
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

    If the `NDData` object has a `mask`, `numpy.asarray` will return a
    Numpy masked array.

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
            self._meta = fits.Header()
        elif isinstance(value, fits.Header):
            self._meta = value
        else:
            h = fits.Header()
            try:
                for k, v in value.iteritems():
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
        """Creates an HDUList object from a CCDData object

           Raises
           -------
           ValueError
              Multi-Exenstion FITS files are not supported

           Returns
           -------
           hdulist : astropy.io.fits.HDUList object

        """
        hdu = fits.PrimaryHDU(self.data, self.header)
        hdulist = fits.HDUList([hdu])
        return hdulist

    def copy(self):
        """
        Return a copy of the CCDData object
        """
        return copy.deepcopy(self)


def fits_ccddata_reader(filename, hdu=0, unit=None, **kwd):
    """
    Generate a CCDData object from a FITS file

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
    for key, msg in unsupport_open_keywords.iteritems():
        if key in kwd:
            prefix = 'Unsupported keyword: {0}.'.format(key)
            raise TypeError(' '.join([prefix, msg]))
    hdus = fits.open(filename, **kwd)
    ccd_data = CCDData(hdus[hdu].data, meta=hdus[hdu].header, unit=unit)
    hdus.close()
    return ccd_data


def fits_ccddata_writer(ccd_data, filename, **kwd):
    """
    Write CCDData object to FITS file

    Parameters
    ----------

    filename : str
        Name of file

    kwd :
        All additional keywords are passed to :py:mod:`astropy.io.fits`
    """
    hdu = fits.PrimaryHDU(data=ccd_data.data, header=ccd_data.header)
    hdu.writeto(filename, **kwd)


registry.register_reader('fits', CCDData, fits_ccddata_reader)
registry.register_writer('fits', CCDData, fits_ccddata_writer)
registry.register_identifier('fits', CCDData, fits.connect.is_fits)
