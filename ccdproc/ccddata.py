# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.

from astropy.nddata import NDData
from astropy.nddata.nduncertainty import StdDevUncertainty, NDUncertainty
from astropy.io import fits
from astropy.utils.compat.odict import OrderedDict
from astropy import units as u

adu = u.def_unit('ADU')
electrons = u.def_unit('electrons')
photons = u.def_unit('photons')


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
            self._meta = OrderedDict()
        elif isinstance(value, fits.Header):
            self._meta = value
        else:
            try:
                self._meta = OrderedDict(value)
            except ValueError:
                raise TypeError('NDData meta attribute must be dict-like')

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

    def create_variance(self, readnoise):
        """Create a variance frame.  The function will update the uncertainty
           plane which gives the variance for the data.  The function assumes
           that the ccd is in electrons and the readnoise is in the same units.

        Parameters
        ----------
        readnoise :  float
          readnoise for each pixel

        Raises
        ------
        TypeError :
          raises TypeError if units are not in electrons

        Returns
        -------
        ccd :  CCDData object
          CCDData object with uncertainty created
        """
        if self.unit != electrons:
            raise TypeError('CCDData object is not in electrons')

        var = (self.data + readnoise ** 2) ** 0.5
        self.uncertainty = StdDevUncertainty(var)


def fromFITS(hdu, units=None):
    """Creates a CCDData object from a FITS file

       Parameters
       ----------
       hdu : astropy.io.fits object
          FITS object fo the CCDData object

       units : astropy.units object
          Unit describing the data

       Raises
       -------
       ValueError
          Multi-Exenstion FITS files are not supported

       Returns
       -------
       ccddata : ccddata.CCDData object

    """
    if len(hdu) > 1:
        raise ValueError('Multi-Exenstion FITS files are not supported')

    return CCDData(hdu[0].data, meta=hdu[0].header, unit=units)


def toFITS(ccddata):
    """Creates an HDUList object from a CCDData object

       Parameters
       ----------
       ccddata : CCDData object
          CCDData object to create FITS file

       Raises
       -------
       ValueError
          Multi-Exenstion FITS files are not supported

       Returns
       -------
       hdulist : astropy.io.fits.HDUList object

    """
    hdu = fits.PrimaryHDU(ccddata.data, ccddata.header)
    hdulist = fits.HDUList([hdu])
    return hdulist
