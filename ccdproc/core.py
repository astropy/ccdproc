# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements the base CCDPROC functions"""

import math
import numbers
import logging
import warnings

import numpy as np
from scipy import ndimage

from astropy.units.quantity import Quantity
from astropy import units as u
from astropy.modeling import fitting
from astropy import stats
from astropy.nddata import utils as nddata_utils
from astropy.nddata import StdDevUncertainty, CCDData
from astropy.wcs.utils import proj_plane_pixel_area
from astropy.utils import deprecated
import astropy  # To get the version.

from .utils.slices import slice_from_string
from .log_meta import log_to_metadata
from .extern.bitfield import bitfield_to_boolean_mask as _bitfield_to_boolean_mask

logger = logging.getLogger(__name__)

__all__ = ['background_deviation_box', 'background_deviation_filter',
           'ccd_process', 'cosmicray_median', 'cosmicray_lacosmic',
           'create_deviation', 'flat_correct', 'gain_correct', 'rebin',
           'sigma_func', 'subtract_bias', 'subtract_dark', 'subtract_overscan',
           'transform_image', 'trim_image', 'wcs_project', 'Keyword',
           'median_filter', 'ccdmask', 'bitfield_to_boolean_mask']

# The dictionary below is used to translate actual function names to names
# that are FITS compliant, i.e. 8 characters or less.
_short_names = {
    'background_deviation_box': 'bakdevbx',
    'background_deviation_filter': 'bakdfilt',
    'ccd_process': 'ccdproc',
    'cosmicray_median': 'crmedian',
    'create_deviation': 'creatvar',
    'flat_correct': 'flatcor',
    'gain_correct': 'gaincor',
    'subtract_bias': 'subbias',
    'subtract_dark': 'subdark',
    'subtract_overscan': 'suboscan',
    'trim_image': 'trimim',
    'transform_image': 'tranim',
    'wcs_project': 'wcsproj'
}


@log_to_metadata
def ccd_process(ccd, oscan=None, trim=None, error=False, master_bias=None,
                dark_frame=None, master_flat=None, bad_pixel_mask=None,
                gain=None, readnoise=None, oscan_median=True, oscan_model=None,
                min_value=None, dark_exposure=None, data_exposure=None,
                exposure_key=None, exposure_unit=None,
                dark_scale=False, gain_corrected=True):
    """Perform basic processing on ccd data.

    The following steps can be included:

    * overscan correction (:func:`subtract_overscan`)
    * trimming of the image (:func:`trim_image`)
    * create deviation frame (:func:`create_deviation`)
    * gain correction (:func:`gain_correct`)
    * add a mask to the data
    * subtraction of master bias (:func:`subtract_bias`)
    * subtraction of a dark frame (:func:`subtract_dark`)
    * correction of flat field (:func:`flat_correct`)

    The task returns a processed `~astropy.nddata.CCDData` object.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        Frame to be reduced.

    oscan : `~astropy.nddata.CCDData`, str or None, optional
        For no overscan correction, set to None. Otherwise provide a region
        of ccd from which the overscan is extracted, using the FITS
        conventions for index order and index start, or a
        slice from ccd that contains the overscan.
        Default is ``None``.

    trim : str or None, optional
        For no trim correction, set to None. Otherwise provide a region
        of ccd from which the image should be trimmed, using the FITS
        conventions for index order and index start.
        Default is ``None``.

    error : bool, optional
        If True, create an uncertainty array for ccd.
        Default is ``False``.

    master_bias : `~astropy.nddata.CCDData` or None, optional
        A master bias frame to be subtracted from ccd. The unit of the
        master bias frame should match the unit of the image **after
        gain correction** if ``gain_corrected`` is True.
        Default is ``None``.

    dark_frame : `~astropy.nddata.CCDData` or None, optional
        A dark frame to be subtracted from the ccd. The unit of the
        master dark frame should match the unit of the image **after
        gain correction** if ``gain_corrected`` is True.
        Default is ``None``.

    master_flat : `~astropy.nddata.CCDData` or None, optional
        A master flat frame to be divided into ccd. The unit of the
        master flat frame should match the unit of the image **after
        gain correction** if ``gain_corrected`` is True.
        Default is ``None``.

    bad_pixel_mask : `numpy.ndarray` or None, optional
        A bad pixel mask for the data. The bad pixel mask should be in given
        such that bad pixels have a value of 1 and good pixels a value of 0.
        Default is ``None``.

    gain : `~astropy.units.Quantity` or None, optional
        Gain value to multiple the image by to convert to electrons.
        Default is ``None``.

    readnoise : `~astropy.units.Quantity` or None, optional
        Read noise for the observations. The read noise should be in
        electrons.
        Default is ``None``.

    oscan_median : bool, optional
        If true, takes the median of each line. Otherwise, uses the mean.
        Default is ``True``.

    oscan_model : `~astropy.modeling.Model` or None, optional
        Model to fit to the data. If None, returns the values calculated
        by the median or the mean.
        Default is ``None``.

    min_value : float or None, optional
        Minimum value for flat field. The value can either be None and no
        minimum value is applied to the flat or specified by a float which
        will replace all values in the flat by the min_value.
        Default is ``None``.

    dark_exposure : `~astropy.units.Quantity` or None, optional
        Exposure time of the dark image; if specified, must also provided
        ``data_exposure``.
        Default is ``None``.

    data_exposure : `~astropy.units.Quantity` or None, optional
        Exposure time of the science image; if specified, must also provided
        ``dark_exposure``.
        Default is ``None``.

    exposure_key : `~ccdproc.Keyword`, str or None, optional
        Name of key in image metadata that contains exposure time.
        Default is ``None``.

    exposure_unit : `~astropy.units.Unit` or None, optional
        Unit of the exposure time if the value in the meta data does not
        include a unit.
        Default is ``None``.

    dark_scale : bool, optional
        If True, scale the dark frame by the exposure times.
        Default is ``False``.

    gain_corrected : bool, optional
        If True, the ``master_bias``, ``master_flat``, and ``dark_frame``
        have already been gain corrected.  Default is ``True``.

    Returns
    -------
    occd : `~astropy.nddata.CCDData`
        Reduded ccd.

    Examples
    --------
    1. To overscan, trim and gain correct a data set::

        >>> import numpy as np
        >>> from astropy import units as u
        >>> from astropy.nddata import CCDData
        >>> from ccdproc import ccd_process
        >>> ccd = CCDData(np.ones([100, 100]), unit=u.adu)
        >>> nccd = ccd_process(ccd, oscan='[1:10,1:100]',
        ...                    trim='[10:100, 1:100]', error=False,
        ...                    gain=2.0*u.electron/u.adu)
    """
    # make a copy of the object
    nccd = ccd.copy()

    # apply the overscan correction
    if isinstance(oscan, CCDData):
        nccd = subtract_overscan(nccd, overscan=oscan,
                                 median=oscan_median,
                                 model=oscan_model)
    elif isinstance(oscan, str):
        nccd = subtract_overscan(nccd, fits_section=oscan,
                                 median=oscan_median,
                                 model=oscan_model)
    elif oscan is None:
        pass
    else:
        raise TypeError('oscan is not None, a string, or CCDData object.')

    # apply the trim correction
    if isinstance(trim, str):
        nccd = trim_image(nccd, fits_section=trim)
    elif trim is None:
        pass
    else:
        raise TypeError('trim is not None or a string.')

    # create the error frame
    if error and gain is not None and readnoise is not None:
        nccd = create_deviation(nccd, gain=gain, readnoise=readnoise)
    elif error and (gain is None or readnoise is None):
        raise ValueError(
            'gain and readnoise must be specified to create error frame.')

    # apply the bad pixel mask
    if isinstance(bad_pixel_mask, np.ndarray):
        nccd.mask = bad_pixel_mask
    elif bad_pixel_mask is None:
        pass
    else:
        raise TypeError('bad_pixel_mask is not None or numpy.ndarray.')

    # apply the gain correction
    if not (gain is None or isinstance(gain, Quantity)):
        raise TypeError('gain is not None or astropy.units.Quantity.')

    if gain is not None and gain_corrected:
        nccd = gain_correct(nccd, gain)

    # subtracting the master bias
    if isinstance(master_bias, CCDData):
        nccd = subtract_bias(nccd, master_bias)
    elif master_bias is None:
        pass
    else:
        raise TypeError(
            'master_bias is not None or a CCDData object.')

    # subtract the dark frame
    if isinstance(dark_frame, CCDData):
        nccd = subtract_dark(nccd, dark_frame, dark_exposure=dark_exposure,
                             data_exposure=data_exposure,
                             exposure_time=exposure_key,
                             exposure_unit=exposure_unit,
                             scale=dark_scale)
    elif dark_frame is None:
        pass
    else:
        raise TypeError(
            'dark_frame is not None or a CCDData object.')

    # test dividing the master flat
    if isinstance(master_flat, CCDData):
        nccd = flat_correct(nccd, master_flat, min_value=min_value)
    elif master_flat is None:
        pass
    else:
        raise TypeError(
            'master_flat is not None or a CCDData object.')

    # apply the gain correction only at the end if gain_corrected is False
    if gain is not None and not gain_corrected:
        nccd = gain_correct(nccd, gain)

    return nccd


@log_to_metadata
def create_deviation(ccd_data, gain=None, readnoise=None, disregard_nan=False):
    """
    Create a uncertainty frame. The function will update the uncertainty
    plane which gives the standard deviation for the data. Gain is used in
    this function only to scale the data in constructing the deviation; the
    data is not scaled.

    Parameters
    ----------
    ccd_data : `~astropy.nddata.CCDData`
        Data whose deviation will be calculated.

    gain : `~astropy.units.Quantity` or None, optional
        Gain of the CCD; necessary only if ``ccd_data`` and ``readnoise``
        are not in the same units. In that case, the units of ``gain``
        should be those that convert ``ccd_data.data`` to the same units as
        ``readnoise``.
        Default is ``None``.

    readnoise : `~astropy.units.Quantity` or None, optional
        Read noise per pixel.
        Default is ``None``.

    disregard_nan: boolean
        If ``True``, any value of nan in the output array will be replaced by
        the readnoise.

    {log}

    Raises
    ------
    UnitsError
        Raised if ``readnoise`` units are not equal to product of ``gain`` and
        ``ccd_data`` units.

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        CCDData object with uncertainty created; uncertainty is in the same
        units as the data in the parameter ``ccd_data``.

    """
    if gain is not None and not isinstance(gain, Quantity):
        raise TypeError('gain must be a astropy.units.Quantity.')

    if readnoise is None:
        raise ValueError('must provide a readnoise.')

    if not isinstance(readnoise, Quantity):
        raise TypeError('readnoise must be a astropy.units.Quantity.')

    if gain is None:
        gain = 1.0 * u.dimensionless_unscaled

    if gain.unit * ccd_data.unit != readnoise.unit:
        raise u.UnitsError("units of data, gain and readnoise do not match.")

    # Need to convert Quantity to plain number because NDData data is not
    # a Quantity. All unit checking should happen prior to this point.
    gain_value = float(gain / gain.unit)
    readnoise_value = float(readnoise / readnoise.unit)

    # remove values that might be negative or treat as nan
    data = gain_value * ccd_data.data
    mask = (data < 0)
    if disregard_nan:
        data[mask] = 0
    else:
        data[mask] = np.nan
        logging.warning('Negative values in array will be replaced with nan')

    # calculate the deviation
    var = (data + readnoise_value ** 2) ** 0.5

    # ensure uncertainty and image data have same unit
    ccd = ccd_data.copy()
    var /= gain_value
    ccd.uncertainty = StdDevUncertainty(var)
    return ccd


@log_to_metadata
def subtract_overscan(ccd, overscan=None, overscan_axis=1, fits_section=None,
                      median=False, model=None):
    """
    Subtract the overscan region from an image.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        Data to have overscan frame corrected.

    overscan : `~astropy.nddata.CCDData` or None, optional
        Slice from ``ccd`` that contains the overscan. Must provide either
        this argument or ``fits_section``, but not both.
        Default is ``None``.

    overscan_axis : 0, 1 or None, optional
        Axis along which overscan should combined with mean or median. Axis
        numbering follows the *python* convention for ordering, so 0 is the
        first axis and 1 is the second axis.

        If overscan_axis is explicitly set to None, the axis is set to
        the shortest dimension of the overscan section (or 1 in case
        of a square overscan).
        Default is ``1``.

    fits_section : str or None, optional
        Region of ``ccd`` from which the overscan is extracted, using the FITS
        conventions for index order and index start. See Notes and Examples
        below. Must provide either this argument or ``overscan``, but not both.
        Default is ``None``.

    median : bool, optional
        If true, takes the median of each line. Otherwise, uses the mean.
        Default is ``False``.

    model : `~astropy.modeling.Model` or None, optional
        Model to fit to the data. If None, returns the values calculated
        by the median or the mean.
        Default is ``None``.

    {log}

    Raises
    ------
    TypeError
        A TypeError is raised if either ``ccd`` or ``overscan`` are not the
        correct objects.

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        CCDData object with overscan subtracted.

    Notes
    -----
    The format of the ``fits_section`` string follow the rules for slices that
    are consistent with the FITS standard (v3) and IRAF usage of keywords like
    TRIMSEC and BIASSEC. Its indexes are one-based, instead of the
    python-standard zero-based, and the first index is the one that increases
    most rapidly as you move through the array in memory order, opposite the
    python ordering.

    The 'fits_section' argument is provided as a convenience for those who are
    processing files that contain TRIMSEC and BIASSEC. The preferred, more
    pythonic, way of specifying the overscan is to do it by indexing the data
    array directly with the ``overscan`` argument.

    Examples
    --------
    Creating a 100x100 array containing ones just for demonstration purposes::

        >>> import numpy as np
        >>> from astropy import units as u
        >>> arr1 = CCDData(np.ones([100, 100]), unit=u.adu)

    The statement below uses all rows of columns 90 through 99 as the
    overscan::

        >>> no_scan = subtract_overscan(arr1, overscan=arr1[:, 90:100])
        >>> assert (no_scan.data == 0).all()

    This statement does the same as the above, but with a FITS-style section::

        >>> no_scan = subtract_overscan(arr1, fits_section='[91:100, :]')
        >>> assert (no_scan.data == 0).all()

    Spaces are stripped out of the ``fits_section`` string.

    """
    if not (isinstance(ccd, CCDData) or isinstance(ccd, np.ndarray)):
        raise TypeError('ccddata is not a CCDData or ndarray object.')

    if ((overscan is not None and fits_section is not None) or
            (overscan is None and fits_section is None)):
        raise TypeError('specify either overscan or fits_section, but not '
                        'both.')

    if (overscan is not None) and (not isinstance(overscan, CCDData)):
        raise TypeError('overscan is not a CCDData object.')

    if (fits_section is not None and
            not isinstance(fits_section, str)):
        raise TypeError('overscan is not a string.')

    if fits_section is not None:
        overscan = ccd[slice_from_string(fits_section, fits_convention=True)]

    if overscan_axis is None:
        overscan_axis = 0 if overscan.shape[1] > overscan.shape[0] else 1

    if median:
        oscan = np.median(overscan.data, axis=overscan_axis)
    else:
        oscan = np.mean(overscan.data, axis=overscan_axis)

    if model is not None:
        of = fitting.LinearLSQFitter()
        yarr = np.arange(len(oscan))
        oscan = of(model, yarr, oscan)
        oscan = oscan(yarr)
        if overscan_axis == 1:
            oscan = np.reshape(oscan, (oscan.size, 1))
        else:
            oscan = np.reshape(oscan, (1, oscan.size))
    else:
        if overscan_axis == 1:
            oscan = np.reshape(oscan, oscan.shape + (1,))
        else:
            oscan = np.reshape(oscan, (1,) + oscan.shape)

    subtracted = ccd.copy()

    # subtract the overscan
    subtracted.data = ccd.data - oscan
    return subtracted


@log_to_metadata
def trim_image(ccd, fits_section=None):
    """
    Trim the image to the dimensions indicated.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        CCD image to be trimmed, sliced if desired.

    fits_section : str or None, optional
        Region of ``ccd`` from which the overscan is extracted; see
        `~ccdproc.subtract_overscan` for details.
        Default is ``None``.

    {log}

    Returns
    -------
    trimmed_ccd : `~astropy.nddata.CCDData`
        Trimmed image.

    Examples
    --------
    Given an array that is 100x100,

        >>> import numpy as np
        >>> from astropy import units as u
        >>> arr1 = CCDData(np.ones([100, 100]), unit=u.adu)

    the syntax for trimming this to keep all of the first index but only the
    first 90 rows of the second index is

        >>> trimmed = trim_image(arr1[:, :90])
        >>> trimmed.shape
        (100, 90)
        >>> trimmed.data[0, 0] = 2
        >>> arr1.data[0, 0]
        1.0

    This both trims *and makes a copy* of the image.

    Indexing the image directly does *not* do the same thing, quite:

        >>> not_really_trimmed = arr1[:, :90]
        >>> not_really_trimmed.data[0, 0] = 2
        >>> arr1.data[0, 0]
        2.0

    In this case, ``not_really_trimmed`` is a view of the underlying array
    ``arr1``, not a copy.
    """
    if (fits_section is not None and
            not isinstance(fits_section, str)):
        raise TypeError("fits_section must be a string.")
    trimmed = ccd.copy()
    if fits_section:
        python_slice = slice_from_string(fits_section, fits_convention=True)
        trimmed = trimmed[python_slice]

    return trimmed


@log_to_metadata
def subtract_bias(ccd, master):
    """
    Subtract master bias from image.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        Image from which bias will be subtracted.

    master : `~astropy.nddata.CCDData`
        Master image to be subtracted from ``ccd``.

    {log}

    Returns
    -------
    result : `~astropy.nddata.CCDData`
        CCDData object with bias subtracted.
    """

    try:
        result = ccd.subtract(master)
    except ValueError as e:
        if 'operand units' in str(e):
            raise u.UnitsError("Unit '{}' of the uncalibrated image does not "
                               "match unit '{}' of the calibration "
                               "image".format(ccd.unit, master.unit))
        else:
            raise e

    result.meta = ccd.meta.copy()
    return result


@log_to_metadata
def subtract_dark(ccd, master, dark_exposure=None, data_exposure=None,
                  exposure_time=None, exposure_unit=None,
                  scale=False):
    """
    Subtract dark current from an image.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        Image from which dark will be subtracted.

    master : `~astropy.nddata.CCDData`
        Dark image.

    dark_exposure : `~astropy.units.Quantity` or None, optional
        Exposure time of the dark image; if specified, must also provided
        ``data_exposure``.
        Default is ``None``.

    data_exposure : `~astropy.units.Quantity` or None, optional
        Exposure time of the science image; if specified, must also provided
        ``dark_exposure``.
        Default is ``None``.

    exposure_time : str or `~ccdproc.Keyword` or None, optional
        Name of key in image metadata that contains exposure time.
        Default is ``None``.

    exposure_unit : `~astropy.units.Unit` or None, optional
        Unit of the exposure time if the value in the meta data does not
        include a unit.
        Default is ``None``.

    scale: bool, optional
        If True, scale the dark frame by the exposure times.
        Default is ``False``.

    {log}

    Returns
    -------
    result : `~astropy.nddata.CCDData`
        Dark-subtracted image.
    """
    if ccd.shape != master.shape:
        err_str = "operands could not be subtracted with shapes {} {}".format(ccd.shape, master.shape)
        raise ValueError(err_str)

    if not (isinstance(ccd, CCDData) and isinstance(master, CCDData)):
        raise TypeError("ccd and master must both be CCDData objects.")

    if (data_exposure is not None and
            dark_exposure is not None and
            exposure_time is not None):
        raise TypeError("specify either exposure_time or "
                        "(dark_exposure and data_exposure), not both.")

    if data_exposure is None and dark_exposure is None:
        if exposure_time is None:
            raise TypeError("must specify either exposure_time or both "
                            "dark_exposure and data_exposure.")
        if isinstance(exposure_time, Keyword):
            data_exposure = exposure_time.value_from(ccd.header)
            dark_exposure = exposure_time.value_from(master.header)
        else:
            data_exposure = ccd.header[exposure_time]
            dark_exposure = master.header[exposure_time]

    if not (isinstance(dark_exposure, Quantity) and
            isinstance(data_exposure, Quantity)):
        if exposure_time:
            try:
                data_exposure *= exposure_unit
                dark_exposure *= exposure_unit
            except TypeError:
                raise TypeError("must provide unit for exposure time.")
        else:
            raise TypeError("exposure times must be astropy.units.Quantity "
                            "objects.")

    try:
        if scale:
            master_scaled = master.copy()
            # data_exposure and dark_exposure are both quantities,
            # so we can just have subtract do the scaling
            master_scaled = master_scaled.multiply(data_exposure /
                                                   dark_exposure)
            result = ccd.subtract(master_scaled)
        else:
            result = ccd.subtract(master)
    except (u.UnitsError, u.UnitConversionError, ValueError) as e:
        # Astropy LTS (v1) returns a ValueError, not a UnitsError, so catch
        # that if it appears to really be a UnitsError.
        if (isinstance(e, ValueError) and
                'operand units' not in str(e) and
                astropy.__version__.startswith('1.0')):
            raise e

        # Make the error message a little more explicit than what is returned
        # by default.
        raise u.UnitsError("Unit '{}' of the uncalibrated image does not "
                           "match unit '{}' of the calibration "
                           "image".format(ccd.unit, master.unit))

    result.meta = ccd.meta.copy()
    return result


@log_to_metadata
def gain_correct(ccd, gain, gain_unit=None):
    """Correct the gain in the image.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
      Data to have gain corrected.

    gain : `~astropy.units.Quantity` or `~ccdproc.Keyword`
      gain value for the image expressed in electrons per adu.

    gain_unit : `~astropy.units.Unit` or None, optional
        Unit for the ``gain``; used only if ``gain`` itself does not provide
        units.
        Default is ``None``.

    {log}

    Returns
    -------
    result : `~astropy.nddata.CCDData`
      CCDData object with gain corrected.
    """
    if isinstance(gain, Keyword):
        gain_value = gain.value_from(ccd.header)
    elif isinstance(gain, numbers.Number) and gain_unit is not None:
        gain_value = gain * u.Unit(gain_unit)
    else:
        gain_value = gain

    result = ccd.multiply(gain_value)
    result.meta = ccd.meta.copy()
    return result


@log_to_metadata
def flat_correct(ccd, flat, min_value=None, norm_value=None):
    """Correct the image for flat fielding.

    The flat field image is normalized by its mean or a user-supplied value
    before flat correcting.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        Data to be transformed.

    flat : `~astropy.nddata.CCDData`
        Flatfield to apply to the data.

    min_value : float or None, optional
        Minimum value for flat field. The value can either be None and no
        minimum value is applied to the flat or specified by a float which
        will replace all values in the flat by the min_value.
        Default is ``None``.

    norm_value : float or None, optional
        If not ``None``, normalize flat field by this argument rather than the
        mean of the image. This allows fixing several different flat fields to
        have the same scale. If this value is negative or 0, a ``ValueError``
        is raised. Default is ``None``.

    {log}

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        CCDData object with flat corrected.
    """
    # Use the min_value to replace any values in the flat
    use_flat = flat
    if min_value is not None:
        flat_min = flat.copy()
        flat_min.data[flat_min.data < min_value] = min_value
        use_flat = flat_min

    # If a norm_value was input and is positive, use it to scale the flat
    if norm_value is not None and norm_value > 0:
        flat_mean_val = norm_value
    elif norm_value is not None:
        # norm_value was set to a bad value
        raise ValueError('norm_value must be greater than zero.')
    else:
        # norm_value was not set, use mean of the image.
        flat_mean_val = use_flat.data.mean()

    # Normalize the flat.
    flat_mean = flat_mean_val * use_flat.unit
    flat_normed = use_flat.divide(flat_mean)

    # divide through the flat
    flat_corrected = ccd.divide(flat_normed)

    flat_corrected.meta = ccd.meta.copy()
    return flat_corrected


@log_to_metadata
def transform_image(ccd, transform_func, **kwargs):
    """Transform the image.

    Using the function specified by transform_func, the transform will
    be applied to data, uncertainty, and mask in ccd.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        Data to be transformed.

    transform_func : callable
        Function to be used to transform the data, mask and uncertainty.

    kwargs :
        Additional keyword arguments to be used by the transform_func.

    {log}

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        A transformed CCDData object.

    Notes
    -----
    At this time, transform will be applied to the uncertainty data but it
    will only transform the data. This will not properly handle uncertainties
    that arise due to correlation between the pixels.

    These should only be geometric transformations of the images. Other
    methods should be used if the units of ccd need to be changed.

    Examples
    --------
    Given an array that is 100x100::

        >>> import numpy as np
        >>> from astropy import units as u
        >>> arr1 = CCDData(np.ones([100, 100]), unit=u.adu)

    The syntax for transforming the array using
    `scipy.ndimage.shift`::

        >>> from scipy.ndimage.interpolation import shift
        >>> from ccdproc import transform_image
        >>> transformed = transform_image(arr1, shift, shift=(5.5, 8.1))
    """
    # check that it is a ccddata object
    if not isinstance(ccd, CCDData):
        raise TypeError('ccd is not a CCDData.')

    # make a copy of the object
    nccd = ccd.copy()

    # transform the image plane
    try:
        nccd.data = transform_func(nccd.data, **kwargs)
    except TypeError as exc:
        if 'is not callable' in str(exc):
            raise TypeError('transform_func is not a callable.')
        raise

    # transform the uncertainty plane if it exists
    if nccd.uncertainty is not None:
        nccd.uncertainty.array = transform_func(nccd.uncertainty.array,
                                                **kwargs)

    # transform the mask plane
    if nccd.mask is not None:
        mask = transform_func(nccd.mask, **kwargs)
        nccd.mask = mask > 0

    if nccd.wcs is not None:
        warn = 'WCS information may be incorrect as no transformation was applied to it'
        logging.warning(warn)

    return nccd


@log_to_metadata
def wcs_project(ccd, target_wcs, target_shape=None, order='bilinear'):
    """
    Given a CCDData image with WCS, project it onto a target WCS and
    return the reprojected data as a new CCDData image.

    Any flags, weight, or uncertainty are ignored in doing the
    reprojection.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        Data to be projected.

    target_wcs : `~astropy.wcs.WCS` object
        WCS onto which all images should be projected.

    target_shape : two element list-like or None, optional
        Shape of the output image. If omitted, defaults to the shape of the
        input image.
        Default is ``None``.

    order : str, optional
        Interpolation order for re-projection. Must be one of:

        + 'nearest-neighbor'
        + 'bilinear'
        + 'biquadratic'
        + 'bicubic'

        Default is ``'bilinear'``.

    {log}

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        A transformed CCDData object.
    """
    from astropy.nddata.ccddata import _generate_wcs_and_update_header
    from reproject import reproject_interp

    if not (ccd.wcs.is_celestial and target_wcs.is_celestial):
        raise ValueError('one or both WCS is not celestial.')

    if target_shape is None:
        target_shape = ccd.shape

    projected_image_raw, _ = reproject_interp((ccd.data, ccd.wcs),
                                              target_wcs,
                                              shape_out=target_shape,
                                              order=order)

    reprojected_mask = None
    if ccd.mask is not None:
        reprojected_mask, _ = reproject_interp((ccd.mask, ccd.wcs),
                                               target_wcs,
                                               shape_out=target_shape,
                                               order=order)
        # Make the mask 1 if the reprojected mask pixel value is non-zero.
        # A small threshold is included to allow for some rounding in
        # reproject_interp.
        reprojected_mask = reprojected_mask > 1e-8

    # The reprojection will contain nan for any pixels for which the source
    # was outside the original image. Those should be masked also.
    output_mask = np.isnan(projected_image_raw)

    if reprojected_mask is not None:
        output_mask = output_mask | reprojected_mask

    # Need to scale counts by ratio of pixel areas
    area_ratio = (proj_plane_pixel_area(target_wcs) /
                  proj_plane_pixel_area(ccd.wcs))

    # If nothing ended up masked, don't create a mask.
    if not output_mask.any():
        output_mask = None

    # If there are any wcs keywords in the header, remove them
    hdr, _ = _generate_wcs_and_update_header(ccd.header)

    nccd = CCDData(area_ratio * projected_image_raw, wcs=target_wcs,
                   mask=output_mask,
                   header=hdr, unit=ccd.unit)

    return nccd


def sigma_func(arr, axis=None):
    """
    Robust method for calculating the deviation of an array. ``sigma_func``
    uses the median absolute deviation to determine the standard deviation.

    Parameters
    ----------
    arr : `~astropy.nddata.CCDData` or `numpy.ndarray`
        Array whose deviation is to be calculated.

    axis : int, tuple of ints or None, optional
        Axis or axes along which the function is performed.
        If ``None`` it is performed over all the dimensions of
        the input array. The axis argument can also be negative, in this case
        it counts from the last to the first axis.
        Default is ``None``.

    Returns
    -------
    uncertainty : float
        uncertainty of array estimated from median absolute deviation.
    """
    return stats.median_absolute_deviation(arr, axis=axis) * 1.482602218505602


def setbox(x, y, mbox, xmax, ymax):
    """
    Create a box of length mbox around a position x,y. If the box will
    be out of [0,len] then reset the edges of the box to be within the
    boundaries.

    Parameters
    ----------
    x : int
        Central x-position of box.

    y : int
        Central y-position of box.

    mbox : int
        Width of box.

    xmax : int
        Maximum x value.

    ymax : int
        Maximum y value.

    Returns
    -------
    x1 : int
        Lower x corner of box.

    x2 : int
        Upper x corner of box.

    y1 : int
        Lower y corner of box.

    y2 : int
        Upper y corner of box.
    """
    mbox = max(int(0.5 * mbox), 1)
    y1 = max(0, y - mbox)
    y2 = min(y + mbox + 1, ymax - 1)
    x1 = max(0, x - mbox)
    x2 = min(x + mbox + 1, xmax - 1)

    return x1, x2, y1, y2


def background_deviation_box(data, bbox):
    """
    Determine the background deviation with a box size of bbox. The algorithm
    steps through the image and calculates the deviation within each box.
    It returns an array with the pixels in each box filled with the deviation
    value.

    Parameters
    ----------
    data : `numpy.ndarray` or `numpy.ma.MaskedArray`
        Data to measure background deviation.

    bbox : int
        Box size for calculating background deviation.

    Raises
    ------
    ValueError
        A value error is raised if bbox is less than 1.

    Returns
    -------
    background : `numpy.ndarray` or `numpy.ma.MaskedArray`
        An array with the measured background deviation in each pixel.
    """
    # Check to make sure the background box is an appropriate size
    # If it is too small, then insufficient statistics are generated
    if bbox < 1:
        raise ValueError('bbox must be greater than 1.')

    # make the background image
    barr = data * 0.0 + data.std()
    ylen, xlen = data.shape
    for i in range(int(0.5 * bbox), xlen, bbox):
        for j in range(int(0.5 * bbox), ylen, bbox):
            x1, x2, y1, y2 = setbox(i, j, bbox, xlen, ylen)
            barr[y1:y2, x1:x2] = sigma_func(data[y1:y2, x1:x2])

    return barr


def background_deviation_filter(data, bbox):
    """
    Determine the background deviation for each pixel from a box with size of
    bbox.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data to measure background deviation.

    bbox : int
        Box size for calculating background deviation.

    Raises
    ------
    ValueError
        A value error is raised if bbox is less than 1.

    Returns
    -------
    background : `numpy.ndarray` or `numpy.ma.MaskedArray`
        An array with the measured background deviation in each pixel.
    """
    # Check to make sure the background box is an appropriate size
    if bbox < 1:
        raise ValueError('bbox must be greater than 1.')

    return ndimage.generic_filter(data, sigma_func, size=(bbox, bbox))


@deprecated('1.1')
def rebin(ccd, newshape):
    """
    Rebin an array to have a new shape.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `numpy.ndarray`
        Data to rebin.

    newshape : tuple
        Tuple containing the new shape for the array.

    Returns
    -------
    output : `~astropy.nddata.CCDData` or `numpy.ndarray`
        An array with the new shape. It will have the same type as the input
        object.

    Raises
    ------
    TypeError
        A type error is raised if data is not an `numpy.ndarray` or
        `~astropy.nddata.CCDData`.

    ValueError
        A value error is raised if the dimension of the new shape is not equal
        to the data's.

    Notes
    -----
    This is based on the scipy cookbook for rebinning:
    http://wiki.scipy.org/Cookbook/Rebinning

    If rebinning a CCDData object to a smaller shape, the masking and
    uncertainty are not handled correctly.

    Examples
    --------
    Given an array that is 100x100::

        import numpy as np
        from astropy import units as u
        arr1 = CCDData(np.ones([10, 10]), unit=u.adu)

    The syntax for rebinning an array to a shape
    of (20,20) is::

        rebin(arr1, (20,20))
    """
    # check to see that is in a nddata type
    if isinstance(ccd, np.ndarray):

        # check to see that the two arrays are going to be the same length
        if len(ccd.shape) != len(newshape):
            raise ValueError('newshape does not have the same dimensions as '
                             'ccd.')

        slices = [slice(0, old, old/new) for old, new in
                  zip(ccd.shape, newshape)]
        coordinates = np.mgrid[slices]
        indices = coordinates.astype('i')
        return ccd[tuple(indices)]

    elif isinstance(ccd, CCDData):
        # check to see that the two arrays are going to be the same length
        if len(ccd.shape) != len(newshape):
            raise ValueError('newshape does not have the same dimensions as '
                             'ccd.')

        nccd = ccd.copy()
        # rebin the data plane
        nccd.data = rebin(nccd.data, newshape)

        # rebin the uncertainty plane
        if nccd.uncertainty is not None:
            nccd.uncertainty.array = rebin(nccd.uncertainty.array, newshape)

        # rebin the mask plane
        if nccd.mask is not None:
            nccd.mask = rebin(nccd.mask, newshape)

        return nccd
    else:
        raise TypeError('ccd is not an ndarray or a CCDData object.')


def block_reduce(ccd, block_size, func=np.sum):
    """Thin wrapper around `astropy.nddata.block_reduce`."""
    data = nddata_utils.block_reduce(ccd, block_size, func)
    if isinstance(ccd, CCDData):
        # unit and meta "should" be unaffected by the change of shape and can
        # be copied. However wcs, mask, uncertainty should not be copied!
        data = CCDData(data, unit=ccd.unit, meta=ccd.meta.copy())
    return data


def block_average(ccd, block_size):
    """Like `block_reduce` but with predefined ``func=np.mean``.
    """
    data = nddata_utils.block_reduce(ccd, block_size, np.mean)
    # Like in block_reduce:
    if isinstance(ccd, CCDData):
        data = CCDData(data, unit=ccd.unit, meta=ccd.meta.copy())
    return data


def block_replicate(ccd, block_size, conserve_sum=True):
    """Thin wrapper around `astropy.nddata.block_replicate`."""
    data = nddata_utils.block_replicate(ccd, block_size, conserve_sum)
    # Like in block_reduce:
    if isinstance(ccd, CCDData):
        data = CCDData(data, unit=ccd.unit, meta=ccd.meta.copy())
    return data


try:
    # Append original docstring to docstrings of these functions
    block_reduce.__doc__ += nddata_utils.block_reduce.__doc__
    block_replicate.__doc__ += nddata_utils.block_replicate.__doc__
    __all__ += ['block_average', 'block_reduce', 'block_replicate']
except AttributeError:
    # Astropy 1.0 has no block_reduce, block_average
    del block_reduce, block_average, block_replicate


def _blkavg(data, newshape):
    """
    Block average an array such that it has the new shape.

    Parameters
    ----------
    data : `numpy.ndarray` or `numpy.ma.MaskedArray`
        Data to average.

    newshape : tuple
        Tuple containing the new shape for the array.

    Returns
    -------
    output : `numpy.ndarray` or `numpy.ma.MaskedArray`
        An array with the new shape and the average of the pixels.

    Raises
    ------
    TypeError
        A type error is raised if data is not an `numpy.ndarray`.

    ValueError
        A value error is raised if the dimensions of new shape is not equal
        to data.

    Notes
    -----
    This is based on the scipy cookbook for rebinning:
    http://wiki.scipy.org/Cookbook/Rebinning
    """
    # check to see that is in a nddata type
    if not isinstance(data, np.ndarray):
        raise TypeError('data is not a ndarray object.')

    # check to see that the two arrays are going to be the same length
    if len(data.shape) != len(newshape):
        raise ValueError('newshape does not have the same dimensions as data.')

    shape = data.shape
    lenShape = len(shape)
    factor = np.asarray(shape)/np.asarray(newshape)

    evList = ['data.reshape('] + \
        ['newshape[%d],int(factor[%d]),' % (i, i) for i in range(lenShape)] + \
        [')'] + ['.mean(%d)' % (i + 1) for i in range(lenShape)]

    return eval(''.join(evList))


def median_filter(data, *args, **kwargs):
    """See `scipy.ndimage.median_filter` for arguments.

    If the ``data`` is a `~astropy.nddata.CCDData` object the result will be another
    `~astropy.nddata.CCDData` object with the median filtered data as ``data`` and
    copied ``unit`` and ``meta``.
    """
    if isinstance(data, CCDData):
        out_kwargs = {'meta': data.meta.copy(),
                      'unit': data.unit}
        result = ndimage.median_filter(data.data, *args, **kwargs)
        return CCDData(result, **out_kwargs)
    else:
        return ndimage.median_filter(data, *args, **kwargs)


def cosmicray_lacosmic(ccd, sigclip=4.5, sigfrac=0.3,
                       objlim=5.0, gain=1.0, readnoise=6.5,
                       satlevel=65535.0, pssl=0.0, niter=4,
                       sepmed=True, cleantype='meanmask', fsmode='median',
                       psfmodel='gauss', psffwhm=2.5, psfsize=7,
                       psfk=None, psfbeta=4.765, verbose=False,
                       gain_apply=True):
    r"""
    Identify cosmic rays through the L.A. Cosmic technique. The L.A. Cosmic
    technique identifies cosmic rays by identifying pixels based on a variation
    of the Laplacian edge detection. The algorithm is an implementation of the
    code describe in van Dokkum (2001) [1]_ as implemented by McCully (2014)
    [2]_. If you use this algorithm, please cite these two works.



    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `numpy.ndarray`
        Data to have cosmic ray cleaned.

    gain_apply : bool, optional
        If ``True``, **return gain-corrected data**, with correct units,
        otherwise do not gain-correct the data. Default is ``True`` to
        preserve backwards compatibility.

    sigclip : float, optional
        Laplacian-to-noise limit for cosmic ray detection. Lower values will
        flag more pixels as cosmic rays. Default: 4.5.

    sigfrac : float, optional
        Fractional detection limit for neighboring pixels. For cosmic ray
        neighbor pixels, a Laplacian-to-noise detection limit of
        sigfrac * sigclip will be used. Default: 0.3.

    objlim : float, optional
        Minimum contrast between Laplacian image and the fine structure image.
        Increase this value if cores of bright stars are flagged as cosmic
        rays. Default: 5.0.

    pssl : float, optional
        Previously subtracted sky level in ADU. We always need to work in
        electrons for cosmic ray detection, so we need to know the sky level
        that has been subtracted so we can add it back in. Default: 0.0.

    gain : float or `~astropy.units.Quantity`, optional
        Gain of the image (electrons / ADU). We always need to work in
        electrons for cosmic ray detection. Default: 1.0

    readnoise : float, optional
        Read noise of the image (electrons). Used to generate the noise model
        of the image. Default: 6.5.

    satlevel : float, optional
        Saturation level of the image (electrons). This value is used to
        detect saturated stars and pixels at or above this level are added to
        the mask. Default: 65535.0.

    niter : int, optional
        Number of iterations of the LA Cosmic algorithm to perform. Default: 4.

    sepmed : bool, optional
        Use the separable median filter instead of the full median filter.
        The separable median is not identical to the full median filter, but
        they are approximately the same, the separable median filter is
        significantly faster, and still detects cosmic rays well. Note, this is
        a performance feature, and not part of the original L.A. Cosmic.
        Default: True

    cleantype : str, optional
        Set which clean algorithm is used:

        - ``"median"``: An unmasked 5x5 median filter.
        - ``"medmask"``: A masked 5x5 median filter.
        - ``"meanmask"``: A masked 5x5 mean filter.
        - ``"idw"``: A masked 5x5 inverse distance weighted interpolation.

        Default: ``"meanmask"``.

    fsmode : str, optional
        Method to build the fine structure image:

        - ``"median"``: Use the median filter in the standard LA Cosmic \
          algorithm.
        - ``"convolve"``: Convolve the image with the psf kernel to calculate \
          the fine structure image.

        Default: ``"median"``.

    psfmodel : str, optional
        Model to use to generate the psf kernel if fsmode == 'convolve' and
        psfk is None. The current choices are Gaussian and Moffat profiles:

        - ``"gauss"`` and ``"moffat"`` produce circular PSF kernels.
        - The ``"gaussx"`` and ``"gaussy"`` produce Gaussian kernels in the x \
          and y directions respectively.

        Default: ``"gauss"``.

    psffwhm : float, optional
        Full Width Half Maximum of the PSF to use to generate the kernel.
        Default: 2.5.

    psfsize : int, optional
        Size of the kernel to calculate. Returned kernel will have size
        psfsize x psfsize. psfsize should be odd. Default: 7.

    psfk : `numpy.ndarray` (with float dtype) or None, optional
        PSF kernel array to use for the fine structure image if
        ``fsmode == 'convolve'``. If None and ``fsmode == 'convolve'``, we
        calculate the psf kernel using ``psfmodel``. Default: None.

    psfbeta : float, optional
        Moffat beta parameter. Only used if ``fsmode=='convolve'`` and
        ``psfmodel=='moffat'``. Default: 4.765.

    verbose : bool, optional
        Print to the screen or not. Default: False.

    Notes
    -----
    Implementation of the cosmic ray identification L.A.Cosmic:
    http://www.astro.yale.edu/dokkum/lacosmic/

    Returns
    -------
    nccd : `~astropy.nddata.CCDData` or `numpy.ndarray`
        An object of the same type as ccd is returned. If it is a
        `~astropy.nddata.CCDData`, the mask attribute will also be updated with
        areas identified with cosmic rays masked. **By default, the image is
        multiplied by the gain.** You can control this behavior with the
        ``gain_apply`` argument.

    crmask : `numpy.ndarray`
        If an `numpy.ndarray` is provided as ccd, a boolean ndarray with the
        cosmic rays identified will also be returned.

    References
    ----------
    .. [1] van Dokkum, P; 2001, "Cosmic-Ray Rejection by Laplacian Edge
       Detection". The Publications of the Astronomical Society of the
       Pacific, Volume 113, Issue 789, pp. 1420-1427.
       doi: 10.1086/323894

    .. [2] McCully, C., 2014, "Astro-SCRAPPY",
       https://github.com/astropy/astroscrappy

    Examples
    --------
    1) Given an numpy.ndarray object, the syntax for running
       cosmicrar_lacosmic would be:

       >>> newdata, mask = cosmicray_lacosmic(data, sigclip=5)  #doctest: +SKIP

       where the error is an array that is the same shape as data but
       includes the pixel error. This would return a data array, newdata,
       with the bad pixels replaced by the local median from a box of 11
       pixels; and it would return a mask indicating the bad pixels.

    2) Given an `~astropy.nddata.CCDData` object with an uncertainty frame, the syntax
       for running cosmicrar_lacosmic would be:

       >>> newccd = cosmicray_lacosmic(ccd, sigclip=5)   # doctest: +SKIP

       The newccd object will have bad pixels in its data array replace and the
       mask of the object will be created if it did not previously exist or be
       updated with the detected cosmic rays.
    """
    from astroscrappy import detect_cosmics

    # If we didn't get a quantity, put them in, with unit specified by the
    # documentation above.
    if not isinstance(gain, u.Quantity):
        # Gain will change the value, so use the proper units
        gain = gain * u.electron / u.adu

    # Set the units of readnoise to electrons, as specified in the
    # documentation, if no unit is present.
    if not isinstance(readnoise, u.Quantity):
        readnoise = readnoise * u.electron

    if isinstance(ccd, np.ndarray):
        data = ccd

        crmask, cleanarr = detect_cosmics(
            data, inmask=None, sigclip=sigclip,
            sigfrac=sigfrac, objlim=objlim, gain=gain.value,
            readnoise=readnoise.value, satlevel=satlevel, pssl=pssl,
            niter=niter, sepmed=sepmed, cleantype=cleantype,
            fsmode=fsmode, psfmodel=psfmodel, psffwhm=psffwhm,
            psfsize=psfsize, psfk=psfk, psfbeta=psfbeta,
            verbose=verbose)

        if not gain_apply and gain != 1.0:
            cleanarr = cleanarr / gain
        return cleanarr, crmask

    elif isinstance(ccd, CCDData):
        # Start with a check for a special case: ccd is in electron, and
        # gain and readnoise have no units. In that case we issue a warning
        # instead of raising an error to avoid crashing user's pipelines.
        if ccd.unit.is_equivalent(u.electron) and gain.value != 1.0:
            warnings.warn("Image unit is electron but gain value "
                          "is not 1.0. Data maybe end up being gain "
                          "corrected twice.")

        else:
            if ((readnoise.unit == u.electron)
                and (ccd.unit == u.electron)
                and (gain.value == 1.0)):
                gain = gain.value * u.one
            # Check unit consistency before taking the time to check for
            # cosmic rays.
            if not (gain * ccd).unit.is_equivalent(readnoise.unit):
                raise ValueError('Inconsistent units for gain ({}) '.format(gain.unit) +
                                 ' ccd ({}) and readnoise ({}).'.format(ccd.unit,
                                                                        readnoise.unit))

        crmask, cleanarr = detect_cosmics(
            ccd.data, inmask=ccd.mask,
            sigclip=sigclip, sigfrac=sigfrac, objlim=objlim, gain=gain.value,
            readnoise=readnoise.value, satlevel=satlevel, pssl=pssl,
            niter=niter, sepmed=sepmed, cleantype=cleantype,
            fsmode=fsmode, psfmodel=psfmodel, psffwhm=psffwhm,
            psfsize=psfsize, psfk=psfk, psfbeta=psfbeta, verbose=verbose)

        # create the new ccd data object
        nccd = ccd.copy()

        # Remove the gain scaling if it wasn't desired
        if not gain_apply and gain != 1.0:
            cleanarr = cleanarr / gain.value

        # Fix the units if the gain is being applied.
        nccd.unit = ccd.unit * gain.unit

        nccd.data = cleanarr
        if nccd.mask is None:
            nccd.mask = crmask
        else:
            nccd.mask = nccd.mask + crmask

        return nccd

    else:
        raise TypeError('ccd is not a CCDData or ndarray object.')


def cosmicray_median(ccd, error_image=None, thresh=5, mbox=11, gbox=0,
                     rbox=0):
    """
    Identify cosmic rays through median technique. The median technique
    identifies cosmic rays by identifying pixels by subtracting a median image
    from the initial data array.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`, `numpy.ndarray` or `numpy.ma.MaskedArray`
        Data to have cosmic ray cleaned.

    thresh : float, optional
        Threshold for detecting cosmic rays.
        Default is ``5``.

    error_image : `numpy.ndarray`, float or None, optional
        Error level. If None, the task will use the standard
        deviation of the data. If an ndarray, it should have the same shape
        as data.
        Default is ``None``.

    mbox : int, optional
        Median box for detecting cosmic rays.
        Default is ``11``.

    gbox : int, optional
        Box size to grow cosmic rays. If zero, no growing will be done.
        Default is ``0``.

    rbox : int, optional
        Median box for calculating replacement values. If zero, no pixels will
        be replaced.
        Default is ``0``.

    Notes
    -----
    Similar implementation to crmedian in iraf.imred.crutil.crmedian.

    Returns
    -------
    nccd : `~astropy.nddata.CCDData` or `numpy.ndarray`
        An object of the same type as ccd is returned. If it is a
        `~astropy.nddata.CCDData`, the mask attribute will also be updated with
        areas identified with cosmic rays masked.

    nccd : `numpy.ndarray`
        If an `numpy.ndarray` is provided as ccd, a boolean ndarray with the
        cosmic rays identified will also be returned.

    Examples
    --------
    1) Given an numpy.ndarray object, the syntax for running
       cosmicray_median would be:

       >>> newdata, mask = cosmicray_median(data, error_image=error,
       ...                                  thresh=5, mbox=11,
       ...                                  rbox=11, gbox=5)   # doctest: +SKIP

       where error is an array that is the same shape as data but
       includes the pixel error. This would return a data array, newdata,
       with the bad pixels replaced by the local median from a box of 11
       pixels; and it would return a mask indicating the bad pixels.

    2) Given an `~astropy.nddata.CCDData` object with an uncertainty frame, the syntax
       for running cosmicray_median would be:

       >>> newccd = cosmicray_median(ccd, thresh=5, mbox=11,
       ...                           rbox=11, gbox=5)   # doctest: +SKIP

       The newccd object will have bad pixels in its data array replace and the
       mask of the object will be created if it did not previously exist or be
       updated with the detected cosmic rays.
    """
    if isinstance(ccd, np.ndarray):
        data = ccd

        if error_image is None:
            error_image = data.std()
        else:
            if not isinstance(error_image, (float, np.ndarray)):
                raise TypeError('error_image is not a float or ndarray.')

        # create the median image
        marr = ndimage.median_filter(data, size=(mbox, mbox))

        # Only look at the data array
        if isinstance(data, np.ma.MaskedArray):
            data = data.data

        # Find the residual image
        rarr = (data - marr) / error_image

        # identify all sources
        crarr = (rarr > thresh)

        # grow the pixels
        if gbox > 0:
            crarr = ndimage.maximum_filter(crarr, gbox)

        # replace bad pixels in the image
        ndata = data.copy()
        if rbox > 0:
            data = np.ma.masked_array(data, (crarr == 1))
            mdata = ndimage.median_filter(data, rbox)
            ndata[crarr == 1] = mdata[crarr == 1]

        return ndata, crarr
    elif isinstance(ccd, CCDData):

        # set up the error image
        if error_image is None and ccd.uncertainty is not None:
            error_image = ccd.uncertainty.array
        if ccd.data.shape != error_image.shape:
            raise ValueError('error_image is not the same shape as data.')

        data, crarr = cosmicray_median(ccd.data, error_image=error_image,
                                       thresh=thresh, mbox=mbox, gbox=gbox,
                                       rbox=rbox)

        # create the new ccd data object
        nccd = ccd.copy()
        nccd.data = data
        if nccd.mask is None:
            nccd.mask = crarr
        else:
            nccd.mask = nccd.mask + crarr
        return nccd

    else:
        raise TypeError('ccd is not an numpy.ndarray or a CCDData object.')


def ccdmask(ratio, findbadcolumns=False, byblocks=False, ncmed=7, nlmed=7,
            ncsig=15, nlsig=15, lsigma=9, hsigma=9, ngood=5):
    """
    Uses method based on the IRAF ccdmask task to generate a mask based on the
    given input.

    .. note::
        This function uses ``lines`` as synonym for the first axis and
        ``columns`` the second axis. Only two-dimensional ``ratio`` is
        currently supported.

    Parameters
    ----------
    ratio : `~astropy.nddata.CCDData`
        Data to used to form mask.  Typically this is the ratio of two flat
        field images.

    findbadcolumns : `bool`, optional
        If set to True, the code will search for bad column sections.  Note
        that this treats columns as special and breaks symmetry between lines
        and columns and so is likely only appropriate for detectors which have
        readout directions.
        Default is ``False``.

    byblocks : `bool`, optional
        If set to true, the code will divide the image up in to blocks of size
        nlsig by ncsig and determine the standard deviation estimate in each
        block (as described in the original IRAF task, see Notes below).  If
        set to False, then the code will use `scipy.ndimage.percentile_filter`
        to generate a running box version of the standard
        deviation estimate and use that value for the standard deviation at
        each pixel.
        Default is ``False``.

    ncmed, nlmed : `int`, optional
        The column and line size of the moving median rectangle used to
        estimate the uncontaminated local signal. The column median size should
        be at least 3 pixels to span single bad columns.
        Default is ``7``.

    ncsig, nlsig : `int`, optional
        The column and line size of regions used to estimate the uncontaminated
        local sigma using a percentile. The size of the box should contain of
        order 100 pixels or more.
        Default is ``15``.

    lsigma, hsigma : `float`, optional
        Positive sigma factors to use for selecting pixels below and above the
        median level based on the local percentile sigma.
        Default is ``9``.

    ngood : `int`, optional
        Gaps of undetected pixels along the column direction of length less
        than this amount are also flagged as bad pixels, if they are between
        pixels masked in that column.
        Default is ``5``.

    Returns
    -------
    mask : `numpy.ndarray`
        A boolean ndarray where the bad pixels have a value of 1 (True) and
        valid pixels 0 (False), following the numpy.ma conventions.

    Notes
    -----
    Similar implementation to IRAF's ccdmask task.
    The Following documentation is copied directly from:
    http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?ccdmask

    The input image is first subtracted by a moving box median. The median is
    unaffected by bad pixels provided the median size is larger that twice the
    size of a bad region. Thus, if 3 pixel wide bad columns are present then
    the column median box size should be at least 7 pixels. The median box can
    be a single pixel wide along one dimension if needed. This may be
    appropriate for spectroscopic long slit data.

    The median subtracted image is then divided into blocks of size nclsig by
    nlsig. In each block the pixel values are sorted and the pixels nearest the
    30.9 and 69.1 percentile points are found; this would be the one sigma
    points in a Gaussian noise distribution. The difference between the two
    count levels divided by two is then the local sigma estimate. This
    algorithm is used to avoid contamination by the bad pixel values. The block
    size must be at least 10 pixels in each dimension to provide sufficient
    pixels for a good estimate of the percentile sigma. The sigma uncertainty
    estimate of each pixel in the image is then the sigma from the nearest
    block.

    The deviant pixels are found by comparing the median subtracted residual to
    a specified sigma threshold factor times the local sigma above and below
    zero (the lsigma and hsigma parameters). This is done for individual pixels
    and then for column sums of pixels (excluding previously flagged bad
    pixels) from two to the number of lines in the image. The sigma of the sums
    is scaled by the square root of the number of pixels summed so that
    statistically low or high column regions may be detected even though
    individual pixels may not be statistically deviant. For the purpose of this
    task one would normally select large sigma threshold factors such as six or
    greater to detect only true bad pixels and not the extremes of the noise
    distribution.

    As a final step each column is examined to see if there are small segments
    of unflagged pixels between bad pixels. If the length of a segment is less
    than that given by the ngood parameter all the pixels in the segment are
    also marked as bad.
    """
    try:
        nlines, ncols = ratio.data.shape
    except (TypeError, ValueError):
        # shape is not iterable or has more or less than two values
        raise ValueError('"ratio" must be two-dimensional.')
    except AttributeError:
        # No data attribute or data has no shape attribute.
        raise ValueError('"ratio" should be a "CCDData".')

    def _sigma_mask(baseline, one_sigma_value, lower_sigma, upper_sigma):
        """Helper function to mask values outside of the specified sigma range.
        """
        return ((baseline < -lower_sigma * one_sigma_value) |
                (baseline > upper_sigma * one_sigma_value))

    mask = ~np.isfinite(ratio.data)
    medsub = (ratio.data -
              ndimage.median_filter(ratio.data, size=(nlmed, ncmed)))

    if byblocks:
        nlinesblock = int(math.ceil(nlines / nlsig))
        ncolsblock = int(math.ceil(ncols / ncsig))
        for i in range(nlinesblock):
            for j in range(ncolsblock):
                l1 = i * nlsig
                l2 = min((i + 1) * nlsig, nlines)
                c1 = j * ncsig
                c2 = min((j + 1) * ncsig, ncols)
                block = medsub[l1:l2, c1:c2]
                high = np.percentile(block.ravel(), 69.1)
                low = np.percentile(block.ravel(), 30.9)
                block_sigma = (high - low) / 2.0
                block_mask = _sigma_mask(block, block_sigma, lsigma, hsigma)
                mblock = np.ma.MaskedArray(block, mask=block_mask, copy=False)

                if findbadcolumns:
                    csum = np.ma.sum(mblock, axis=0)
                    csum[csum <= 0] = 0
                    csum_sigma = np.ma.MaskedArray(np.sqrt(c2 - c1 - csum))
                    colmask = _sigma_mask(csum.filled(1), csum_sigma,
                                          lsigma, hsigma)
                    block_mask[:, :] |= colmask[np.newaxis, :]

                mask[l1:l2, c1:c2] = block_mask
    else:
        high = ndimage.percentile_filter(medsub, 69.1, size=(nlsig, ncsig))
        low = ndimage.percentile_filter(medsub, 30.9, size=(nlsig, ncsig))
        sigmas = (high - low) / 2.0
        mask |= _sigma_mask(medsub, sigmas, lsigma, hsigma)

    if findbadcolumns:
        # Loop through columns and look for short segments (<ngood pixels long)
        # which are unmasked, but are surrounded by masked pixels and mask them
        # under the assumption that the column region is bad.
        for col in range(0, ncols):
            for line in range(0, nlines - ngood - 1):
                if mask[line, col]:
                    for i in range(2, ngood + 2):
                        lend = line + i
                        if (mask[lend, col] and
                                not np.all(mask[line:lend + 1, col])):
                            mask[line:lend, col] = True
    return mask


def bitfield_to_boolean_mask(bitfield, ignore_bits=0, flip_bits=None):
    """Convert an integer bit field to a boolean mask.

    Parameters
    ----------
    bitfield : `numpy.ndarray` of integer dtype
        The array of bit flags.

    ignore_bits : int, None or str, optional
        The bits to ignore when converting the bitfield.

        - If it's an integer it's binary representation is interpreted as the
          bits to ignore. ``0`` means that all bit flags are taken into
          account while a binary representation of all ``1`` means that all
          flags would be ignored.
        - If it's ``None`` then all flags are ignored
        - If it's a string then it must be a ``,`` or ``+`` separated string
          of integers that bits to ignore. If the string starts with an ``~``
          the integers are interpreted as **the only flags** to take into
          account.

        Default is ``0``.

    Returns
    -------
    mask : `numpy.ndarray` of boolean dtype
        The bitfield converted to a boolean mask that can be used for
        `numpy.ma.MaskedArray` or `~astropy.nddata.CCDData`.

    Examples
    --------
    Bitfields (or data quality arrays) are integer arrays where the binary
    representation of the values indicates whether a specific flag is set or
    not. The convention is that a value of ``0`` represents a **good value**
    and a value that is ``!= 0`` represents a value that is in some (or
    multiple) ways considered a **bad value**. The ``bitfield_to_boolean_mask``
    function can be used by default to create a boolean mask wherever any bit
    flag is set::

        >>> import ccdproc
        >>> import numpy as np
        >>> ccdproc.bitfield_to_boolean_mask(np.arange(8))
        array([False,  True,  True,  True,  True,  True,  True,  True]...)

    To ignore all bit flags ``ignore_bits=None`` can be used::

        >>> ccdproc.bitfield_to_boolean_mask(np.arange(8), ignore_bits=None)
        array([False, False, False, False, False, False, False, False]...)

    To ignore only specific bit flags one can use a ``list`` of bits flags to
    ignore::

        >>> ccdproc.bitfield_to_boolean_mask(np.arange(8), ignore_bits=[1, 4])
        array([False, False,  True,  True, False, False,  True,  True]...)

    There are some equivalent ways::

        >>> # pass in the sum of the "ignore_bits" directly
        >>> ccdproc.bitfield_to_boolean_mask(np.arange(8), ignore_bits=5)  # 1 + 4
        array([False, False,  True,  True, False, False,  True,  True]...)
        >>> # use a comma seperated string of integers
        >>> ccdproc.bitfield_to_boolean_mask(np.arange(8), ignore_bits='1, 4')
        array([False, False,  True,  True, False, False,  True,  True]...)
        >>> # use a + seperated string of integers
        >>> ccdproc.bitfield_to_boolean_mask(np.arange(8), ignore_bits='1+4')
        array([False, False,  True,  True, False, False,  True,  True]...)

    Instead of directly specifying the **bits flags to ignore** one can also
    pass in the **only bits that shouldn't be ignored** by prepending a ``~``
    to the string of ``ignore_bits`` (or if it's not a string in
    ``ignore_bits`` one can set ``flip_bits=True``)::

        >>> # ignore all bit flags except the one for 2.
        >>> ccdproc.bitfield_to_boolean_mask(np.arange(8), ignore_bits='~(2)')
        array([False, False,  True,  True, False, False,  True,  True]...)
        >>> # ignore all bit flags except the one for 1, 8 and 32.
        >>> ccdproc.bitfield_to_boolean_mask(np.arange(8), ignore_bits='~(1, 8, 32)')
        array([False,  True, False,  True, False,  True, False,  True]...)

        >>> # Equivalent for a list using flip_bits.
        >>> ccdproc.bitfield_to_boolean_mask(np.arange(8), ignore_bits=[1, 8, 32], flip_bits=True)
        array([False,  True, False,  True, False,  True, False,  True]...)

    """
    return _bitfield_to_boolean_mask(
        bitfield, ignore_bits, flip_bits=flip_bits,
        good_mask_value=False, dtype=bool)


class Keyword:
    """
    """
    def __init__(self, name, unit=None, value=None):
        self._name = name
        self._unit = unit
        self.value = value

    @property
    def name(self):
        return self._name

    @property
    def unit(self):
        return self._unit

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value is None:
            self._value = value
        elif isinstance(value, Quantity):
            self._unit = value.unit
            self._value = value
        elif isinstance(value, str):
            if self.unit is not None:
                raise ValueError("keyword with a unit cannot have a "
                                 "string value.")
            else:
                self._value = value
        else:
            if self.unit is None:
                raise ValueError("no unit provided. Set value with "
                                 "an astropy.units.Quantity.")
            self._value = value * self.unit

    def value_from(self, header):
        """
        Set value of keyword from FITS header.

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            FITS header containing a value for this keyword.
        """
        value_from_header = header[self.name]
        self.value = value_from_header
        return self.value
