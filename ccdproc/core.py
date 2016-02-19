# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDPROC functions
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numbers

import numpy as np
from astropy.extern import six
from astropy.units.quantity import Quantity
from astropy import units as u
from astropy.modeling import fitting
from astropy import stats
from astropy.nddata import StdDevUncertainty

from scipy import ndimage

from .ccddata import CCDData
from .utils.slices import slice_from_string
from .log_meta import log_to_metadata

__all__ = ['background_deviation_box', 'background_deviation_filter',
           'cosmicray_median', 'cosmicray_lacosmic',
           'create_deviation', 'flat_correct', 'gain_correct', 'rebin',
           'sigma_func', 'subtract_bias', 'subtract_dark', 'subtract_overscan',
           'transform_image', 'trim_image', 'Keyword']

# The dictionary below is used to translate actual function names to names
# that are FITS compliant, i.e. 8 characters or less.
_short_names = {
    'background_deviation_box': 'bakdevbx',
    'background_deviation_filter': 'bakdfilt',
    'cosmicray_median': 'crmedian',
    'create_deviation': 'creatvar',
    'flat_correct': 'flatcor',
    'gain_correct': 'gaincor',
    'subtract_bias': 'subbias',
    'subtract_dark': 'subdark',
    'subtract_overscan': 'suboscan',
    'trim_image': 'trimim',
    'transform_image': 'tranim',
}


@log_to_metadata
def create_deviation(ccd_data, gain=None, readnoise=None):
    """
    Create a uncertainty frame.  The function will update the uncertainty
    plane which gives the standard deviation for the data. Gain is used in
    this function only to scale the data in constructing the deviation; the
    data is not scaled.

    Parameters
    ----------

    ccd_data : `~ccdproc.CCDData`
        Data whose deviation will be calculated.

    gain : `~astropy.units.Quantity`, optional
        Gain of the CCD; necessary only if ``ccd_data`` and ``readnoise``
        are not in the same units. In that case, the units of ``gain``
        should be those that convert ``ccd_data.data`` to the same units as
        ``readnoise``.

    readnoise :  `~astropy.units.Quantity`
        Read noise per pixel.

    {log}

    Raises
    ------
    UnitsError
        Raised if ``readnoise`` units are not equal to product of ``gain`` and
        ``ccd_data`` units.

    Returns
    -------
    ccd :  `~ccdproc.CCDData`
        CCDData object with uncertainty created; uncertainty is in the same
        units as the data in the parameter ``ccd_data``.

    """
    if gain is not None and not isinstance(gain, Quantity):
        raise TypeError('gain must be a astropy.units.Quantity')

    if readnoise is None:
        raise ValueError('Must provide a readnoise.')

    if not isinstance(readnoise, Quantity):
        raise TypeError('readnoise must be a astropy.units.Quantity')

    if gain is None:
        gain = 1.0 * u.dimensionless_unscaled

    if gain.unit * ccd_data.unit != readnoise.unit:
        raise u.UnitsError("Units of data, gain and readnoise do not match")

    # Need to convert Quantity to plain number because NDData data is not
    # a Quantity. All unit checking should happen prior to this point.
    gain_value = float(gain / gain.unit)
    readnoise_value = float(readnoise / readnoise.unit)

    var = (gain_value * ccd_data.data + readnoise_value ** 2) ** 0.5
    ccd = ccd_data.copy()
    # ensure uncertainty and image data have same unit
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
    ccd : `~ccdproc.CCDData`
        Data to have overscan frame corrected

    overscan : `~ccdproc.CCDData`
        Slice from ``ccd`` that contains the overscan. Must provide either
        this argument or ``fits_section``, but not both.

    overscan_axis : None, 0 or 1, optional
        Axis along which overscan should combined with mean or median. Axis
        numbering follows the *python* convention for ordering, so 0 is the
        first axis and 1 is the second axis.

        If overscan_axis is explicitly set to None, the axis is set to
        the shortest dimension of the overscan section (or 1 in case
        of a square overscan).

    fits_section :  str
        Region of ``ccd`` from which the overscan is extracted, using the FITS
        conventions for index order and index start. See Notes and Examples
        below. Must provide either this argument or ``overscan``, but not both.

    median :  bool, optional
        If true, takes the median of each line.  Otherwise, uses the mean

    model :  `~astropy.modeling.Model`, optional
        Model to fit to the data.  If None, returns the values calculated
        by the median or the mean.

    {log}

    Raises
    ------
    TypeError
        A TypeError is raised if either ``ccd`` or ``overscan`` are not the
        correct objects.

    Returns
    -------
    ccd :  `~ccdproc.CCDData`
        CCDData object with overscan subtracted


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

    >>> import numpy as np
    >>> from astropy import units as u
    >>> arr1 = CCDData(np.ones([100, 100]), unit=u.adu)

    The statement below uses all rows of columns 90 through 99 as the
    overscan.

    >>> no_scan = subtract_overscan(arr1, overscan=arr1[:, 90:100])
    >>> assert (no_scan.data == 0).all()

    This statement does the same as the above, but with a FITS-style section.

    >>> no_scan = subtract_overscan(arr1, fits_section='[91:100, :]')
    >>> assert (no_scan.data == 0).all()

    Spaces are stripped out of the ``fits_section`` string.

    """
    if not (isinstance(ccd, CCDData) or isinstance(ccd, np.ndarray)):
        raise TypeError('ccddata is not a CCDData or ndarray object')

    if ((overscan is not None and fits_section is not None) or
            (overscan is None and fits_section is None)):
        raise TypeError('Specify either overscan or fits_section, but not both')

    if (overscan is not None) and (not isinstance(overscan, CCDData)):
        raise TypeError('overscan is not a CCDData object')

    if (fits_section is not None) and not isinstance(fits_section, six.string_types):
        raise TypeError('overscan is not a string')

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

    ccd : `~ccdproc.CCDData`
        CCD image to be trimmed, sliced if desired.

    fits_section : str
        Region of ``ccd`` from which the overscan is extracted; see
        `~ccdproc.subtract_overscan` for details.

    {log}

    Returns
    -------

    trimmed_ccd : `~ccdproc.CCDData`
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
    if fits_section is not None and not isinstance(fits_section, six.string_types):
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

    ccd : `~ccdproc.CCDData`
        Image from which bias will be subtracted

    master : `~ccdproc.CCDData`
        Master image to be subtracted from ``ccd``

    {log}

    Returns
    -------

    result :  `~ccdproc.CCDData`
        CCDData object with bias subtracted
    """
    result = ccd.subtract(master)
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

    ccd : `~ccdproc.CCDData`
        Image from which dark will be subtracted

    master : `~ccdproc.CCDData`
        Dark image

    dark_exposure : `~astropy.units.Quantity`
        Exposure time of the dark image; if specified, must also provided
        ``data_exposure``.

    data_exposure : `~astropy.units.Quantity`
        Exposure time of the science image; if specified, must also provided
        ``dark_exposure``.

    exposure_time : str or `~ccdproc.Keyword`
        Name of key in image metadata that contains exposure time.

    exposure_unit : `~astropy.units.Unit`
        Unit of the exposure time if the value in the meta data does not
        include a unit.

    {log}

    Returns
    -------

    result : `~ccdproc.CCDData`
        Dark-subtracted image
    """
    if not (isinstance(ccd, CCDData) and isinstance(master, CCDData)):
        raise TypeError("ccd and master must both be CCDData objects")

    if (data_exposure is not None and
            dark_exposure is not None and
            exposure_time is not None):
        raise TypeError("Specify either exposure_time or "
                        "(dark_exposure and data_exposure), not both.")

    if data_exposure is None and dark_exposure is None:
        if exposure_time is None:
            raise TypeError("Must specify either exposure_time or both "
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
                raise TypeError("Must provide unit for exposure time")
        else:
            raise TypeError("exposure times must be astropy.units.Quantity "
                            "objects")

    if scale:
        master_scaled = master.copy()
        # data_exposure and dark_exposure are both quantities,
        # so we can just have subtract do the scaling
        master_scaled = master_scaled.multiply(data_exposure / dark_exposure)
        result = ccd.subtract(master_scaled)
    else:
        result = ccd.subtract(master)

    result.meta = ccd.meta.copy()
    return result


@log_to_metadata
def gain_correct(ccd, gain, gain_unit=None):
    """Correct the gain in the image.

    Parameters
    ----------
    ccd : `~ccdproc.CCDData`
      Data to have gain corrected

    gain :  `~astropy.units.Quantity` or `~ccdproc.Keyword`
      gain value for the image expressed in electrons per adu

    gain_unit : `~astropy.units.Unit`, optional
        Unit for the ``gain``; used only if ``gain`` itself does not provide
        units.

    {log}

    Returns
    -------
    result :  `~ccdproc.CCDData`
      CCDData object with gain corrected
    """
    if isinstance(gain, Keyword):
        gain_value = gain.value_from(ccd.header)
    elif isinstance(gain, numbers.Number) and gain_unit is not None:
        gain_value = gain * u.Unit(gain_unit)
    else:
        gain_value = gain

    result = ccd.multiply(gain_value)
    return result


@log_to_metadata
def flat_correct(ccd, flat, min_value=None):
    """Correct the image for flat fielding.

    The flat field image is normalized by its mean before flat correcting.

    Parameters
    ----------
    ccd : `~ccdproc.CCDData`
        Data to be flatfield corrected

    flat : `~ccdproc.CCDData`
        Flatfield to apply to the data

    min_value : None or float
        Minimum value for flat field.  The value can either be None and no
        minimum value is applied to the flat or specified by a float which
        will replace all values in the flat by the min_value.

    {log}

    Returns
    -------
    ccd :  `~ccdproc.CCDData`
        CCDData object with flat corrected
    """
    # Use the min_value to replace any values in the flat
    use_flat = flat
    if min_value is not None:
        flat_min = flat.copy()
        flat_min.data[flat_min.data < min_value] = min_value
        use_flat = flat_min

    # divide through the flat
    flat_corrected = ccd.divide(use_flat)
    # multiply by the mean of the flat
    flat_corrected = flat_corrected.multiply(use_flat.data.mean() *
                                             use_flat.unit)

    flat_corrected.meta = ccd.meta.copy()
    return flat_corrected


@log_to_metadata
def transform_image(ccd, transform_func, **kwargs):
    """Transform the image

    Using the function specified by transform_func, the transform will
    be applied to data, uncertainty, and mask in ccd.

    Parameters
    ----------
    ccd : `~ccdproc.CCDData`
        Data to be flatfield corrected

    transform_func : function
        Function to be used to transform the data

    kwargs: dict
        Dictionary of arguments to be used by the transform_func.

    {log}

    Returns
    -------
    ccd :  `~ccdproc.CCDData`
        A transformed CCDData object

    Notes
    -----

    At this time, transform will be applied to the uncertainy data but it
    will only transform the data.  This will not properly handle uncertainties
    that arise due to correlation between the pixels.

    These should only be geometric transformations of the images.  Other
    methods should be used if the units of ccd need to be changed.

    Examples
    --------

    Given an array that is 100x100,

    >>> import numpy as np
    >>> from astropy import units as u
    >>> arr1 = CCDData(np.ones([100, 100]), unit=u.adu)

    the syntax for transforming the array using
    scipy.ndimage.interpolation.shift

    >>> from scipy.ndimage.interpolation import shift
    >>> from ccdproc import transform_image
    >>> transformed = transform_image(arr1, shift, shift=(5.5, 8.1))

    """
    # check that it is a ccddata object
    if not (isinstance(ccd, CCDData)):
        raise TypeError('ccd is not a CCDData')

    # check that transform is a callable function
    if not hasattr(transform_func, '__call__'):
        raise TypeError('transform is not a function')

    # make a copy of the object
    nccd = ccd.copy()

    # transform the image plane
    nccd.data = transform_func(nccd.data, **kwargs)

    # transform the uncertainty plane if it exists
    if nccd.uncertainty is not None:
        nccd.uncertainty.array = transform_func(nccd.uncertainty.array,
                                                **kwargs)

    # transform the mask plane
    if nccd.mask is not None:
        mask = transform_func(nccd.mask, **kwargs)
        nccd.mask = (mask > 0)

    return nccd


def sigma_func(arr, axis=None):
    """
    Robust method for calculating the deviation of an array. ``sigma_func``
    uses the median absolute deviation to determine the standard deviation.

    Parameters
    ----------
    arr : `~ccdproc.CCDData` or `~numpy.ndarray`
        Array whose deviation is to be calculated.

    axis : None or int or tuple of ints, optional
        Axis or axes along which the function is performed.
        If ``None`` (the default) it is performed over all the dimensions of the input array.
        The axis argument can also be negative, in this case it counts from
        the last to the first axis.


    Returns
    -------
    float
        standard deviation of array
    """
    return 1.482602218505602 * stats.median_absolute_deviation(arr)


def setbox(x, y, mbox, xmax, ymax):
    """Create a box of length mbox around a position x,y.   If the box will
       be out of [0,len] then reset the edges of the box to be within the
       boundaries

       Parameters
       ----------
       x : int
           Central x-position of box

       y : int
           Central y-position of box

       mbox : int
           Width of box

       xmax : int
           Maximum x value

       ymax : int
           Maximum y value

        Returns
        -------
        x1 :  int
           Lower x corner of box

        x2 :  int
           Upper x corner of box

        y1 :  int
           Lower y corner of box

        y2 :  int
           Upper y corner of box
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

    data : `~numpy.ndarray` or `~numpy.ma.MaskedArray`
        Data to measure background deviation

    bbox :  int
        Box size for calculating background deviation

    Raises
    ------

    ValueError
        A value error is raised if bbox is less than 1

    Returns
    -------
    background : `~numpy.ndarray` or `~numpy.ma.MaskedArray`
        An array with the measured background deviation in each pixel

    """
    # Check to make sure the background box is an appropriate size
    # If it is too small, then insufficient statistics are generated
    if bbox < 1:
        raise ValueError('bbox must be greater than 1')

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
    data : `~numpy.ndarray`
        Data to measure background deviation

    bbox :  int
        Box size for calculating background deviation

    Raises
    ------
    ValueError
        A value error is raised if bbox is less than 1

    Returns
    -------
    background : `~numpy.ndarray` or `~numpy.ma.MaskedArray`
        An array with the measured background deviation in each pixel

    """
    # Check to make sure the background box is an appropriate size
    if bbox < 1:
        raise ValueError('bbox must be greater than 1')

    return ndimage.generic_filter(data, sigma_func, size=(bbox, bbox))


def rebin(ccd, newshape):
    """
    Rebin an array to have a new shape.

    Parameters
    ----------
    data : `~ccdproc.CCDData` or `~numpy.ndarray`
        Data to rebin

    newshape : tuple
        Tuple containing the new shape for the array

    Returns
    -------
    output : `~ccdproc.CCDData` or `~numpy.ndarray`
        An array with the new shape.  It will have the same type as the input
        object.

    Raises
    ------
    TypeError
        A type error is raised if data is not an `~numpy.ndarray` or
        `~ccdproc.CCDData`

    ValueError
        A value error is raised if the dimenisions of new shape is not equal
        to data

    Notes
    -----
    This is based on the scipy cookbook for rebinning:
    http://wiki.scipy.org/Cookbook/Rebinning

    If rebinning a CCDData object to a smaller shape, the masking and
    uncertainty are not handled correctly.

    Examples
    --------
    Given an array that is 100x100,

    >>> import numpy as np
    >>> from astropy import units as u
    >>> arr1 = CCDData(np.ones([10, 10]), unit=u.adu)

    the syntax for rebinning an array to a shape
    of (20,20) is

    >>> rebinned = rebin(arr1, (20,20))

    """
    # check to see that is in a nddata type
    if isinstance(ccd, np.ndarray):

        # check to see that the two arrays are going to be the same length
        if len(ccd.shape) != len(newshape):
            raise ValueError('newshape does not have the same dimensions as ccd')

        slices = [slice(0, old, old/new) for old, new in
                  zip(ccd.shape, newshape)]
        coordinates = np.mgrid[slices]
        indices = coordinates.astype('i')
        return ccd[tuple(indices)]

    elif isinstance(ccd, CCDData):
        # check to see that the two arrays are going to be the same length
        if len(ccd.shape) != len(newshape):
            raise ValueError('newshape does not have the same dimensions as ccd')

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
        raise TypeError('ccd is not an ndarray or a CCDData object')


def _blkavg(data, newshape):
    """
    Block average an array such that it has the new shape

    Parameters
    ----------
    data : `~numpy.ndarray` or `~numpy.ma.MaskedArray`
        Data to average

    newshape : tuple
        Tuple containing the new shape for the array

    Returns
    -------
    output : `~numpy.ndarray` or `~numpy.ma.MaskedArray`
        An array with the new shape and the average of the pixels

    Raises
    ------
    TypeError
        A type error is raised if data is not an `numpy.ndarray`

    ValueError
        A value error is raised if the dimensions of new shape is not equal
        to data

    Notes
    -----
    This is based on the scipy cookbook for rebinning:
    http://wiki.scipy.org/Cookbook/Rebinning
    """
    # check to see that is in a nddata type
    if not isinstance(data, np.ndarray):
        raise TypeError('data is not a ndarray object')

    # check to see that the two arrays are going to be the same length
    if len(data.shape) != len(newshape):
        raise ValueError('newshape does not have the same dimensions as data')

    shape = data.shape
    lenShape = len(shape)
    factor = np.asarray(shape)/np.asarray(newshape)

    evList = ['data.reshape('] + \
        ['newshape[%d],int(factor[%d]),' % (i, i) for i in range(lenShape)] + \
        [')'] + ['.mean(%d)' % (i + 1) for i in range(lenShape)]

    return eval(''.join(evList))


def cosmicray_lacosmic(ccd, sigclip=4.5, sigfrac=0.3,
                       objlim=5.0,  gain=1.0,  readnoise=6.5,
                       satlevel=65536.0,  pssl=0.0, niter=4,
                       sepmed=True, cleantype='meanmask', fsmode='median',
                       psfmodel='gauss',  psffwhm=2.5, psfsize=7,
                       psfk=None,  psfbeta=4.765, verbose=False):
    r"""
    Identify cosmic rays through the lacosmic technique. The lacosmic technique
    identifies cosmic rays by identifying pixels based on a variation of the
    Laplacian edge detection.  The algorithm is an implementation of the
    code describe in van Dokkum (2001) :ref:[1]_ as implemented by McCully (2014)
    [2]_.  If you use this algorithm, please cite these two works.

    Parameters
    ----------
    ccd: `~ccdproc.CCDData` or `~numpy.ndarray`
        Data to have cosmic ray cleaned
    sigclip : float, optional
        Laplacian-to-noise limit for cosmic ray detection. Lower values will
        flag more pixels as cosmic rays. Default: 4.5.
    sigfrac : float, optional
        Fractional detection limit for neighboring pixels. For cosmic ray
        neighbor pixels, a lapacian-to-noise detection limit of
        sigfrac * sigclip will be used. Default: 0.3.
    objlim : float, optional
        Minimum contrast between Laplacian image and the fine structure image.
        Increase this value if cores of bright stars are flagged as cosmic
        rays. Default: 5.0.
    pssl : float, optional
        Previously subtracted sky level in ADU. We always need to work in
        electrons for cosmic ray detection, so we need to know the sky level
        that has been subtracted so we can add it back in. Default: 0.0.
    gain : float, optional
        Gain of the image (electrons / ADU). We always need to work in
        electrons for cosmic ray detection. Default: 1.0
    readnoise : float, optional
        Read noise of the image (electrons). Used to generate the noise model
        of the image. Default: 6.5.
    satlevel : float, optional
        Saturation of level of the image (electrons). This value is used to
        detect saturated stars and pixels at or above this level are added to
        the mask. Default: 65536.0.
    niter : int, optional
        Number of iterations of the LA Cosmic algorithm to perform. Default: 4.
    sepmed : boolean, optional
        Use the separable median filter instead of the full median filter.
        The separable median is not identical to the full median filter, but
        they are approximately the same and the separable median filter is
        significantly faster and still detects cosmic rays well. Default: True
    cleantype : {'median', 'medmask', 'meanmask', 'idw'}, optional
        Set which clean algorithm is used:\n
        'median': An umasked 5x5 median filter\n
        'medmask': A masked 5x5 median filter\n
        'meanmask': A masked 5x5 mean filter\n
        'idw': A masked 5x5 inverse distance weighted interpolation\n
        Default: "meanmask".
    fsmode : {'median', 'convolve'}, optional
        Method to build the fine structure image:\n
        'median': Use the median filter in the standard LA Cosmic algorithm
        'convolve': Convolve the image with the psf kernel to calculate the
        fine structure image.
        Default: 'median'.
    psfmodel : {'gauss', 'gaussx', 'gaussy', 'moffat'}, optional
        Model to use to generate the psf kernel if fsmode == 'convolve' and
        psfk is None. The current choices are Gaussian and Moffat profiles.
        'gauss' and 'moffat' produce circular PSF kernels. The 'gaussx' and
        'gaussy' produce Gaussian kernels in the x and y directions
        respectively. Default: "gauss".
    psffwhm : float, optional
        Full Width Half Maximum of the PSF to use to generate the kernel.
        Default: 2.5.
    psfsize : int, optional
        Size of the kernel to calculate. Returned kernel will have size
        psfsize x psfsize. psfsize should be odd. Default: 7.
    psfk : float numpy array, optional
        PSF kernel array to use for the fine structure image if
        fsmode == 'convolve'. If None and fsmode == 'convolve', we calculate
        the psf kernel using 'psfmodel'. Default: None.
    psfbeta : float, optional
        Moffat beta parameter. Only used if fsmode=='convolve' and
        psfmodel=='moffat'. Default: 4.765.
    verbose : boolean, optional
        Print to the screen or not. Default: False.
    {log}

    Notes
    -----
    Implementation of the cosmic ray identification L.A.Cosmic:
    http://www.astro.yale.edu/dokkum/lacosmic/

    Returns
    -------
    nccd : `~ccdproc.CCDData` or `~numpy.ndarray`
        An object of the same type as ccd is returned.   If it is a
        `~ccdproc.CCDData`, the mask attribute will also be updated with
        areas identified with cosmic rays masked.
    crmask : `~numpy.ndarray`
        If an `~numpy.ndarray` is provided as ccd, a boolean ndarray with the
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
       includes the pixel error.  This would return a data array, newdata,
       with the bad pixels replaced by the local median from a box of 11
       pixels; and it would return a mask indicating the bad pixels.

    2) Given an `~ccdproc.CCDData` object with an uncertainty frame, the syntax
       for running cosmicrar_lacosmic would be:

       >>> newccd = cosmicray_lacosmic(ccd, sigclip=5)   # doctest: +SKIP

       The newccd object will have bad pixels in its data array replace and the
       mask of the object will be created if it did not previously exist or be
       updated with the detected cosmic rays.
    """
    from astroscrappy import detect_cosmics
    if isinstance(ccd, np.ndarray):
        data = ccd

        crmask, cleanarr = detect_cosmics(
                   data, inmask=None, sigclip=sigclip,
                   sigfrac=sigfrac, objlim=objlim, gain=gain,
                   readnoise=readnoise, satlevel=satlevel,  pssl=pssl,
                   niter=niter, sepmed=sepmed,  cleantype=cleantype,
                   fsmode=fsmode, psfmodel=psfmodel, psffwhm=psffwhm,
                   psfsize=psfsize, psfk=psfk, psfbeta=psfbeta,
                   verbose=verbose)

        return cleanarr, crmask

    elif isinstance(ccd, CCDData):

        crmask, cleanarr = detect_cosmics(
            ccd.data, inmask=ccd.mask,
            sigclip=sigclip, sigfrac=sigfrac, objlim=objlim, gain=gain,
            readnoise=readnoise, satlevel=satlevel,  pssl=pssl,
            niter=niter, sepmed=sepmed,  cleantype=cleantype,
            fsmode=fsmode, psfmodel=psfmodel, psffwhm=psffwhm,
            psfsize=psfsize, psfk=psfk, psfbeta=psfbeta, verbose=verbose)

        # create the new ccd data object
        nccd = ccd.copy()
        nccd.data = cleanarr
        if nccd.mask is None:
            nccd.mask = crmask
        else:
            nccd.mask = nccd.mask + crmask

        return nccd

    else:
        raise TypeError('ccddata is not a CCDData or ndarray object')


def cosmicray_median(ccd, error_image=None, thresh=5, mbox=11, gbox=0,
                     rbox=0):
    """
    Identify cosmic rays through median technique.  The median technique
    identifies cosmic rays by identifying pixels by subtracting a median image
    from the initial data array.

    Parameters
    ----------

    ccd : `~ccdproc.CCDData` or numpy.ndarray or numpy.MaskedArary
        Data to have cosmic ray cleaned

    thresh :  float
        Threshold for detecting cosmic rays

    error_image : None, float, or `~numpy.ndarray`
        Error level.   If None, the task will use the standard
        deviation of the data. If an ndarray, it should have the same shape
        as data.

    mbox :  int
        Median box for detecting cosmic rays

    gbox :  int
        Box size to grow cosmic rays. If zero, no growing will be done.

    rbox :  int
        Median box for calculating replacement values.  If zero, no pixels will
        be replaced.

    {log}

    Notes
    -----
    Similar implementation to crmedian in iraf.imred.crutil.crmedian

    Returns
    -------
    nccd : `~ccdproc.CCDData` or `~numpy.ndarray`
        An object of the same type as ccd is returned.   If it is a
        `~ccdproc.CCDData`, the mask attribute will also be updated with
        areas identified with cosmic rays masked.

    nccd : `~numpy.ndarray`
        If an `~numpy.ndarray` is provided as ccd, a boolean ndarray with the
        cosmic rays identified will also be returned.

    Examples
    --------

    1) Given an numpy.ndarray object, the syntax for running
       cosmicray_median would be:

       >>> newdata, mask = cosmicray_median(data, error_image=error,
       ...                                  thresh=5, mbox=11,
       ...                                  rbox=11, gbox=5)   # doctest: +SKIP

       where error is an array that is the same shape as data but
       includes the pixel error.  This would return a data array, newdata,
       with the bad pixels replaced by the local median from a box of 11
       pixels; and it would return a mask indicating the bad pixels.

    2) Given an `~ccdproc.CCDData` object with an uncertainty frame, the syntax
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
                raise TypeError('error_image is not a float or ndarray')

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
            raise ValueError('error_image is not the same shape as data')

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
        raise TypeError('ccd is not an ndarray or a CCDData object')


class Keyword(object):
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
        elif isinstance(value, six.string_types):
            if self.unit is not None:
                raise ValueError("Keyword with a unit cannot have a "
                                 "string value.")
            else:
                self._value = value
        else:
            if self.unit is None:
                raise ValueError("No unit provided. Set value with "
                                 "an astropy.units.Quantity")
            self._value = value * self.unit

    def value_from(self, header):
        """
        Set value of keyword from FITS header

        Parameters
        ----------

        header : `~astropy.io.fits.Header`
            FITS header containing a value for this keyword
        """

        value_from_header = header[self.name]
        self.value = value_from_header
        return self.value
