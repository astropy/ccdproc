# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDPROC functions
import inspect
import numpy as np

from ccddata import CCDData, electrons

from astropy import units as u
from astropy.units.quantity import Quantity
from astropy.nddata.nduncertainty import StdDevUncertainty
from astropy.modeling import fitting
from astropy import stats

from scipy import ndimage


def subtract_overscan(ccd, overscan, median=False, model=None):
    """
    Subtract the overscan region from an image.  This will first
    has an uncertainty plane which gives the variance for the data. The
    function assumes that the ccd is in electrons and the readnoise is in the
    same units.

    Parameters
    ----------
    ccd : CCDData object
        Data to have variance frame corrected

    overscan :  CCDData object
        section from CCDData object that is defined as the overscan region

    median :  boolean
        If true, takes the median of each line.  Otherwise, uses the mean

    model :  astropy.model object
        Model to fit to the data.  If None, returns the values calculated
        by the median or the mean

    Raises
    ------
    TypeError
        A TypeError is raised if either ccd or overscan are not the correct
        objects.

    Returns
    -------
    ccd :  CCDData object
        CCDData object with overscan subtracted
    """
    if not (isinstance(ccd, CCDData) or isinstance(ccd, np.ndarray)):
        raise TypeError('ccddata is not a CCDData or ndarray object')
    if not (isinstance(overscan, CCDData) or isinstance(overscan, np.ndarray)):
        raise TypeError('overscan is not a CCDData or ndarray object')

    if median:
        oscan = np.median(overscan.data, axis=1)
    else:
        oscan = np.mean(overscan.data, axis=1)

    if model is not None:
        of = fitting.LinearLSQFitter()
        yarr = np.arange(len(oscan))
        oscan = of(model, yarr, oscan)
        oscan = oscan(yarr)
        oscan = np.reshape(oscan, (oscan.size, 1))
    else:
        oscan = np.reshape(oscan, oscan.shape + (1,))

    # subtract the overscan
    ccd.data = ccd.data - oscan
    return ccd


def gain_correct(ccd, gain):
    """Correct the gain in the image.

       Parameters
       ----------
       ccd : CCDData object
          Data to have variance frame corrected

       gain :  float or quantity
          gain value for the image expressed in electrions per adu


       Returns
       -------
       ccd :  CCDData object
          CCDData object with gain corrected
    """
    if isinstance(gain, Quantity):
        ccd.data = ccd.data * gain.value
        ccd.unit = ccd.unit * gain.unit
    else:
        ccd.data = ccd.data * gain
    return ccd


def flat_correct(ccd, flat):
    """Correct the image for flatfielding

       Parameters
       ----------
       ccd : CCDData object
          Data to be flatfield corrected

       flat : CCDData object
          Flatfield to apply to the data

       Returns
       -------
       ccd :  CCDData object
          CCDData object with flat corrected
    """
    # normalize the flat
    flat.data = flat.data / flat.data.mean()
    if flat.uncertainty is not None:
        flat.uncertainty.array = flat.uncertainty.array / flat.data.mean()

    # divide through the flat
    ccd.divide(flat)

    return ccd


def sigma_func(arr):
    """
    Robust method for calculating the variance of an array. ``sigma_func`` uses
    the median absolute deviation to determine the variance.

    Parameters
    ----------
    arr : ccdproc.CCDData or np.array
        Array whose variance is to be calculated.

    Returns
    -------
    float
        variance of array
    """
    return 1.4826 * stats.median_absolute_deviation(arr)


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


def background_variance_box(data, bbox):
    """
    Determine the background variance with a box size of bbox. The algorithm
    steps through the image and calculates the variance within each box.
    It returns an array with the pixels in each box filled with the variance
    value.

    Parameters
    ----------
    data : numpy ndarray or Mask arary object
        Data to measure background variance

    bbox :  int
        Box size for calculating background variance

    Raises
    ------
    ValueError
        A value error is raised if bbox is less than 1

    Returns
    -------
    background : numpy ndarray or Mask arary object
        An array with the measured background variance in each pixel

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


def background_variance_filter(data, bbox):
    """
    Determine the background variance for each pixel from a box with size of
    bbox.

    Parameters
    ----------
    data : numpy ndarray or Mask arary object
        Data to measure background variance

    bbox :  int
        Box size for calculating background variance

    Raises
    ------
    ValueError
        A value error is raised if bbox is less than 1

    Returns
    -------
    background : numpy ndarray or Mask arary object
        An array with the measured background variance in each pixel

    """
    # Check to make sure the background box is an appropriate size
    if bbox < 1:
        raise ValueError('bbox must be greater than 1')

    return ndimage.generic_filter(data, sigma_func, size=(bbox, bbox))


def cosmicray_median(data, thresh,  background=None, mbox=11):
    """
    Identify cosmic rays through median technique.  The median technique
    identifies cosmic rays by identifying pixels by subtracting a median image
    from the initial data array.

    Parameters
    ----------

    ccd : numpy ndarray or Mask arary object
        Data to have cosmic ray cleans

    thresh :  float
        Threshhold for detecting cosmic rays

    background : None, float, or ndarray
        Background variance level.   If None, the task will use the standard
        deviation of the data. If an ndarray, it should have the same shape
        as data.

    mbox :  int
        Median box for detecting cosmic rays


    Notes
    -----
    Similar implimentation to crmedian in iraf.imred.crutil.crmedian

    Returns
    -------
    crarr : numpy ndarray
      A boolean ndarray with the cosmic rays identified

    """

    # create the median image
    marr = ndimage.median_filter(data, size=(mbox, mbox))

    # Find the residual image
    rarr = (data - marr) / background

    # identify all sources
    crarr = (rarr > thresh)

    return crarr


def cosmicray_clean(ccddata, thresh, cr_func, crargs=(),
                    background=None, bargs=(), gbox=0, rbox=0):
    """
    Cosmic ray clean a ccddata object.  This process will apply a cosmic ray
    cleaning method, cr_func, to a data set.  The cosmic rays will be
    identified based on being above a threshold, thresh, above the background.
    The background can either be supplied by a function

    Parameters
    ----------

    ccddata : CCDData object
        Data to have cosmic ray cleans

    thresh :  float
        Threshhold for detecting cosmic rays

    cr_func :  function
        Function for identifying cosmic rays

    cargs :  tuple
        This countains any extra arguments needed for the cosmic ray function

    background : None, float, ndarray, or function
        Background variance level. If None, the task will use the standard
        deviation of the data.   If an ndarray, it should have the same shape
        as data.

    bargs :  tuple
        If background is a function, any extra arguments that are needed should
        be passed via bargs.

    gbox :  int
        Box size to grow cosmic rays. If zero, no growing will be done.

    rbox :  int
        Median box for calculating replacement values.  If zero, no pixels will
        be replaced.


    Returns
    -------
    ccddata : CCDData obejct
        A CCDData object with cosmic rays cleaned.  The ccddata.mask object
        will be updated to flag cosmic rays in the mask. If replace is set,
        then the ccddata object will be replaced with median of the
        surrounding unmasked pixels

    Examples
    --------

    This will use the median method to clean cosmic rays based on a background
    estimated in a box around the image.  It will then replace bad pixel value
    with the median of the pixels in an 11 pixel wide box around the bad pixel.

        >>> from ccdproc import background_variance_box,cosmicray_median, cosmicray_clean
        >>> cosmicray_clean(ccddata, 10, cosmicray_median, crargs(11,),
               background=background_variance_box, bargs=(25,), rbox=11)


    """

    # make a masked array that will be used for all calculations
    if ccddata.mask is None:
        data = ccddata.data
    else:
        data = np.ma.masked_array(ccddata.data, ccddata.mask)

    if background is None:
        background = sigma_func(data)
    elif hasattr(background, '__call__'):
        background = background(data, *bargs)

    # identify the cosmic rays
    crarr = cr_func(data, thresh, background, *crargs)

    # upate the mask
    if ccddata.mask is None:
        ccddata.mask = crarr
    else:
        ccddata.mask = ccddata.mask + crarr.mask

    # grow the pixels
    if gbox > 0:
        ccddata.mask = ndimage.maximum_filter(ccddata.mask, gbox)

    if rbox > 0:
        data = np.ma.masked_array(ccddata.data, (ccddata.mask == 0))
        mdata = ndimage.median_filter(data, rbox)
        ccddata.data[ccddata.mask > 0] = mdata[ccddata.mask > 0]
    return ccddata
