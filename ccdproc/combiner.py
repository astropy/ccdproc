# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the combiner class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy import ma
from .ccddata import CCDData

from astropy.stats import median_absolute_deviation
from astropy.nddata import StdDevUncertainty

__all__ = ['Combiner']


class Combiner(object):

    """A class for combining CCDData objects.

    The Combiner class is used to combine together CCDData objects
    including the method for combining the data, rejecting outlying data,
    and weighting used for combining frames

    Parameters
    -----------
    ccd_list : `list`
        A list of CCDData objects that will be combined together.

    Raises
    ------
    TypeError
        If the `ccd_list` are not `~ccdproc.CCDData` objects, have different
        units, or are different shapes

    Notes
    -----
    The following is an example of combining together different
    `~ccdproc.CCDData` objects:

        >>> from combiner import combiner
        >>> c = combiner([ccddata1, cccdata2, ccddata3])
        >>> ccdall = c.median_combine()

    """

    def __init__(self, ccd_list):
        if ccd_list is None:
            raise TypeError("ccd_list should be a list of CCDData objects")

        default_shape = None
        default_unit = None
        for ccd in ccd_list:
            #raise an error if the objects aren't CCDDAata objects
            if not isinstance(ccd, CCDData):
                raise TypeError("ccd_list should only contain CCDData objects")

            #raise an error if the shape is different
            if default_shape is None:
                default_shape = ccd.shape
            else:
                if not (default_shape == ccd.shape):
                    raise TypeError("CCDData objects are not the same size")

            #raise an error if the units are different
            if default_unit is None:
                default_unit = ccd.unit
            else:
                if not (default_unit == ccd.unit):
                    raise TypeError("CCDdata objects are not the same unit")

        self.ccd_list = ccd_list
        self.unit = default_unit
        self.weights = None

        #set up the data array
        ydim, xdim = default_shape
        new_shape = (len(ccd_list), ydim, xdim)
        self.data_arr = ma.masked_all(new_shape)

        #populate self.data_arr
        for i, ccd in enumerate(ccd_list):
            self.data_arr[i] = ccd.data
            if ccd.mask is not None:
                self.data_arr.mask[i] = ccd.mask
            else:
                self.data_arr.mask[i] = ma.zeros((ydim, xdim))

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                if value.shape == self.data_arr.data.shape:
                    self._weights = value
                else:
                    raise ValueError("dimensions of weights do not match data")
            else:
                raise TypeError("mask must be a Numpy array")
        else:
            self._weights = None

    #set up min/max clipping algorithms
    def minmax_clipping(self, min_clip=None, max_clip=None):
        """Mask all pixels that are below min_clip or above max_clip.

         Parameters
         -----------
         min_clip : None or float
             If specified, all pixels with values below min_clip will be masked

         max_clip : None or float
             If specified, all pixels with values above min_clip will be masked

        """
        if min_clip is not None:
            mask = (self.data_arr < min_clip)
            self.data_arr.mask[mask] = True
        if max_clip is not None:
            mask = (self.data_arr > max_clip)
            self.data_arr.mask[mask] = True

    #set up sigma  clipping algorithms
    def sigma_clipping(self, low_thresh=3, high_thresh=3,
                       func=ma.mean, dev_func=ma.std):
        """Pixels will be rejected if they have deviations greater than those
           set by the threshold values.   The algorithm will first calculated
           a baseline value using the function specified in func and deviation
           based on dev_func and the input data array.   Any pixel with a
           deviation from the baseline value greater than that set by
           high_thresh or lower than that set by low_thresh will be rejected.

        Parameters
        -----------
        low_thresh : positive float or None
            Threshold for rejecting pixels that deviate below the baseline
            value.  If negative value, then will be convert to a positive
            value.   If None, no rejection will be done based on low_thresh.

        high_thresh : positive float or None
            Threshold for rejecting pixels that deviate above the baseline
            value. If None, no rejection will be done based on high_thresh.

        func : function
            Function for calculating the baseline values (i.e. mean or median).
            This should be a function that can handle
            numpy.ma.core.MaskedArray objects.

        dev_func : function
            Function for calculating the deviation from the baseline value
            (i.e. std).  This should be a function that can handle
            numpy.ma.core.MaskedArray objects.

        """
        #check for negative numbers in low_thresh

        #setup baseline values
        baseline = func(self.data_arr, axis=0)
        dev = dev_func(self.data_arr, axis=0)
        #reject values
        if low_thresh is not None:
            if low_thresh < 0:
                low_thresh = abs(low_thresh)
            mask = (self.data_arr - baseline < -low_thresh * dev)
            self.data_arr.mask[mask] = True
        if high_thresh is not None:
            mask = (self.data_arr - baseline > high_thresh * dev)
            self.data_arr.mask[mask] = True

    #set up the combining algorithms
    def median_combine(self, median_func=ma.median,
                       scale_func=None, scale_to=1.0):
        """Median combine a set of arrays.

           A CCDData object is returned
           with the data property set to the median of the arrays.  If the data
           was masked or any data have been rejected, those pixels will not be
           included in the median.   A mask will be returned, and if a pixel
           has been rejected in all images, it will be masked.   The
           uncertainty of the combined image is set by 1.4826 times the median
           absolute deviation of all input images.

           Parameters
           ----------
           median_func : function, optional
               Function that calculates median of a ``numpy.ma.masked_array``.
               Default is to use ``np.ma.median`` to calculate median.

           scale_func : function, optional
               A function (e.g. `~numpy.ma.median`) that should be used to
               calculate a value for each image by which each image should
               be scaled before combining.

           scale_to : float, optional
               Value to which data should be scaled using `scale_func`.

           Returns
           -------
           combined_image: `~ccdproc.CCDData`
               CCDData object based on the combined input of CCDData objects.

           Warnings
           --------
           The uncertainty currently calculated using the median absolute
           deviation does not account for rejected pixels

        """
        if scale_func:
            scalings = scale_func(self.data_arr, axis=2)
            scalings = scale_to/scale_func(scalings, axis=1)
            scalings = scalings[:, np.newaxis, np.newaxis]
        else:
            scalings = 1.0

        #set the data
        data = median_func(scalings * self.data_arr, axis=0)

        #set the mask
        mask = self.data_arr.mask.sum(axis=0)
        mask = (mask == len(self.data_arr))

        #set the uncertainty
        uncertainty = 1.4826 * median_absolute_deviation(self.data_arr.data,
                                                         axis=0)

        #create the combined image
        combined_image = CCDData(data.data, mask=mask, unit=self.unit,
                                 uncertainty=StdDevUncertainty(uncertainty))

        #update the meta data
        combined_image.meta['NCOMBINE'] = len(self.data_arr)

        #return the combined image
        return combined_image

    def average_combine(self, scale_func=None, scale_to=1.0):
        """Average combine together a set of arrays.   A CCDData object is
           returned with the data property set to the average of the arrays.
           If the data was masked or any data have been rejected, those pixels
           will not be included in the median.   A mask will be returned, and
           if a pixel has been rejected in all images, it will be masked.   The
           uncertainty of the combined image is set by the standard deviation
           of the input images.

           Parameters
           ----------

           scale_func : function, optional
               A function (e.g. `~numpy.ma.median`) that should be used to
               calculate a value for each image by which each image should
               be scaled before combining.

           scale_to : float, optional
               Value to which data should be scaled using `scale_func`.

           Returns
           -------
           combined_image: `~ccdproc.CCDData`
               CCDData object based on the combined input of CCDData objects.

        """
        if scale_func:
            scalings = scale_func(self.data_arr, axis=2)
            scalings = scale_to/scale_func(scalings, axis=1)
            scalings = scalings[:, np.newaxis, np.newaxis]
        else:
            scalings = 1.0
        #set up the data
        data, wei = ma.average(scalings * self.data_arr,
                               axis=0, weights=self.weights,
                               returned=True)

        #set up the mask
        mask = self.data_arr.mask.sum(axis=0)
        mask = (mask == len(self.data_arr))

        #set up the variance
        uncertainty = ma.std(self.data_arr, axis=0)

        #create the combined image
        combined_image = CCDData(data.data, mask=mask, unit=self.unit,
                                 uncertainty=StdDevUncertainty(uncertainty))

        #update the meta data
        combined_image.meta['NCOMBINE'] = len(self.data_arr)

        #return the combined image
        return combined_image
