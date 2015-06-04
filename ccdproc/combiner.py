# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the combiner class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy import ma
from .ccddata import CCDData
from .core import trim_image

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

    dtype : 'numpy dtype'
        Allows user to set dtype.

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

    def __init__(self, ccd_list, dtype=None):
        if ccd_list is None:
            raise TypeError("ccd_list should be a list of CCDData objects")

        if dtype is None:
            dtype = np.float64

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
        self._dtype = dtype

        #set up the data array
        ydim, xdim = default_shape
        new_shape = (len(ccd_list), ydim, xdim)
        self.data_arr = ma.masked_all(new_shape, dtype=dtype)

        #populate self.data_arr
        for i, ccd in enumerate(ccd_list):
            self.data_arr[i] = ccd.data
            if ccd.mask is not None:
                self.data_arr.mask[i] = ccd.mask
            else:
                self.data_arr.mask[i] = ma.zeros((ydim, xdim))

        # Must be after self.data_arr is defined because it checks the
        # length of the data array.
        self.scaling = None

    @property
    def dtype(self):
        return self._dtype

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

    @property
    def scaling(self):
        """
        Scaling factor used in combining images.

        Parameters
        ----------

        scale : function or array-like or None, optional
            Images are multiplied by scaling prior to combining them. Scaling
            may be either a function, which will be applied to each image
            to determine the scaling factor, or a list or array whose length
            is the number of images in the `Combiner`. Default is ``None``.
        """
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        if value is None:
            self._scaling = value
        else:
            n_images = self.data_arr.data.shape[0]
            if callable(value):
                self._scaling = [value(self.data_arr[i]) for
                                 i in range(n_images)]
                self._scaling = np.array(self._scaling)
            else:
                try:
                    len(value) == n_images
                    self._scaling = np.array(value)
                except TypeError:
                    raise TypeError("Scaling must be a function or an array "
                                    "the same length as the number of images.")
            # reshape so that broadcasting occurs properly
            self._scaling = self.scaling[:, np.newaxis, np.newaxis]

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
    def median_combine(self, median_func=ma.median):
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

           Returns
           -------
           combined_image: `~ccdproc.CCDData`
               CCDData object based on the combined input of CCDData objects.

           Warnings
           --------
           The uncertainty currently calculated using the median absolute
           deviation does not account for rejected pixels

        """
        if self.scaling is not None:
            scalings = self.scaling
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

        # create the combined image with a dtype matching the combiner
        combined_image = CCDData(np.asarray(data.data, dtype=self.dtype),
                                 mask=mask, unit=self.unit,
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

           Returns
           -------
           combined_image: `~ccdproc.CCDData`
               CCDData object based on the combined input of CCDData objects.

        """
        if self.scaling is not None:
            scalings = self.scaling
        else:
            scalings = 1.0
        #set up the data
        data, wei = ma.average(scalings * self.data_arr,
                               axis=0, weights=self.weights,
                               returned=True)

        #set up the mask
        mask = self.data_arr.mask.sum(axis=0)
        mask = (mask == len(self.data_arr))

        #set up the deviation
        uncertainty = ma.std(self.data_arr, axis=0)

        # create the combined image with a dtype that matches the combiner
        combined_image = CCDData(np.asarray(data.data, dtype=self.dtype),
                                 mask=mask, unit=self.unit,
                                 uncertainty=StdDevUncertainty(uncertainty))

        #update the meta data
        combined_image.meta['NCOMBINE'] = len(self.data_arr)

        #return the combined image
        return combined_image


class Combine_fits(object):

    """A class for combining fits images with limited memory usage

    The Combine_fits class is used to combine together fits images with
    limit on the maximum memory usage, it including all the  methods
    of the Combiner class for combining the data, rejecting outlying data,
    and weighting used for combining frames etc.

    Parameters
    -----------
    img_list : `list`
        A list of fits filenames that will be combined together.
    memlimit : float (default 16e9)
        Maximum memory which should be used for combining in bytes.
    output_fits: 'string'
        Optional output fits filename
    
             : Other keyword arguments for CCD Object fits reader
    """
    def __init__(self, img_list, mem_limit=16e9, output_fits=None, **kwargs):
        self.img_list = img_list
        self.mem_limit = mem_limit
        self.output_fits = output_fits
        self.ccdkwargs = kwargs
        # We create a dummy output CCDObject from first image for storing output
        self.ccd_dummy = CCDData.read(img_list[0],**self.ccdkwargs)
        
        size_of_an_img = self.ccd_dummy.data.nbytes
        no_of_img = len(self.img_list)
        
        #determine the number of chunks to split the images into
        no_chunks = int((size_of_an_img*no_of_img)/self.mem_limit)+1
        print('Spliting each image into {1} chunks to limit memory usage to {0} bytes.'.format(self.mem_limit,no_chunks))
        self.Xs, self.Ys = self.ccd_dummy.data.shape
        self.Xstep = max(1, int(self.Xs/no_chunks)) # First we try to split only along fast x axis
        # If more chunks need to be created we split in Y axis for remaining number of chunks
        self.Ystep = max(1, int(self.Ys/(1+ no_chunks - int(self.Xs/self.Xstep)) ) ) 
        
        # List of Combiner properties to set and methods to call before combining
        self.to_set_in_combiner = {}
        self.to_call_in_combiner = {}

        # Replace all the doc strings with the original doc strings in Combiner class
        self.set_scaling.__func__.__doc__ = Combiner.scaling.__doc__ 
        self.minmax_clipping.__func__.__doc__ = Combiner.minmax_clipping.__doc__ 
        self.sigma_clipping.__func__.__doc__ = Combiner.sigma_clipping.__doc__ 
        self.median_combine.__func__.__doc__ = Combiner.median_combine.__doc__ 
        self.average_combine.__func__.__doc__ = Combiner.average_combine.__doc__ 

    # Define all the Combiner properties one wants to apply before combining images
    def set_weights(self, value):
        """ Sets the weights in the Combiner"""
        self.to_set_in_combiner['weights'] = value

    def set_scaling(self, value):
        """ Sets the scaling property of Combiner."""
        self.to_set_in_combiner['scaling'] = value


    def minmax_clipping(self, **kwargs):
        """Sets to Mask all pixels that are below min_clip or above max_clip."""
        self.to_call_in_combiner['minmax_clipping'] = kwargs

    def sigma_clipping(self, **kwargs):
        """Sets to Mask all pixels that are below or above certain sigma."""
        self.to_call_in_combiner['sigma_clipping'] = kwargs


    # Define Combiner's combining methods
    def median_combine(self, **kwargs):
        """ Run Combiner to Median combine a set of images."""
        self.run_on_all_tiles('median_combine', **kwargs)
        return self.ccd_dummy

    def average_combine(self, **kwargs):
        """ Run Combiner to Average combine a set of images."""
        self.run_on_all_tiles('average_combine', **kwargs)
        return self.ccd_dummy

    def run_on_all_tiles(self, method, **kwargs):
        """ Runs the input method on all the subsections of the image and return final stitched image"""
        try:
            calculate_scalevalue = callable(self.to_set_in_combiner['scaling'])
        except KeyError:
            # No scaling requested..
            calculate_scalevalue = False

        if calculate_scalevalue:
            scalevalues = []

        for x in range(0,self.Xs,self.Xstep):
            for y in range(0,self.Ys,self.Ystep):
                xend, yend = min(self.Xs, x + self.Xstep), min(self.Ys, y + self.Ystep)
                ccd_list = []
                for image in self.img_list:
                    imgccd = CCDData.read(image,**self.ccdkwargs)
                    #Scaling function need to be applied on full image to obtain scaling factor
                    if calculate_scalevalue:
                        scalevalues.append(self.to_set_in_combiner['scaling'](imgccd.data))
                        
                    ccd_list.append(trim_image(imgccd[x:xend, y:yend])) #Trim image

                if calculate_scalevalue:
                    self.to_set_in_combiner['scaling'] = np.array(scalevalues) #Replace callable with array
                    calculate_scalevalue = False

                Tile_combiner = Combiner(ccd_list) # Create Combiner for tile
                # Set all properties and call all methods
                for to_set in self.to_set_in_combiner:
                    setattr(Tile_combiner, to_set, self.to_set_in_combiner[to_set])
                for to_call in self.to_call_in_combiner:
                    getattr(Tile_combiner, to_call)(**self.to_call_in_combiner[to_call])

                # Finally call the combine algorithm
                comb_tile = getattr(Tile_combiner, method)(**kwargs)
 
                #add it back into the master image
                self.ccd_dummy.data[x:xend, y:yend] = comb_tile.data
                if self.ccd_dummy.mask is not None:
                    self.ccd_dummy.mask[x:xend, y:yend] = comb_tile.mask
                if self.ccd_dummy.uncertainty is not None:
                    self.ccd_dummy.uncertainty.array[x:xend, y:yend] = comb_tile.uncertainty.array
  
        # Write fits file if filename was provided
        if self.output_fits is not None:
            self.ccd_dummy.write(self.output_fits)
        
