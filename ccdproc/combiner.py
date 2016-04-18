# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the combiner class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy import ma
from .ccddata import CCDData
from .core import trim_image, sigma_func

from astropy.stats import median_absolute_deviation
from astropy.nddata import StdDevUncertainty
from astropy import log

import math

__all__ = ['Combiner', 'combine']


class Combiner(object):
    """
    A class for combining CCDData objects.

    The Combiner class is used to combine together `~ccdproc.CCDData` objects
    including the method for combining the data, rejecting outlying data,
    and weighting used for combining frames.

    Parameters
    -----------
    ccd_list : list
        A list of CCDData objects that will be combined together.

    dtype : str or `numpy.dtype` or None, optional
        Allows user to set dtype. See `numpy.array` ``dtype`` parameter
        description.
        Default is ``None``.

    Raises
    ------
    TypeError
        If the ``ccd_list`` are not `~ccdproc.CCDData` objects, have different
        units, or are different shapes.

    Notes
    -----
    The following is an example of combining together different
    `~ccdproc.CCDData` objects::

        >>> import numpy as np
        >>> import astropy.units as u
        >>> from ccdproc import Combiner, CCDData
        >>> ccddata1 = CCDData(np.ones((4, 4)), unit=u.adu)
        >>> ccddata2 = CCDData(np.zeros((4, 4)), unit=u.adu)
        >>> ccddata3 = CCDData(np.ones((4, 4)), unit=u.adu)
        >>> c = Combiner([ccddata1, ccddata2, ccddata3])
        >>> ccdall = c.average_combine()
        >>> ccdall
        CCDData([[ 0.66666667,  0.66666667,  0.66666667,  0.66666667],
                 [ 0.66666667,  0.66666667,  0.66666667,  0.66666667],
                 [ 0.66666667,  0.66666667,  0.66666667,  0.66666667],
                 [ 0.66666667,  0.66666667,  0.66666667,  0.66666667]])
    """
    def __init__(self, ccd_list, dtype=None):
        if ccd_list is None:
            raise TypeError("ccd_list should be a list of CCDData objects.")

        if dtype is None:
            dtype = np.float64

        default_shape = None
        default_unit = None
        for ccd in ccd_list:
            # raise an error if the objects aren't CCDDAata objects
            if not isinstance(ccd, CCDData):
                raise TypeError(
                    "ccd_list should only contain CCDData objects.")

            # raise an error if the shape is different
            if default_shape is None:
                default_shape = ccd.shape
            else:
                if not (default_shape == ccd.shape):
                    raise TypeError("CCDData objects are not the same size.")

            # raise an error if the units are different
            if default_unit is None:
                default_unit = ccd.unit
            else:
                if not (default_unit == ccd.unit):
                    raise TypeError("CCDdata objects are not the same unit.")

        self.ccd_list = ccd_list
        self.unit = default_unit
        self.weights = None
        self._dtype = dtype

        # set up the data array
        ydim, xdim = default_shape
        new_shape = (len(ccd_list), ydim, xdim)
        self.data_arr = ma.masked_all(new_shape, dtype=dtype)

        # populate self.data_arr
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
        """
        Weights used when combining the `~ccdproc.CCDData` objects.

        Parameters
        ----------
        weight_values : `numpy.ndarray` or None
            An array with the weight values. The dimensions should match the
            the dimensions of the data arrays being combined.
        """
        return self._weights

    @weights.setter
    def weights(self, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                if value.shape == self.data_arr.data.shape:
                    self._weights = value
                else:
                    raise ValueError(
                        "dimensions of weights do not match data.")
            else:
                raise TypeError("mask must be a numpy ndarray.")
        else:
            self._weights = None

    @property
    def scaling(self):
        """
        Scaling factor used in combining images.

        Parameters
        ----------
        scale : function or `numpy.ndarray`-like or None, optional
            Images are multiplied by scaling prior to combining
            them. Scaling may be either a function, which will be applied to
            each image to determine the scaling factor, or a list or array
            whose length is the number of images in the `~ccdproc.Combiner`.
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

    # set up min/max clipping algorithms
    def minmax_clipping(self, min_clip=None, max_clip=None):
        """Mask all pixels that are below min_clip or above max_clip.

         Parameters
         -----------
         min_clip : None or float, optional
             If not None, all pixels with values below min_clip will be masked.
             Default is ``None``.

         max_clip : None or float, optional
             If not None, all pixels with values above min_clip will be masked.
             Default is ``None``.
        """
        if min_clip is not None:
            mask = (self.data_arr < min_clip)
            self.data_arr.mask[mask] = True
        if max_clip is not None:
            mask = (self.data_arr > max_clip)
            self.data_arr.mask[mask] = True

    # set up sigma  clipping algorithms
    def sigma_clipping(self, low_thresh=3, high_thresh=3,
                       func=ma.mean, dev_func=ma.std):
        """
        Pixels will be rejected if they have deviations greater than those
        set by the threshold values. The algorithm will first calculated
        a baseline value using the function specified in func and deviation
        based on dev_func and the input data array. Any pixel with a
        deviation from the baseline value greater than that set by
        high_thresh or lower than that set by low_thresh will be rejected.

        Parameters
        -----------
        low_thresh : positive float or None, optional
            Threshold for rejecting pixels that deviate below the baseline
            value. If negative value, then will be convert to a positive
            value. If None, no rejection will be done based on low_thresh.
            Default is 3.

        high_thresh : positive float or None, optional
            Threshold for rejecting pixels that deviate above the baseline
            value. If None, no rejection will be done based on high_thresh.
            Default is 3.

        func : function, optional
            Function for calculating the baseline values (i.e. `numpy.ma.mean`
            or `numpy.ma.median`). This should be a function that can handle
            `numpy.ma.MaskedArray` objects.
            Default is `numpy.ma.mean`.

        dev_func : function, optional
            Function for calculating the deviation from the baseline value
            (i.e. `numpy.ma.std`). This should be a function that can handle
            `numpy.ma.MaskedArray` objects.
            Default is `numpy.ma.std`.
        """
        # setup baseline values
        baseline = func(self.data_arr, axis=0)
        dev = dev_func(self.data_arr, axis=0)
        # reject values
        if low_thresh is not None:
            # check for negative numbers in low_thresh
            if low_thresh < 0:
                low_thresh = abs(low_thresh)
            mask = (self.data_arr - baseline < -low_thresh * dev)
            self.data_arr.mask[mask] = True
        if high_thresh is not None:
            mask = (self.data_arr - baseline > high_thresh * dev)
            self.data_arr.mask[mask] = True

    # set up the combining algorithms
    def median_combine(self, median_func=ma.median, scale_to=None,
                       uncertainty_func=sigma_func):
        """
        Median combine a set of arrays.

        A `~ccdproc.CCDData` object is returned with the data property set to
        the median of the arrays. If the data was masked or any data have been
        rejected, those pixels will not be included in the median. A mask will
        be returned, and if a pixel has been rejected in all images, it will be
        masked. The uncertainty of the combined image is set by 1.4826 times
        the median absolute deviation of all input images.

        Parameters
        ----------
        median_func : function, optional
            Function that calculates median of a `numpy.ma.MaskedArray`.
            Default is `numpy.ma.median`.

        scale_to : float or None, optional
            Scaling factor used in the average combined image. If given,
            it overrides ``CCDData.scaling``.
            Defaults to None.

        uncertainty_func : function, optional
            Function to calculate uncertainty.
            Defaults is `ccdproc.sigma_func`.

        Returns
        -------
        combined_image: `~ccdproc.CCDData`
            CCDData object based on the combined input of CCDData objects.

        Warnings
        --------
        The uncertainty currently calculated using the median absolute
        deviation does not account for rejected pixels.
        """
        if scale_to is not None:
            scalings = scale_to
        elif self.scaling is not None:
            scalings = self.scaling
        else:
            scalings = 1.0

        # set the data
        data = median_func(scalings * self.data_arr, axis=0)

        # set the mask
        masked_values = self.data_arr.mask.sum(axis=0)
        mask = (masked_values == len(self.data_arr))

        # set the uncertainty
        uncertainty = uncertainty_func(self.data_arr.data, axis=0)
        # Divide uncertainty by the number of pixel (#309)
        # TODO: This should be np.sqrt(len(self.data_arr) - masked_values) but
        # median_absolute_deviation ignores the mask... so it
        # would yield inconsistent results.
        uncertainty /= math.sqrt(len(self.data_arr))

        # create the combined image with a dtype matching the combiner
        combined_image = CCDData(np.asarray(data.data, dtype=self.dtype),
                                 mask=mask, unit=self.unit,
                                 uncertainty=StdDevUncertainty(uncertainty))

        # update the meta data
        combined_image.meta['NCOMBINE'] = len(self.data_arr)

        # return the combined image
        return combined_image

    def average_combine(self, scale_func=ma.average, scale_to=None,
                        uncertainty_func=ma.std):
        """
        Average combine together a set of arrays.

        A `~ccdproc.CCDData` object is returned with the data property
        set to the average of the arrays. If the data was masked or any
        data have been rejected, those pixels will not be included in the
        average. A mask will be returned, and if a pixel has been
        rejected in all images, it will be masked. The uncertainty of
        the combined image is set by the standard deviation of the input
        images.

        Parameters
        ----------
        scale_func : function, optional
            Function to calculate the average. Defaults to
            `numpy.ma.average`.

        scale_to : float or None, optional
            Scaling factor used in the average combined image. If given,
            it overrides ``CCDData.scaling``. Defaults to ``None``.

        uncertainty_func : function, optional
            Function to calculate uncertainty. Defaults to `numpy.ma.std`.

        Returns
        -------
        combined_image: `~ccdproc.CCDData`
            CCDData object based on the combined input of CCDData objects.
        """
        if scale_to is not None:
            scalings = scale_to
        elif self.scaling is not None:
            scalings = self.scaling
        else:
            scalings = 1.0

        # set up the data
        data, wei = scale_func(scalings * self.data_arr,
                               axis=0, weights=self.weights,
                               returned=True)

        # set up the mask
        masked_values = self.data_arr.mask.sum(axis=0)
        mask = (masked_values == len(self.data_arr))

        # set up the deviation
        uncertainty = uncertainty_func(self.data_arr, axis=0)
        # Divide uncertainty by the number of pixel (#309)
        uncertainty /= np.sqrt(len(self.data_arr) - masked_values)

        # create the combined image with a dtype that matches the combiner
        combined_image = CCDData(np.asarray(data.data, dtype=self.dtype),
                                 mask=mask, unit=self.unit,
                                 uncertainty=StdDevUncertainty(uncertainty))

        # update the meta data
        combined_image.meta['NCOMBINE'] = len(self.data_arr)

        # return the combined image
        return combined_image


def combine(img_list, output_file=None, method='average', weights=None,
            scale=None, mem_limit=16e9,
            minmax_clip=False, minmax_clip_min=None, minmax_clip_max=None,
            sigma_clip=False,
            sigma_clip_low_thresh=3, sigma_clip_high_thresh=3,
            sigma_clip_func=ma.mean, sigma_clip_dev_func=ma.std, **ccdkwargs):
    """
    Convenience function for combining multiple images.

    Parameters
    -----------
    img_list : list or string
        A list of fits filenames or CCDData objects that will be combined
        together. Or a string of fits filenames seperated by comma ",".

    output_file : string or None, optional
        Optional output fits filename to which the final output can be directly
        written.
        Default is ``None``.

    method : string
        Method to combine images:

        - ``'average'`` : To combine by calculating the average.
        - ``'median'`` : To combine by calculating the median.

        Default is ``'average'``.

    weights : `numpy.ndarray` or None, optional
        Weights to be used when combining images.
        An array with the weight values. The dimensions should match the
        the dimensions of the data arrays being combined.
        Default is ``None``.

    scale : function or `numpy.ndarray`-like or None, optional
        Scaling factor to be used when combining images.
        Images are multiplied by scaling prior to combining them. Scaling
        may be either a function, which will be applied to each image
        to determine the scaling factor, or a list or array whose length
        is the number of images in the `Combiner`. Default is ``None``.

    mem_limit : float, optional (default 16e9)
        Maximum memory which should be used while combining (in bytes).

    minmax_clip : bool, optional (default False)
        Set to True if you want to mask all pixels that are below
        minmax_clip_min or above minmax_clip_max before combining.

        Parameters below are valid only when minmax_clip is set to True, see
        :meth:`Combiner.minmax_clipping` for the parameter description:

        - ``minmax_clip_min`` : None, float, optional
        - ``minmax_clip_max`` : None, float, optional

    sigma_clip : bool, optional (default False)
        Set to True if you want to reject pixels which have deviations greater
        than those
        set by the threshold values. The algorithm will first calculated
        a baseline value using the function specified in func and deviation
        based on sigma_clip_dev_func and the input data array. Any pixel with
        a deviation from the baseline value greater than that set by
        sigma_clip_high_thresh or lower than that set by sigma_clip_low_thresh
        will be rejected.

        Parameters below are valid only when sigma_clip is set to True. See
        :meth:`Combiner.sigma_clipping` for the parameter description.

        - ``sigma_clip_low_thresh`` : positive float or None, optional
        - ``sigma_clip_high_thresh`` : positive float or None, optional
        - ``sigma_clip_func`` : function, optional
        - ``sigma_clip_dev_func`` : function, optional

    ccdkwargs : Other keyword arguments for CCD Object's fits reader.

    Returns
    -------
    combined_image : `~ccdproc.CCDData`
        CCDData object based on the combined input of CCDData objects.
    """
    if not isinstance(img_list, list):
        # If not a list, check wheter it is a string of filenames seperated
        # by comma
        if isinstance(img_list, str) and (',' in img_list):
            img_list = img_list.split(',')
        else:
            raise ValueError(
                "Unrecognised input for list of images to combine.")

    # Select Combine function to call in Combiner
    if method == 'average':
        combine_function = 'average_combine'
    elif method == 'median':
        combine_function = 'median_combine'
    else:
        raise ValueError("Unrecognised combine method : {0}.".format(method))

    # First we create a CCDObject from first image for storing output
    if isinstance(img_list[0], CCDData):
        ccd = img_list[0].copy()
    else:
        # User has provided fits filenames to read from
        ccd = CCDData.read(img_list[0], **ccdkwargs)

    size_of_an_img = ccd.data.nbytes
    if ccd.uncertainty is not None:
        size_of_an_img += ccd.uncertainty.nbytes
    if ccd.mask is not None:
        size_of_an_img += ccd.mask.nbytes
    if ccd.flags is not None:
        size_of_an_img += ccd.flags.nbytes

    no_of_img = len(img_list)

    # determine the number of chunks to split the images into
    no_chunks = int((size_of_an_img*no_of_img)/mem_limit)+1
    log.info('Splitting each image into {0} chunks to limit memory usage to '
             '{1} bytes.'.format(no_chunks, mem_limit))
    xs, ys = ccd.data.shape
    # First we try to split only along fast x axis
    xstep = max(1, int(xs/no_chunks))
    # If more chunks need to be created we split in Y axis for remaining number
    # of chunks
    ystep = max(1, int(ys / (1 + no_chunks - int(xs / xstep))))

    # Dictionary of Combiner properties to set and methods to call before
    # combining
    to_set_in_combiner = {}
    to_call_in_combiner = {}

    # Define all the Combiner properties one wants to apply before combining
    # images
    if weights is not None:
        to_set_in_combiner['weights'] = weights

    if scale is not None:
        # If the scale is a function, then scaling function need to be applied
        # on full image to obtain scaling factor and create an array instead.
        if callable(scale):
            scalevalues = []
            for image in img_list:
                if isinstance(image, CCDData):
                    imgccd = image
                else:
                    imgccd = CCDData.read(image, **ccdkwargs)

                scalevalues.append(scale(imgccd.data))

            to_set_in_combiner['scaling'] = np.array(scalevalues)
        else:
            to_set_in_combiner['scaling'] = scale

    if minmax_clip:
        to_call_in_combiner['minmax_clipping'] = {'min_clip': minmax_clip_min,
                                                  'max_clip': minmax_clip_max}

    if sigma_clip:
        to_call_in_combiner['sigma_clipping'] = {
            'low_thresh': sigma_clip_low_thresh,
            'high_thresh': sigma_clip_high_thresh,
            'func': sigma_clip_func,
            'dev_func': sigma_clip_dev_func}

    # Finally Run the input method on all the subsections of the image
    # and write final stitched image to ccd
    for x in range(0, xs, xstep):
        for y in range(0, ys, ystep):
            xend, yend = min(xs, x + xstep), min(ys, y + ystep)
            ccd_list = []
            for image in img_list:
                if isinstance(image, CCDData):
                    imgccd = image
                else:
                    imgccd = CCDData.read(image, **ccdkwargs)

                # Trim image
                ccd_list.append(trim_image(imgccd[x:xend, y:yend]))

            # Create Combiner for tile
            tile_combiner = Combiner(ccd_list)
            # Set all properties and call all methods
            for to_set in to_set_in_combiner:
                setattr(tile_combiner, to_set, to_set_in_combiner[to_set])
            for to_call in to_call_in_combiner:
                getattr(tile_combiner, to_call)(**to_call_in_combiner[to_call])

            # Finally call the combine algorithm
            comb_tile = getattr(tile_combiner, combine_function)()

            # add it back into the master image
            ccd.data[x:xend, y:yend] = comb_tile.data
            if ccd.mask is not None:
                ccd.mask[x:xend, y:yend] = comb_tile.mask
            if ccd.uncertainty is not None:
                ccd.uncertainty.array[x:xend, y:yend] = comb_tile.uncertainty.array

    # Write fits file if filename was provided
    if output_file is not None:
        ccd.write(output_file)

    return ccd
