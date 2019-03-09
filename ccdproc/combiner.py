# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements the combiner class."""

from abc import ABC, abstractmethod

import numpy as np
from numpy import ma
from .core import sigma_func

from astropy.nddata import CCDData, StdDevUncertainty
from astropy import log

__all__ = ['MedianCombiner', 'SumCombiner', 'AverageCombiner', 'combine']


class CombinerBase(ABC):
    """

    A class for base class combining CCDData objects.

    The Combiner class is used to combine together `~astropy.nddata.CCDData` objects
    including the method for combining the data, rejecting outlying data,
    and weighting used for combining frames.

    Parameters
    -----------
    ccd_list : list
        A list of CCDData objects that will be combined together.

    dtype : str or `numpy.dtype` or None, optional
        Allows user to set dtype. See `numpy.array` ``dtype`` parameter
        description. If ``None`` it uses ``np.float64``.
        Default is ``None``.

    use_input_uncertainties : boolean
        default value here is False, in which case the original uncertainty
        calculations will be used. If True we can generate uncertainties using
        the input CCDData uncertainties.

    uncertainty_func : function, optional
            Function to calculate uncertainty. Defaults to `numpy.ma.std`.

    Raises
    ------
    TypeError
        If the ``ccd_list`` are not `~astropy.nddata.CCDData` objects, have different
        units, or are different shapes.

    Examples
    --------
    The following is an example of combining together different
    `~astropy.nddata.CCDData` objects::

        >>> import numpy as np
        >>> import astropy.units as u
        >>> from astropy.nddata import CCDData
        >>> from ccdproc import AverageCombiner
        >>> ccddata1 = CCDData(np.ones((4, 4)), unit=u.adu)
        >>> ccddata2 = CCDData(np.zeros((4, 4)), unit=u.adu)
        >>> ccddata3 = CCDData(np.ones((4, 4)), unit=u.adu)
        >>> c = AverageCombiner([ccddata1, ccddata2, ccddata3])
        >>> ccdall = c.combiner_method()
        >>> ccdall
        CCDData([[ 0.66666667,  0.66666667,  0.66666667,  0.66666667],
                 [ 0.66666667,  0.66666667,  0.66666667,  0.66666667],
                 [ 0.66666667,  0.66666667,  0.66666667,  0.66666667],
                 [ 0.66666667,  0.66666667,  0.66666667,  0.66666667]])
    """
    # TODO: add uncertainty_func setter?
    def __init__(self, ccd_list, dtype=None, use_input_uncertainties=False,
                 uncertainty_func=None):
        if ccd_list is None:
            raise TypeError("ccd_list should be a list of CCDData objects.")

        if dtype is None:
            dtype = np.float64

        default_shape = None
        default_unit = None
        for ccd in ccd_list:
            # raise an error if the objects aren't CCDData objects
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
                    raise TypeError("CCDData objects don't have the same unit.")

        self.ccd_list = ccd_list
        self.unit = default_unit
        self.weights = None
        self._dtype = dtype
        self.use_input_uncer = use_input_uncertainties
        self.uncertainty_func = uncertainty_func

        # set up the data array
        new_shape = (len(ccd_list),) + default_shape
        self.data_arr = ma.masked_all(new_shape, dtype=dtype)

        # populate self.data_arr
        for i, ccd in enumerate(ccd_list):
            self.data_arr[i] = ccd.data
            if ccd.mask is not None:
                self.data_arr.mask[i] = ccd.mask
            else:
                self.data_arr.mask[i] = ma.zeros(default_shape)

        # Must be after self.data_arr is defined because it checks the
        # length of the data array.
        self.scaling = None

    @property
    def dtype(self):
        return self._dtype

    @property
    def weights(self):
        """
        Weights used when combining the `~astropy.nddata.CCDData` objects.

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
                raise TypeError("weights must be a numpy.ndarray.")
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
                    raise TypeError("scaling must be a function or an array "
                                    "the same length as the number of images.")
            # reshape so that broadcasting occurs properly
            for i in range(len(self.data_arr.data.shape)-1):
                self._scaling = self.scaling[:, np.newaxis]

    # set up IRAF-like minmax clipping
    def clip_extrema(self, nlow=0, nhigh=0):
        """Mask pixels using an IRAF-like minmax clipping algorithm.  The
        algorithm will mask the lowest nlow values and the highest nhigh values
        before combining the values to make up a single pixel in the resulting
        image.  For example, the image will be a combination of
        Nimages-nlow-nhigh pixel values instead of the combination of Nimages.

        Parameters
        -----------
        nlow : int or None, optional
            If not None, the number of low values to reject from the
            combination.
            Default is 0.

        nhigh : int or None, optional
            If not None, the number of high values to reject from the
            combination.
            Default is 0.

        Notes
        -----
        Note that this differs slightly from the nominal IRAF imcombine
        behavior when other masks are in use.  For example, if ``nhigh>=1`` and
        any pixel is already masked for some other reason, then this algorithm
        will count the masking of that pixel toward the count of nhigh masked
        pixels.

        Here is a copy of the relevant IRAF help text [0]_:

        nlow = 1, nhigh = (minmax)
            The number of low and high pixels to be rejected by the "minmax"
            algorithm. These numbers are converted to fractions of the total
            number of input images so that if no rejections have taken place
            the specified number of pixels are rejected while if pixels have
            been rejected by masking, thresholding, or nonoverlap, then the
            fraction of the remaining pixels, truncated to an integer, is used.

        References
        ----------
        .. [0] image.imcombine help text.
           http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?imcombine
        """

        if nlow is None:
            nlow = 0
        if nhigh is None:
            nhigh = 0

        argsorted = np.argsort(self.data_arr.data, axis=0)
        mg = np.mgrid[[slice(ndim)
                       for i, ndim in enumerate(self.data_arr.shape) if i > 0]]
        for i in range(-1*nhigh, nlow):
            # create a tuple with the indices
            where = tuple([argsorted[i, :, :].ravel()] +
                          [i.ravel() for i in mg])
            self.data_arr.mask[where] = True

    # set up min/max clipping algorithms
    def minmax_clipping(self, min_clip=None, max_clip=None):
        """Mask all pixels that are below min_clip or above max_clip.

         Parameters
         -----------
         min_clip : float or None, optional
             If not None, all pixels with values below min_clip will be masked.
             Default is ``None``.

         max_clip : float or None, optional
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

    def _get_scaled_data(self, scale_arg):
        if scale_arg is not None:
            return self.data_arr * scale_arg
        if self.scaling is not None:
            return self.data_arr * self.scaling
        return self.data_arr

    @abstractmethod
    def combiner_method(self, combine_func=None, scale_to=None):
        """
        Abstract container method for combiner's defined in subclasses. I
        actually think it might be worthwile to re-tool this so we don't have
        repeat code, but my initial solution to that might make the implementation
        a little harder to understand for someone reading the code, as the
        top level combiner fucntion in this abstrac base class would no longer
        be the abstract method.
        """

        pass

    @abstractmethod
    def uncertainty_fresh(self, masked_values=None):
        pass

    @abstractmethod
    def uncertainty_provided(self):
        pass


class MedianCombiner(CombinerBase):

    def __init__(self, ccd_list, dtype=None, use_input_uncertainties=False,
                 uncertainty_func=sigma_func):
        super().__init__(ccd_list, dtype, use_input_uncertainties,
                         uncertainty_func)

    # set up the combining algorithms
    def combiner_method(self, combine_func=ma.median, scale_to=None):
        """
        Median combine a set of arrays.

        A `~astropy.nddata.CCDData` object is returned with the data property set to
        the median of the arrays. If the data was masked or any data have been
        rejected, those pixels will not be included in the median. A mask will
        be returned, and if a pixel has been rejected in all images, it will be
        masked. The uncertainty of the combined image is set by 1.4826 times
        the median absolute deviation of all input images.

        Parameters
        ----------
        combine_func : function, optional
            Function that calculates median of a `numpy.ma.MaskedArray`.
            Default is `numpy.ma.median`.

        scale_to : float or None, optional
            Scaling factor used in the sum combined image. If given,
            it overrides `scaling`. Defaults to ``None``.

        Returns
        -------
        combined_image: `~astropy.nddata.CCDData`
            CCDData object based on the combined input of CCDData objects.

        Warnings
        --------
        The uncertainty currently calculated using the median absolute
        deviation does not account for rejected pixels.
        """

        # set the data
        data = combine_func(self._get_scaled_data(scale_to), axis=0)

        # set the mask
        masked_values = self.data_arr.mask.sum(axis=0)
        mask = (masked_values == len(self.data_arr))

        # handle uncertainty, this may change
        if self.use_input_uncer:
            uncertainty = self.uncertainty_provided()
        else:
            uncertainty = self.uncertainty_fresh(masked_values)

        # create the combined image with a dtype matching the combiner
        combined_image = CCDData(np.asarray(data.data, dtype=self.dtype),
                                 mask=mask, unit=self.unit,
                                 uncertainty=StdDevUncertainty(uncertainty))

        # update the meta data
        combined_image.meta['NCOMBINE'] = len(self.data_arr)

        # return the combined image
        return combined_image

    def uncertainty_fresh(self, masked_values=None):
        """

        :return:
        """
        # set the uncertainty
        uncertainty = self.uncertainty_func(self.data_arr, axis=0)
        # Divide uncertainty by the number of pixel (#309)
        uncertainty /= np.sqrt(len(self.data_arr) - masked_values)
        # Convert uncertainty to plain numpy array (#351)
        # There is no need to care about potential masks because the
        # uncertainty was calculated based on the data so potential masked
        # elements are also masked in the data. No need to keep two identical
        # masks.
        uncertainty = np.asarray(uncertainty)

        return uncertainty

    def uncertainty_provided(self):
        """



        """
        # TODO: this function
        return None


class AverageCombiner(CombinerBase):
    def __init__(self, ccd_list, dtype=None, use_input_uncertainties=False,
                 uncertainty_func=ma.std):
        super().__init__(ccd_list, dtype, use_input_uncertainties,
                         uncertainty_func)

    # set up the combining algorithms
    def combiner_method(self, combine_func=ma.average, scale_to=None):

        """
        Average combine together a set of arrays.

        A `~astropy.nddata.CCDData` object is returned with the data property
        set to the average of the arrays. If the data was masked or any
        data have been rejected, those pixels will not be included in the
        average. A mask will be returned, and if a pixel has been
        rejected in all images, it will be masked. The uncertainty of
        the combined image is set by the standard deviation of the input
        images.

        Parameters
        ----------
        combine_func : function, optional
            Function to calculate the average. Defaults to
            `numpy.ma.average`.

         scale_to : float or None, optional
            Scaling factor used in the sum combined image. If given,
            it overrides `scaling`. Defaults to ``None``.

        Returns
        -------
        combined_image: `~astropy.nddata.CCDData`
            CCDData object based on the combined input of CCDData objects.
        """

        # set up the data
        data, wei = combine_func(self._get_scaled_data(scale_to),
                                 axis=0, weights=self.weights,
                                 returned=True)

        # set up the mask
        masked_values = self.data_arr.mask.sum(axis=0)
        mask = (masked_values == len(self.data_arr))

        # handle uncertainty, this may change
        if self.use_input_uncer:
            uncertainty = self.uncertainty_provided()
        else:
            uncertainty = self.uncertainty_fresh(masked_values)

        # create the combined image with a dtype that matches the combiner
        combined_image = CCDData(np.asarray(data.data, dtype=self.dtype),
                                 mask=mask, unit=self.unit,
                                 uncertainty=StdDevUncertainty(uncertainty))

        # update the meta data
        combined_image.meta['NCOMBINE'] = len(self.data_arr)

        # return the combined image
        return combined_image

    def uncertainty_fresh(self, masked_values=None):
        """

        :return:
        """
        # set up the deviation
        uncertainty = self.uncertainty_func(self.data_arr, axis=0)
        # Divide uncertainty by the number of pixel (#309)
        uncertainty /= np.sqrt(len(self.data_arr) - masked_values)
        # Convert uncertainty to plain numpy array (#351)
        uncertainty = np.asarray(uncertainty)

        return uncertainty

    def uncertainty_provided(self):
        """

        """
        # TODO: this function
        return None


class SumCombiner(CombinerBase):
    def __init__(self, ccd_list, dtype=None, use_input_uncertainties=False,
                 uncertainty_func=ma.std):
        super().__init__(ccd_list, dtype, use_input_uncertainties,
                         uncertainty_func)

    # set up the combining algorithms
    def combiner_method(self, combine_func=ma.sum, scale_to=None):
        """
        Sum combine together a set of arrays.

        A `~astropy.nddata.CCDData` object is returned with the data property
        set to the sum of the arrays. If the data was masked or any
        data have been rejected, those pixels will not be included in the
        sum. A mask will be returned, and if a pixel has been
        rejected in all images, it will be masked. The uncertainty of
        the combined image is set by the multiplication of summation of
        standard deviation of the input by square root of number of images.
        Because sum_combine returns 'pure sum' with masked pixels ignored, if
        re-scaled sum is needed, average_combine have to be used with
        multiplication by number of images combined.

        Parameters
        ----------
        combine_func : function, optional
            Function to calculate the sum. Defaults to
            `numpy.ma.sum`.

        scale_to : float or None, optional
            Scaling factor used in the sum combined image. If given,
            it overrides `scaling`. Defaults to ``None``.

        Returns
        -------
        combined_image: `~astropy.nddata.CCDData`
            CCDData object based on the combined input of CCDData objects.
        """

        # set up the data
        data = combine_func(self._get_scaled_data(scale_to), axis=0)

        # set up the mask
        masked_values = self.data_arr.mask.sum(axis=0)
        mask = (masked_values == len(self.data_arr))

        # handle uncertainty, this may change
        if self.use_input_uncer:
            uncertainty = self.uncertainty_provided()
        else:
            uncertainty = self.uncertainty_fresh(masked_values)

        # create the combined image with a dtype that matches the combiner
        combined_image = CCDData(np.asarray(data.data, dtype=self.dtype),
                                 mask=mask, unit=self.unit,
                                 uncertainty=StdDevUncertainty(uncertainty))

        # update the meta data
        combined_image.meta['NCOMBINE'] = len(self.data_arr)

        # return the combined image
        return combined_image

    def uncertainty_fresh(self, masked_values=None):
        """

        :return:
        """
        # set up the deviation
        uncertainty = self.uncertainty_func(self.data_arr, axis=0)
        # Divide uncertainty by the number of pixel (#309)
        uncertainty /= np.sqrt(len(self.data_arr) - masked_values)
        # Convert uncertainty to plain numpy array (#351)
        uncertainty = np.asarray(uncertainty)
        # Multiply uncertainty by square root of the number of images
        uncertainty *= len(self.data_arr) - masked_values

        return uncertainty

    def uncertainty_provided(self):
        """


        """
        # TODO: this function
        return None


def _calculate_step_sizes(x_size, y_size, num_chunks):
    """
    Calculate the strides in x and y to achieve at least
    the ``num_chunks`` pieces.

    Parameters
    ----------
    """
    # First we try to split only along fast x axis
    xstep = max(1, int(x_size / num_chunks))

    # More chunks are needed only if xstep gives us fewer chunks than
    # requested.
    x_chunks = int(x_size / xstep)

    if x_chunks >= num_chunks:
        ystep = y_size
    else:
        # The x and y loops are nested, so the number of chunks
        # is multiplicative, not additive. Calculate the number
        # of y chunks we need to get at num_chunks.
        y_chunks = int(num_chunks / x_chunks) + 1
        ystep = max(1, int(y_size / y_chunks))

    return xstep, ystep


def combine(img_list, output_file=None,
            method='average', weights=None, scale=None, mem_limit=16e9,
            clip_extrema=False, nlow=1, nhigh=1,
            minmax_clip=False, minmax_clip_min=None, minmax_clip_max=None,
            sigma_clip=False,
            sigma_clip_low_thresh=3, sigma_clip_high_thresh=3,
            sigma_clip_func=ma.mean, sigma_clip_dev_func=ma.std,
            dtype=None, use_input_uncertainties=False, combine_uncertainty_function=None,
            **ccdkwargs):
    """
    Convenience function for combining multiple images.

    Parameters
    -----------
    img_list : `numpy.ndarray`, list or str
        A list of fits filenames or `~astropy.nddata.CCDData` objects that will be
        combined together. Or a string of fits filenames separated by comma
        ",".

    output_file : str or None, optional
        Optional output fits file-name to which the final output can be
        directly written.
        Default is ``None``.

    method : str, optional
        Method to combine images:

        - ``'average'`` : To combine by calculating the average.
        - ``'median'`` : To combine by calculating the median.
        - ``'sum'`` : To combine by calculating the sum.

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

    mem_limit : float, optional
        Maximum memory which should be used while combining (in bytes).
        Default is ``16e9``.

    clip_extrema : bool, optional
        Set to True if you want to mask pixels using an IRAF-like minmax
        clipping algorithm.  The algorithm will mask the lowest nlow values and
        the highest nhigh values before combining the values to make up a
        single pixel in the resulting image.  For example, the image will be a
        combination of Nimages-low-nhigh pixel values instead of the
        combination of Nimages.

        Parameters below are valid only when clip_extrema is set to True,
        see :meth:`Combiner.clip_extrema` for the parameter description:

        - ``nlow`` : int or None, optional
        - ``nhigh`` : int or None, optional


    minmax_clip : bool, optional
        Set to True if you want to mask all pixels that are below
        minmax_clip_min or above minmax_clip_max before combining.
        Default is ``False``.

        Parameters below are valid only when minmax_clip is set to True, see
        :meth:`Combiner.minmax_clipping` for the parameter description:

        - ``minmax_clip_min`` : float or None, optional
        - ``minmax_clip_max`` : float or None, optional

    sigma_clip : bool, optional
        Set to True if you want to reject pixels which have deviations greater
        than those
        set by the threshold values. The algorithm will first calculated
        a baseline value using the function specified in func and deviation
        based on sigma_clip_dev_func and the input data array. Any pixel with
        a deviation from the baseline value greater than that set by
        sigma_clip_high_thresh or lower than that set by sigma_clip_low_thresh
        will be rejected.
        Default is ``False``.

        Parameters below are valid only when sigma_clip is set to True. See
        :meth:`Combiner.sigma_clipping` for the parameter description.

        - ``sigma_clip_low_thresh`` : positive float or None, optional
        - ``sigma_clip_high_thresh`` : positive float or None, optional
        - ``sigma_clip_func`` : function, optional
        - ``sigma_clip_dev_func`` : function, optional

    dtype : str or `numpy.dtype` or None, optional
        The intermediate and resulting ``dtype`` for the combined CCDs. See
        `ccdproc.Combiner`. If ``None`` this is set to ``float64``.
        Default is ``None``.

    combine_uncertainty_function : callable, None, optional
        If ``None`` use the default uncertainty func when using average, median or
        sum combine, otherwise use the function provided.
        Default is ``None``.

    ccdkwargs : Other keyword arguments for `astropy.nddata.fits_ccddata_reader`.

    Returns
    -------
    combined_image : `~astropy.nddata.CCDData`
        CCDData object based on the combined input of CCDData objects.
    """
    if not isinstance(img_list, list):
        # If not a list, check whether it is a numpy ndarray or string of
        # filenames separated by comma
        if isinstance(img_list, np.ndarray):
            img_list = img_list.tolist()
        elif isinstance(img_list, str) and (',' in img_list):
            img_list = img_list.split(',')
        else:
            raise ValueError(
                "unrecognised input for list of images to combine.")

    # Select Combine function to call in Combiner
    if method == 'average':
        Combine_class = AverageCombiner
    elif method == 'median':
        Combine_class = MedianCombiner
    elif method == 'sum':
        Combine_class = SumCombiner
    else:
        raise ValueError("unrecognised combine method : {0}.".format(method))

    # First we create a CCDObject from first image for storing output
    if isinstance(img_list[0], CCDData):
        ccd = img_list[0].copy()
    else:
        # User has provided fits filenames to read from
        ccd = CCDData.read(img_list[0], **ccdkwargs)

    # If uncertainty_func is given for combine this will create an uncertainty
    # even if the originals did not have one. In that case we need to create
    # an empty placeholder.
    if ccd.uncertainty is None and combine_uncertainty_function is not None:
        ccd.uncertainty = StdDevUncertainty(np.zeros(ccd.data.shape))

    if dtype is None:
        dtype = np.float64

    # Convert the master image to the appropriate dtype so when overwriting it
    # later the data is not downcast and the memory consumption calculation
    # uses the internally used dtype instead of the original dtype. #391
    if ccd.data.dtype != dtype:
        ccd.data = ccd.data.astype(dtype)

    size_of_an_img = ccd.data.nbytes
    try:
        size_of_an_img += ccd.uncertainty.array.nbytes
    # In case uncertainty is None it has no "array" and in case the "array" is
    # not a numpy array:
    except AttributeError:
        pass
    # Mask is enforced to be a numpy.array across astropy versions
    if ccd.mask is not None:
        size_of_an_img += ccd.mask.nbytes
    # flags is not necessarily a numpy array so do not fail with an
    # AttributeError in case something was set!
    # TODO: Flags are not taken into account in Combiner. This number is added
    #       nevertheless for future compatibility.
    try:
        size_of_an_img += ccd.flags.nbytes
    except AttributeError:
        pass

    no_of_img = len(img_list)

    # determine the number of chunks to split the images into
    no_chunks = int((size_of_an_img * no_of_img) / mem_limit) + 1
    if no_chunks > 1:
        log.info('splitting each image into {0} chunks to limit memory usage '
                 'to {1} bytes.'.format(no_chunks, mem_limit))
    xs, ys = ccd.data.shape

    # Calculate strides for loop
    xstep, ystep = _calculate_step_sizes(xs, ys, no_chunks)

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

    if clip_extrema:
        to_call_in_combiner['clip_extrema'] = {'nlow': nlow,
                                               'nhigh': nhigh}

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

                # Trim image and copy
                # The copy is *essential* to avoid having a bunch
                # of unused file references around if the files
                # are memory-mapped. See this PR for details
                # https://github.com/astropy/ccdproc/pull/630
                ccd_list.append(imgccd[x:xend, y:yend].copy())

            # if uncertainty function is provided, use that function
            combine_kwds_class = {}
            if combine_uncertainty_function is not None:
                combine_kwds_class['uncertainty_func'] = combine_uncertainty_function

            # Create Combiner for tile
            tile_combiner = Combine_class(ccd_list, dtype=dtype,
                                          use_input_uncertainties=use_input_uncertainties,
                                          **combine_kwds_class)

            # Set all properties and call all methods
            for to_set in to_set_in_combiner:
                setattr(tile_combiner, to_set, to_set_in_combiner[to_set])
            for to_call in to_call_in_combiner:
                getattr(tile_combiner, to_call)(**to_call_in_combiner[to_call])

            # Finally call the combine method
            #comb_tile = getattr(tile_combiner, combine_function)(**combine_kwds)
            comb_tile = tile_combiner.combiner_method()

            # add it back into the master image
            ccd.data[x:xend, y:yend] = comb_tile.data
            if ccd.mask is not None:
                ccd.mask[x:xend, y:yend] = comb_tile.mask
            if ccd.uncertainty is not None:
                ccd.uncertainty.array[x:xend, y:yend] = comb_tile.uncertainty.array
            # Free up memory to try to stay under user's limit
            del comb_tile
            del tile_combiner
            del ccd_list

    # Write fits file if filename was provided
    if output_file is not None:
        ccd.write(output_file)

    return ccd
