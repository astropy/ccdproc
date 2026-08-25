# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module implements the combiner class."""

from copy import deepcopy
from functools import partial

try:  # pragma: no cover
    import bottleneck as bn
except ImportError:
    HAS_BOTTLENECK = False
else:  # pragma: no cover
    HAS_BOTTLENECK = True

import array_api_compat
import array_api_extra as xpx
from astropy import log
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.stats import sigma_clip
from astropy.utils import deprecated_renamed_argument

from ._nanfuncs import nanmean, nanmedian, nanstd, nansum
from .core import _namespace_dtype, _native_numpy, _to_numpy, sigma_func

__all__ = ["Combiner", "combine"]


def _default_median(xp=None):
    if HAS_BOTTLENECK and (xp is None or array_api_compat.is_numpy_namespace(xp)):
        return bn.nanmedian
    if xp is None:
        return None

    # No bottleneck, but we have a namespace.
    try:
        return xp.nanmedian
    except AttributeError:
        # nanmedian is not part of the array API standard; fall back to a
        # (slower, sort-based) implementation written purely in terms of it.
        return partial(nanmedian, xp=xp)


def _default_average(xp=None):
    if HAS_BOTTLENECK and (xp is None or array_api_compat.is_numpy_namespace(xp)):
        return bn.nanmean
    if xp is None:
        return None

    # No bottleneck, but we have a namespace.
    try:
        return xp.nanmean
    except AttributeError:
        # nanmean is not part of the array API standard; fall back to an
        # implementation written purely in terms of it.
        return partial(nanmean, xp=xp)


def _default_sum(xp=None):
    if HAS_BOTTLENECK and (xp is None or array_api_compat.is_numpy_namespace(xp)):
        return bn.nansum
    if xp is None:
        return None

    # No bottleneck, but we have a namespace.
    try:
        return xp.nansum
    except AttributeError:
        # nansum is not part of the array API standard; fall back to an
        # implementation written purely in terms of it.
        return partial(nansum, xp=xp)


def _default_std(xp=None):
    if HAS_BOTTLENECK and (xp is None or array_api_compat.is_numpy_namespace(xp)):
        return bn.nanstd
    if xp is None:
        return None

    # No bottleneck, but we have a namespace.
    try:
        return xp.nanstd
    except AttributeError:
        # nanstd is not part of the array API standard; fall back to an
        # implementation written purely in terms of it.
        return partial(nanstd, xp=xp)


class Combiner:
    """
    A class for combining CCDData objects.

    The Combiner class is used to combine together `~astropy.nddata.CCDData` objects
    including the method for combining the data, rejecting outlying data,
    and weighting used for combining frames.

    Parameters
    ----------
    ccd_iter : list or generator
        A list or generator of CCDData objects that will be combined together.

    dtype : dtype-like or None, optional
        The dtype for the stacked data and the results: a dtype object of
        the array namespace, or anything NumPy accepts as a dtype (e.g.
        ``int``, ``"float32"``, `numpy.float32`), which is mapped to the
        namespace's dtype of the same name. If ``None`` the namespace's
        ``float64`` is used.
        Default is ``None``.

    xp : array namespace, optional
        The array namespace to use for the data. If `None` or not provided, it will
        be inferred from the first `~astropy.nddata.CCDData` object in
        ``ccd_iter``. A plain module (e.g. ``numpy``) is accepted and is
        converted to its array-API-compatible namespace.
        Default is `None`.

    Raises
    ------
    TypeError
        If the ``ccd_iter`` are not `~astropy.nddata.CCDData` objects, have different
        units, or are different shapes.

    Examples
    --------
    The following is an example of combining together different
    `~astropy.nddata.CCDData` objects::

        >>> import numpy as np
        >>> import astropy.units as u
        >>> from astropy.nddata import CCDData
        >>> from ccdproc import Combiner
        >>> ccddata1 = CCDData(np.ones((4, 4)), unit=u.adu)
        >>> ccddata2 = CCDData(np.zeros((4, 4)), unit=u.adu)
        >>> ccddata3 = CCDData(np.ones((4, 4)), unit=u.adu)
        >>> c = Combiner([ccddata1, ccddata2, ccddata3])
        >>> ccdall = c.average_combine()
        >>> ccdall  # doctest: +FLOAT_CMP
        CCDData([[ 0.66666667,  0.66666667,  0.66666667,  0.66666667],
                 [ 0.66666667,  0.66666667,  0.66666667,  0.66666667],
                 [ 0.66666667,  0.66666667,  0.66666667,  0.66666667],
                 [ 0.66666667,  0.66666667,  0.66666667,  0.66666667]]...)
    """

    def __init__(self, ccd_iter, dtype=None, xp=None):
        if ccd_iter is None:
            raise TypeError(
                "ccd_iter should be a list or a generator of CCDData objects."
            )

        default_shape = None
        default_unit = None

        ccd_list = list(ccd_iter)

        for ccd in ccd_list:
            # raise an error if the objects aren't CCDData objects
            if not isinstance(ccd, CCDData):
                raise TypeError("ccd_list should only contain CCDData objects.")

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

        # Set array namespace. A raw module such as ``numpy`` or ``dask.array``
        # may lack array-API features that are used below (``xp.bool``, the
        # ``device`` keyword), so normalise whatever the caller passed to the
        # array-api-compat namespace of one of its arrays.
        if xp is None:
            xp = array_api_compat.array_namespace(ccd_list[0].data)
        else:
            xp = array_api_compat.array_namespace(xp.asarray(0))
        self._xp = xp
        if dtype is None:
            dtype = xp.float64
        else:
            dtype = _namespace_dtype(dtype, xp)

        self.unit = default_unit
        self.weights = None
        self._dtype = dtype

        # set up the data array
        # new_shape = (len(ccd_list),) + default_shape
        # Stack the individual images rather than passing a nested list to
        # xp.asarray: the array API does not allow nested sequences of arrays.
        # Keep the stack on the device of the input data, but only when the
        # data already belong to ``xp``; a device object from a different
        # namespace (e.g. numpy's 'cpu' for a jax namespace) is meaningless
        # to ``xp``, so let ``xp`` use its default device instead.
        data_xp = array_api_compat.array_namespace(ccd_list[0].data)
        device = array_api_compat.device(ccd_list[0].data) if data_xp is xp else None
        self._data_arr = xp.stack(
            [xp.asarray(ccd.data, dtype=dtype, device=device) for ccd in ccd_list]
        )

        # populate self._data_arr_mask. The mask of a CCDData may be a numpy
        # array even when its data is not, so coerce each mask into the data
        # namespace and onto the data device before stacking.
        mask_list = [
            (
                xp.asarray(ccd.mask, dtype=xp.bool, device=device)
                if ccd.mask is not None
                else xp.zeros(default_shape, dtype=xp.bool, device=device)
            )
            for ccd in ccd_list
        ]
        self._data_arr_mask = xp.stack(mask_list)

        # Must be after self.data_arr is defined because it checks the
        # length of the data array.
        self.scaling = None

    @property
    def dtype(self):
        """The dtype of the data array to be combined."""
        return self._dtype

    @property
    def data(self):
        """The data array to be combined."""
        return self._data_arr

    @property
    def mask(self):
        """The mask array to be used in image combination. This is *not* the mask
        of the combined image, but the mask of the data array to be combined."""
        return self._data_arr_mask

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
            try:
                _ = array_api_compat.array_namespace(value)
            except TypeError as err:
                raise TypeError("weights must be an array.") from err

            if value.shape != self._data_arr.shape:
                if value.ndim != 1:
                    raise ValueError(
                        "1D weights expected when shapes of the "
                        "data and weights differ."
                    )
                if value.shape[0] != self._data_arr.shape[0]:
                    raise ValueError(
                        "Length of weights not compatible with specified axis."
                    )
            self._weights = value

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
        xp = self._xp
        if value is None:
            self._scaling = value
        else:
            n_images = self._data_arr.shape[0]
            device = array_api_compat.device(self._data_arr)
            dtype = self._data_arr.dtype
            if callable(value):
                # The callable may return a Python float or a 0-d array of
                # the backend; stack per-element conversions rather than
                # passing a list of arrays to asarray, which array-api-strict
                # rejects as a nested sequence of arrays.
                # Cast to the data dtype so that scaling by, e.g., an integer
                # does not require type promotion, which the array API does
                # not guarantee between integer and floating dtypes.
                self._scaling = xp.stack(
                    [
                        xp.asarray(
                            value(self._data_arr[i, ...]), dtype=dtype, device=device
                        )
                        for i in range(n_images)
                    ]
                )
            else:
                # Array API arrays need not implement __len__, so use the
                # shape where there is one and fall back to len() for lists
                # and tuples.
                try:
                    n_values = getattr(value, "shape", None)
                    n_values = n_values[0] if n_values else len(value)
                except (TypeError, IndexError) as err:
                    raise TypeError(
                        "scaling must be a function or an array "
                        "the same length as the number of images.",
                    ) from err
                if n_values != n_images:
                    raise ValueError(
                        "scaling must be a function or an array "
                        "the same length as the number of images."
                    )
                self._scaling = xp.asarray(value, dtype=dtype, device=device)
            # reshape so that broadcasting occurs properly
            self._scaling = xp.reshape(
                self._scaling, (n_images,) + (1,) * (self._data_arr.ndim - 1)
            )

    # set up IRAF-like minmax clipping
    def clip_extrema(self, nlow=0, nhigh=0):
        """Mask pixels using an IRAF-like minmax clipping algorithm.  The
        algorithm will mask the lowest nlow values and the highest nhigh values
        before combining the values to make up a single pixel in the resulting
        image.  For example, the image will be a combination of
        Nimages-nlow-nhigh pixel values instead of the combination of Nimages.

        Parameters
        ----------
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

        If ``nlow`` or ``nhigh`` is at least the number of images, every pixel
        is masked.

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
        xp = self._xp
        if nlow is None:
            nlow = 0
        if nhigh is None:
            nhigh = 0

        n_images = self._data_arr.shape[0]
        # argsorted[i, ...] is, per pixel, the index of the image whose value
        # has rank i (0 = lowest, n_images - 1 = highest). Sorting those
        # indices again inverts the permutation: ranks[k, ...] is the rank of
        # image k's value at each pixel. Comparing ranks avoids scattering
        # into the mask with per-pixel indices, which the array API standard
        # does not support.
        argsorted = xp.argsort(self._data_arr, axis=0)
        ranks = xp.argsort(argsorted, axis=0)
        clip = (ranks < nlow) | (ranks >= n_images - nhigh)
        self._data_arr_mask = self._data_arr_mask | clip

    # set up min/max clipping algorithms
    def minmax_clipping(self, min_clip=None, max_clip=None):
        """Mask all pixels that are below min_clip or above max_clip.

        Parameters
        ----------
        min_clip : float or None, optional
            If not None, all pixels with values below min_clip will be masked.
            Default is ``None``.

        max_clip : float or None, optional
            If not None, all pixels with values above min_clip will be masked.
            Default is ``None``.
        """
        if min_clip is not None:
            mask = self._data_arr < min_clip
            # Do "or" in-place if possible...
            self._data_arr_mask |= mask
        if max_clip is not None:
            mask = self._data_arr > max_clip
            # Do "or" in-place if possible...
            self._data_arr_mask |= mask

    # set up sigma  clipping algorithms
    @deprecated_renamed_argument(
        "use_astropy",
        None,
        arg_in_kwargs=True,
        since="2.4.0",
        message="The use_astropy argument has been removed because "
        "astropy sigma clipping is now always used.",
    )
    def sigma_clipping(
        self, low_thresh=3, high_thresh=3, func="mean", dev_func="std", **kwd
    ):
        """
        Pixels will be rejected if they have deviations greater than those
        set by the threshold values. The algorithm will first calculated
        a baseline value using the function specified in func and deviation
        based on dev_func and the input data array. Any pixel with a
        deviation from the baseline value greater than that set by
        high_thresh or lower than that set by low_thresh will be rejected.

        Parameters
        ----------
        low_thresh : positive float or None, optional
            Threshold for rejecting pixels that deviate below the baseline
            value. If negative value, then will be convert to a positive
            value. If None, no rejection will be done based on low_thresh.
            Default is 3.

        high_thresh : positive float or None, optional
            Threshold for rejecting pixels that deviate above the baseline
            value. If None, no rejection will be done based on high_thresh.
            Default is 3.

        func : {'median', 'mean'} or callable, optional
            The statistic or callable function/object used to compute
            the center value for the clipping. If using a callable
            function/object and the ``axis`` keyword is used, then it must
            be able to ignore NaNs (e.g., `numpy.nanmean`) and it must have
            an ``axis`` keyword to return an array with axis dimension(s)
            removed. The default is ``'median'``.

        dev_func : {'std', 'mad_std'} or callable, optional
            The statistic or callable function/object used to compute the
            standard deviation about the center value. If using a callable
            function/object and the ``axis`` keyword is used, then it must
            be able to ignore NaNs (e.g., `numpy.nanstd`) and it must have
            an ``axis`` keyword to return an array with axis dimension(s)
            removed. The default is ``'std'``.

        kwd
            Any remaining keyword arguments are passed to astropy's
            :func:`~astropy.stats.sigma_clip` function.
        """

        # Remove in 3.0
        _ = kwd.pop("use_astropy", True)

        self._data_arr_mask = (
            self._data_arr_mask
            | sigma_clip(
                self._data_arr,
                sigma_lower=low_thresh,
                sigma_upper=high_thresh,
                axis=kwd.get("axis", 0),
                copy=kwd.get("copy", False),
                maxiters=kwd.get("maxiters", 1),
                cenfunc=func,
                stdfunc=dev_func,
                masked=True,
                **kwd,
            ).mask
        )

    def _get_scaled_data(self, scale_arg):
        if scale_arg is not None:
            return self._data_arr * scale_arg
        if self.scaling is not None:
            return self._data_arr * self.scaling
        return self._data_arr

    def _get_nan_substituted_data(self, data):
        xp = self._xp

        # Get the data as an unmasked array with masked values filled as NaN
        if xp.any(self._data_arr_mask):
            # Use array_api_extra so that we can use at with all array libraries
            data = xpx.at(data)[self._data_arr_mask].set(xp.nan)
        else:
            data = data
        return data

    def _combination_setup(self, user_func, default_func, scale_to):
        """
        Handle the common pieces of image combination data/mask setup.
        """
        data = self._get_scaled_data(scale_to)
        xp = self._xp
        # Play it safe for now and only do the nan thing if the user is using
        # the default combination function.
        if user_func is None:
            combo_func = default_func
            # Subtitute NaN for masked entries
            data = self._get_nan_substituted_data(data)
            masked_values = xp.count_nonzero(xp.isnan(data), axis=0)
        else:
            masked_values = xp.count_nonzero(self._data_arr_mask, axis=0)
            combo_func = user_func

        return data, masked_values, combo_func

    # set up the combining algorithms
    def median_combine(
        self, median_func=None, scale_to=None, uncertainty_func=sigma_func
    ):
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
        median_func : function, optional
            Function that calculates median of an array.

        scale_to : float or None, optional
            Scaling factor used in the average combined image. If given,
            it overrides `scaling`.
            Defaults to None.

        uncertainty_func : function, optional
            Function to calculate uncertainty.
            Defaults is `~ccdproc.sigma_func`.

        Returns
        -------
        combined_image: `~astropy.nddata.CCDData`
            CCDData object based on the combined input of CCDData objects.

        Warnings
        --------
        The uncertainty currently calculated using the median absolute
        deviation does not account for rejected pixels.
        """
        xp = self._xp

        _default_median_func = _default_median(xp=xp)

        data, masked_values, median_func = self._combination_setup(
            median_func, _default_median_func, scale_to
        )

        medianed = median_func(data, axis=0)

        # set the mask
        mask = masked_values == self._data_arr.shape[0]

        # set the uncertainty

        # The default uncertainty function (sigma_func) takes ignore_nan,
        # which makes it handle both NaNs and masked values (which were
        # converted to NaN in _combination_setup); other callables are
        # called with only the data and axis.
        if uncertainty_func is sigma_func:
            uncertainty = uncertainty_func(data, axis=0, ignore_nan=True)
        else:
            uncertainty = uncertainty_func(data, axis=0)
        # Depending on how the uncertainty was calculated it may or may not
        # be an array of the same class as the data, so make sure it is.
        # There is no need to carry a mask on the uncertainty: it was
        # calculated from the data, so masked elements are already masked
        # in the data.
        uncertainty = xp.asarray(uncertainty)
        # Divide uncertainty by the number of pixel (#309)
        uncertainty = uncertainty / xp.sqrt(
            xp.astype(self._data_arr.shape[0] - masked_values, xp.float64)
        )

        # create the combined image with a dtype matching the combiner
        combined_image = CCDData(
            xp.asarray(medianed, dtype=self.dtype),
            unit=self.unit,
            uncertainty=StdDevUncertainty(uncertainty),
        )
        # TODO: the private _mask attribute is set here to avoid the
        # CCDData.mask setter, which converts the mask to a numpy array.
        # This can be removed when CCDData supports array namespaces.
        combined_image._mask = mask

        # update the meta data
        combined_image.meta["NCOMBINE"] = self._data_arr.shape[0]

        # return the combined image
        return combined_image

    def _weighted_sum(self, data, sum_func, xp=None):
        """
        Perform weighted sum, used by both ``sum_combine`` and in some cases
        by ``average_combine``.
        """
        xp = xp or array_api_compat.array_namespace(data)
        if self.weights.shape != data.shape:
            # Add extra axes to the weights for broadcasting
            weights = xp.reshape(self.weights, (self.weights.shape[0], 1, 1))
        else:
            weights = self.weights

        # Turns out bn.nansum has an implementation that is not
        # precise enough for float32 sums. Doing this should
        # ensure the sums are carried out as float64
        weights = xp.astype(weights, xp.float64)
        weighted_sum = sum_func(data * weights, axis=0)
        return weighted_sum, weights

    def average_combine(
        self,
        scale_func=None,
        scale_to=None,
        uncertainty_func=None,
        sum_func=None,
    ):
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
        scale_func : function, optional
            Function to calculate the average.

        scale_to : float or None, optional
            Scaling factor used in the average combined image. If given,
            it overrides `scaling`. Defaults to ``None``.

        uncertainty_func : function, optional
            Function to calculate uncertainty.

        sum_func : function, optional
            Function used to calculate sums, including the one done to
            find the weighted average.

        Returns
        -------
        combined_image: `~astropy.nddata.CCDData`
            CCDData object based on the combined input of CCDData objects.
        """
        xp = self._xp

        _default_average_func = _default_average(xp=xp)

        if sum_func is None:
            sum_func = _default_sum(xp=xp)

        if uncertainty_func is None:
            uncertainty_func = _default_std(xp=xp)

        use_default_scale_func = scale_func is None
        data, masked_values, scale_func = self._combination_setup(
            scale_func, _default_average_func, scale_to
        )

        # Do NOT modify data after this -- we need it to be intact when we
        # we get to the uncertainty calculation.
        if self.weights is not None:
            weighted_sum, weights = self._weighted_sum(data, sum_func, xp=xp)
            if use_default_scale_func and xp.any(masked_values):
                weights = xp.where(xp.isnan(data), xp.zeros_like(weights), weights)
            mean = weighted_sum / sum_func(weights, axis=0)
        else:
            mean = scale_func(data, axis=0)

        # calculate the mask

        mask = masked_values == self._data_arr.shape[0]

        # set up the deviation
        uncertainty = uncertainty_func(data, axis=0)
        # Divide uncertainty by the number of pixel (#309)
        uncertainty = uncertainty / xp.sqrt(
            xp.astype(data.shape[0] - masked_values, xp.float64)
        )
        # Make sure the uncertainty is an array in the combiner's namespace
        uncertainty = xp.asarray(uncertainty)

        # create the combined image with a dtype that matches the combiner
        combined_image = CCDData(
            xp.asarray(mean, dtype=self.dtype),
            unit=self.unit,
            uncertainty=StdDevUncertainty(uncertainty),
        )
        # TODO: the private _mask attribute is set here to avoid the
        # CCDData.mask setter, which converts the mask to a numpy array.
        # This can be removed when CCDData supports array namespaces.
        combined_image._mask = mask

        # update the meta data
        combined_image.meta["NCOMBINE"] = data.shape[0]

        # return the combined image
        return combined_image

    def sum_combine(self, sum_func=None, scale_to=None, uncertainty_func=None):
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
        sum_func : function, optional
            Function to calculate the sum. Defaults to
            `numpy.nansum` or ``bottleneck.nansum``.

        scale_to : float or None, optional
            Scaling factor used in the sum combined image. If given,
            it overrides `scaling`. Defaults to ``None``.

        uncertainty_func : function, optional
            Function to calculate uncertainty.

        Returns
        -------
        combined_image: `~astropy.nddata.CCDData`
            CCDData object based on the combined input of CCDData objects.
        """

        xp = self._xp

        _default_sum_func = _default_sum(xp=xp)

        if uncertainty_func is None:
            uncertainty_func = _default_std(xp=xp)

        data, masked_values, sum_func = self._combination_setup(
            sum_func, _default_sum_func, scale_to
        )

        if self.weights is not None:
            summed, weights = self._weighted_sum(data, sum_func, xp=xp)
        else:
            summed = sum_func(data, axis=0)

        # set up the mask
        mask = masked_values == self._data_arr.shape[0]

        # set up the deviation
        uncertainty = uncertainty_func(data, axis=0)
        # Divide uncertainty by the number of pixel (#309)
        uncertainty = uncertainty / xp.sqrt(
            xp.astype(data.shape[0] - masked_values, xp.float64)
        )
        # Make sure the uncertainty is an array in the combiner's namespace
        uncertainty = xp.asarray(uncertainty)
        # Multiply uncertainty by square root of the number of images
        uncertainty = uncertainty * xp.astype(data.shape[0] - masked_values, xp.float64)

        # create the combined image with a dtype that matches the combiner
        combined_image = CCDData(
            xp.asarray(summed, dtype=self.dtype),
            unit=self.unit,
            uncertainty=StdDevUncertainty(uncertainty),
        )
        # TODO: the private _mask attribute is set here to avoid the
        # CCDData.mask setter, which converts the mask to a numpy array.
        # This can be removed when CCDData supports array namespaces.
        combined_image._mask = mask

        # update the meta data
        combined_image.meta["NCOMBINE"] = self._data_arr.shape[0]

        # return the combined image
        return combined_image


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


def _array_size_in_bytes(arr):
    # ``nbytes`` is not part of the array API standard, so get the size from
    # the element count and the dtype's bit width instead. ``finfo``/``iinfo``
    # report the bit width of a single component, so complex dtypes need a
    # factor of two.
    xp = array_api_compat.array_namespace(arr)
    dtype = arr.dtype
    if xp.isdtype(dtype, "bool"):
        bits = 8
    elif xp.isdtype(dtype, "integral"):
        bits = xp.iinfo(dtype).bits
    elif xp.isdtype(dtype, "complex floating"):
        bits = 2 * xp.finfo(dtype).bits
    else:
        bits = xp.finfo(dtype).bits
    return array_api_compat.size(arr) * bits // 8


def _calculate_size_of_image(ccd):
    # If uncertainty_func is given for combine this will create an uncertainty
    # even if the originals did not have one. In that case we need to create
    # an empty placeholder.

    size_of_an_img = _array_size_in_bytes(ccd.data)
    try:
        size_of_an_img += _array_size_in_bytes(ccd.uncertainty.array)
    # In case uncertainty is None it has no "array" and in case the "array" is
    # not an array at all:
    except (AttributeError, TypeError):
        pass
    if ccd.mask is not None:
        size_of_an_img += _array_size_in_bytes(ccd.mask)
    # flags is not necessarily an array so do not fail in case something
    # was set!
    # TODO: Flags are not taken into account in Combiner. This number is added
    #       nevertheless for future compatibility.
    try:
        size_of_an_img += _array_size_in_bytes(ccd.flags)
    except (AttributeError, TypeError):
        pass

    return size_of_an_img


def combine(
    img_list,
    output_file=None,
    method="average",
    weights=None,
    scale=None,
    mem_limit=16e9,
    clip_extrema=False,
    nlow=1,
    nhigh=1,
    minmax_clip=False,
    minmax_clip_min=None,
    minmax_clip_max=None,
    sigma_clip=False,
    sigma_clip_low_thresh=3,
    sigma_clip_high_thresh=3,
    sigma_clip_func=None,
    sigma_clip_dev_func=None,
    dtype=None,
    combine_uncertainty_function=None,
    overwrite_output=False,
    array_package=None,
    **ccdkwargs,
):
    """
    Convenience function for combining multiple images.

    Parameters
    ----------
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

    dtype : dtype-like or None, optional
        The intermediate and resulting ``dtype`` for the combined CCDs; see
        `ccdproc.Combiner` for the accepted forms. If ``None`` this is set to
        the namespace's ``float64``.
        Default is ``None``.

    combine_uncertainty_function : callable, None, optional
        If ``None`` use the default uncertainty func when using average, median or
        sum combine, otherwise use the function provided.
        Default is ``None``.

    overwrite_output : bool, optional
        If ``output_file`` is specified, this is passed to the
        `astropy.nddata.fits_ccddata_writer` under the keyword ``overwrite``;
        has no effect otherwise.
        Default is ``False``.

    array_package : array namespace or module, optional
        The array package to use for data read in from files; ignored if
        ``ccd_list`` is already a list of `~astropy.nddata.CCDData` objects.
        Either an array namespace or a plain module that follows the array
        API standard (e.g. ``numpy`` or ``dask.array``); it is normalised to
        its array-api-compat namespace the same way `~ccdproc.Combiner`
        handles ``xp``. Default is NumPy.

    ccdkwargs : Other keyword arguments for `astropy.nddata.fits_ccddata_reader`.

    Returns
    -------
    combined_image : `~astropy.nddata.CCDData`
        CCDData object based on the combined input of CCDData objects.
    """
    # Handle case where the input is an array of file names first
    if not isinstance(img_list, list):
        try:
            _ = array_api_compat.array_namespace(img_list)
        except TypeError:
            pass
        else:
            # If it is an array, convert it to a list
            img_list = list(img_list)
    if (
        not isinstance(img_list, list)
        and isinstance(img_list, str)
        and ("," in img_list)
    ):
        # Handle case where the input is a string of file names separated by comma
        img_list = img_list.split(",")
    else:
        try:
            # Maybe the input can be made into a list, so try that
            img_list = list(img_list)
        except TypeError as err:
            raise ValueError(
                "unrecognised input for list of images to combine."
            ) from err

    # Select Combine function to call in Combiner
    if method == "average":
        combine_function = "average_combine"
    elif method == "median":
        combine_function = "median_combine"
    elif method == "sum":
        combine_function = "sum_combine"
    else:
        raise ValueError(f"unrecognised combine method : {method}.")

    # First we create a CCDObject from first image for storing output
    if isinstance(img_list[0], CCDData):
        ccd = img_list[0].copy()
    else:
        # User has provided fits filenames to read from
        ccd = CCDData.read(img_list[0], **ccdkwargs)
        # The ccd object will always read as numpy, so convert it to the
        # requested namespace if there is one.
        if array_package is not None:
            # ``array_package`` may be a raw module such as ``numpy`` or
            # ``dask.array``; normalise it to the array-api-compat namespace
            # the same way ``Combiner.__init__`` does, so the conversions
            # below can rely on array-API features (e.g. the ``device``
            # keyword) that a raw module may not provide.
            xp = array_api_compat.array_namespace(array_package.asarray(0))

            # ccd.data (and its uncertainty, if any) were just read from a
            # FITS file, so they are NumPy arrays, possibly in big-endian
            # byte order. Convert to native byte order before handing them
            # to a non-NumPy namespace: some namespaces reject non-native
            # dtypes outright, and array_api_strict warns (which becomes an
            # error under this project's warning filters) when a NumPy
            # dtype object is compared against one of its own.
            ccd.data = xp.asarray(_native_numpy(ccd.data))
            if ccd.uncertainty is not None:
                ccd.uncertainty.array = xp.asarray(_native_numpy(ccd.uncertainty.array))
            # The mask is converted below, once the namespace is known.

    # Get the array namespace; if array_package was not None and files were read in,
    # then xp the ccd.data will be the same as the array_package.
    xp = array_api_compat.array_namespace(ccd.data)
    if dtype is None:
        dtype = xp.float64
    else:
        dtype = _namespace_dtype(dtype, xp)

    if sigma_clip_func is None:
        sigma_clip_func = xp.mean
    if sigma_clip_dev_func is None:
        sigma_clip_dev_func = xp.std

    # Convert the master image to the appropriate dtype so when overwriting it
    # later the data is not downcast and the memory consumption calculation
    # uses the internally used dtype instead of the original dtype. #391
    if ccd.data.dtype != dtype:
        ccd.data = xp.astype(ccd.data, dtype)

    # If the template image doesn't have an uncertainty, add one, because the
    # result always has an uncertainty.
    if ccd.uncertainty is None:
        ccd.uncertainty = StdDevUncertainty(xp.zeros_like(ccd.data))

    # If the template doesn't have a mask, add one, because the result may have
    # a mask. If it does have one, it may be a numpy array even when the data
    # is not (the CCDData.mask setter converts to numpy), so coerce it into the
    # data's namespace and onto the data's device: the combined tiles are
    # written into it below.
    # TODO: the private _mask attribute is set here to avoid the CCDData.mask
    # setter. This can be removed when CCDData supports array namespaces.
    if ccd.mask is None:
        ccd._mask = xp.zeros_like(ccd.data, dtype=xp.bool)
    else:
        ccd._mask = xp.asarray(
            ccd.mask, dtype=xp.bool, device=array_api_compat.device(ccd.data)
        )

    size_of_an_img = _calculate_size_of_image(ccd)

    no_of_img = len(img_list)

    # Set a memory use factor based on profiling
    if method == "median":
        memory_factor = 3
    else:
        memory_factor = 2

    memory_factor *= 1.3

    # determine the number of chunks to split the images into
    no_chunks = int((memory_factor * size_of_an_img * no_of_img) / mem_limit) + 1
    if no_chunks > 1:
        log.info(
            f"splitting each image into {no_chunks} chunks to limit memory usage "
            f"to {mem_limit} bytes."
        )
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
        to_set_in_combiner["weights"] = weights

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
                    if array_package is not None:
                        imgccd.data = xp.asarray(imgccd.data, dtype=dtype)
                        if imgccd.uncertainty is not None:
                            imgccd.uncertainty.array = xp.asarray(
                                imgccd.uncertainty.array, dtype=dtype
                            )
                        if imgccd.mask is not None:
                            imgccd._mask = xp.asarray(imgccd.mask, dtype=xp.bool)

                scalevalues.append(scale(imgccd.data))

            # See Combiner.scaling: stack per-element conversions so that a
            # callable returning 0-d backend arrays works on array-api-strict.
            device = array_api_compat.device(ccd.data)
            to_set_in_combiner["scaling"] = xp.stack(
                [xp.asarray(value, device=device) for value in scalevalues]
            )
        else:
            to_set_in_combiner["scaling"] = scale

    if clip_extrema:
        to_call_in_combiner["clip_extrema"] = {"nlow": nlow, "nhigh": nhigh}

    if minmax_clip:
        to_call_in_combiner["minmax_clipping"] = {
            "min_clip": minmax_clip_min,
            "max_clip": minmax_clip_max,
        }

    if sigma_clip:
        to_call_in_combiner["sigma_clipping"] = {
            "low_thresh": sigma_clip_low_thresh,
            "high_thresh": sigma_clip_high_thresh,
            "func": sigma_clip_func,
            "dev_func": sigma_clip_dev_func,
        }

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
                    if array_package is not None:
                        imgccd.data = xp.asarray(imgccd.data, dtype=dtype)
                        if imgccd.uncertainty is not None:
                            imgccd.uncertainty.array = xp.asarray(
                                imgccd.uncertainty.array, dtype=dtype
                            )
                        if imgccd.mask is not None:
                            imgccd._mask = xp.asarray(imgccd.mask, dtype=xp.bool)

                # Trim image and copy
                # The copy is *essential* to avoid having a bunch
                # of unused file references around if the files
                # are memory-mapped. See this PR for details
                # https://github.com/astropy/ccdproc/pull/630
                ccd_list.append(deepcopy(imgccd[x:xend, y:yend]))

            # Create Combiner for tile
            tile_combiner = Combiner(ccd_list, dtype=dtype)

            # Set all properties and call all methods
            for to_set in to_set_in_combiner:
                setattr(tile_combiner, to_set, to_set_in_combiner[to_set])
            for to_call in to_call_in_combiner:
                getattr(tile_combiner, to_call)(**to_call_in_combiner[to_call])

            # Finally call the combine algorithm
            combine_kwds = {}
            if combine_uncertainty_function is not None:
                combine_kwds["uncertainty_func"] = combine_uncertainty_function

            comb_tile = getattr(tile_combiner, combine_function)(**combine_kwds)

            # add it back into the master image
            # Use array_api_extra so that we can use at with all array libraries
            ccd.data = xpx.at(ccd.data)[x:xend, y:yend].set(comb_tile.data)

            if ccd.mask is not None:
                # Handle immutable arrays with array_api_extra; copy=True
                # also covers a read-only mask. The private attribute is set
                # to avoid the CCDData.mask setter (see above).
                ccd._mask = xpx.at(ccd.mask)[x:xend, y:yend].set(
                    comb_tile.mask, copy=True
                )

            if ccd.uncertainty is not None:
                # Handle immutable arrays with array_api_extra
                ccd.uncertainty.array = xpx.at(ccd.uncertainty.array)[
                    x:xend, y:yend
                ].set(xp.astype(comb_tile.uncertainty.array, ccd.dtype))
            # Free up memory to try to stay under user's limit
            del comb_tile
            del tile_combiner
            del ccd_list

    # Write fits file if filename was provided
    if output_file is not None:
        # astropy.io.fits needs NumPy arrays, so write from a NumPy copy and
        # leave the returned result in its array namespace.
        to_write = CCDData(
            _to_numpy(ccd.data), unit=ccd.unit, meta=ccd.meta, wcs=ccd.wcs
        )
        if ccd.mask is not None:
            to_write.mask = _to_numpy(ccd.mask)
        if ccd.uncertainty is not None:
            to_write.uncertainty = ccd.uncertainty.__class__(
                _to_numpy(ccd.uncertainty.array)
            )
        to_write.write(output_file, overwrite=overwrite_output)

    return ccd
