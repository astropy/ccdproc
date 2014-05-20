:orphan:

.. _bottleneck_example:

Faster medians using bottleneck
===============================

The `bottleneck`_ package provides very fast implementations of numpy functions like median that aggregate data. It accommodates masking by replacing masked values with ``numpy.NaN``.

How much faster is `bottleneck`_? The median on masked data is roughly 1000x faster than numpy.

.. note::
    The latest version of `bottleneck`_ works only with numpy 1.8.0 or later.

Installing bottleneck
---------------------

This should be easy: ``pip install bottleneck`` will do the trick.

Using `bottleneck`_ with `ccdproc`
----------------------------------

To use `bottleneck`_, we need to do three things:

1. Fill any mask values in the data array with ``numpy.NaN``.
2. Pass the data into, e.g., ``bottleneck.nanmedian()``
3. Create a mask for the result by masking all ``numpy.NaN`` values.

The function below can be used as a replacement for `numpy.ma.median`::

    def bn_median(masked_array, axis=None):
        """
        Perform fast median on masked array
        
        Parameters
        
        masked_array : `numpy.ma.masked_array`
            Array of which to find the median.
        
        axis : int, optional
            Axis along which to perform the median. Default is to find the median of
            the flattened array.
        """
        import numpy as np
        import bottleneck as bn
        data = masked_array.filled(fill_value=np.NaN)
        med = bn.nanmedian(data, axis=axis)
        # construct a masked array result, setting the mask from any NaN entries
        return np.ma.array(med, mask=np.isnan(med))

To use this with `~ccdproc.Combiner.sigma_clipping` (assuming you have done all of the necessary imports first and created the combiner):

    >>> my_combiner.sigma_clipping(func=bn_median)  # doctest: +SKIP

To perform `~ccdproc.Combiner.median_combine` with `bottleneck`_:

    >>> my_combiner.median_combine(median_func=bn_median)  # doctest: +SKIP

Feedback on whether this should be incorporated directly into `ccdproc` would be
appreciated.

.. _bottleneck: http://berkeleyanalytics.com/bottleneck/
