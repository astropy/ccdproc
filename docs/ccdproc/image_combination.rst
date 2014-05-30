.. _image_combination:

Combining images and generating masks from clipping
===================================================

.. note::
    No attempt has been made yet to optimize memory usage in
    `~ccdproc.Combiner`. A copy is made, and a mask array
    constructed, for each input image.


The first step in combining a set of images is creating a
`~ccdproc.Combiner` instance:

    >>> from ccdproc import CCDData, Combiner
    >>> import numpy as np
    >>> ccd1 = CCDData(np.random.normal(size=(10,10)))
    >>> ccd2 = ccd1.copy()
    >>> ccd3 = ccd1.copy()
    >>> combiner = Combiner([ccd1, ccd2, ccd3])

The combiner task really combines two things: generation of masks for
individual images via several clipping techniques and combination of images.

.. _clipping:

Image masks/clipping
--------------------

There are currently two methods of clipping. Neither affects the data
directly; instead each constructs a mask that is applied when images are
combined.

Masking done by clipping operations is combined with the image mask provided
when the `~ccdproc.Combiner` is created.

Min/max clipping
++++++++++++++++

`~ccdproc.Combiner.minmax_clipping` masks all pixels above or below
user-specified levels. For example, to mask all values above the value
``0.1`` and below the value ``-0.3``:

    >>> combiner.minmax_clipping(min_clip=-0.3, max_clip=0.1)

Either ``min_clip`` or ``max_clip`` can be omitted.

Sigma clipping
++++++++++++++

For each pixel of an image in the combiner,
`~ccdproc.combiner.Combiner.sigma_clipping` masks the pixel if is more than a
user-specified number of deviations from the central value of that pixel in
the list of images.

The `~ccdproc.combiner.Combiner.sigma_clipping` method is very flexible: you can
specify both the function for calculating the central value and the function
for calculating the deviation. The default is to use the mean (ignoring any
masked pixels) for the central value and the standard deviation (again
ignoring any masked values) for the deviation.

You can mask pixels more than 5 standard deviations above or 2 standard
deviations below the median with

    >>> combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.median)

.. note::
    Numpy masked median can be very slow in exactly the situation typically
    encountered in reducing ccd data: a cube of data in which one dimension
    (in the case the number of frames in the combiner) is much smaller than
    the number of pixels.

    A much faster library for this case is `bottleneck`_; a detailed example
    which uses `bottleneck`_ is at :ref:`bottleneck_example`.

Iterative clipping
++++++++++++++++++

To clip iteratively, continuing the clipping process until no more pixels are
rejected, loop in the code calling the clipping method:

    >>> old_n_masked = 0  # dummy value to make loop execute at least once
    >>> new_n_masked = combiner.data_array.mask.sum()
    >>> while (new_n_masked > old_n_masked):
    ...     combiner.sigma_clipping(func=np.ma.median)
    ...     old_n_masked = new_n_masked
    ...     new_n_masked = combiner.data_array.mask.sum()

Note that the default values for the high and low thresholds for rejection are
3 standard deviations.

Image combination
-----------------

Image combination is straightforward; to combine by taking the average,
excluding any pixels mapped by clipping:

    >>> combined_average = combination.average_combine()

Performing a median combination is also straightforward,

    >>> combined_median = combiner.median_combine()  # can be slow, see below 

The masked median function provided by numpy (and used by default in
``median_combine``) can be very slow, so ``median_combine`` accepts an
argument ``median_func`` for calculating the median instead. One fast
alternative is provided by the `bottleneck`_ package; an example using it is
at :ref:`bottleneck_example`.

With image transformation
-------------------------

TBD

.. _bottleneck: http://berkeleyanalytics.com/bottleneck/
