.. _image_combination:

Combining images and generating masks from clipping
===================================================

.. note::
    There are currently two interfaces to image combination. One is through
    the `~ccdproc.Combiner` class, the other through the `~ccdproc.combine`
    function. They offer *almost* identical capabilities. The primary
    difference is that `~ccdproc.combine` allows you to place an upper
    limit on the amount of memory used.


.. note::
    Image combination performance is substantially better if you install
    the `bottleneck`_ package, especially when using a median.

    .. _bottleneck:  https://github.com/pydata/bottleneck


The first step in combining a set of images is creating a
`~ccdproc.Combiner` instance:

    >>> from astropy import units as u
    >>> from astropy.nddata import CCDData
    >>> from ccdproc import Combiner
    >>> import numpy as np
    >>> ccd1 = CCDData(np.random.normal(size=(10,10)),
    ...                unit=u.adu)
    >>> ccd2 = ccd1.copy()
    >>> ccd3 = ccd1.copy()
    >>> combiner = Combiner([ccd1, ccd2, ccd3])

The combiner task really combines two things: generation of masks for
individual images via several clipping techniques and combination of images,
with optional weighting of images for some of the combination methods.

.. _clipping:

Image masks and clipping
------------------------

There are currently three methods of clipping. None affect the data
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


Extrema clipping
++++++++++++++++

For each pixel position in the input arrays, the algorithm will mask the
highest ``nhigh`` and lowest ``nlow`` pixel values.  The resulting image will be
a combination of ``Nimages-nlow-nhigh`` pixel values instead of the combination
of ``Nimages`` worth of pixel values.

You can mask the lowest pixel value and the highest two pixel values with:

    >>> combiner.clip_extrema(nlow=1, nhigh=2)


Iterative clipping
++++++++++++++++++

To clip iteratively, continuing the clipping process until no more pixels are
rejected, loop in the code calling the clipping method:

    >>> old_n_masked = 0  # dummy value to make loop execute at least once
    >>> new_n_masked = combiner.data_arr.mask.sum()
    >>> while (new_n_masked > old_n_masked):
    ...     combiner.sigma_clipping(func=np.ma.median)
    ...     old_n_masked = new_n_masked
    ...     new_n_masked = combiner.data_arr.mask.sum()

Note that the default values for the high and low thresholds for rejection are
3 standard deviations.

Image combination
-----------------

Image combination is straightforward; to combine by taking the average,
excluding any pixels mapped by clipping:

    >>> combined_average = combiner.average_combine()  # doctest: +IGNORE_WARNINGS

Performing a median combination is also straightforward, but can be slow:

    >>> combined_median = combiner.median_combine()  #  doctest: +IGNORE_WARNINGS



Combination with image scaling
++++++++++++++++++++++++++++++

In some circumstances it may be convenient to scale all images to some value
before combining them. Do so by setting `~ccdproc.Combiner.scaling`:

    >>> scaling_func = lambda arr: 1/np.ma.average(arr)
    >>> combiner.scaling = scaling_func  # doctest: +IGNORE_WARNINGS
    >>> combined_average_scaled = combiner.average_combine()  # doctest: +IGNORE_WARNINGS

This will normalize each image by its mean before combining (note that the
underlying images are *not* scaled; scaling is only done as part of combining
using `~ccdproc.Combiner.average_combine` or
`~ccdproc.Combiner.median_combine`).

Weighting images during image combination
+++++++++++++++++++++++++++++++++++++++++

There are times when different images need to have different weights during
image combination. For example, different images may have different exposure
times. When combining image mosaics, each pixel may need a different weight
depending on how much overlap there is between the images that make up the
mosaic.

Both weighting by image and pixel-wise weighting are done by setting
`~ccdproc.Combiner.weights`.

Recall that in the example on this page three images, each ``10 x 10`` pixels,
are being combined. To weight the three images differently, set
`~ccdproc.Combiner.weights` to an array for length three:

    >>> combiner.weights = np.array([0.5, 1, 2.0])
    >>> combine_weighted_by_image = combiner.average_combine()  # doctest: +IGNORE_WARNINGS

To use pixel-wise weighting set `~ccdproc.Combiner.weights` to an array that
matches the number of images and image shape, in this case ``3 x 10 x 10``:

    >>> combiner.weights = np.random.random_sample([3, 10, 10])
    >>> combine_weighted_by_image = combiner.average_combine()  # doctest: +IGNORE_WARNINGS

.. note::
    Weighting does **not** work when using the median to combine images.
    It works only for combining by average or by summation.


.. _combination_with_IFC:

Image combination using `~ccdproc.ImageFileCollection`
------------------------------------------------------

There are a couple of ways that image combination can be done if you are using
`~ccdproc.ImageFileCollection` to
:ref:`manage a folder of images <image_management>`.

For this example, a temporary folder with images in it is created:

    >>> from tempfile import mkdtemp
    >>> from pathlib import Path
    >>> import numpy as np
    >>> from astropy.nddata import CCDData
    >>> from ccdproc import ImageFileCollection, Combiner, combine
    >>>
    >>> ccd = CCDData(np.ones([5, 5]), unit='adu')
    >>>
    >>> # Make a temporary folder as a path object
    >>> image_folder = Path(mkdtemp())
    >>> # Put several copies ccd in the temporary folder
    >>> _ = [ccd.write(image_folder / f"ccd-{i}.fits") for i in range(3)]
    >>> ifc = ImageFileCollection(image_folder)

To combine images using the `~ccdproc.Combiner` class you can use the ``ccds``
method of the `~ccdproc.ImageFileCollection`:

    >>> c = Combiner(ifc.ccds())
    >>> avg_combined = c.average_combine()

There two ways combine images using the `~ccdproc.combine` function. If the
images are large enough to combine in memory, then use the file names as the argument to `~ccdproc.combine`, like this:

    >>> avg_combo_mem_lim = combine(ifc.files_filtered(include_path=True),
    ...                             mem_limit=1e9)

If memory use is not an issue, then the ``ccds`` method can be used here too:

    >>> avg_combo = combine(ifc.ccds())



.. _reprojection:

Combination with image transformation and alignment
---------------------------------------------------

.. note::

    **Flux conservation** Whether flux is conserved in performing the
    reprojection depends on the method you use for reprojecting and the
    extent to which pixel area varies across the image.
    `~ccdproc.wcs_project` rescales counts by the ratio of pixel area
    *of the pixel indicated by the keywords* ``CRPIX`` of the input and
    output images.

    The reprojection methods available are described in detail in the
    documentation for the `reproject project`_; consult those
    documents for details.

    You should carefully check whether flux conservation provided in CCDPROC
    is adequate for your needs. Suggestions for improvement are welcome!

Align and then combine images based on World Coordinate System (WCS)
information in the image headers in two steps.

First, reproject each image onto the same footprint using
`~ccdproc.wcs_project`. The example below assumes you have an image with WCS
information and another image (or WCS) onto which you want to project your
images:

.. doctest-skip::

    >>> from ccdproc import wcs_project
    >>> reprojected_image = wcs_project(input_image, target_wcs)

Repeat this for each of the images you want to combine, building up a list of
reprojected images:

.. doctest-skip::

    >>> reprojected = []
    >>> for img in my_list_of_images:
    ...     new_image = wcs_project(img, target_wcs)
    ...     reprojected.append(new_image)

Then, combine the images as described above for any set of images:

.. doctest-skip::

    >>> combiner = Combiner(reprojected)
    >>> stacked_image = combiner.average_combine()  # doctest: +IGNORE_WARNINGS

.. _reproject project: http://reproject.readthedocs.io/
