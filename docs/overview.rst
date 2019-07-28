Overview
========

.. note::
    `ccdproc` works only with astropy version 2.0 or later.

The `ccdproc` package provides:

+ An image class, `~astropy.nddata.CCDData`, that includes an uncertainty for the
  data, units and methods for performing arithmetic with images including the
  propagation of uncertainties.
+ A set of functions performing common CCD data reduction steps (e.g. dark
  subtraction, flat field correction) with a flexible mechanism for logging
  reduction steps in the image metadata.
+ A function for reprojecting an image onto another WCS, useful for stacking
  science images. The actual reprojection is done by the
  `reproject package <http://reproject.readthedocs.io/en/stable/>`_.
+ A class for combining and/or clipping images, `~ccdproc.Combiner`, and
  associated functions.
+ A class, `~ccdproc.ImageFileCollection`, for working with a directory of
  images.
