CCD Data reduction (`ccdproc`)
==============================

Introduction
------------

.. note::
    `ccdproc` works only with astropy version 0.4.0 or later.

The `ccdproc` package provides:

+ An image class, `~ccdproc.CCDData`, that includes an uncertainty for the 
  data, units and methods for performing arithmetic with images including the
  propagation of uncertainties.
+ A set of functions performing common CCD data reduction steps (e.g. dark
  subtraction, flat field correction) with a flexible mechanism for logging
  reduction steps in the image metadata.
+ A class for combining and/or clipping images, `~ccdproc.Combiner`, and
  associated functions.

Getting Started
---------------

.. warning::
    `ccdproc` is still under active development. The API will almost
    certainly change.

    In addition, testing of `ccdproc` on real data is currently very limited.
    Use with caution, and please report any errors you find at the 
    `GitHub repo`_ for this project.

A ``CCDData`` object can be created from a numpy array (masked or not) or from
a FITS file:

    >>> import numpy as np
    >>> from astropy import units as u
    >>> import ccdproc
    >>> image_1 = ccdproc.CCDData(np.ones((10, 10)), unit="adu")

An example of reading from a FITS file is
``image_2 = ccdproc.CCDData.read('my_image.fits', unit="electron")`` (the 
``electron`` unit is defined as part of ``ccdproc``).

The metadata of a ``CCDData`` object is a case-insensitive dictionary (though 
this may change in future versions).

The data is accessible either by indexing directly or through the ``data``
attribute:

    >>> sub_image = image_1[:, 1:-3]  # a CCDData object
    >>> sub_data =  image_1.data[:, 1:-3]  # a numpy array

See the documentation for `~ccdproc.CCDData` for a complete list of attributes.

Most operations are performed by functions in `ccdproc`:

    >>> dark = ccdproc.CCDData(np.random.normal(size=(10, 10)), unit="adu")
    >>> dark_sub = ccdproc.subtract_dark(image_1, dark,
    ...                                  dark_exposure=30*u.second,
    ...                                  data_exposure=15*u.second,
    ...                                  scale=True)

Every function returns a *copy* of the data with the operation performed. If,
for some reason, you wanted to modify the data in-place, do this:

    >>> image_2 = ccdproc.subtract_dark(image_1, dark, dark_exposure=30*u.second, data_exposure=15*u.second, scale=True)

See the documentation for `~ccdproc.subtract_dark` for more compact
ways of providing exposure times.

Every function in `ccdproc` supports logging through the addition of
information to the image metadata.

Logging can be simple -- add a string to the metadata:

    >>> image_2_gained = ccdproc.gain_correct(image_2, 1.5 * u.photon/u.adu, add_keyword='gain_corrected')

Logging can be more complicated -- add several keyword/value pairs by passing
a dictionary to ``add_keyword``:

    >>> my_log = {'gain_correct': 'Gain value was 1.5',
    ...           'calstat': 'G'}
    >>> image_2_gained = ccdproc.gain_correct(image_2,
    ...                                       1.5 * u.photon/u.adu,
    ...                                       add_keyword=my_log)

The `~ccdproc.ccdproc.Keyword` class provides a compromise between the simple
and complicated cases for providing a single key/value pair:

    >>> key = ccdproc.Keyword('gain_corrected', value='Yes')
    >>> image_2_gained = ccdproc.gain_correct(image_2,
    ...                                       1.5 * u.photon/u.adu,
    ...                                       add_keyword=key)

`~ccdproc.ccdproc.Keyword` also provides a convenient way to get a value from
image metadata and specify its unit:

    >>> image_2.header['gain']  = 1.5
    >>> gain = ccdproc.Keyword('gain', unit=u.photon/u.adu)
    >>> image_2_var = ccdproc.create_deviation(image_2,
    ...                                       gain=gain.value_from(image_2.header),
    ...                                       readnoise=3.0 * u.photon)

You might wonder why there is a `~ccdproc.gain_correct` at all, since the implemented
gain correction simple multiplies by a constant. There are two things you get
with `~ccdproc.gain_correct` that you do not get with multiplication:

+ Appropriate scaling of uncertainties.
+ Units

The same advantages apply to operations that are more complex, like flat
correction, in which one image is divided by another:

    >>> flat = ccdproc.CCDData(np.random.normal(1.0, scale=0.1, size=(10, 10)),
    ...                        unit='adu')
    >>> image_1_flat = ccdproc.flat_correct(image_1, flat)

In addition to doing the necessary division, `~ccdproc.flat_correct` propagates
uncertainties (if they are set).


Using `ccdproc`
---------------

.. toctree::
    :maxdepth: 1

    ccddata.rst
    image_combination.rst
    reduction_toolbox.rst
    reduction_examples.rst

.. automodapi:: ccdproc

.. _GitHub repo: https://github.com/astropy/ccdproc
