Getting Started
===============

A ``CCDData`` object can be created from a numpy array (masked or not) or from
a FITS file:

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropy.nddata import CCDData
    >>> import ccdproc
    >>> image_1 = CCDData(np.ones((10, 10)), unit="adu")

An example of reading from a FITS file is
``image_2 = astropy.nddata.CCDData.read('my_image.fits', unit="electron")`` (the
``electron`` unit is defined as part of ``ccdproc``).

The metadata of a ``CCDData`` object may be any dictionary-like object, including a FITS header. When a ``CCDData`` object is initialized from FITS file its metadata is a FITS header.

The data is accessible either by indexing directly or through the ``data``
attribute:

    >>> sub_image = image_1[:, 1:-3]  # a CCDData object
    >>> sub_data =  image_1.data[:, 1:-3]  # a numpy array

See the documentation for `~astropy.nddata.CCDData` for a complete list of attributes.

Most operations are performed by functions in `ccdproc`:

    >>> dark = CCDData(np.random.normal(size=(10, 10)), unit="adu")
    >>> dark_sub = ccdproc.subtract_dark(image_1, dark,
    ...                                  dark_exposure=30*u.second,
    ...                                  data_exposure=15*u.second,
    ...                                  scale=True)

See the documentation for `~ccdproc.subtract_dark` for more compact
ways of providing exposure times.

Every function returns a *copy* of the data with the operation performed.

Every function in `ccdproc` supports logging through the addition of
information to the image metadata.

Logging can be simple -- add a string to the metadata:

    >>> dark_sub_gained = ccdproc.gain_correct(dark_sub, 1.5 * u.photon/u.adu, add_keyword='gain_corrected')

Logging can be more complicated -- add several keyword/value pairs by passing
a dictionary to ``add_keyword``:

    >>> my_log = {'gain_correct': 'Gain value was 1.5',
    ...           'calstat': 'G'}
    >>> dark_sub_gained = ccdproc.gain_correct(dark_sub,
    ...                                        1.5 * u.photon/u.adu,
    ...                                        add_keyword=my_log)

You might wonder why there is a `~ccdproc.gain_correct` at all, since the implemented
gain correction simple multiplies by a constant. There are two things you get
with `~ccdproc.gain_correct` that you do not get with multiplication:

+ Appropriate scaling of uncertainties.
+ Units

The same advantages apply to operations that are more complex, like flat
correction, in which one image is divided by another:

    >>> flat = CCDData(np.random.normal(1.0, scale=0.1, size=(10, 10)),
    ...                        unit='adu')
    >>> image_1_flat = ccdproc.flat_correct(image_1, flat)

In addition to doing the necessary division, `~ccdproc.flat_correct` propagates
uncertainties (if they are set).

The function `~ccdproc.wcs_project` allows you to reproject an image onto a different WCS.

To make applying the same operations to a set of files in a directory easier,
use an `~ccdproc.image_collection.ImageFileCollection`. It constructs, given a directory, a `~astropy.table.Table` containing the values of user-selected keywords in the directory. It also provides methods for iterating over the files. The example below was used to find an image in which the sky background was high for use in a talk:

    >>> from ccdproc import ImageFileCollection
    >>> import numpy as np
    >>> from glob import glob
    >>> dirs = glob('/Users/mcraig/Documents/Data/feder-images/fixed_headers/20*-??-??')

    >>> for d in dirs:
    ...     print(d)
    ...     ic = ImageFileCollection(d, keywords='*')
    ...     for data, fname in ic.data(imagetyp='LIGHT', return_fname=True):
    ...         if data.mean() > 4000.:
    ...             print(fname)
