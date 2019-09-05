.. _image_management:

Image Management
================


.. _image_collection:

Working with a directory of images
----------------------------------

For the sake of argument all of the examples below assume you are working in
a directory that contains FITS images.

The class :class:`~ccdproc.image_collection.ImageFileCollection` is meant to
make working with a directory of FITS images easier by allowing you select the
files you act on based on the values of FITS keywords in their headers or based
on Unix shell-style filename matching.

It is initialized with the name of a directory containing FITS images and a
list of FITS keywords you want the
:class:`~ccdproc.image_collection.ImageFileCollection` to be aware of. An
example initialization looks like::

    >>> from ccdproc import ImageFileCollection
    >>> from ccdproc.utils.sample_directory import sample_directory_with_files
    >>> keys = ['imagetyp', 'object', 'filter', 'exposure']
    >>> dir = sample_directory_with_files()
    >>> ic1 = ImageFileCollection(dir, keywords=keys) # only keep track of keys

You can use the wildcard ``*`` in place of a list to indicate you want the
collection to use all keywords in the headers::

    >>> ic_all = ImageFileCollection(dir, keywords='*')

Normally identification of FITS files is done by looking at the file extension
and including all files with the correct extension.

If the files are not compressed (e.g. not gzipped) then you can force the image
collection to open each file and check from its contents whether it is FITS by
using the ``find_fits_by_reading`` argument::

    >> ic_from_content = ImageFileCollection(dir, find_fits_by_reading=True)

You can indicate filename patterns to include or exclude using Unix shell-style
expressions. For example, to include all filenames that begin with ``1d_`` but
not ones that include the word ``bad``, you could do::

    >>> ic_all = ImageFileCollection(dir, glob_include='1d_*',
    ...                              glob_exclude='*bad*')

Alternatively, you can create the collection with an explicit list of file names::

    >>> ic_names = ImageFileCollection(filenames=['a.fits', '/some/path/b.fits.gz'])

Most of the useful interaction with the image collection is via its
``.summary`` property, a :class:`~astropy.table.Table` of the value of each keyword for each
file in the collection::

    >>> ic1.summary.colnames
    ['file', 'imagetyp', 'object', 'filter', 'exposure']
    >>> ic_all.summary.colnames # doctest: +SKIP
    # long list of keyword names omitted

Note that the name of the file is automatically added to the table as a
column named ``file``.

Selecting files
---------------

Selecting the files that match a set of criteria, for example all images in
the I band with exposure time less than 60 seconds you could do::

    >>> matches = (ic1.summary['filter'] == 'R') & (ic1.summary['exposure'] < 15)
    >>> my_files = ic1.summary['file'][matches]

The column ``file`` is added automatically when the image collection is created.

For more simple selection, when you just want files whose keywords exactly
match particular values, say all I band images with exposure time of 30
seconds, there is a convenience method ``.files_filtered``::

    >>> my_files = ic1.files_filtered(filter='R', exposure=15)

The optional arguments to ``files_filtered`` are used to filter the list of
files.

Python regular expression patterns can also be used as the value if the
``regex_match`` flag is set. For example, to find all of the images whose
object is in the Kelt exoplanet survey, you might do::

    >>> my_files = ic1.files_filtered(regex_match=True, object='kelt.*')

To get all of the images that have image type ``BIAS`` or ``LIGHT`` you
can also use a regular expression pattern::

    >>> my_files = ic1.files_filtered(regex_match=True,
    ...                               imagetyp='bias|light')

Note that regular expression is different, and much more flexible than,
file name matching (or "globbing") at the command line. The
`Python documentation on the re module <https://docs.python.org/3.7/library/re.html#module-re>`_
is useful for learning about regular expressions.

Finally, a new `~ccdproc.ImageFileCollection` can be created with by providing
a list of keywords. The example below makes a new collection containing the
files whose ``imagetyp`` is ``BIAS`` or ``LIGHT``::

    >>> new_ic = ic1.filter(regex_match=True,
    ...                     imagetyp='bias|light')

Sorting files
-------------

Sometimes it is useful to bring the files into a specific order, e.g. if you
make a plot for each object you probably want all images of the same object
next to each other. To do this, the images in a collection can be sorted with
the ``sort`` method using the fits header keys in the same way you would sort a
:class:`~astropy.table.Table`::

    >>> ic1.sort(['exposure', 'imagetyp'])

Iterating over hdus, headers, data, or ccds
-------------------------------------------

Four methods are provided for iterating over the images in the collection,
optionally filtered by keyword values.

For example, to iterate over all of the I band images with exposure of
30 seconds, performing some basic operation on the data (very contrived
example)::

    >>> for hdu in ic1.hdus(imagetyp='LiGhT', filter='R', exposure=15):
    ...     hdu.header['exposure']
    ...     new_data = hdu.data - hdu.data.mean()
    15.0

Note that the names of the arguments to ``hdus`` here are the names of FITS
keywords in the collection and the values are the values of those keywords you
want to select. Note also that string comparisons are not case sensitive.

The other iterators are ``headers``, ``data``, and ``ccds``.

All of them have the option to also provide the file name in addition to the
hdu (or header or data)::

    >>> for hdu, fname in ic1.hdus(return_fname=True,
    ...                            imagetyp='LiGhT', filter='R', exposure=15):
    ...    hdu.header['meansub'] = True
    ...    hdu.data = hdu.data - hdu.data.mean()
    ...    hdu.writeto(fname + '.new')

That last use case, doing something to several files and saving them
somewhere afterwards, is common enough that the iterators provide arguments to
automate it.

Automatic saving from the iterators
-----------------------------------

There are three ways of triggering automatic saving.

1. One is with the argument ``save_with_name``; it adds the value of the
argument to the file name between the original base name and extension. The
example below has (almost) the same effect of the example above, subtracting
the mean from each image and saving to a new file::

    >>> for hdu in ic1.hdus(save_with_name='_new',
    ...                     imagetyp='LiGhT', filter='R', exposure=15):
    ...    hdu.header['meansub'] = True
    ...    hdu.data = hdu.data - hdu.data.mean()

It saves, in the ``location`` of the image collection, a new FITS file with
the mean subtracted from the data, with ``_new`` added to the name; as an
example, if one of the files iterated over was ``intput001.fit`` then a new
file, in the same directory, called ``input001_new.fit`` would be created.

2. You can also provide the directory to which you want to save the files with
``save_location``; note that you do not need to actually do anything to the
hdu (or header or data) to cause the copy to be made. The example below copies
all of the I band images with 30 second exposure from the original
location to ``other_dir``::

    >>> for hdu in ic1.hdus(save_location='other_dir',
    ...                     imagetyp='LiGhT', filter='I', exposure=30):  # doctest: +SKIP
    ...     pass

This option can be combined with the previous one to also give the files a
new name.

3. Finally, if you want to live dangerously, you can overwrite the files in
the same location with the ``overwrite`` argument; use it carefully because it
preserves no backup. The example below replaces each of the I band images
with 30 second exposure with a file that has had the mean subtracted::

    >>> for hdu in ic1.hdus(overwrite=True,
    ...                     imagetyp='LiGhT', filter='R', exposure=15): # doctest: +SKIP
    ...    hdu.header['meansub'] = True
    ...    hdu.data = hdu.data - hdu.data.mean()

.. note::
    This functionality is not currently available on Windows.
