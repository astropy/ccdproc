Array library options in ccdproc
================================

.. note::

    Who needs this? If you are currently using numpy for your image processing
    there is no need to change anything about what you currently do. The changes
    made in `ccdproc` to adopt the `array API`_ were made with the intent of
    requiring no change to existing code that continues to use `numpy`_.

What is the "Array API"?
------------------------

The Python `array API`_ specifies an interface that has been adopted by many
different array libraries (e.g. jax, dask, CuPy). The API is very similar to the
familiar `numpy`_ interface. The `array API`_ was constructed to allow users
with specialized needs to use any of the large variety of array options
available in Python.

What array libraries are supported?
-----------------------------------

The best list of array libraries that implement the array API is at `array-api-compat`_.
`ccdproc`_ is currently regularly tested against `numpy`_, `dask`_, and `jax`_. It
is occasionally tested against `CuPy`_; any errors your encounter running `ccdproc`_
on a GPU using `CuPy`_ should be
`reported as an issue <https://github.com/astropy/ccdproc/issues>`_.

Though the
`sparse`_ array library supports the array API, `ccdproc`_ does not currently work
with `sparse`_. A `pull request <https://github.com/astropy/ccdproc/pulls>`_ to FIX
that would be a welcome contribution to the project.

What limitations shuold I be aware of?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+ The ``median`` function is not part of the array API, but most array libraries
  do provide a ``median``. If the array library you choose does not have a ``median``
  function then `ccdproc`_ will automatically fall back to using a ``median`` function from
  `bottleneck`_, if that is installed, or to `numpy`_.

Which array library should I use?
---------------------------------

If you have access to a GPU then using `cupy`_ will be noticeably faster than
using `numpy`_. If you routinely use very large datasets, consider using `dask`_.
The array library that the maintainers of `ccdproc` most often use is `numpy`_.

How do I use the array API?
---------------------------

There are two ways to use the array API in `ccdproc`_:

1. Use the `ccdproc`_ functions as you normally would, but pass in an array from
   the array library of your choice. For example, if you want to use `dask`_ arrays,
   you can do this:

   .. code-block:: python

       import dask.array as da
       import ccdproc
       from astropy.nddata import CCDData

       data = da.random.random((1000, 1000))
       ccd = CCDData(data, unit='adu')
       ccd = ccdproc.trim_image(ccd[:900, :900])

2. Use `ccdproc`_ functions to read/write data in addition to
   using `ccdproc`_ functions to process the data. For example, if you want to
   use `dask`_ arrays to process a set of images, you can do this:

   .. code-block:: python

       import dask.array as da
       import ccdproc
       from astropy.nddata import CCDData

       images = ccdproc.ImageFileCollection('path/to/images/*.fits',
                                            array_package=da)
       for ccd in images.ccds():
           ccd = ccdproc.trim_image(ccd[:900, :900])
           # Do more processing with ccdproc functions
           # ...

   If you do this, image combination will also be done using the array library
   you specified.

   To do image combination with the array library of your choice without doing
   any other processing, you can either create a `ccdproc.Combiner` object with a
   list of file names and the ``array_package`` argument set to the array library
   you want to use, or use the `ccdproc.combine` function a list of file names and
   the ``array_package`` argument set to the array library you want to use. For
   example, to combine images using `dask`_ arrays, you can do this:

   .. code-block:: python

       import dask.array as da
       import ccdproc
       from astropy.nddata import CCDData

       images = ccdproc.ImageFileCollection('path/to/images/*.fits',
                                            array_package=da)
       combined = ccdproc.combine_images(images.ccds(), method='median')

.. _array API: https://data-apis.org/array-api/latest/index.html
.. _array-api-compat: https://data-apis.org/array-api-compat
.. _bottleneck: https://bottleneck.readthedocs.io/en/latest/
.. _ccdproc: https://ccdproc.readthedocs.io/en/latest/
.. _cupy: https://docs.cupy.dev/en/stable/
.. _dask: https://docs.dask.org/en/stable/
.. _jax: https://docs.jax.dev/en/latest/index.html
.. _numpy: https://numpy.org/doc/stable/reference/array_api.html
.. _sparse: https://sparse.pydata.org/en/stable/
