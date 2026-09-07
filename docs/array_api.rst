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
is occasionally tested against `CuPy`_; any errors you encounter running `ccdproc`_
on a GPU using `CuPy`_ should be
`reported as an issue <https://github.com/astropy/ccdproc/issues>`_.

Though the
`sparse`_ array library supports the array API, `ccdproc`_ does not currently work
with `sparse`_. A `pull request <https://github.com/astropy/ccdproc/pulls>`_ to add
support for `sparse`_ would be a welcome contribution to the project.

For development purposes, `ccdproc`_'s test suite can also be run against
`array-api-strict`_, a thin wrapper around `numpy`_ that strictly enforces
the array API, rejecting any usage outside the standard, and simulates
multiple devices. This
is set with the environment variable ``CCDPROC_ARRAY_LIBRARY=array-api-strict``
(``array_api_strict`` is also accepted). By default the test suite creates
arrays on one of `array-api-strict`_'s non-default devices, which causes
``numpy.asarray`` to raise an error, the same way it would on an array still
resident on a `CuPy`_ GPU device. This makes `array-api-strict`_ a convenient
CPU-only proxy for catching places where `ccdproc`_ silently (and
incorrectly) converts a non-numpy array back to numpy. The device used can be
overridden with the ``CCDPROC_ARRAY_DEVICE`` environment variable (its value
is passed to ``array_api_strict.Device``); set it to ``default`` to use the
library's normal CPU device instead.

A few more developer tools help triage failures on non-numpy backends:

+ Setting ``CCDPROC_TRIAGE_ESCAPES=1`` prints a summary at the end of the
  test session that groups failures by "escape site" -- the innermost frame
  inside `ccdproc`_ (but outside its test suite) in each failure's
  traceback -- so a large batch of backend failures collapses to a short
  list of root-cause call sites.
+ Setting ``CCDPROC_LOG_ARRAY_ESCAPES=1`` logs a warning whenever a
  non-numpy array-API array is passed to ``numpy.asarray``,
  ``numpy.asanyarray`` or ``numpy.ma.asanyarray``. This catches backends
  like `dask`_ and `jax`_ where the conversion succeeds silently and the
  test passes anyway. Because the messages go through Python's ``logging``
  and pytest only shows captured logs for *failing* tests, run with
  ``-o log_cli=true`` to see escapes from passing tests (the log level is
  already configured in ``pyproject.toml``).
+ The ``backend_xfail(*backends, reason=...)`` marker marks a test as an
  expected (non-strict) failure only when ``CCDPROC_ARRAY_LIBRARY`` matches
  one of the named backends. The ``backend_skip(*backends, reason=...)``
  marker skips a test entirely for the named backends. Because these
  xfails are *non-strict* (unlike the suite-wide ``xfail_strict = true``
  default), a backend bug that later gets fixed becomes a silent XPASS
  rather than a failure, and the stale marker lingers forever -- check for
  XPASSes occasionally (run the backend suite with ``-rX``) and prune the
  markers that no longer fail. Before deleting one, confirm the XPASS in
  the CI logs rather than only on your own machine: backend behavior can be
  platform-dependent (see
  `issue #943 <https://github.com/astropy/ccdproc/issues/943>`_, where a
  jax-marked test passes on macOS but still fails on Linux CI).
+ Setting ``CCDPROC_ENFORCE_ESCAPE_BASELINE=1`` fails the test session if a
  new library escape site appears that is not in the checked-in baseline,
  and setting ``CCDPROC_WRITE_ESCAPE_BASELINE=1`` regenerates that baseline.
  Both are described in "The escape-baseline ratchet" below.

The escape-baseline ratchet
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The file ``ccdproc/tests/array_escape_baseline.txt`` is a checked-in list of
the known places in the `ccdproc`_ library where a non-numpy array is still
silently converted to numpy, as reported by the escape logger. Each entry is
one line of ``<file> <function> <coercion>`` followed by an optional
free-text reason (e.g. ``TODO`` for sites still to migrate, or ``BOUNDARY``
for calls into numpy-only dependencies such as scipy that will never leave).

Setting ``CCDPROC_ENFORCE_ESCAPE_BASELINE=1`` turns that list into a
ratchet: at the end of the test session, any library escape site *not* in
the baseline fails the session, so new numpy escapes cannot creep in while
the existing ones are migrated. Enforcement checks the sites recorded by the
escape logger, so it is only meaningful with ``CCDPROC_LOG_ARRAY_ESCAPES=1``
and a non-numpy ``CCDPROC_ARRAY_LIBRARY`` (on numpy there are no foreign
arrays to escape); the session errors out if these are not set. The
``enforce`` tox factor sets the required test-tooling variables and composes
with a backend factor, e.g.::

    tox -e py312-alldeps-dask-enforce

To regenerate the baseline, run the *full* test suite from the source tree
(not under tox, which runs the tests against an installed copy of the
package from a temporary directory) with all three of
``CCDPROC_WRITE_ESCAPE_BASELINE=1``, ``CCDPROC_LOG_ARRAY_ESCAPES=1`` and a
non-numpy backend set -- write mode errors out if any of them is missing::

    CCDPROC_ARRAY_LIBRARY=dask CCDPROC_LOG_ARRAY_ESCAPES=1 \
        CCDPROC_WRITE_ESCAPE_BASELINE=1 pytest

The file is rewritten from the escapes actually observed during the run, so
a partial run (a subset of the tests) silently drops the entries for code
that was not exercised -- always regenerate over the whole suite.
Hand-written reasons on entries that are still observed are preserved.

If the enforce CI job (e.g. ``py312-alldeps-dask-enforce``) fails on your
pull request, look for the "NEW escapes" list in the ``ccdproc array-API
escape baseline`` section of the pytest terminal summary. For each new site,
either fix the call site so the data stays in the array-API world
(preferred), or -- if the escape is a deliberate boundary with a numpy-only
dependency -- add the new ``<file> <function> <coercion>`` line, with a
reason, to ``ccdproc/tests/array_escape_baseline.txt``. The same summary
also lists baseline entries that were not hit this run; delete them if your
change removed the escape.

Reading the strict CI signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ubuntu-py313-strict`` job in the main CI matrix runs the test suite
against `array-api-strict`_ on a non-default device, a CPU-only proxy for
GPU-style device behavior. It is a regular matrix job: a test failure fails
the job and the pull-request checks rollup, like any other backend. A
failure there usually means a numpy-ism (a numpy-only method or type, a
missing ``device=``) crept into a code path that the more permissive
backends accept silently; reproduce it locally with ``tox -e strict``.

What limitations should I be aware of?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+ The NaN-aware reductions ``nansum``, ``nanmean``, ``nanstd`` and
  ``nanmedian`` are not part of the array API, but most array libraries do
  provide them. When combining images, `ccdproc`_ uses the versions from
  `bottleneck`_ for `numpy`_ arrays if `bottleneck`_ is installed; otherwise
  it uses the ones the selected array library provides. If that library has
  none (for example ``array-api-strict``), `ccdproc`_ falls back to
  implementations written purely in terms of the array API standard. They
  are correct but slower: the sum, mean and standard deviation take a few
  extra passes over the data, and the median is sort-based (O(n log n)
  along the combination axis rather than O(n)). The fallbacks promote
  integer and boolean input to the library's default real floating dtype
  and yield NaN silently for slices that are entirely NaN.
+ ``median`` is not part of the array API either. ``subtract_overscan``
  uses the ``median`` of the selected array library when there is one, and
  otherwise the same sort-based fallback, propagating NaNs as
  ``numpy.median`` does.
+ ``sigma_func``, the default uncertainty estimate of ``median_combine``,
  is a median absolute deviation. For `numpy`_ arrays it calls
  ``astropy.stats.median_absolute_deviation``, which is numpy-only; for
  every other array library it uses a version written purely in terms of
  the array API standard, preferring the library's own ``median``/
  ``nanmedian`` and falling back to the sort-based medians above only when
  the library has neither, so the extra sort cost and the silence on
  all-NaN slices apply only there.
+ ``Combiner.sigma_clipping`` uses ``astropy.stats.sigma_clip`` only for
  `numpy`_ arrays. For any other array library it uses an implementation
  written in terms of the array API standard that reproduces astropy's
  result up to floating-point rounding of the reductions: a value lying
  exactly on a bound can be classified differently from astropy.
  Astropy-only options such as ``grow`` are not available there and raise
  ``TypeError``; ``masked`` and ``return_bounds`` are not accepted by
  ``sigma_clipping`` at all, on any array library, because the wrapper
  always asks astropy for the mask itself. On a lazy library such as
  `dask`_, prefer an integer ``maxiters``: ``maxiters=None`` has to
  compute the data after every iteration to find out whether anything
  else was rejected.
+ ``block_reduce``, ``block_average`` and ``block_replicate`` call
  ``astropy.nddata``, which is numpy-only, for `numpy`_ arrays and an
  implementation written purely in terms of the array API standard for
  every other array library, so the result stays in the array library you
  passed in. The two differ only in dtype handling. First,
  ``block_average`` and ``block_replicate(..., conserve_sum=True)`` promote
  integer and boolean input to the array library's default real floating
  dtype, because some libraries (``array-api-strict``) refuse to divide or
  average integers rather than promoting them. `numpy`_ returns ``float64``
  in both cases anyway, so this differs only for a library whose default
  real dtype is not ``float64``. Second, on non-NumPy backends
  ``block_replicate(..., conserve_sum=True)`` preserves a real floating
  input's dtype (``float32`` stays ``float32``), which ``astropy.nddata``
  only does from the fix for
  `astropy/astropy#20360 <https://github.com/astropy/astropy/issues/20360>`_
  (astropy 7.2.3 and 8.0.2); earlier versions upcast ``float32`` and
  ``float16`` input to ``float64`` there. Note also
  that the default ``block_reduce`` sum of integer or boolean input uses
  the array library's default integer width, which is ``int32`` on `jax`_
  unless 64-bit mode is enabled, so large blocks can overflow there. All
  three functions need a fully known shape, so a `dask`_ array with unknown
  chunk sizes must have ``compute_chunk_sizes()`` called on it first.
+ The local window filters -- ``median_filter``, and the ones inside
  ``cosmicray_median``, ``background_deviation_filter`` and ``ccdmask`` --
  come from `scipy.ndimage`_ for `numpy`_ arrays, which is numpy-only. For
  every other array library they are computed by an implementation written
  purely in terms of the array API standard, which stacks each pixel's
  window along a new axis and reduces over it. It agrees with
  `scipy.ndimage`_ exactly on finite input, but it is markedly more
  expensive: a k-by-k window costs O(k**2 log k**2) per pixel, from a sort,
  against ndimage's O(k**2) selection, and the stack itself holds k**2 copies
  of the image (processed in bands of rows to bound the peak memory).
+ Those filters promote integer input to the library's default real
  floating dtype; `scipy.ndimage`_ keeps an integer dtype. Only
  ndimage's ``'reflect'`` and ``'nearest'`` boundary modes are
  implemented, and on a non-`numpy`_ array ``median_filter`` accepts only
  ``size`` and ``mode`` -- ``footprint``, ``origin``, ``output``, ``cval``
  and ``axes`` raise ``TypeError`` naming the argument. Convert the data
  to `numpy`_ to use `scipy.ndimage`_'s full interface.
+ The window filters that take an order statistic -- the median and the
  percentile -- **exclude NaNs from a window** and rank among the values
  that remain, where `scipy.ndimage`_ **sorts NaNs in with them**, above
  every real number. Results on data containing NaN therefore differ
  between `numpy`_ and every other array library: ``median_filter`` and
  ``cosmicray_median`` accept such data, and ``ccdmask``'s flat ratio
  routinely contains it, so a ratio with NaN in it can give a slightly
  different mask off `numpy`_. Infinities are ordinary values to both.
  These filters also need a fully known shape, so a `dask`_ array with
  unknown chunk sizes must have ``compute_chunk_sizes()`` called on it
  first.

Which operations run on the CPU?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A few `ccdproc`_ functions are built on libraries that only understand
`numpy`_ arrays. Three of them are:

+ ``wcs_project``, which reprojects through `reproject`_;
+ ``subtract_overscan`` when a ``model`` is given, which fits it with
  `astropy.modeling`;
+ ``cosmicray_lacosmic``, which detects cosmic rays with `astroscrappy`_.

These functions copy their input to host memory, run there, and copy every
array they return -- the data and the mask -- back to the array namespace
and the device of the array you passed in. You never get a `numpy`_ array
back in place of what you handed over, whatever array library you use.

Because that round trip can be expensive -- it computes a lazy `dask`_
array, or moves data off a GPU and back -- each of these functions warns
once per call site with a ``HostCopyWarning``, naming the function that
made the copy. The warning is a subclass of ``AstropyUserWarning`` and is
silenced like any other warning:

.. code-block:: python

    import warnings
    import ccdproc

    warnings.filterwarnings("ignore", category=ccdproc.HostCopyWarning)

`numpy`_ input is never copied and never warns.

There is one deliberate exception to the warning. ``combine`` with an
``output_file`` writes the combined image through `astropy.io.fits`, which
also needs a host copy, but nothing from that copy comes back into your
pipeline -- the combined image ``combine`` returns is still in your array
namespace -- so it copies silently.

`ccdproc`_ does not offer a way to turn the conversion off: without it the
operation would simply fail. Nor is a native reimplementation planned;
wcslib, `astroscrappy`_'s C code and astropy's fitters are outside the
scope of this project.

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
.. _astroscrappy: https://astroscrappy.readthedocs.io/en/latest/
.. _array-api-strict: https://data-apis.org/array-api-strict/
.. _bottleneck: https://bottleneck.readthedocs.io/en/latest/
.. _ccdproc: https://ccdproc.readthedocs.io/en/latest/
.. _cupy: https://docs.cupy.dev/en/stable/
.. _dask: https://docs.dask.org/en/stable/
.. _jax: https://docs.jax.dev/en/latest/index.html
.. _numpy: https://numpy.org/doc/stable/reference/array_api.html
.. _reproject: https://reproject.readthedocs.io/en/stable/
.. _scipy.ndimage: https://docs.scipy.org/doc/scipy/reference/ndimage.html
.. _sparse: https://sparse.pydata.org/en/stable/
