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
against `array-api-strict`_, but it is marked ``continue-on-error``, so it
always shows green in the pull-request checks rollup regardless of the test
outcome. The real result is posted as a separate check run, also named
``ubuntu-py313-strict``, by the "Strict array API status" workflow. Its
conclusion is:

+ ``success`` -- the strict test suite passed;
+ ``neutral``, with a "failing (expected)" message -- the strict suite
  failed, which is expected while known array-API bugs remain;
+ ``failure`` -- the job itself broke (an infrastructure problem rather
  than the expected test failures).

Because that workflow is triggered by ``workflow_run``, it executes from the
repository's default branch: the separate check only appears once
the workflow file exists on the default branch, and changes to it take
effect only after they are merged.

What limitations should I be aware of?
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
.. _array-api-strict: https://data-apis.org/array-api-strict/
.. _bottleneck: https://bottleneck.readthedocs.io/en/latest/
.. _ccdproc: https://ccdproc.readthedocs.io/en/latest/
.. _cupy: https://docs.cupy.dev/en/stable/
.. _dask: https://docs.dask.org/en/stable/
.. _jax: https://docs.jax.dev/en/latest/index.html
.. _numpy: https://numpy.org/doc/stable/reference/array_api.html
.. _sparse: https://sparse.pydata.org/en/stable/
