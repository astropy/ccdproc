************
Installation
************

Requirements
============

Ccdproc requires Python 3.12 or later and the following packages:

- `Astropy`_ v7.0 or later
- `NumPy <https://numpy.org/>`_ v2.2 or later
- `SciPy <https://scipy.org/>`_ v1.15 or later
- `array-api-compat <https://data-apis.org/array-api-compat/>`_ v1.12 or later
- `array-api-extra <https://data-apis.org/array-api-extra/>`_ v0.7 or later
- `astroscrappy <https://github.com/astropy/astroscrappy>`_ v1.2 or later
- `reproject <https://github.com/astropy/reproject>`_ v0.14 or later

The minimum supported versions of Python, NumPy and SciPy follow
`SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_: support for a
Python version is dropped three years after its release, and support for a
release of a core package two years after its release. Ccdproc applies the
same two-year window to Astropy.

One easy way to get these dependencies is to install a python distribution
like `anaconda`_.

Installing ccdproc
==================

Using pip
-------------

To install ccdproc with `pip <https://pip.pypa.io/en/latest/>`_, simply run::

    pip install ccdproc

Using conda
-------------

To install ccdproc with `anaconda`_, run::

    conda install -c conda-forge ccdproc


Building from source
====================

Obtaining the source packages
-----------------------------

Source packages
^^^^^^^^^^^^^^^

The latest stable source package for ccdproc can be `downloaded here
<https://pypi.org/project/ccdproc/#files>`_.

Development repository
^^^^^^^^^^^^^^^^^^^^^^

The latest development version of ccdproc can be cloned from github
using this command::

   git clone git://github.com/astropy/ccdproc.git

Building and Installing
-----------------------

To build ccdproc (from the root of the source tree)::

    python setup.py build

To install ccdproc (from the root of the source tree)::

    pip install .

To set up a development install in which changes to the source are immediately
reflected in the installed package (from the root of the source tree)::

    pip install -e .

Testing a source code build of ccdproc
--------------------------------------

The easiest way to test that your ccdproc built correctly (without
installing ccdproc) is to run this from the root of the source tree::

    python setup.py test

.. _anaconda: https://anaconda.com/
