************
Installation
************

Requirements
============

Ccdproc has the following requirements:

- `Astropy`_ v2.0 or later
- `NumPy <http://www.numpy.org/>`_
- `SciPy <https://www.scipy.org/>`_
- `scikit-image <http://scikit-image.org/>`_
- `astroscrappy <https://github.com/astropy/astroscrappy>`_
- `reproject  <https://github.com/astrofrog/reproject>`_

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
