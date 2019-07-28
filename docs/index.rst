.. the "raw" directive below is used to hide the title in favor of
   just the logo being visible
.. raw:: html

    <style media="screen" type="text/css">
      h1 {display:none;}
    </style>

========
 CCDPROC
========

.. raw:: html

   <img src="_static/ccdproc_banner.svg" onerror="this.src='_static/ccdproc_banner.png'; this.onerror=null;" width="495"/>

.. only:: latex

    .. image:: _static/ccdproc_banner.pdf

**Ccdproc** is is an `Astropy`_ `affiliated package
<https://www.astropy.org/affiliated/index.html>`_  for basic data reductions
of CCD images. It provides the essential tools for processing of CCD images
in a framework that provides error propagation and bad pixel tracking
throughout the reduction process.

.. toctree::
  :maxdepth: 2

  ccdproc/install.rst

.. toctree::
  :maxdepth: 3

  ccdproc/index.rst

.. toctree::
  :maxdepth: 1

  authors_for_sphinx
  changelog
  license
