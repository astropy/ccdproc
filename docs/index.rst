.. the "raw" directive below is used to hide the title in favor of
   just the logo being visible
.. raw:: html

    <style media="screen" type="text/css">
      h1 {display:none;}
    </style>

=======
ccdproc
=======

.. raw:: html

   <img src="_static/ccdproc_banner.svg" onerror="this.src='_static/ccdproc_banner.png'; this.onerror=null;" width="495"/>

.. only:: latex

    .. image:: _static/ccdproc_banner.pdf

**Ccdproc** is is an `Astropy`_ `affiliated package
<https://www.astropy.org/affiliated/index.html>`_  for basic data reductions
of CCD images. It provides the essential tools for processing of CCD images
in a framework that provides error propagation and bad pixel tracking
throughout the reduction process.

.. Important::
    If you use `ccdproc`_ for a project that leads to a publication,
    whether directly or as a dependency of another package, please
    include an :doc:`acknowledgment and/or citation <citation>`.

Getting started
---------------

.. toctree::
    :maxdepth: 1

    install.rst
    overview.rst
    citation.rst
    contributing.rst
    conduct.rst


.. toctree::
    :maxdepth: 3

    ccdproc.rst

.. toctree::
    :maxdepth: 1

    authors_for_sphinx
    changelog
    license
