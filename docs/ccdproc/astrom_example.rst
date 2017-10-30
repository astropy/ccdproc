.. _astrom_example:

Astrometry Tuturial
===================

.. note::

    The astrometry package is still under development and no guarantee
    of the quality of the results is given at this time.  The code
    here is still being tested and any feedback would be appreciated.

An important aspect in calibrating imaging data is the positions of the
objects in the frame.  In this tutorial, we work through the steps using
various ``astropy`` tools to measure the astrometry in your image.
Calculating accurate positions in your images requires a
set up stars with known positions matched to the positions of stars
in the image.

Our example here assumes that the pointing and field of view of the
observations are known.  If not, it may be better to determine the
astrometry using something like `astrometry.net <http://astrometry.net/>`_.

Reference Sample
----------------

Several different reference samples are availabe in the vizier catalog.
For this example, we will use stars from the GAIA DR1 data release [1]_.
These sources can be downloaded using the astroquery package [2]_.  We
assume that we will be trying to calibrate observations with a known
center and field of view:

.. code::

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier
    center = SkyCoord(12.717536711448934, -27.778460891 ,unit=('deg','deg'))
    viz_gaia = Vizier(catalog='I/337/gaia')
    viz_gaia.ROW_LIMIT=-1
    gaia = viz_gaia.query_region(coordinates=center, radius='6 arcmin')[0]

This will provide a table with coordinates of known stars in that field of view.

Object Positions
----------------

A number of different tools are available to determine the position of objects
in an image.   In this case, the photutils [3]_ package will be used to determine
the position of objects in our image.  For our purposes of matching data, we
extract only the brightest 20 stars in the field to use for matching with
the catalog, initially.

.. code::

    from ccdproc import CCDData
    from photutils import daofind
    from astropy.stats import mad_std
    ccd = CCDData.read('sgpR.fits', unit='electron')
    bkg_sigma = mad_std(ccd.data)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    sources = daofind(ccd.data)
    bright = sources[sources['mag'].argsort()[:20]]

The current algorithm in photutils will result in duplicate objects in the list,
especially for saturated stars or stars with diffraction spikes.   These should
be removed from the list before proceeding.   A convenience function is provided
in `ccdproc.astrometry`:

.. code::

    from ccdproc import astrometry as astrom
    x, y = bright['xcentroid'], bright['ycentroid']
    x, y = astrom.remove_duplicates(x, y, tolerance=10)

This will remove all detections within 10 pixels of another detection, but leaving the first
detection.

Matching Coordinates
--------------------

Once the two coordinate lists have been produced, the next step is to match
the two lists.   The function `ccdproc.astrometry.match_by_triangle` will
produced a match list of coordinates.  The program first creates a series of
triangles and looks for objects that have matching ratios to the two legs
of the triangle created from the brightest object.  Further filtering can
be done by matching that brightest triangle to a descending order of objects
before doing a solution that matches all objects in the field.

.. code::

    r, d = gaia['RA_ICRS'], gaia['DE_ICRS']
    matches = astrom.match_by_triangle(
        x, y, r, d, n_limit=30, tolerance=0.02,
        clean_limit=5, match_tolerance=0.02, m_init=models.Polynomial2D(1),
        fitter = fitting.LinearLSQFitter())n_groups=5)
    i1, i2 = matches

This will produced a match catalog of stars that can be used to determine
the transformation between the (x,y) -> (r,d).


Deeper Matched Catalog
----------------------

Once an initial match is found, it is much easier to match a larger
number of objects based on the initial transformation based on those
coordinates.  The following command produces a list of indices that
match between the two coordinate frames based on matched coordinates
in the data set:

.. code::

    idp, idx = astrom.match_by_fit(
        x, y, r.data, d.data, i1, i2, tolerance=5*u.arcsec)

The match indices provide the full list of objects that match
within the tolerance.  This can then be used to calculate the
World Coordinate System (WCS) information for the image.

Setting the WCS
---------------

Finally, from the set of matched frames and the image transformation, set
the WCS for the image:

.. code::

    wcs = astrom.create_wcs_from_fit(x, y, r.data, d.data, idp, idw)
    ccd.wcs = wcs

The function calculates a WCS based on a linear transformation of the
coordinates.  This does not currently include any higher distortion terms
and so for the most accurate results, any distortion should be already
removed from the image.


References
----------

.. [1] Gaia Collaboration et al. 2016, Astronomy and Astrophysics, 595, A2

.. [2] Ginsburg, A. et al. 2017, astropy/astroquery: v0.3.6 with fixed 
       license., Zenodo, http://doi.org/10.5281/zenodo.826911

.. [3] Bradley, L. et al. 2016, astropy/photutils: v0.3. Zenodo.,
       http://doi.org/10.5281/zenodo.164986
