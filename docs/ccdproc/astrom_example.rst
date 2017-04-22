.. _astrom_example:

Astrometry Tuturial 
===================

.. note::

    This is not intended to be an example for measuring the astrometry
    in a specific type of data.  While performing the steps presented 
    here may be the correct way to measure the astrometry 
    in some cases, it is not correct in all cases.

An important aspect in calibrating imaging data is the positions of the 
objects in the frame.  In this tutorial, we work through the steps using
various `astropy` tools to measure the astrometry in your image.   
Calculating accurate positions in your images requires a 
set up stars with known positions matched to the positions of stars
in the image.  

Our example here assumes that the pointing and field of view of the
observations are known.  If not, it may be better to determine the 
astrometry using something like [astrometry.net](http://astrometry.net/).

Reference Sample
----------------

Several different reference samples are availabe in the vizier catalog.  
For this example, we will use stars from the GAIA DR1 data release [1]_. 
These sources can be downloaded using the astroquery package [2]_.  We
assume that we will be trying to calibrate observations with a known
center and field of view.  


    >>> from astropy import units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from astroquery.vizier import Vizier
    >>> center = SkyCoord(12.717536711448934, -27.778460891 ,unit=('deg','deg'))
    >>> viz_gaia = Vizier(catalog='	I/324/igsl3')
    >>> viz_gaia.ROW_LIMIT=-1
    >>> gaia = viz_gaia.query_region(coordinates=center, radius='6 arcmin')[0]

This will provide a table with coordinates of known stars in that field of view. 

Object Positions
----------------

A number of different tools are available to determine the position of objects
in an image.   In this case, the photutils [3]_ package will be used to determine
the position of objects in our image.  For our purposes of matching data, we
extract only the brightest 20 stars in the field to use for matching with
the catalog, initially. 

    >>> from ccdproc import CCDData
    >>> from photutils import daofind
    >>> from astropy.stats import mad_std
    >>> ccd = CCDData.read('sgpR.fits', unit='electron')
    >>> bkg_sigma = mad_std(ccd.data) 
    >>> sources = daofind(ccd.data, fwhm=4., threshold=5.*bkg_sigma)
    >>> bright = sources[sources['mag'].argsort()[:20]]

The current algorithm in photutils will result in duplicate objects in the list, 
especially for saturated stars or stars with diffraction spikes.   These should
be removed from the list before proceeding.   A convenience function is provided
in `ccdproc.astrom`:
 
    >>> from ccdproc import astrometry as astrom
    >>> x, y = astrom.remove_duplicates(x, y, tol=10)

This will remove all detections within 10 pixels of another detection, but leaving the first
detection. 

Matching Coordinates
--------------------

Once the two coordinate lists have been produced, the next step is to match 
the two lists.   The function `ccdproc.astrom.match_coordinates` will 
produced a match list of coordinates.  The program first creates a series of
triangles and looks for objects that have matching ratios to the two legs
of the triangle created from the brightest object.  Further filtering can 
be done by matching that brightest triangle to a descending order of objects
before doing a solution that matches all objects in the field.   

    >>> r, d = gaia['RAJ2000'], gaia['DEJ2000']
    >>> matches = astrom.match_by_triangle(x, y, r, d, n_triangle=5)

This will produced a match catalog of stars that can be used to determine
the transformation between the (x,y) -> (r,d).


Deeper Matched Catalog
----------------------

Once an initial transformation is found, it is much easier to match a larger
number of objects

    >>> deep_matches = astrom.match_by_distance(x, y, r, d, transform=t, tol=5)

Setting the WCS
---------------

Finally, from the set of matched frames and the image transformation, set
the wcs for the image:

    >>> wcs = astrom.calculate_wcs(deep_matches, function)
    >>> ccd.wcs = wcs


References
----------

.. [1] 

.. [2]
