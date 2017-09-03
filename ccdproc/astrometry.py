# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the combiner class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import itertools as it
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.modeling import models
from astropy.modeling import fitting
from astropy.stats import median_absolute_deviation

import math

__all__ = ['remove_duplicates', 'distance', 'distance_ratios']

def remove_duplicates(x, y, tolerance):
    """Remove duplicates within a certain tolerance from a pair of coordiantes

    Parameters
    ----------
    x: ~numpy.ndarray
        x-position of objects

    y: ~numpy.ndarray
        y-position of objects

    tolerance: float
        tolerance for removal of dupliactes

 
    """
    mask = np.ones(len(x), dtype=bool)

    for i in range(len(x)):
        if mask[i]:
            d = ((x - x[i])**2 + (y - y[i])**2)**0.5
            j = np.where((d<tolerance))
            mask[j] = False
            mask[i] = True

    return x[mask], y[mask]


def distance(x1, y1, x2, y2):
    """Calcluate the distance between points

    
    Parameters
    ----------
    x1: float or ~numpy.ndarray
        x-position of objects

    y1: float or ~numpy.ndarray
        y-position of objects

    x2: float or ~numpy.ndarray
        x-position of objects

    y2: float or ~numpy.ndarray
        y-position of objects

    Return
    ------
    d: float or ~numpy.ndarray
        Distance between x1,y1 and x2,y2
    """
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def _get_index(z, n_stars):
    """Return the indices from a set of permutations of length n_stars
    
    For example, it will return the indices of the three stars
    corresponding to the output from distance ratio

    Parameters
    ----------
    z: int
        Index from 1-d permuation array 
  
    n_stars: int
        Number of stars in the original array

    Returns
    -------
    indices: tuple
       i, j, k indices 
    """
    count=0
    for i, j, k in it.permutations(range(0,n_stars),3):       
        if count==z: return i, j, k
        count += 1

def triangle_angle(a, b, c):
    """Given the length of three sides, calculate
       the angle of side a
   
    Parameters
    ----------
    a: float
       Side of a triangle

    b: float
       Side of a triangle

    c: float
       Side of a triangle


    Returns
    -------
    theta: float
       Angle of a triangle in radians
    """
    return np.arccos((b**2+c**2-a**2)/(2*b*c))


def calc_triangle(x, y, i, j, k):
    """From three points, create a description of a triangle

    Parameters
    ----------
    x: ~numpy.ndarray
        Array of x-positions
        
    y: ~numpy.ndarray
        Array of y-positions
        
    i: int
        index of first object
        
    j: int
        index of second object
        
    k: int
        index of third object
    
    Returns
    -------
    sides: ~numpy.ndarray
        Array of the length of each side of the triangle normalized to 
        the longest side
        
    angles: ~numpy.ndarray
        Array of the angles of each vertices of the triangle
        
    order: ~numpy.ndarray
        Order of the shortest to longest side 

    """
    d1 = distance(x[i], y[i], x[j], y[j])
    d2 = distance(x[i], y[i], x[k], y[k])
    d3 = distance(x[j], y[j], x[k], y[k])
    sides = np.array([d1,d2,d3])
    a1 = triangle_angle(d1, d2, d3)
    a2 = triangle_angle(d2, d1, d3)
    a3 = triangle_angle(d3, d1, d2)
    angles = np.array([a1, a2, a3])
    order = sides.argsort()
    return sides/sides.max(), angles, order

def _calc_ratio(x, y, i, j, k):
    """Calculate the ratio of the distance between the vertex at index i to 
    points j and k. 

    Parameters
    ----------
    x: ~numpy.ndarray
        x-position of objects

    y: ~numpy.ndarray
        y-position of objects

    i: int
        Index of the vertex

    j: int
        Index for the first side

    k: int
        Index for the second side

    Returns
    -------
    ratio: float
        Ratio between the distance from point i to j to the distance between i
        and k

    """
    d1 = distance(x[i], y[i], x[j], y[j])
    d2 = distance(x[i], y[i], x[k], y[k])
    if d2==0: ratio = np.nan
    return d1/d2 

def distance_ratios(x, y):
    """For all permutations of points in x,y calculatio the ratio between
    the distance from one point to two other points.

    Parameters
    ----------
    x: ~numpy.ndarray
        x-position of objects

    y: ~numpy.ndarray
        y-position of objects

    Returns
    -------
    ratio: ~numpy.ndarray
        flat array of all the permutations of the ratio between the distances

    """
    ratio = []
    for i, j, k in it.permutations(range(len(x)),3):
        ratio.append(_calc_ratio(x, y, i, j, k))
    
    return np.array(ratio)


def match_by_triangle(x, y, ra, dec, n_limit=30, tolerance=0.02, clean_limit=5,
                      match_tolerance=1*u.arcsec, m_init=models.Polynomial2D(1), 
                      fitter = fitting.LinearLSQFitter()):
    """Use triangles with in the set of objects to find the matched coordinates

    The algorithm creates a set of triangles representing every set of three 
    stars in the input list of objects.  The first step is to limit the 
    input list of right ascension and declination to the number of objects
    provided by n_limit.   Then, a triangle is created for each unique
    triplet in the remaining list.  The triangle is described by the
    length of each side, the angle of each vertices, and a ranking of each 
    side from shortest to longest.   The length of the sides is normalized
    such that the longest side has unit length.   

    In order of the given x,y coordinates, a list of matched coordinates is 
    created.   A triangle is created for each  unique triplet in the x,y coordinates 
    and the length of the normalized sides  and angle of the verticies 
    is compared to all of the triangles created for the input ra,dec coordinates.
    A match is considered to occur if the absolute difference between the 
    sides and the angles between the two triangles is less than the tolerance.

    If clean_limt=0, the indices of each triplet that matches is added to a
    list that includes the indices of the x,y and ra,dec sources.  This list 
    of indices is then returned to the user.

    If clean_limit>0, then a second step is implement to verify the match.  Each
    triplet that satisfies the tolerance limit is then passed onto a second 
    function, `match_by_fit`, that uses the three match sources to calculate 
    a transformtion, and then matches the entire list of sources based on the 
    transformation and how close it is to another source within match_tolerance.
    If a triplet results in a number of matches greater than clean_limit, then 
    the indices of the matched triplet are returned. 

    Parameters
    ----------
    x: ~numpy.ndarray
        x-position of objects

    y: ~numpy.ndarray
        y-position of objects

    ra: ~numpy.ndarray
        RA position of objects

    dec: ~numpy.ndarray
        DEC position of objects

    n_limit: int
        Limit on the number of objects to match

    tolerance: float
        Tolerance for matching ratio of distances

    clean_limit: int 
        Numbe of stars required for a match.  Set to zero to return 
        all stars whose ratios satisify the tolerance limit. 

    match_tolerance: ~astropy.Quantity
        Tolerance for when matching all sources based on their distance.

    m_init: astropy.modeling.models
        A model instance for describing the transformation between coordinate
        systems.  It should be a FittableModel2D instance. 

    fitter: astropy.modeling.fitting
        A fitting routine for fitting the transformations. 

    Returns
    -------
    matches: list
        If clean_limit=0, this returns a list of inices for all triplets of stars 
        with side and angle ratios belwo the tolerance.   If clean_limit>0, then
        this returns the indices of the first match to also satisfy the 
        clean_limit requirement.  

    """
    
    # shortent the ra lists if necessary
    if len(ra)>n_limit:
        r = ra[:n_limit]
        d = dec[:n_limit]
    else:
        r = ra
        d = dec
        
    
    # create the potential list of matches
    world_ratio = {}
    for i, j, k in it.combinations(range(len(r)),3):
            world_ratio[(i,j,k)]=calc_triangle(r, d, i, j, k)
                                   
    
    matches = []
    for i, j, k in it.combinations(range(len(x)),3):
        base = calc_triangle(x, y, i, j, k)
        for key in world_ratio:
            tri = world_ratio[key]  
            if np.allclose(base[0][base[2]], tri[0][tri[2]], atol=tolerance) and np.allclose(base[1][base[2]], tri[1][tri[2]], atol=tolerance):
                idp, idx = match_by_fit(x, y, r, d, (i,j,k), key, match_tolerance, 
                                    m_init=m_init, fitter = fitter)  
                if clean_limit==0:
                    matches.append([(i,j,k), key])
                elif len(idp)>clean_limit:
                    return [(i,j,k), key]
                
    return matches


def match_by_fit(x, y, ra, dec, idp, idw, tolerance, 
                 m_init=models.Polynomial2D(1), 
                 fitter = fitting.LinearLSQFitter()):
    """Match coordinates after determing the transformation

    Given two sets of coodrindates, determine the number of matches between the two by determining
    a transformaiton based on a set of  matched objects as specified by their indices.

    Parameters
    ----------
    x: ~numpy.ndarray
        x-position of objects

    y: ~numpy.ndarray
        y-position of objects

    ra: ~numpy.ndarray
        RA position of objects in degrees

    dec: ~numpy.ndarray
        DEC position of objects in degrees

    idp: ~numpy.ndarray
        Indices of x,y that match with objects in ra,dec

    idw: ~numpy.ndarray
        Indices of ra,dec that match with objects in x,y

    tolerance: ~astropy.Quantity
        Tolerance for when matching all sources based on their distance.

    m_init: astropy.modeling.models
        A model instance for describing the transformation between coordinate
        systems.  It should be a FittableModel2D instance. 

    Returns
    -------
    match_idx:  ~numpy.ndarray
        Indices of x,y that match with objects in ra,dec

    match_idw: ~numpy.ndarray
        Indices of ra,dec that match with objects in x,y

    """
    r_fit = fitter(m_init, x[[idp]], y[[idp]], ra[[idw]])
    d_fit = fitter(m_init, x[[idp]], y[[idp]], dec[[idw]])
            
    # apply to coordinates and determine the matches
    c = SkyCoord(ra*u.degree, dec*u.degree)
    c2 = SkyCoord(r_fit(x,y)*u.degree, d_fit(x,y)*u.degree)
    idx, d2d, d3d = c2.match_to_catalog_sky(c)
    
    return idx[d2d<tolerance], np.where(d2d<tolerance)[0], 
    

