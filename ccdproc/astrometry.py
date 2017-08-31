# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the combiner class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import itertools as it
import numpy as np

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


def match_by_triangle(x, y, r, d, n_groups=2, n_limit=30, limit=0.02):
    """Use triangles with in the set of objects to find the matched coordinates

    Parameters
    ----------
    x: ~numpy.ndarray
        x-position of objects

    y: ~numpy.ndarray
        y-position of objects

    r: ~numpy.ndarray
        RA position of objects

    d: ~numpy.ndarray
        DEC position of objects

    n_groups: int
        Number of triangles to use for the matching

    n_limit: int
        Limit on the number of objects to match

    limit: float
        Tolerance for matching ratio of distances

    Returns
    -------
    match_list: ~numpy.ndarray
        flat array of all the permutations of the ratio between the distances 

    """

    # first step is to find possible matches based on 
    # triangles that have the same ratio between their 
    # legs.  Due to the possibiity of centroiding errors
    # or missing data sets, we follow a two step process
    # The second step is to then match objects which 
    # are close to gether
    
    w_ratio =  distance_ratios(r, d)
    for i in range(n_groups):
        p = _calc_ratio(x, y, 0, 1, i+2)
        diff = abs(w_ratio - p)
        guess=[]
        for m in np.where(diff<limit)[0]:
            n=_get_index(m, len(r))
            guess.append([n[0], n[1], n[2]])
            
        if guess_all is None: 
            guess_all = guess.copy() 
        else:
            for g in guess_all:
                if not ((g[0], g[1]) in [(z[0], z[1]) for z in guess]):
                    guess_all.remove(g)
