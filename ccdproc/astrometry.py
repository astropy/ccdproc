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
    if d2==0: d2 = np.inf
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

