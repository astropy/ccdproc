# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the combiner class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy import ma
from .ccddata import CCDData

from astropy.stats import median_absolute_deviation
from astropy.nddata import StdDevUncertainty
from astropy import log

import math

__all__ = ['remove_duplicates']

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

