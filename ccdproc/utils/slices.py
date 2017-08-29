# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Define utility functions and classes for ccdproc
"""

__all__ = ["slice_from_string"]


def slice_from_string(string, fits_convention=False):
    """
    Convert a string to a tuple of slices.

    Parameters
    ----------

    string : str
        A string that can be converted to a slice.

    fits_convention : bool, optional
        If True, assume the input string follows the FITS convention for
        indexing: the indexing is one-based (not zero-based) and the first
        axis is that which changes most rapidly as the index increases.

    Returns
    -------

    slice_tuple : tuple of slice objects
        A tuple able to be used to index a numpy.array

    Notes
    -----

    The ``string`` argument can be anything that would work as a valid way to
    slice an array in Numpy. It must be enclosed in matching brackets; all
    spaces are stripped from the string before processing.

    Examples
    --------

    >>> import numpy as np
    >>> arr1d = np.arange(5)
    >>> a_slice = slice_from_string('[2:5]')
    >>> arr1d[a_slice]
    array([2, 3, 4])
    >>> a_slice = slice_from_string('[ : : -2] ')
    >>> arr1d[a_slice]
    array([4, 2, 0])
    >>> arr2d = np.array([arr1d, arr1d + 5, arr1d + 10])
    >>> arr2d
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> a_slice = slice_from_string('[1:-1, 0:4:2]')
    >>> arr2d[a_slice]
    array([[5, 7]])
    >>> a_slice = slice_from_string('[0:2,0:3]')
    >>> arr2d[a_slice]
    array([[0, 1, 2],
           [5, 6, 7]])
    """
    no_space = string.replace(' ', '')

    if not no_space:
        return ()

    if not (no_space.startswith('[') and no_space.endswith(']')):
        raise ValueError('Slice string must be enclosed in square brackets.')

    no_space = no_space.strip('[]')
    if fits_convention:
        # Special cases first
        # Flip dimension, with step
        no_space = no_space.replace('-*:', '::-')
        # Flip dimension
        no_space = no_space.replace('-*', '::-1')
        # Normal wildcard
        no_space = no_space.replace('*', ':')
    string_slices = no_space.split(',')
    slices = []
    for string_slice in string_slices:
        slice_args = [int(arg) if arg else None
                      for arg in string_slice.split(':')]
        a_slice = slice(*slice_args)
        slices.append(a_slice)

    if fits_convention:
        slices = _defitsify_slice(slices)

    return tuple(slices)


def _defitsify_slice(slices):
    """
    Convert a FITS-style slice specification into a python slice.

    This means two things:
    + Subtract 1 from starting index because in the FITS
      specification arrays are one-based.
    + Do **not** subtract 1 from the ending index because the python
      convention for a slice is for the last value to be one less than the
      stop value. In other words, this subtraction is already built into
      python.
    + Reverse the order of the slices, because the FITS specification dictates
      that the first axis is the one along which the index varies most rapidly
      (aka FORTRAN order).
    """

    python_slice = []
    for a_slice in slices[::-1]:
        new_start = a_slice.start - 1 if a_slice.start is not None else None
        if new_start is not None and new_start < 0:
            raise ValueError("Smallest permissible FITS index is 1")
        if a_slice.stop is not None and a_slice.stop < 0:
            raise ValueError("Negative final index not allowed for FITS slice")
        new_slice = slice(new_start, a_slice.stop, a_slice.step)
        if (a_slice.start is not None and a_slice.stop is not None and
            a_slice.start > a_slice.stop):
            # FITS use a positive step index when dimension are inverted
            new_step = -1 if a_slice.step is None else -a_slice.step
            # Special case to prevent -1 as slice stop value
            new_stop = None if a_slice.stop == 1 else a_slice.stop-2
            new_slice = slice(new_start, new_stop, new_step)
        python_slice.append(new_slice)

    return python_slice
