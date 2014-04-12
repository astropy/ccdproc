"""
Define untility functions and classes for ccdproc
"""


def slice_from_string(string):
    """
    Convert a string to a tuple of slices

    Parameters
    ----------

    string : str
        A string that can be converted to a slice.

    Returns
    -------

    slice_tuple : tuple of slice objects
        A tuple able to be used to index a numpy.array
    """
    no_space = string.replace(' ', '')

    if not no_space:
        return ()

    if not (no_space.startswith('[') and no_space.endswith(']')):
        raise ValueError('Slice string must be enclosed in square brackets.')

    no_space = no_space.strip('[]')
    string_slices = no_space.split(',')
    slices = []
    for string_slice in string_slices:
        slice_args = [int(arg) if arg else None
                      for arg in string_slice.split(':')]
        a_slice = slice(*slice_args)
        slices.append(a_slice)

    return tuple(slices)
