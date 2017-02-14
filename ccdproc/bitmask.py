# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import itertools

import numpy as np

_INTEGER_DTYPE_KINDS = {'u', 'i'}

# Powers of two 2**0 - 2**63
_POWERS = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
    16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824,
    2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736,
    137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552,
    4398046511104, 8796093022208, 17592186044416, 35184372088832,
    70368744177664, 140737488355328, 281474976710656, 562949953421312,
    1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992,
    18014398509481984, 36028797018963968, 72057594037927936,
    144115188075855872, 288230376151711744, 576460752303423488,
    1152921504606846976, 2305843009213693952, 4611686018427387904,
    9223372036854775808
)

# Use signed integers for more controlled behaviour expect for 64 layers then
# only an unsigned may hold the result.
_REPRESENTABLE_FLAGS = {
    'int8': 7, 'int16': 15, 'int32': 31, 'int64': 63,
    'uint8': 8, 'uint16': 16, 'uint32': 32, 'uint64': 64
}

# The following could also be calculated using this statement (except for the
# lower uint types - which should be avoided!)
# {i: min((dtype for dtype, maxsize in REPRESENTABLE_FLAGS.items()
#          if maxsize>=i),
#         key=REPRESENTABLE_FLAGS.get)
#  for i in range(65)}
_LAYERS_TO_DTYPE = {
    0:  'int8',   1: 'int8',   2: 'int8',   3: 'int8',   4: 'int8',
    5:  'int8',   6: 'int8',   7: 'int8',   8: 'int16',  9: 'int16',
    10: 'int16', 11: 'int16', 12: 'int16', 13: 'int16', 14: 'int16',
    15: 'int16', 16: 'int32', 17: 'int32', 18: 'int32', 19: 'int32',
    20: 'int32', 21: 'int32', 22: 'int32', 23: 'int32', 24: 'int32',
    25: 'int32', 26: 'int32', 27: 'int32', 28: 'int32', 29: 'int32',
    30: 'int32', 31: 'int32', 32: 'int64', 33: 'int64', 34: 'int64',
    35: 'int64', 36: 'int64', 37: 'int64', 38: 'int64', 39: 'int64',
    40: 'int64', 41: 'int64', 42: 'int64', 43: 'int64', 44: 'int64',
    45: 'int64', 46: 'int64', 47: 'int64', 48: 'int64', 49: 'int64',
    50: 'int64', 51: 'int64', 52: 'int64', 53: 'int64', 54: 'int64',
    55: 'int64', 56: 'int64', 57: 'int64', 58: 'int64', 59: 'int64',
    60: 'int64', 61: 'int64', 62: 'int64', 63: 'int64', 64: 'uint64'
}


class Bitmask(object):
    """Bitmask representation that can be converted to normal masks.

    Parameters
    ----------
    bitmask : `numpy.ndarray` of integer data type
        The bitmask.

    name_map : sequence or mapping or None, optional
        Provide names for the different flags of the bitmask. This allows
        accessing the flags by name rather than by position or value of the
        bit. If it's a sequence then the names should be listed by increasing
        bitvalue. If it's a mapping the names should be the keys and the
        bit position should be the key.

        For example if ``1`` represents ``bad pixel`` and ``2``
        represents ``hot pixel`` then the ``name_map`` should be either:
        ``["bad pixel", "hot pixel"]`` or ``{"bad pixel": 0, "hot pixel": 1}``

        Default is ``None``.

    Examples
    --------
    Creating an unnamed bitmask::

        >>> import numpy as np
        >>> from ccdproc import Bitmask
        >>> Bitmask(np.arange(10).reshape(2,5))
        Bitmask (
        array([[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]])
        name_map={}
        )

    Or creating a named bitmask::

        >>> Bitmask(np.arange(10),
        ...         name_map=['bad', 'hot', 'artifact', 'low response'])
        Bitmask (
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        name_map=OrderedDict([('bad', 0), ('hot', 1), ('artifact', 2), ('low response', 3)])
        )

        >>> names = {'bad': 0, 'hot': 1, 'artifact': 2, 'low response': 3}
        >>> bm = Bitmask(np.arange(10), name_map=names)
        >>> bm  # doctest: +SKIP
        Bitmask (
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        name_map={'hot': 1, 'low response': 3, 'bad': 0, 'artifact': 2}
        )

    Providing the names may be of help when creating a ``mask``:

        >>> bm.to_mask(bitname=['low response'])
        array([False, False, False, False, False, False, False, False,  True,  True], dtype=bool)
    """
    def __init__(self, bitmask, name_map=None):
        self.data = bitmask
        self.name_map = name_map

    def __repr__(self):
        return ('{self.__class__.__name__} (\n{self.data!r}'
                '\nname_map={self.name_map}\n)'.format(self=self))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        # Check if it looks like a "bitmask", meaning that it should be
        # a `numpy.ndarray` and of integer data type.
        val = np.asarray(val)
        if val.dtype.kind not in _INTEGER_DTYPE_KINDS:
            raise TypeError('"bitmask" must be of integer data type.')
        self._data = val

    @property
    def name_map(self):
        return self._name_map

    @name_map.setter
    def name_map(self, val):
        if val is None:
            val = {}
        elif isinstance(val, (list, tuple)):
            val = collections.OrderedDict(zip(val, itertools.count()))
        elif isinstance(val, dict):
            val = val
        # Do the slow checks (collections) after the more common cases
        elif isinstance(val, collections.Mapping):
            val = val
        elif isinstance(val, collections.Sequence):
            val = collections.OrderedDict(zip(val, itertools.count()))
        else:
            raise TypeError('"name_map" must be a list/tuple/dict or another '
                            'type of sequence/mapping, not a {}.'
                            .format(type(val)))
        self._name_map = val

    @classmethod
    def from_boolarray(cls, boolarray):
        """Convert a boolean array representation of a bitmask to an integer
        bitmask.

        Parameters
        ----------
        boolarray : `numpy.ndarray` of boolean dtype with at least 2 dimensions
            The boolean array to convert to a ``bitmask``. The function assumes
            that the first dimension of the ``boolarray`` represents the
            different flags.

        Returns
        -------
        bitmask : `numpy.ndarray` of integer dtype
            The bitmask representation of the ``boolarray``. It will have the
            same shape as ``boolarray.shape[1:]``.

        See also
        --------
        to_boolarray
        ccdproc.boolarray_to_mask
        """
        boolarray = _prep_boolarray(boolarray)
        # Number of layers (first dimension)
        num = boolarray.shape[0]
        # Choose the minimal dtype to represent the bitmask
        dtype = _LAYERS_TO_DTYPE[num]
        bitmask = np.zeros(boolarray.shape[1:], dtype=dtype)
        # Calculate the bitmask, note that 1 << exp is equivalent to
        # 2 ** exp and combine it with bitwise "or"!
        for exp in range(num):
            bitmask |= boolarray[exp] << exp
        return cls(bitmask)

    def to_boolarray(self, num=None):
        """Convert an integer bitmask to a boolean array representation.

        Parameters
        ----------
        num : `int` in the range 0 - 64 (inclusive) or `None`, optional
            The number of "layers" of the boolean array. This parameter is
            mainly so that the function doesn't have to guess how many
            different flags the bitmask could have and to minimize the result
            of the final boolean array. If ``None`` the number of flags is
            determined by the ``dtype`` of the ``bitmask``.
            Default is ``None``.

        Returns
        -------
        boolarray : `numpy.ndarray` of boolean dtype
            The boolean representation of the bitmask. It will have one more
            dimension that the original ``bitmask`` where the first dimension
            represents the "flags". So ``boolarray[0]`` will return an array of
            the size of the original ``bitmask`` and with values ``False`` (if
            the first flag wasn't set) and ``True`` (if the first flag was
            set).

        Examples
        --------
        This converts the bitmask to its boolean representation. For example
        if it's easier to manipulate::

            >>> from ccdproc import Bitmask
            >>> bm = Bitmask(np.arange(7))
            >>> boolbitmask = bm.to_boolarray(num=3)
            >>> boolbitmask
            array([[False,  True, False,  True, False,  True, False],
                   [False, False,  True,  True, False, False,  True],
                   [False, False, False, False,  True,  True,  True]], dtype=bool)

        The first index represents a flag, so by accessing the booleans by
        single integers allows to manipulate, reset, overwrite a given flag
        for the bitmask easily::

            >>> boolbitmask[1]  # Get the values for the second flag
            array([False, False,  True,  True, False, False,  True], dtype=bool)

            >>> boolbitmask[0] = 0  # set all values for the first flag to 0

        Similarly the boolean array can be converted to a bitmask again::

            >>> Bitmask.from_boolarray(boolbitmask)
            Bitmask (
            array([0, 0, 2, 2, 4, 4, 6], dtype=int8)
            name_map={}
            )

        Notes
        -----
        The conversion from and to boolean representation can be quite slow so
        shouldn't be used excessivly.
        """
        bitmask = self.data
        if num is not None and 0 > num > len(_POWERS) - 1:
            raise ValueError(
                '"num" must be between 0 and {} (both inclusive).'
                .format(len(_POWERS) - 1))
        # num is a space-optimization, if there is no "num" given
        # get the number of representable flags by the dtype of
        # the bitmask. An alternative would be to find the maximum
        # and use the maximal power of 2 that is smaller or equal
        # to the maximum. However hat would require a whole pass
        # over the array and using the dtype makes sure we use
        # exactly the same memory as the original array!
        if num is None:
            num = _REPRESENTABLE_FLAGS[bitmask.dtype.name]
        arr = np.zeros(tuple(itertools.chain([num], bitmask.shape)),
                       dtype=bool)
        # Bitwise and checks if the bitmask contains a set flag
        # for this exponent (layer of the boolmask).
        for exp in range(num):
            arr[exp] = bitmask & _POWERS[exp]
        return arr

    def to_mask(self, bitpos=None, bitval=None, bitname=None,
                reverse_flag_interpretation=False):
        """Convert a bitmask to a real mask assuming that ``bitpos`` are set.

        Parameters
        ----------
        bitpos : iterable of integers or None, optional
            The bitpos that should be taken into account when converting the
            bitmask to a mask.

            .. note::
               Must be ``None`` if ``bitval`` or ``bitname`` is not ``None``.

        bitval : iterable of integers or None, optional
            The bitvalues (powers of two) that should be taken into account
            when converting the bitmask to a mask.

            .. note::
               Must be ``None`` if ``bitpos`` or ``bitname`` is not ``None``.

        bitname : iterable of any type or None, optional
            The names (need to be present in ``self.name_map``) that should be
            taken into account when converting the bitmask to a mask.

            .. note::
               Must be ``None`` if ``bitpos`` or ``bitval`` is not ``None``.

        reverse_flag_interpretation : bool, optional
            If ``False`` then the ``bitpos`` will be interpreted as active
            bitpos. To reverse the meaning so that ``bitpos`` indicate which
            bitpos to ignore set this to ``True``.
            Default is ``False``.

        Returns
        -------
        mask : `numpy.ndarray` of boolean dtype
            The mask created from the bitmask.

        Notes
        -----
        If all ``flags`` should be taken into account then consider using
        ``bitmask.data.astype(bool)``.

        Duplicate items in the ``bitpos``, ``bitval`` and ``bitname`` are
        ignored by the function.

        Examples
        --------
        The active flags can be given by their bit position (from right to
        left)::

            >>> from ccdproc import Bitmask
            >>> bm = Bitmask(np.arange(8))
            >>> bm.to_mask(bitpos=[1])
            array([False, False,  True,  True, False, False,  True,  True], dtype=bool)

        or by their bit value (``position ** 2``)::

            >>> bm.to_mask(bitval=[2])
            array([False, False,  True,  True, False, False,  True,  True], dtype=bool)

        or if a ``name_map`` is given also by their name::

            >>> bm.name_map['bad'] = 1
            >>> bm.to_mask(bitname=['bad'])
            array([False, False,  True,  True, False, False,  True,  True], dtype=bool)

        It is also possible to specify only the **inactive** flags::

            >>> bm.to_mask(bitname=['bad'], reverse_flag_interpretation=True)
            array([False,  True, False,  True,  True,  True,  True,  True], dtype=bool)

        and to specify multiple flags::

            >>> bm.to_mask(bitpos=[0, 1])
            array([False,  True,  True,  True, False,  True,  True,  True], dtype=bool)
        """
        bitmask = self.data
        # Make sure one and only one of the three options to provide the
        # "active" flags is actually given.
        if sum(inp is not None for inp in (bitpos, bitval, bitname)) != 1:
            raise TypeError('One and only one of "bitpos", "bitval" or '
                            '"bitname" may be not "None".')

        # So that the logic and validation is only applied once process
        # first "bitname" to "bitpos" and then "bitpos" to "bitval". So each
        # step is quite short and well defined and the logic can be implemented
        # only in the "bitval" branch.

        # Map the names to bit positions
        if bitname is not None:
            try:
                bitpos = [self.name_map[name] for name in bitname]
            except KeyError:
                raise ValueError(
                    'The "name(s)" {} were not found in "self.name_map" ({})'
                    .format([name for name in bitname
                             if name not in self.name_map],
                            self.name_map))
            bitname = None

        # Map the bit positions to bit values
        if bitpos is not None:
            try:
                bitval = [_POWERS[pos] for pos in bitpos]
            except KeyError:
                raise ValueError(
                    'The "bit position(s)" {} are not in the allowed range of '
                    'values.'.format([pos for pos in bitpos
                                      if pos not in _POWERS]))
            bitpos = None

        # If bitname or bitvals were given these are now converted to bitvalues
        # so we can do all the real processing in here.
        if bitval is not None:
            # There is no zero'th flag so subtract one from the representable
            # flags to get the real number of flags
            maximum = _REPRESENTABLE_FLAGS[bitmask.dtype.name] - 1
            # Make sure the maximum value doesn't exceed the limit given by the
            # dtype of the bitmask
            if max(bitval) > _POWERS[maximum]:
                raise ValueError(
                    'The range of values ({}) exceeds the limit of the data '
                    'type of the bitmask ({}).'
                    .format(max(bitval), _POWERS[maximum]))
            uniques = set(bitval)
            # Make sure only valid powers of two are inside the bitvals
            if not uniques.issubset(_POWERS):
                raise ValueError(
                    'The "bitvals" contain illegal (not powers of two) values:'
                    ' {}'.format(sorted(uniques.difference(_POWERS))))

            # Independant of the flag interpretation we need to sum all
            # unique bitvalues
            cmpbitval = sum(uniques)

            # If we want to reverse the flag interpretation we must subtract
            # the value from the sum of all possible flags:
            if reverse_flag_interpretation:
                cmpbitval = np.array(sum(_POWERS[:maximum+1]) - cmpbitval,
                                     dtype=bitmask.dtype)

        # Finally create the mask using a bitwise and operation and return it.
        mask = np.zeros(bitmask.shape, dtype=bool)
        mask = np.bitwise_and(bitmask, cmpbitval, out=mask, casting='unsafe')
        return mask


def _prep_boolarray(boolarray):
    # Check if it looks like a "boolarray", meaning that it should be
    # a `numpy.ndarray` and of boolean data type with at least one dimension.
    boolarray = np.asarray(boolarray, dtype=bool)
    if boolarray.ndim < 2:
        raise TypeError('"boolarray" is expected to have at least '
                        'two dimensions.')
    return boolarray


def boolarray_to_mask(boolarray, flags):
    """Convenience method to convert a boolarray to a real mask assuming that
    ``flags`` (bit positions from right to left) are set.

    Parameters
    ----------
    boolarray : `numpy.ndarray` of boolean data type
        The boolarray that should be converted to a mask depending on the
        ``flags``.

    flags : iterable of integers
        The flags that should be taken into account when converting the
        boolarray to a mask.

    Returns
    -------
    mask : `numpy.ndarray` of boolean dtype
        The mask created from the boolarray.

    Notes
    -----
    If all ``flags`` should be taken into account then consider using
    ``boolarray.max(axis=0)``.

    Duplicate items in the ``flags`` are ignored by the function.
    """
    boolarray = _prep_boolarray(boolarray)
    if boolarray.ndim < 2:
        raise TypeError('"boolarray" is expected to have at least '
                        'two dimensions.')
    if max(flags) > boolarray.shape[0]:
        raise ValueError(
            '"layers" must not contain values below 0 or '
            'above {}.'.format(boolarray.shape[0]))
    # Duplicate layers would mess up the calculation so use "set" on it.
    # This uses advanced integer slicing together with a ``reduce`` function.
    # This is equivalent to:
    # mask = np.zeros(boolarray.shape[1:], dtype=bool)
    # for flag in flags:
    #     mask |= boolarray[flag]
    mask = np.logical_or.reduce(boolarray[list(set(flags))], axis=0)
    return mask
