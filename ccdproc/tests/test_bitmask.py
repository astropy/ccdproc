# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals

import numpy as np

from astropy.tests.helper import pytest, catch_warnings

from ..core import bitmask2mask


def test_bitmask_not_integer():
    with pytest.raises(TypeError):
        bitmask2mask(np.random.random((10, 10)))


def test_bitmask_negative_flags():
    bm = np.random.randint(0, 10, (10, 10))
    with pytest.raises(ValueError):
        bitmask2mask(bm, [-1])


def test_bitmask_non_poweroftwo_flags():
    bm = np.random.randint(0, 10, (10, 10))
    with pytest.raises(ValueError):
        bitmask2mask(bm, [3])


def test_bitmask_flipbits_when_no_bits():
    bm = np.random.randint(0, 10, (10, 10))
    with pytest.raises(TypeError):
        bitmask2mask(bm, None, flip_bits=1)


def test_bitmask_flipbits_when_stringbits():
    bm = np.random.randint(0, 10, (10, 10))
    with pytest.raises(TypeError):
        bitmask2mask(bm, '3', flip_bits=1)


def test_bitmask_string_flag_flip_not_start_of_string():
    bm = np.random.randint(0, 10, (10, 10))
    with pytest.raises(ValueError):
        bitmask2mask(bm, '1, ~4')


def test_bitmask_string_flag_unbalanced_parens():
    bm = np.random.randint(0, 10, (10, 10))
    with pytest.raises(ValueError):
        bitmask2mask(bm, '(1, 4))')


def test_bitmask_string_flag_wrong_positioned_parens():
    bm = np.random.randint(0, 10, (10, 10))
    with pytest.raises(ValueError):
        bitmask2mask(bm, '((1, )4)')


def test_bitmask_string_flag_empty():
    bm = np.random.randint(0, 10, (10, 10))
    with pytest.raises(ValueError):
        bitmask2mask(bm, '~')


def test_bitmask_flag_non_integer():
    bm = np.random.randint(0, 10, (10, 10))
    with pytest.raises(TypeError):
        bitmask2mask(bm, [1.3])


def test_bitmask_duplicate_flag_throws_warning():
    bm = np.random.randint(0, 10, (10, 10))
    with catch_warnings(UserWarning) as w:
        bitmask2mask(bm, [1, 1])
    assert len(w)


def test_bitmask_none_identical_to_strNone():
    bm = np.random.randint(0, 10, (10, 10))
    m1 = bitmask2mask(bm, None)
    m2 = bitmask2mask(bm, 'None')
    np.testing.assert_array_equal(m1, m2)
