# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.nddata import NDData
from astropy.tests.helper import pytest

from ..ccddata import CCDData
from ..support_nddata_patched import support_nddata


def test_setup_failures1():
    # repack but no returns
    with pytest.raises(ValueError):
        support_nddata(repack=True)


def test_setup_failures2():
    # returns but no repack
    with pytest.raises(ValueError):
        support_nddata(returns=['data'])


def test_setup_failures9():
    # keeps but no repack
    with pytest.raises(ValueError):
        support_nddata(keeps=['unit'])


def test_setup_failures3():
    # same attribute in keeps and returns
    with pytest.raises(ValueError):
        support_nddata(repack=True, keeps=['mask'], returns=['data', 'mask'])


def test_setup_failures4():
    # function accepts *args
    with pytest.raises(ValueError):
        @support_nddata
        def test(data, *args):
            pass


def test_setup_failures10():
    # function accepts **kwargs
    with pytest.raises(ValueError):
        @support_nddata
        def test(data, **kwargs):
            pass


def test_setup_failures5():
    # function accepts *args (or **kwargs)
    with pytest.raises(ValueError):
        @support_nddata
        def test(data, *args):
            pass


def test_setup_failures6():
    # First argument is not data
    with pytest.raises(ValueError):
        @support_nddata
        def test(img):
            pass


def test_setup_failures7():
    # accepts CCDData but was given just an NDData
    with pytest.raises(TypeError):
        @support_nddata(accepts=CCDData)
        def test(data):
            pass
        test(NDData(np.ones((3, 3))))


def test_setup_failures8():
    # function returns a different amount of arguments than specified. Using
    # NDData here so we don't get into troubles when creating a CCDData without
    # unit!
    with pytest.raises(ValueError):
        @support_nddata(repack=True, returns=['data', 'mask'])
        def test(data):
            return 10
        test(NDData(np.ones((3, 3))))  # do NOT use CCDData here.


def test_setup_failures11():
    # function accepts no arguments
    with pytest.raises(ValueError):
        @support_nddata
        def test():
            pass


def test_still_accepts_other_input():
    @support_nddata(repack=True, returns=['data'])
    def test(data):
        return data
    assert isinstance(test(NDData(np.ones((3, 3)))), NDData)
    assert isinstance(test(10), int)
    assert isinstance(test([1, 2, 3]), list)


def test_accepting_property_normal():
    # Accepts a mask attribute and takes it from the input
    @support_nddata
    def test(data, mask=None):
        return mask

    ndd = NDData(np.ones((3, 3)))
    assert test(ndd) is None
    ndd._mask = np.zeros((3, 3))
    assert np.all(test(ndd) == 0)
    # Use the explicitly given one (raises a Warning but too lazy to check!)
    assert test(ndd, mask=10) == 10


def test_accepting_property_notexist():
    # Accepts flags attribute but NDData doesn't have one
    @support_nddata
    def test(data, flags=10):
        return flags

    ndd = NDData(np.ones((3, 3)))
    test(ndd)


def test_accepting_property_translated():
    # Accepts a error attribute and we want to pass in uncertainty!
    @support_nddata(mask='masked')
    def test(data, masked=None):
        return masked

    ndd = NDData(np.ones((3, 3)))
    assert test(ndd) is None
    ndd._mask = np.zeros((3, 3))
    assert np.all(test(ndd) == 0)
    # Use the explicitly given one (raises a Warning but too lazy to check!)
    assert test(ndd, masked=10) == 10


def test_accepting_property_meta_empty():
    # Meta is always set (OrderedDict) so it has a special case that it's
    # ignored if it's empty but not None
    @support_nddata
    def test(data, meta=None):
        return meta

    ndd = NDData(np.ones((3, 3)))
    assert test(ndd) is None
    ndd._meta = {'a': 10}
    assert test(ndd) == {'a': 10}
