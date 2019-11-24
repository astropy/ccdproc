# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from astropy.nddata import StdDevUncertainty, CCDData

from scipy import ndimage

from .. import core


def test_medianfilter_correct():
    ccd = CCDData([[2, 6, 6, 1, 7, 2, 4, 5, 9, 1],
                   [10, 10, 9, 0, 2, 10, 8, 3, 9, 7],
                   [2, 4, 0, 4, 4, 10, 0, 5, 6, 5],
                   [7, 10, 8, 7, 7, 0, 5, 3, 5, 9],
                   [9, 6, 3, 8, 6, 9, 2, 8, 10, 10],
                   [6, 5, 1, 7, 8, 0, 8, 2, 9, 3],
                   [0, 6, 0, 6, 3, 10, 8, 9, 7, 8],
                   [5, 8, 3, 2, 3, 0, 2, 0, 3, 5],
                   [9, 6, 3, 7, 1, 0, 5, 4, 8, 3],
                   [5, 6, 9, 9, 0, 4, 9, 1, 7, 8]], unit='adu')
    result = core.median_filter(ccd, 3)
    assert isinstance(result, CCDData)
    assert np.all(result.data == [[6, 6, 6, 6, 2, 4, 4, 5, 5, 7],
                                  [4, 6, 4, 4, 4, 4, 5, 5, 5, 6],
                                  [7, 8, 7, 4, 4, 5, 5, 5, 5, 7],
                                  [7, 6, 6, 6, 7, 5, 5, 5, 6, 9],
                                  [7, 6, 7, 7, 7, 6, 3, 5, 8, 9],
                                  [6, 5, 6, 6, 7, 8, 8, 8, 8, 8],
                                  [5, 5, 5, 3, 3, 3, 2, 7, 5, 5],
                                  [6, 5, 6, 3, 3, 3, 4, 5, 5, 5],
                                  [6, 6, 6, 3, 2, 2, 2, 4, 4, 5],
                                  [6, 6, 7, 7, 4, 4, 4, 7, 7, 8]])
    assert result.unit == 'adu'
    assert all(getattr(result, attr) is None
               for attr in ['mask', 'uncertainty', 'wcs', 'flags'])
    # The following test could be deleted if log_to_metadata is also applied.
    assert not result.meta


def test_medianfilter_unusued():
    ccd = CCDData(np.ones((3, 3)), unit='adu',
                  mask=np.ones((3, 3)),
                  uncertainty=StdDevUncertainty(np.ones((3, 3))),
                  flags=np.ones((3, 3)))
    result = core.median_filter(ccd, 3)
    assert isinstance(result, CCDData)
    assert result.unit == 'adu'
    assert all(getattr(result, attr) is None
               for attr in ['mask', 'uncertainty', 'wcs', 'flags'])
    # The following test could be deleted if log_to_metadata is also applied.
    assert not result.meta


def test_medianfilter_ndarray():
    arr = np.random.random((5, 5))
    result = core.median_filter(arr, 3)
    reference = ndimage.median_filter(arr, 3)
    # It's a wrapped function so we can use the equal comparison.
    np.testing.assert_array_equal(result, reference)
