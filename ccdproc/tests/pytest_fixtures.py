from __future__ import division, print_function

import numpy as np

from astropy.tests.helper import pytest
from astropy import units as u
from astropy.utils import NumpyRNGContext

from ..ccddata import CCDData

# If additional pytest markers are defined the key in the dictionary below
# should be the name of the marker.
DEFAULTS = {
    'seed': 123,
    'data_size': 100,
    'data_scale': 1.0,
    'data_mean': 0.0
}

DEFAULT_SEED = 123
DEFAULT_DATA_SIZE = 100
DEFAULT_DATA_SCALE = 1.0


def value_from_markers(key, request):
    try:
        val = request.keywords[key].args[0]
    except KeyError:
        val = DEFAULTS[key]
    return val


@pytest.fixture
def ccd_data(request):
    """
    Return a CCDData object with units of ADU.

    The size of the data array is 100x100 but can be changed using the marker
    @pytest.mark.data_size(N) on the test function, where N should be the
    desired dimension.

    Data values are initialized to random numbers drawn from a normal
    distribution with mean of 0 and scale 1.

    The scale can be changed with the marker @pytest.marker.scale(s) on the
    test function, where s is the desired scale.

    The mean can be changed with the marker @pytest.marker.scale(m) on the
    test function, where m is the desired mean.
    """
    size = value_from_markers('data_size', request)
    scale = value_from_markers('data_scale', request)
    mean = value_from_markers('data_mean', request)

    with NumpyRNGContext(DEFAULTS['seed']):
        data = np.random.normal(loc=mean, size=[size, size], scale=scale)

    return CCDData(data, unit=u.adu)
