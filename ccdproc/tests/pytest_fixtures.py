from __future__ import division, print_function

import numpy as np
import pytest
from astropy import units as u
from ..ccddata import CCDData


@pytest.fixture
def ccd_data(request):
    """
    Return a CCDData object with units of ADU.

    The size of the data array is 100x100 but can be changed using the marker
    @pytest.mark.data_size(N) on the test function, where N should be the
    desired dimension.

    Data values are initialized to random numbers drawn from a uniform
    distribution.
    """
    try:
        size = request.keywords['data_size'].args[0]
    except KeyError:
        size = 100
    return CCDData(np.zeros([size, size]), unit=u.adu)
