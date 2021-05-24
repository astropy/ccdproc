# Licensed under a 3-clause BSD style license - see LICENSE.rst

import gzip
from tempfile import mkdtemp
import os
from shutil import rmtree

import numpy as np

import pytest
from astropy import units as u
from astropy.utils import NumpyRNGContext
from astropy.io import fits
from astropy.nddata import CCDData

from ..utils.sample_directory import directory_for_testing

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
DEFAULT_DATA_MEAN = 0.0


def value_from_markers(key, request):
    m = request.node.get_closest_marker(key)
    if m is not None:
        return m.args[0]
    else:
        return DEFAULTS[key]


def ccd_data(data_size=DEFAULT_DATA_SIZE,
             data_scale=DEFAULT_DATA_SCALE,
             data_mean=DEFAULT_DATA_MEAN,
             rng_seed=DEFAULT_SEED):
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
    size = data_size
    scale = data_scale
    mean = data_mean

    with NumpyRNGContext(rng_seed):
        data = np.random.normal(loc=mean, size=[size, size], scale=scale)

    fake_meta = {'my_key': 42, 'your_key': 'not 42'}
    ccd = CCDData(data, unit=u.adu)
    ccd.header = fake_meta
    return ccd


@pytest.fixture
def triage_setup(request):

    n_test, test_dir = directory_for_testing()

    def teardown():
        try:
            rmtree(test_dir)
        except OSError:
            # If we cannot clean up just keep going.
            pass

    request.addfinalizer(teardown)

    class Result:
        def __init__(self, n, directory):
            self.n_test = n
            self.test_dir = directory
    return Result(n_test, test_dir)
