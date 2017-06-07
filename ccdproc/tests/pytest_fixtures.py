# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import gzip
from tempfile import mkdtemp
import os
from shutil import rmtree

import numpy as np

import pytest
from astropy import units as u
from astropy.utils import NumpyRNGContext
from astropy.io import fits


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

    fake_meta = {'my_key': 42, 'your_key': 'not 42'}
    ccd = CCDData(data, unit=u.adu)
    ccd.header = fake_meta
    return ccd


@pytest.fixture
def triage_setup(request):
    n_test = {'files': 0, 'need_object': 0,
              'need_filter': 0, 'bias': 0,
              'compressed': 0, 'light': 0,
              'need_pointing': 0}

    test_dir = ''

    for key in n_test.keys():
        n_test[key] = 0

    test_dir = mkdtemp()
    original_dir = os.getcwd()
    os.chdir(test_dir)
    img = np.uint16(np.arange(100))

    no_filter_no_object = fits.PrimaryHDU(img)
    no_filter_no_object.header['imagetyp'] = 'light'.upper()
    no_filter_no_object.writeto('no_filter_no_object_light.fit')
    n_test['files'] += 1
    n_test['need_object'] += 1
    n_test['need_filter'] += 1
    n_test['light'] += 1
    n_test['need_pointing'] += 1

    no_filter_no_object.header['imagetyp'] = 'bias'.upper()
    no_filter_no_object.writeto('no_filter_no_object_bias.fit')
    n_test['files'] += 1
    n_test['bias'] += 1

    filter_no_object = fits.PrimaryHDU(img)
    filter_no_object.header['imagetyp'] = 'light'.upper()
    filter_no_object.header['filter'] = 'R'
    filter_no_object.writeto('filter_no_object_light.fit')
    n_test['files'] += 1
    n_test['need_object'] += 1
    n_test['light'] += 1
    n_test['need_pointing'] += 1

    filter_no_object.header['imagetyp'] = 'bias'.upper()
    filter_no_object.writeto('filter_no_object_bias.fit')
    n_test['files'] += 1
    n_test['bias'] += 1

    filter_object = fits.PrimaryHDU(img)
    filter_object.header['imagetyp'] = 'light'.upper()
    filter_object.header['filter'] = 'R'
    filter_object.header['OBJCTRA'] = '00:00:00'
    filter_object.header['OBJCTDEC'] = '00:00:00'
    filter_object.writeto('filter_object_light.fit')
    n_test['files'] += 1
    n_test['light'] += 1
    n_test['need_object'] += 1
    with open('filter_object_light.fit', 'rb') as f_in:
        with gzip.open('filter_object_light.fit.gz', 'wb') as f_out:
            f_out.write(f_in.read())
    n_test['files'] += 1
    n_test['compressed'] += 1
    n_test['light'] += 1
    n_test['need_object'] += 1

    filter_object.header['RA'] = filter_object.header['OBJCTRA']
    filter_object.header['Dec'] = filter_object.header['OBJCTDEC']
    filter_object.writeto('filter_object_RA_keyword_light.fit')
    n_test['files'] += 1
    n_test['light'] += 1
    n_test['need_object'] += 1

    def teardown():
        for key in n_test.keys():
            n_test[key] = 0
        try:
            rmtree(test_dir)
        except OSError:
            # If we cannot clean up just keep going.
            pass
        os.chdir(original_dir)
    request.addfinalizer(teardown)

    class Result(object):
        def __init__(self, n, directory):
            self.n_test = n
            self.test_dir = directory
    return Result(n_test, test_dir)
