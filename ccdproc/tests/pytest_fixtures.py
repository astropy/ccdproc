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
    m = request.node.get_closest_marker(key)
    if m is not None:
        return m.args[0]
    else:
        return DEFAULTS[key]


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


def _make_file_for_testing(file_name='', **kwd):
    img = np.uint16(np.arange(100))

    hdu = fits.PrimaryHDU(img)

    for k, v in kwd.items():
        hdu.header[k] = v

    hdu.writeto(file_name)


@pytest.fixture
def triage_setup(request):
    """
    Set up directory with these contents:

    One file with imagetyp BIAS. It has an the keyword EXPTIME in
    the header, but no others beyond IMAGETYP and the bare minimum
    created with the FITS file.

    File name(s)
    ------------

    no_filter_no_object_bias.fit

    Five (5) files with imagetyp LIGHT, including two compressed
    files.

    + One file for each compression type, currently .gz and .fz.
    + ALL of the files will have the keyword EXPTIME
      in the header.
    + Only ONE of them will have the value EXPTIME=15.0.
    + All of the files EXCEPT ONE will have the keyword
      FILTER with the value 'R'.
    + NONE of the files have the keyword OBJECT

    File names
    ----------

    test.fits.fz
    filter_no_object_light.fit
    filter_object_light.fit.gz
    filter_object_light.fit
    no_filter_no_object_light.fit    <---- this one has no filter
    """
    n_test = {
        'files': 6,
        'missing_filter_value': 1,
        'bias': 1,
        'compressed': 2,
        'light': 5
    }

    test_dir = mkdtemp()

    # Directory is reset on teardown.
    original_dir = os.getcwd()
    os.chdir(test_dir)

    _make_file_for_testing(file_name='no_filter_no_object_bias.fit',
                           imagetyp='BIAS',
                           exptime=0.0)

    _make_file_for_testing(file_name='no_filter_no_object_light.fit',
                           imagetyp='LIGHT',
                           exptime=1.0)

    _make_file_for_testing(file_name='filter_no_object_light.fit',
                           imagetyp='LIGHT',
                           exptime=1.0,
                           filter='R')

    _make_file_for_testing(file_name='filter_object_light.fit',
                           imagetyp='LIGHT',
                           exptime=1.0,
                           filter='R')

    with open('filter_object_light.fit', 'rb') as f_in:
        with gzip.open('filter_object_light.fit.gz', 'wb') as f_out:
            f_out.write(f_in.read())

    # filter_object.writeto('filter_object_RA_keyword_light.fit')

    _make_file_for_testing(file_name='test.fits.fz',
                           imagetyp='LIGHT',
                           exptime=15.0,
                           filter='R')

    def teardown():
        os.chdir(original_dir)
        try:
            rmtree(test_dir)
        except OSError:
            # If we cannot clean up just keep going.
            pass

    try:
        request.addfinalizer(teardown)
    except AttributeError:
        # Apparently this is not really a pytest test, just ignore it.
        pass

    class Result:
        def __init__(self, n, directory):
            self.n_test = n
            self.test_dir = directory
    return Result(n_test, test_dir)
