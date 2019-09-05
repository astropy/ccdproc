import gzip
from tempfile import mkdtemp
import os

import numpy as np

from astropy.io import fits


def _make_file_for_testing(file_name='', **kwd):
    img = np.uint16(np.arange(100))

    hdu = fits.PrimaryHDU(img)

    for k, v in kwd.items():
        hdu.header[k] = v

    hdu.writeto(file_name)


def directory_for_testing():
    """
    Set up directory with these contents:

    One file with imagetyp BIAS. It has an the keyword EXPOSURE in
    the header, but no others beyond IMAGETYP and the bare minimum
    created with the FITS file.

    File name(s)
    ------------

    no_filter_no_object_bias.fit

    Five (5) files with imagetyp LIGHT, including two compressed
    files.

    + One file for each compression type, currently .gz and .fz.
    + ALL of the files will have the keyword EXPOSURE
      in the header.
    + Only ONE of them will have the value EXPOSURE=15.0.
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
                           EXPOSURE=0.0)

    _make_file_for_testing(file_name='no_filter_no_object_light.fit',
                           imagetyp='LIGHT',
                           EXPOSURE=1.0)

    _make_file_for_testing(file_name='filter_no_object_light.fit',
                           imagetyp='LIGHT',
                           EXPOSURE=1.0,
                           filter='R')

    _make_file_for_testing(file_name='filter_object_light.fit',
                           imagetyp='LIGHT',
                           EXPOSURE=1.0,
                           filter='R')

    with open('filter_object_light.fit', 'rb') as f_in:
        with gzip.open('filter_object_light.fit.gz', 'wb') as f_out:
            f_out.write(f_in.read())

    # filter_object.writeto('filter_object_RA_keyword_light.fit')

    _make_file_for_testing(file_name='test.fits.fz',
                           imagetyp='LIGHT',
                           EXPOSURE=15.0,
                           filter='R')

    os.chdir(original_dir)

    return n_test, test_dir


def sample_directory_with_files():
    """
    Returns the path to the small sample directory used
    in the tests of ``ImageFileCollection``. Primarily intended
    for use in the doctests.
    """

    n_test, tmpdir = directory_for_testing()
    return tmpdir
