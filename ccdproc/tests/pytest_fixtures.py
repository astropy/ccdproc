# Licensed under a 3-clause BSD style license - see LICENSE.rst

from shutil import rmtree

import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import CCDData

from ..core import _to_numpy
from ..utils.sample_directory import directory_for_testing

# If additional pytest markers are defined the key in the dictionary below
# should be the name of the marker.
DEFAULTS = {"seed": 123, "data_size": 100, "data_scale": 1.0, "data_mean": 0.0}

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


def ccd_data(
    data_size=DEFAULT_DATA_SIZE,
    data_scale=DEFAULT_DATA_SCALE,
    data_mean=DEFAULT_DATA_MEAN,
    rng_seed=DEFAULT_SEED,
):
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
    # Need the import here to avoid circular import issues
    from ..conftest import testing_array_device as xp_device
    from ..conftest import testing_array_library as xp

    size = data_size
    scale = data_scale
    mean = data_mean

    ##Create random number generator with a specified state
    rng = np.random.default_rng(seed=rng_seed)

    data = rng.normal(loc=mean, size=[size, size], scale=scale)

    fake_meta = {"my_key": 42, "your_key": "not 42"}
    ccd = CCDData(xp.asarray(data, device=xp_device), unit=u.adu)
    ccd.header = fake_meta
    return ccd


def numpy_ccddata(ccd):
    """
    Return a copy of a ``CCDData`` whose arrays are all NumPy arrays.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        Image whose ``data``, and ``mask`` and ``uncertainty`` if present,
        may be in any array namespace.

    Returns
    -------
    `~astropy.nddata.CCDData`
        A new ``CCDData`` with ``data``, ``mask`` and ``uncertainty.array``
        converted with `ccdproc.core._to_numpy`; ``unit`` and ``meta`` are
        carried over unchanged and the uncertainty keeps its class.

    Notes
    -----
    Use this to hand a namespace ``CCDData`` to code that requires NumPy,
    such as ``CCDData.write``. The strict tests run on ``Device("device1")``,
    on which array-api-strict refuses to export to NumPy; ``_to_numpy``
    moves the arrays to the default device first.
    """
    new_ccd = CCDData(_to_numpy(ccd.data), unit=ccd.unit, meta=ccd.meta)
    if ccd.mask is not None:
        new_ccd.mask = _to_numpy(ccd.mask)
    if ccd.uncertainty is not None:
        new_ccd.uncertainty = ccd.uncertainty.__class__(
            _to_numpy(ccd.uncertainty.array)
        )
    return new_ccd


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
