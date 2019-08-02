# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import pytest

try:
    from .run_for_memory_profile import run_memory_profile, generate_fits_files, TMPPATH
except ImportError:
    memory_profile_present = False
else:
    memory_profile_present = True

image_size = 2000   # Square image, so 4000 x 4000
num_files = 10


def setup_module():
    if memory_profile_present:
        generate_fits_files(num_files, size=image_size)


def teardown_module():
    if memory_profile_present:
        for fil in TMPPATH.glob('*.fit'):
            fil.unlink()


@pytest.mark.skipif(not memory_profile_present,
                    reason='memory_profiler not installed')
@pytest.mark.parametrize('combine_method',
                         ['average', 'sum', 'median'])
def test_memory_use_in_combine(combine_method):
    # This is essentially a regression test for
    # https://github.com/astropy/ccdproc/issues/638
    #
    sampling_interval = 0.01  # sec
    memory_limit = 500000000  # bytes, roughly 0.5GB

    mem_use, _ = run_memory_profile(num_files, sampling_interval,
                                    size=image_size, memory_limit=memory_limit,
                                    combine_method=combine_method)

    # We do not expect memory use to be strictly less than memory_limit
    # throughout the combination. The factor below allows for that.
    # It may need to be raised in the future...that is fine, there is a
    # separate test for average memory use.
    overhead_allowance = 1.75

    # memory_profile reports in MB (no, this is not the correct conversion)
    memory_limit_mb = memory_limit / 1e6

    # Checks for TOO MUCH MEMORY USED

    # Check peak memory use
    assert np.max(mem_use) <= overhead_allowance * memory_limit_mb

    # Also check average, which gets no allowance
    assert np.mean(mem_use) < memory_limit_mb

    # Checks for NOT ENOUGH MEMORY USED; if these fail it means that
    # memory_factor in the combine function should perhaps be modified

    # If the peak is coming in under the limit something need to be fixed
    assert np.max(mem_use) >= 0.95 * memory_limit_mb

    # If the average is really low perhaps we should look at reducing peak
    # usage. Nothing special, really, about the factor 0.4 below.
    assert np.mean(mem_use) > 0.4 * memory_limit_mb
