# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from ..combiner import Combiner, combine
from .run_for_memory_profile import run_memory_profile, generate_fits_files


@pytest.mark.parametrize('combine_method',
                         ['average', 'sum', 'median'])
def test_memory_use_in_combine(combine_method):
    # This is essentially a regression test for
    # https://github.com/astropy/ccdproc/issues/638
    #
    # Parameters are taken from profiling notebook
    image_size = 2000   # Square image, so 4000 x 4000
    num_files = 10
    sampling_interval = 0.01  # sec
    memory_limit = 500000000  # bytes, roughly 0.5GB

    generate_fits_files(num_files, size=image_size)

    mem_use, _ = run_memory_profile(num_files, sampling_interval,
                                    size=image_size, memory_limit=memory_limit,
                                    combine_method=combine_method)

    # We do not expect memory use to be strictly less than memory_limit
    # throughout the combination. The factor below allows for that.
    # It may need to be raised in the future...that is fine, there is a
    # separate test for average memory use.
    overhead_allowance = 1.5

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
    # usage. Nothing special, really, about the factor 0.5 below.
    assert np.mean(mem_use) > 0.5 * memory_limit_mb