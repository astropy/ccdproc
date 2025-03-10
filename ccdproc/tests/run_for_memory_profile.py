import pytest

pytest.importorskip("memory_profiler")

import gc
import sys
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import psutil
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import median_absolute_deviation
from memory_profiler import memory_usage

# This bit of hackery ensures that we can see ccdproc from within
# the test suite
sys.path.append(str(Path().cwd()))
from ccdproc import ImageFileCollection, combine
from ccdproc.combiner import _calculate_size_of_image

# Do not combine these into one statement. When all references are lost
# to a TemporaryDirectory the directory is automatically deleted. _TMPDIR
# creates a reference that will stick around.
_TMPDIR = TemporaryDirectory()
TMPPATH = Path(_TMPDIR.name)


def generate_fits_files(n_images, size=None, seed=1523):
    if size is None:
        use_size = (2024, 2031)
    else:
        use_size = (size, size)

    rng = np.random.default_rng(seed=seed)

    base_name = "test-combine-{num:03d}.fits"

    for num in range(n_images):
        data = rng.normal(size=use_size)
        # Now add some outlying pixels so there is something to clip
        n_bad = 50000
        bad_x = rng.integers(0, high=use_size[0] - 1, size=n_bad)
        bad_y = rng.integers(0, high=use_size[1] - 1, size=n_bad)
        data[bad_x, bad_y] = rng.choice([-1, 1], size=n_bad) * (10 + rng.random(n_bad))
        hdu = fits.PrimaryHDU(data=np.asarray(data, dtype="float32"))
        hdu.header["for_prof"] = "yes"
        hdu.header["bunit"] = "adu"
        path = TMPPATH.resolve() / base_name.format(num=num)
        hdu.writeto(path, overwrite=True)


def run_memory_profile(
    n_files,
    sampling_interval,
    sigma_clip=False,
    combine_method=None,
    memory_limit=None,
):
    """
    Try opening a bunch of files with a relatively low limit on the number
    of open files.

    Parameters
    ----------

    n_files : int
        Number of files to combine.

    sampling_interval : float
        Time, in seconds, between memory samples.

    size : int, optional
        Size of one side of the image (the image is always square).

    sigma_clip : bool, optional
        If true, sigma clip the data before combining.

    combine_method : str, optional
        Should be one of the combine methods accepted by
        ccdproc.combine

    memory_limit : int, optional
        Cap on memory use during image combination.
    """
    # Do a little input validation
    if n_files <= 0:
        raise ValueError("Argument 'n' must be a positive integer")

    proc = psutil.Process()

    print("Process ID is: ", proc.pid, flush=True)
    ic = ImageFileCollection(str(TMPPATH))
    files = ic.files_filtered(for_prof="yes", include_path=True)

    kwargs = {"method": combine_method}

    if sigma_clip:
        kwargs.update(
            {
                "sigma_clip": True,
                "sigma_clip_low_thresh": 5,
                "sigma_clip_high_thresh": 5,
                "sigma_clip_func": np.ma.median,
                "sigma_clip_dev_func": median_absolute_deviation,
            }
        )

    ccd = CCDData.read(files[0])
    expected_img_size = _calculate_size_of_image(ccd, None)

    if memory_limit:
        kwargs["mem_limit"] = memory_limit

    pre_mem_use = memory_usage(-1, interval=sampling_interval, timeout=1)
    baseline = np.mean(pre_mem_use)
    print(f"Subtracting baseline memory before profile: {baseline}")
    mem_use = memory_usage(
        (combine, (files,), kwargs), interval=sampling_interval, timeout=None
    )
    mem_use = [m - baseline for m in mem_use]
    return mem_use, expected_img_size


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("number", type=int, help="Number of files to combine.")
    parser.add_argument(
        "--size",
        type=int,
        action="store",
        help="Size of one side of image to create. "
        "All images are square, so only give "
        "a single number for the size.",
    )
    parser.add_argument(
        "--combine-method",
        "-c",
        choices=("average", "median"),
        help="Method to use to combine images.",
    )
    parser.add_argument(
        "--memory-limit", type=int, help="Limit combination to this amount of memory"
    )
    parser.add_argument(
        "--sigma-clip",
        action="store_true",
        help="If set, sigma-clip before combining. Clipping "
        "will be done with high/low limit of 5. "
        "The central function is the median, the "
        "deviation is the median_absolute_deviation.",
    )
    parser.add_argument(
        "--sampling-freq",
        type=float,
        default=0.05,
        help="Time, in seconds, between memory samples.",
    )
    parser.add_argument(
        "--frequent-gc",
        action="store_true",
        help="If set, perform garbage collection "
        "much more frequently than the default.",
    )
    args = parser.parse_args()

    if args.frequent_gc:
        gc.set_threshold(10, 10, 10)

    print("Garbage collection thresholds: ", gc.get_threshold())

    mem_use = run_memory_profile(
        args.number,
        args.sampling_freq,
        sigma_clip=args.sigma_clip,
        combine_method=args.combine_method,
        memory_limit=args.memory_limit,
    )
    print("Max memory usage (MB): ", np.max(mem_use))
    print("Baseline memory usage (MB): ", mem_use[0])
