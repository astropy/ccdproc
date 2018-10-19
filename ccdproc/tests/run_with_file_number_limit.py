from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from pathlib import Path
import resource
import mmap
import sys
import gc

import psutil

import numpy as np
from astropy.io import fits

# This bit of hackery ensures that we can see ccdproc from within
# the test suite
sys.path.append(str(Path().cwd()))
from ccdproc import combine

# Do not combine these into one statement. When all references are lost
# to a TemporaryDirectory the directory is automatically deleted. _TMPDIR
# creates a reference that will stick around.
_TMPDIR = TemporaryDirectory()
TMPPATH = Path(_TMPDIR.name)

ALLOWED_EXTENSIONS = {
    'fits': 'fits',
    'plain': 'txt'
}


def generate_fits_files(number, size=None):
    if size is None:
        use_size = [250, 250]
    else:
        int_size = int(size)
        use_size = [int_size, int_size]

    base_name = 'test-combine-{num:03d}.' + ALLOWED_EXTENSIONS['fits']

    for num in range(number):
        data = np.zeros(shape=use_size)
        hdu = fits.PrimaryHDU(data=data)
        hdu.header['bunit'] = 'adu'
        name = base_name.format(num=num)
        path = TMPPATH / name
        hdu.writeto(path, overwrite=True)


def generate_plain_files(number):
    for i in range(number):
        file = TMPPATH / ("{i:03d}.".format(i=i) + ALLOWED_EXTENSIONS['plain'])
        file.write_bytes(np.random.random(100))


def open_files_with_open(kind):
    """
    Open files with plain open.
    """
    # Ensure the file references persist until end of script. Not really
    # necessary, but convenient while debugging the script.
    global fds
    fds = []

    paths = TMPPATH.glob('**/*.' + ALLOWED_EXTENSIONS[kind])

    for p in paths:
        fds.append(p.open())


def open_files_as_mmap(kind):
    """
    Open files as mmaps.
    """
    # Ensure the file references persist until end of script. Not really
    # necessary, but convenient while debugging the script.
    global fds
    fds = []

    paths = TMPPATH.glob('**/*.' + ALLOWED_EXTENSIONS[kind])

    for p in paths:
        with p.open() as f:
            fds.append(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY))


def open_files_ccdproc_combine_chunk(kind):
    """
    Open files indirectly as part of ccdproc.combine, ensuring that the
    task is broken into chunks.
    """
    global combo
    paths = sorted(list(TMPPATH.glob('**/*.' + ALLOWED_EXTENSIONS[kind])))
    # We want to force combine to break the task into chunks even
    # if the task really would fit in memory; it is in that case that
    # we end up with too many open files. We'll open one file, determine
    # the size of the data in bytes, and set the memory limit to that.
    # That will mean lots of chunks (however many files there are plus one),
    # but lots of chunks is fine.
    with fits.open(paths[0]) as hdulist:
        array_size = hdulist[0].data.nbytes

    combo = combine(paths, mem_limit=array_size)


def open_files_ccdproc_combine_nochunk(kind):
    """
    Open files indirectly as part of ccdproc.combine, ensuring that the
    task is not broken into chunks.
    """
    global combo
    paths = sorted(list(TMPPATH.glob('**/*.' + ALLOWED_EXTENSIONS[kind])))

    # We ensure there are no chunks by setting a memory limit large
    # enough to hold everything.
    with fits.open(paths[0]) as hdulist:
        array_size = hdulist[0].data.nbytes

    # Why 2x the number of files? To make absolutely sure we don't
    # end up chunking the job.
    array_size *= 2 * len(paths)
    combo = combine(paths)


ALLOWED_OPENERS = {
    'open': open_files_with_open,
    'mmap': open_files_as_mmap,
    'combine-chunk': open_files_ccdproc_combine_chunk,
    'combine-nochunk': open_files_ccdproc_combine_nochunk
}


def run_with_limit(n, kind='fits', size=None, overhead=6,
                   open_method='mmap'):
    """
    Try opening a bunch of files with a relatively low limit on the number
    of open files.

    Parameters
    ----------

    n : int
        Limit on number of open files in this function. The number of files
        to create is calculated from this to be just below the maximum number
        of files controlled by this function that can be opened.

    kind : one of 'fits', 'plain', optional
        The type of file to generate. The plain files are intended mainly for
        testing this script, while the FITS files are for testing
        ccdproc.combine.

    size : int, optional
        Size of file to create. If the kind is 'plain; this is the size
        of the file, in bytes. If the kind is 'fits', this is the size
        of one side of the image (the image is always square).

    overhead : int, optional
        Number of open files to assume the OS is using for this process. The
        default value is chosen so that this succeeds on MacOS or Linux.
        Setting it to a value lower than default should cause a SystemExit
        exception to be raised because of too many open files. This is meant
        for testing that this script is actually testing something.

    Notes
    -----

    .. warning::

        You should run this in a subprocess. Running as part of a larger python
        process will lower the limit on the number of open files for that
        **entire python process** which will almost certainly lead to nasty
        side effects.
    """
    # Do a little input validation
    if n <= 0:
        raise ValueError("Argument 'n' must be a positive integer")

    if kind not in ALLOWED_EXTENSIONS.keys():
        raise ValueError("Argument 'kind' must be one of "
                         "{}".format(ALLOWED_EXTENSIONS.keys()))

    # Set the limit on the number of open files to n. The try/except
    # is the catch the case where this change would *increase*, rather than
    # decrease, the limit. That apparently can only be done by a superuser.
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (n, n))
    except ValueError as e:
        if 'not allowed to raise maximum limit' not in str(e):
            raise
        max_n_this_process = resource.getrlimit(resource.RLIMIT_NOFILE)
        raise ValueError('Maximum number of open '
                         'files is {}'.format(max_n_this_process))

    # The "-1" is to leave a little wiggle room. overhead is based on the
    # the number of open files that a process running on linux has open.
    # These typically include stdin and stout, and apparently others.
    n_files = n - 1 - overhead

    proc = psutil.Process()

    print('Process ID is: ', proc.pid, flush=True)
    print("Making {} files".format(n_files))
    if kind == 'plain':
        generate_plain_files(n_files)
    elif kind == 'fits':
        generate_fits_files(n_files, size=size)

    # Print number of open files before we try opening anything for debugging
    # purposes.
    print("Before opening, files open is {}".format(len(proc.open_files())),
          flush=True)
    print("    Note well: this number is different than what lsof reports.")

    try:
        ALLOWED_OPENERS[open_method](kind)
            # fds.append(p.open())
    except OSError as e:
        # Capture the error and re-raise as a SystemExit because this is
        # run in a subprocess. This ensures that the original error message
        # is reported back to the calling process; we add on the number of
        # open files.
        raise SystemExit(str(e) + '; number of open files: ' +
                         '{}, with target {}'.format(len(proc.open_files()),
                                                     n_files))
    else:
        print('Opens succeeded, files currently open:',
              len(proc.open_files()),
              flush=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('number', type=int,
                        help='Limit on number of open files.')
    parser.add_argument('--kind', action='store', default='plain',
                        choices=ALLOWED_EXTENSIONS.keys(),
                        help='Kind of file to generate for test; '
                             'default is plain')
    parser.add_argument('--overhead', type=int, action='store',
                        help='Number of files to assume the OS is using.',
                        default=6)
    parser.add_argument('--open-by', action='store', default='mmap',
                        choices=ALLOWED_OPENERS.keys(),
                        help='How to open the files. Default is mmap')
    parser.add_argument('--size', type=int, action='store',
                        help='Size of one side of image to create. '
                             'All images are square, so only give '
                             'a single number for the size.')
    parser.add_argument('--frequent-gc', action='store_true',
                        help='If set, perform garbage collection '
                             'much more frequently than the default.')
    args = parser.parse_args()
    if args.frequent_gc:
        gc.set_threshold(10, 10, 10)
    print("Garbage collection thresholds: ", gc.get_threshold())
    run_with_limit(args.number, kind=args.kind, overhead=args.overhead,
                   open_method=args.open_by, size=args.size)
