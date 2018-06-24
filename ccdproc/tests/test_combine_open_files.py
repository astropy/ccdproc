from pathlib import Path
import subprocess
import sys
import os

import pytest

run_dir = Path(__file__).parent

# Why? So that we get up to the file above ccdproc, so that in the
# subprocess we can add that direction to sys.path.
subprocess_dir = run_dir.parent.parent

OVERHEAD = '4'
NUM_FILE_LIMIT = '20'
common_args = [sys.executable, str(run_dir / 'run_with_file_number_limit.py'),
               '--kind', 'fits', '--overhead', OVERHEAD]


# Regression test for #629
@pytest.mark.skipif(os.environ.get('APPVEYOR') or os.sys.platform == 'win32',
                    reason='Test relies on linux/osx features of psutil')
@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason='Test requires subprocess.run, introduced in 3.5')
def test_open_files_combine_no_chunks():
    """
    Test that we are not opening (much) more than the number of files
    we are processing.
    """
    # Make a copy
    args = list(common_args)
    args.extend(['--open-by', 'combine-nochunk', NUM_FILE_LIMIT])
    p = subprocess.run(args=args, stderr=subprocess.PIPE,
                       cwd=str(subprocess_dir))
    # If we have succeeded the test passes. We are only checking that
    # we don't have too many files open.
    assert p.returncode == 0


# Regression test for #629
@pytest.mark.skipif(os.environ.get('APPVEYOR') or os.sys.platform == 'win32',
                    reason='Test relies on linux/osx features of psutil')
@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason='Test requires subprocess.run, introduced in 3.5')
def test_open_files_combine_chunks():
    """
    Test that we are not opening (much) more than the number of files
    we are processing when combination is broken into chunks.
    """
    # Make a copy
    args = list(common_args)
    args.extend(['--open-by', 'combine-chunk', NUM_FILE_LIMIT])
    p = subprocess.run(args=args, stderr=subprocess.PIPE,
                       cwd=str(subprocess_dir))
    # If we have succeeded the test passes. We are only checking that
    # we don't have too many files open.
    assert p.returncode == 0
