from pathlib import Path
import subprocess

p = Path(__file__).parent

OVERHEAD = '4'
NUM_FILE_LIMIT = '20'
common_args = ['python', str(p / 'run_with_file_number_limit.py'),
               '--kind', 'fits', '--overhead', OVERHEAD]


# Regression test for #629
def test_open_files_combine_no_chunks():
    """
    Test that we are not opening (much) more than the number of files
    we are processing.
    """
    # Make a copy
    args = list(common_args)
    args.extend(['--open-by', 'combine-nochunk', NUM_FILE_LIMIT])
    p = subprocess.run(args=args, stderr=subprocess.PIPE)
    # If we have succeeded the test passes. We are only checking that
    # we don't have too many files open.
    assert p.returncode == 0


# Regression test for #629
def test_open_files_combine_chunks():
    """
    Test that we are not opening (much) more than the number of files
    we are processing when combination is broken into chunks.
    """
    # Make a copy
    args = list(common_args)
    args.extend(['--open-by', 'combine-chunk', NUM_FILE_LIMIT])
    p = subprocess.run(args=args, stderr=subprocess.PIPE)
    # If we have succeeded the test passes. We are only checking that
    # we don't have too many files open.
    assert p.returncode == 0
