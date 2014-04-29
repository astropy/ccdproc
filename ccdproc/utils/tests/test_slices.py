import numpy as np

from astropy.tests.helper import pytest

from ..slices import slice_from_string


# none of these are properly enclosed in brackets; is an error raised?
@pytest.mark.parametrize('arg',
                         ['1:2', '[1:2', '1:2]'])
def test_slice_from_string_needs_enclosing_brackets(arg):
    with pytest.raises(ValueError):
        slice_from_string(arg)


@pytest.mark.parametrize('start,stop,step', [
                         (None, None, -1),
                         (5, 10, None),
                         (None, 25, None),
                         (2, 30, 3),
                         (30, None, -2),
                         (None, None, None)
                         ])
def test_slice_from_string_1d(start, stop, step):
    an_array = np.zeros([100])

    stringify = lambda n: str(n) if n else ''
    start_str = stringify(start)
    stop_str = stringify(stop)
    step_str = stringify(step)

    if step_str:
        slice_str = ':'.join([start_str, stop_str, step_str])
    else:
        slice_str = ':'.join([start_str, stop_str])
    sli = slice_from_string('[' + slice_str + ']')
    expected = an_array[slice(start, stop, step)]
    np.testing.assert_array_equal(expected,
                                  an_array[sli])


@pytest.mark.parametrize('arg',
                         ['  [ 1:  45]', '[ 1  :4 5]', '  [1:45] '])
def test_slice_from_string_spaces(arg):
    an_array = np.zeros([100])
    np.testing.assert_array_equal(an_array[1:45],
                                  an_array[slice_from_string(arg)])


def test_slice_from_string_2d():
    an_array = np.zeros([100, 200])

    # manually writing a few cases here rather than parametrizing because the
    # latter seems not worth the trouble.
    sli = slice_from_string('[:-1:2, :]')
    np.testing.assert_array_equal(an_array[:-1:2, :],
                                  an_array[sli])

    sli = slice_from_string('[:, 15:90]')
    np.testing.assert_array_equal(an_array[:, 15:90],
                                  an_array[sli])

    sli = slice_from_string('[10:80:5, 15:90:-1]')
    np.testing.assert_array_equal(an_array[10:80:5, 15:90:-1],
                                  an_array[sli])


def test_slice_from_string_fits_style():
    sli = slice_from_string('[1:5, :]', fits_convention=True)
    # order is reversed, so is the *first* slice one that includes everything?
    assert (sli[0].start is None and
            sli[0].stop is None and
            sli[0].step is None)
    # In the second slice, has the first index been reduced by 1 and the
    # second index left unchanged?
    assert (sli[1].start == 0 and
            sli[1].stop == 5)


def test_slice_from_string_empty():
    assert len(slice_from_string('')) == 0


def test_slice_from_string_bad_fits_slice():
    with pytest.raises(ValueError):
        # Do I error because 0 is an illegal lower bound?
        slice_from_string('[0:10, 1:5]', fits_convention=True)
    with pytest.raises(ValueError):
        # Same as above, but switched order
        slice_from_string('[1:5, 0:10]', fits_convention=True)
    with pytest.raises(ValueError):
        # Do I error if an ending index is negative?
        slice_from_string('[1:10, 10:-1]', fits_convention=True)
