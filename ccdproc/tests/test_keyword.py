# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
from astropy import units as u
from astropy.io import fits

from ..core import Keyword


def test_keyword_init():
    key_name = 'some_key'
    key = Keyword(key_name, unit=u.second)
    assert key.name == key_name
    assert key.unit == u.second


def test_keyword_properties_read_only():
    key = Keyword('observer')
    with pytest.raises(AttributeError):
        key.name = 'error'
    with pytest.raises(AttributeError):
        key.unit = u.hour


unit = u.second
numerical_value = 30


# The variable "expected" below is
#     True if the expected result is key.value == numerical_value * key.unit
#     Name of an error if an error is expected
#     A string if the expected value is a string
@pytest.mark.parametrize('value,unit,expected', [
                         (numerical_value, unit, True),
                         (numerical_value, None, ValueError),
                         (numerical_value * unit, None, True),
                         (numerical_value * unit, unit, True),
                         (numerical_value * unit, u.km, True),
                         ('some string', None, 'some string'),
                         ('no strings with unit', unit, ValueError)
                         ])
def test_value_setting(value, unit, expected):
    name = 'exposure'
    # Setting at initialization time with
    try:
        expected_is_error = issubclass(expected, Exception)
    except TypeError:
        expected_is_error = False
    if expected_is_error:
        with pytest.raises(expected):
            key = Keyword(name, unit=unit, value=value)
    else:
        key = Keyword(name, unit=unit, value=value)
        if isinstance(expected, str):
            assert key.value == expected
        else:
            assert key.value == numerical_value * key.unit


def test_keyword_value_from_header():
    name = 'exposure'
    numerical_value = 30
    unit = u.second
    h = fits.Header()
    h[name] = numerical_value

    key = Keyword(name, unit=unit)
    assert key.value_from(h) == numerical_value * unit
    assert key.value == numerical_value * unit
