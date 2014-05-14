from astropy.extern import six
from astropy.tests.helper import pytest
import astropy.units as u

from ..ccdproc import create_variance, Keyword


@pytest.mark.parametrize('key', [
                         'short',
                         'toolongforfits'])
def test_log_string(ccd_data, key):
    add_key = key
    new = create_variance(ccd_data, readnoise=3 * ccd_data.unit,
                          add_keyword=add_key)
    # Keys should be added to new but not to ccd_data and should have
    # no value.
    assert add_key in new.meta
    assert add_key not in ccd_data.meta
    # Long keyword names should be accessible with just the keyword name
    # without HIERARCH -- is it?
    assert new.meta[add_key] is None


def test_log_keyword(ccd_data):
    key = 'filter'
    key_val = 'V'
    kwd = Keyword(key, value=key_val)
    new = create_variance(ccd_data, readnoise=3 * ccd_data.unit,
                          add_keyword=kwd)
    # Was the Keyword added with the correct value?
    assert kwd.name in new.meta
    assert kwd.name not in ccd_data.meta
    assert new.meta[kwd.name] == key_val


def test_log_dict(ccd_data):
    keys_to_add = {
        'process': 'Added variance',
        'n_images_input': 1,
        'current_temp': 42.9
    }
    new = create_variance(ccd_data, readnoise=3 * ccd_data.unit,
                          add_keyword=keys_to_add)
    for k, v in six.iteritems(keys_to_add):
        # Were all dictionary items added?
        assert k in new.meta
        assert k not in ccd_data.meta
        assert new.meta[k] == v


def test_log_bad_type_fails(ccd_data):
    add_key = 15   # anything not string and not dict-like will work here
    # Do we fail with non-string, non-Keyword, non-dict-like value?
    with pytest.raises(AttributeError):
        create_variance(ccd_data, readnoise=3 * ccd_data.unit,
                        add_keyword=add_key)


def test_log_set_to_None_does_not_change_header(ccd_data):
    new = create_variance(ccd_data, readnoise=3 * ccd_data.unit,
                          add_keyword=None)
    assert new.meta.keys() == ccd_data.header.keys()
