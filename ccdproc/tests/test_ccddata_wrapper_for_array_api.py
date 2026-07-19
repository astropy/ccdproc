# Licensed under a 3-clause BSD style license - see LICENSE.rst

import array_api_extra as xpx
import astropy.units as u
import pytest
from astropy.nddata import (
    CCDData,
    InverseVariance,
    StdDevUncertainty,
    VarianceUncertainty,
)
from astropy.wcs import WCS

from ccdproc import flat_correct, trim_image
from ccdproc._ccddata_wrapper_for_array_api import (
    _InverseVarianceWrapper,
    _StdDevUncertaintyWrapper,
    _unwrap_ccddata_for_array_api,
    _VarianceUncertaintyWrapper,
)
from ccdproc.conftest import testing_array_library as xp


def test_trim_image_returns_plain_ccddata():
    data = xp.asarray([[1.0, 2.0], [3.0, 4.0]])
    uncertainty = StdDevUncertainty(xp.asarray([[0.1, 0.2], [0.3, 0.4]]))
    wcs = WCS(naxis=2)
    ccd = CCDData(
        data,
        unit=u.adu,
        uncertainty=uncertainty,
        meta={"source": "test"},
        wcs=wcs,
    )

    result = trim_image(ccd, add_keyword=None)

    assert type(result) is CCDData
    assert type(result.uncertainty) is StdDevUncertainty
    assert xp.all(xpx.isclose(result.data, ccd.data))
    assert result.mask is None
    assert xp.all(xpx.isclose(result.uncertainty.array, ccd.uncertainty.array))
    assert result.meta == ccd.meta
    assert result.unit == ccd.unit
    assert result.wcs.wcs.compare(ccd.wcs.wcs)


@pytest.mark.backend_xfail(
    "array-api-strict",
    reason="Astropy uncertainty propagation mixes NumPy and strict arrays",
)
def test_flat_correct_returns_public_uncertainty():
    ccd = CCDData(
        xp.ones((2, 2)),
        unit=u.adu,
        uncertainty=StdDevUncertainty(xp.ones((2, 2))),
    )
    flat = CCDData(xp.ones((2, 2)), unit=u.adu)

    result = flat_correct(ccd, flat, add_keyword=None)

    assert type(result) is CCDData
    assert type(result.uncertainty) is StdDevUncertainty


@pytest.mark.parametrize(
    ("wrapper_type", "public_type"),
    [
        (_StdDevUncertaintyWrapper, StdDevUncertainty),
        (_VarianceUncertaintyWrapper, VarianceUncertainty),
        (_InverseVarianceWrapper, InverseVariance),
    ],
)
def test_unwrap_plain_ccddata_returns_public_uncertainty(wrapper_type, public_type):
    ccd = CCDData(
        xp.ones((1, 1)),
        unit=u.adu,
        uncertainty=wrapper_type(xp.ones((1, 1))),
    )

    result = _unwrap_ccddata_for_array_api(ccd)

    assert result is ccd
    assert type(result.uncertainty) is public_type


def test_unwrap_plain_ccddata_is_identity():
    uncertainty = StdDevUncertainty(xp.asarray([[0.1]]))
    ccd = CCDData(xp.asarray([[1.0]]), unit=u.adu, uncertainty=uncertainty)
    assigned_uncertainty = ccd.uncertainty

    result = _unwrap_ccddata_for_array_api(ccd)

    assert result is ccd
    assert result.uncertainty is assigned_uncertainty


def test_unwrap_ccddata_subclass_is_identity():
    class CustomCCDData(CCDData):
        pass

    ccd = CustomCCDData(xp.asarray([[1.0]]), unit=u.adu)

    assert _unwrap_ccddata_for_array_api(ccd) is ccd


def test_unwrap_rejects_non_ccddata():
    with pytest.raises(
        TypeError,
        match="Input must be a CCDData or _CCDDataWrapperForArrayAPI instance",
    ):
        _unwrap_ccddata_for_array_api(object())
