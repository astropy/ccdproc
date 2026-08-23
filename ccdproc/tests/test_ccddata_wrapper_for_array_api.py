# Licensed under a 3-clause BSD style license - see LICENSE.rst

import array_api_compat
import array_api_extra as xpx
import astropy.units as u
import numpy as np
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
    _ArrayAPIPropagationMixin,
    _CCDDataWrapperForArrayAPI,
    _InverseVarianceWrapper,
    _StdDevUncertaintyWrapper,
    _unwrap_ccddata_for_array_api,
    _VarianceUncertaintyWrapper,
    _wrap_ccddata_for_array_api,
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


def test_unwrap_wrapper_preserves_backend_arrays():
    ccd = CCDData(xp.ones((2, 2)), unit=u.adu)
    ccd._mask = xp.asarray([[True, False], [False, True]])
    ccd.uncertainty = StdDevUncertainty(xp.ones((2, 2)))

    wrapped = _wrap_ccddata_for_array_api(ccd)
    assert isinstance(wrapped, _CCDDataWrapperForArrayAPI)
    assert wrapped is not ccd
    wrapped_data = wrapped.data
    wrapped_mask = wrapped.mask
    wrapped_uncertainty_array = wrapped.uncertainty.array

    result = _unwrap_ccddata_for_array_api(wrapped)

    assert result is wrapped
    assert type(result) is CCDData
    assert result.data is wrapped_data
    assert result.mask is wrapped_mask
    assert type(result.uncertainty.array) is type(wrapped_uncertainty_array)
    assert xp.all(xpx.isclose(result.uncertainty.array, wrapped_uncertainty_array))
    assert type(result.uncertainty) is StdDevUncertainty
    assert type(ccd) is CCDData


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


_STRICT_STDDEV_MULDIV_XFAIL = pytest.mark.backend_xfail(
    "array-api-strict",
    reason="astropy's _propagate_multiply_divide applies np.sqrt/np.abs to the "
    "std-dev result, which fails on a non-default strict device (see #940)",
)


def test_propagation_mixin_requires_variance_hooks():
    """The mixin is abstract: a subclass that forgets ``_variance_hooks`` fails
    loudly rather than silently propagating with the wrong conversions."""
    with pytest.raises(NotImplementedError):
        _ArrayAPIPropagationMixin._variance_hooks(xp)


@pytest.mark.parametrize(
    ("uncertainty_type", "operation"),
    [
        pytest.param(
            unc,
            op,
            marks=(
                [_STRICT_STDDEV_MULDIV_XFAIL]
                if unc is StdDevUncertainty and op in ("multiply", "divide")
                else []
            ),
        )
        for unc in (StdDevUncertainty, VarianceUncertainty, InverseVariance)
        for op in ("add", "subtract", "multiply", "divide")
    ],
)
def test_wrapped_arithmetic_keeps_uncertainty_in_namespace(uncertainty_type, operation):
    data1 = [[1.0, 2.0], [3.0, 4.0]]
    data2 = [[2.0, 2.0], [4.0, 8.0]]
    unc1 = [[0.1, 0.2], [0.3, 0.4]]
    unc2 = [[0.2, 0.1], [0.4, 0.3]]

    def make(data, unc, asarray):
        return CCDData(
            asarray(data), unit=u.adu, uncertainty=uncertainty_type(asarray(unc))
        )

    ccd1 = _wrap_ccddata_for_array_api(make(data1, unc1, xp.asarray))
    ccd2 = _wrap_ccddata_for_array_api(make(data2, unc2, xp.asarray))
    result = getattr(ccd1, operation)(ccd2)

    # Reference: astropy's own propagation on plain numpy CCDData.
    ref1 = make(data1, unc1, np.asarray)
    ref2 = make(data2, unc2, np.asarray)
    expected = getattr(ref1, operation)(ref2)

    assert array_api_compat.array_namespace(result.uncertainty.array) is xp
    assert isinstance(result.uncertainty, uncertainty_type)
    assert xp.all(
        xpx.isclose(result.uncertainty.array, xp.asarray(expected.uncertainty.array))
    )
