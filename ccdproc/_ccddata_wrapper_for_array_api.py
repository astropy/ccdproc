# This file is a rough draft of the changes that will be needed
# in astropy.nddata to adopt the array API. This does not cover all
# of the changes that will be needed, but it is a start.

import array_api_compat
import numpy as np
from astropy import units as u
from astropy.nddata import (
    CCDData,
    StdDevUncertainty,
)
from astropy.nddata.compat import NDDataArray
from astropy.units import UnitsError


class _NDDataArray(NDDataArray):
    @NDDataArray.mask.setter
    def mask(self, value):
        xp = array_api_compat.array_namespace(self.data)
        # Check that value is not either type of null mask.
        if (value is not None) and (value is not np.ma.nomask):
            mask = xp.asarray(value, dtype=bool)
            if mask.shape != self.data.shape:
                raise ValueError(
                    f"dimensions of mask {mask.shape} and data "
                    f"{self.data.shape} do not match"
                )
            else:
                self._mask = mask
        else:
            # internal representation should be one numpy understands
            self._mask = np.ma.nomask


class _CCDDataWrapperForArrayAPI(CCDData):
    """
    Thin wrapper around CCDData to allow arithmetic operations with
    arbitray array API backends.
    """

    def _arithmetic_wrapper(self, operation, operand, result_unit, **kwargs):
        """ "
        Use NDDataArray for arthmetic because that does not force conversion
        to Quantity (and hence numpy array). If there are units on the operands
        then NDArithmeticMixin will convert to Quantity.
        """
        # Take the units off to make sure the arithmetic operation
        # does not try to convert to Quantity.
        if hasattr(self, "unit"):
            self_unit = self.unit
            self._unit = None
        else:
            self_unit = None

        if hasattr(operand, "unit"):
            operand_unit = operand.unit
            operand._unit = None
        else:
            operand_unit = None

        # Also take the units off of the uncertainty
        if self_unit is not None and hasattr(self.uncertainty, "unit"):
            self.uncertainty._unit = None

        if (
            operand_unit is not None
            and hasattr(operand, "uncertainty")
            and hasattr(operand.uncertainty, "unit")
        ):
            operand.uncertainty._unit = None

        _result = _NDDataArray._prepare_then_do_arithmetic(
            operation, self, operand, **kwargs
        )
        if self_unit:
            self._unit = self_unit
        if operand_unit:
            operand._unit = operand_unit
        # Also take the units off of the uncertainty
        if hasattr(self, "uncertainty") and self.uncertainty is not None:
            self.uncertainty._unit = self_unit

        if hasattr(operand, "uncertainty") and operand.uncertainty is not None:
            operand.uncertainty._unit = operand_unit

        # We need to handle the mask separately if we want to return a
        # genuine CCDDatta object and CCDData does not understand the
        # array API.
        result_mask = None
        if _result.mask is not None:
            result_mask = _result._mask
            _result._mask = None
        result = CCDData(_result, unit=result_unit)
        result._mask = result_mask
        return result

    def subtract(self, operand, xp=None, **kwargs):
        """
        Determine the right operation to use and figure out
        the units of the result.
        """
        xp = xp or array_api_compat.array_namespace(self.data)
        if not self.unit.is_equivalent(operand.unit):
            raise UnitsError("Units must be equivalent for subtraction.")
        result_unit = self.unit
        handle_mask = kwargs.pop("handle_mask", xp.logical_or)
        return self._arithmetic_wrapper(
            xp.subtract, operand, result_unit, handle_mask=handle_mask, **kwargs
        )

    def add(self, operand, xp=None, **kwargs):
        """
        Determine the right operation to use and figure out
        the units of the result.
        """
        xp = xp or array_api_compat.array_namespace(self.data)
        if not self.unit.is_equivalent(operand.unit):
            raise UnitsError("Units must be equivalent for addition.")
        result_unit = self.unit
        handle_mask = kwargs.pop("handle_mask", xp.logical_or)
        return self._arithmetic_wrapper(
            xp.add, operand, result_unit, handle_mask=handle_mask, **kwargs
        )

    def multiply(self, operand, xp=None, **kwargs):
        """
        Determine the right operation to use and figure out
        the units of the result.
        """
        xp = xp or array_api_compat.array_namespace(self.data)
        # The "1 *" below is because quantities do arithmetic properly
        # but units do not necessarily.
        if not hasattr(operand, "unit"):
            operand_unit = 1 * u.dimensionless_unscaled
        else:
            operand_unit = operand.unit
        result_unit = (1 * self.unit) * (1 * operand_unit)
        handle_mask = kwargs.pop("handle_mask", xp.logical_or)
        return self._arithmetic_wrapper(
            xp.multiply, operand, result_unit, handle_mask=handle_mask, **kwargs
        )

    def divide(self, operand, xp=None, **kwargs):
        """
        Determine the right operation to use and figure out
        the units of the result.
        """
        xp = xp or array_api_compat.array_namespace(self.data)
        if not hasattr(operand, "unit"):
            operand_unit = 1 * u.dimensionless_unscaled
        else:
            operand_unit = operand.unit
        result_unit = (1 * self.unit) / (1 * operand_unit)
        handle_mask = kwargs.pop("handle_mask", xp.logical_or)
        return self._arithmetic_wrapper(
            xp.divide, operand, result_unit, handle_mask=handle_mask, **kwargs
        )

    @NDDataArray.mask.setter
    def mask(self, value):
        xp = array_api_compat.array_namespace(self.data)
        # Check that value is not either type of null mask.
        if (value is not None) and (value is not np.ma.nomask):
            mask = xp.asarray(value, dtype=bool)
            if mask.shape != self.data.shape:
                raise ValueError(
                    f"dimensions of mask {mask.shape} and data "
                    f"{self.data.shape} do not match"
                )
            else:
                self._mask = mask
        else:
            # internal representation should be one numpy understands
            self._mask = np.ma.nomask


class _StdDevUncertaintyWrapper(StdDevUncertainty):
    """
    Override propagate methods to make sure they use the array API.
    """

    def _propagate_add(self, other_uncert, result_data, correlation):
        xp = array_api_compat.array_namespace(self.array, other_uncert.array)
        return super()._propagate_add_sub(
            other_uncert,
            result_data,
            correlation,
            subtract=False,
            to_variance=xp.square,
            from_variance=xp.sqrt,
        )

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        xp = array_api_compat.array_namespace(self.array, other_uncert.array)
        return super()._propagate_add_sub(
            other_uncert,
            result_data,
            correlation,
            subtract=True,
            to_variance=xp.square,
            from_variance=xp.sqrt,
        )

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        xp = array_api_compat.array_namespace(self.array, other_uncert.array)
        return super()._propagate_multiply_divide(
            other_uncert,
            result_data,
            correlation,
            divide=False,
            to_variance=xp.square,
            from_variance=xp.sqrt,
        )

    def _propagate_divide(self, other_uncert, result_data, correlation):
        xp = array_api_compat.array_namespace(self.array, other_uncert.array)
        return super()._propagate_multiply_divide(
            other_uncert,
            result_data,
            correlation,
            divide=True,
            to_variance=xp.square,
            from_variance=xp.sqrt,
        )


def _wrap_ccddata_for_array_api(ccd):
    """
    Wrap a CCDData object for use with array API backends.
    """
    if isinstance(ccd, _CCDDataWrapperForArrayAPI):
        return ccd

    _ccd = _CCDDataWrapperForArrayAPI(ccd)
    if isinstance(_ccd.uncertainty, StdDevUncertainty):
        _ccd.uncertainty = _StdDevUncertaintyWrapper(_ccd.uncertainty)
    return _ccd


def _unwrap_ccddata_for_array_api(ccd):
    """
    Unwrap a CCDData object from array API backends to the original CCDData.
    """

    if isinstance(ccd.uncertainty, _StdDevUncertaintyWrapper):
        ccd.uncertainty = StdDevUncertainty(ccd.uncertainty.array)

    if isinstance(ccd, CCDData):
        return ccd

    if not isinstance(ccd, _CCDDataWrapperForArrayAPI):
        raise TypeError(
            "Input must be a CCDData or _CCDDataWrapperForArrayAPI instance."
        )

    # Convert back to CCDData
    return CCDData(ccd)
