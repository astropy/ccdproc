# This file is a rough draft of the changes that will be needed
# in astropy.nddata to adopt the array API. This does not cover all
# of the changes that will be needed, but it is a start.

import array_api_compat
import numpy as np
from astropy import units as u
from astropy.nddata import (
    CCDData,
    InverseVariance,
    StdDevUncertainty,
    VarianceUncertainty,
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


class _CupyOperationNamesMixin:
    # Override the method below solely to allow CuPy operation names
    def propagate(self, operation, other_nddata, result_data, correlation, axis=None):
        """Calculate the resulting uncertainty given an operation on the data.

        .. versionadded:: 1.2

        Parameters
        ----------
        operation : callable
            The operation that is performed on the `NDData`. Supported are
            `numpy.add`, `numpy.subtract`, `numpy.multiply` and
            `numpy.true_divide` (or `numpy.divide`).

        other_nddata : `NDData` instance
            The second operand in the arithmetic operation.

        result_data : `~astropy.units.Quantity` or ndarray
            The result of the arithmetic operations on the data.

        correlation : `numpy.ndarray` or number
            The correlation (rho) is defined between the uncertainties in
            sigma_AB = sigma_A * sigma_B * rho. A value of ``0`` means
            uncorrelated operands.

        axis : int or tuple of ints, optional
            Axis over which to perform a collapsing operation.

        Returns
        -------
        resulting_uncertainty : `NDUncertainty` instance
            Another instance of the same `NDUncertainty` subclass containing
            the uncertainty of the result.

        Raises
        ------
        ValueError
            If the ``operation`` is not supported or if correlation is not zero
            but the subclass does not support correlated uncertainties.

        Notes
        -----
        First this method checks if a correlation is given and the subclass
        implements propagation with correlated uncertainties.
        Then the second uncertainty is converted (or an Exception is raised)
        to the same class in order to do the propagation.
        Then the appropriate propagation method is invoked and the result is
        returned.
        """
        # Check if the subclass supports correlation
        if not self.supports_correlated:
            if isinstance(correlation, np.ndarray) or correlation != 0:
                raise ValueError(
                    f"{type(self).__name__} does not support uncertainty propagation"
                    " with correlation."
                )

        if other_nddata is not None:
            # Get the other uncertainty (and convert it to a matching one)
            other_uncert = self._convert_uncertainty(other_nddata.uncertainty)

            if operation.__name__ in ["add", "cupy_add"]:
                result = self._propagate_add(other_uncert, result_data, correlation)
            elif operation.__name__ in ["subtract", "cupy_subtract"]:
                result = self._propagate_subtract(
                    other_uncert, result_data, correlation
                )
            elif operation.__name__ in ["multiply", "cupy_multiply"]:
                result = self._propagate_multiply(
                    other_uncert, result_data, correlation
                )
            elif operation.__name__ in [
                "true_divide",
                "cupy_true_divide",
                "divide",
                "cupy_divide",
            ]:
                result = self._propagate_divide(other_uncert, result_data, correlation)
            else:
                raise ValueError(f"unsupported operation: {operation.__name__}")
        else:
            # assume this is a collapsing operation:
            result = self._propagate_collapse(operation, axis)

        return self.__class__(array=result, copy=False)


class _StdDevUncertaintyWrapper(_CupyOperationNamesMixin, StdDevUncertainty):
    """
    Override operation propagate methods to make sure they use the array API.

    Override overall propagate method to allow cupy_-prefixed operation names.
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


class _VarianceUncertaintyWrapper(_CupyOperationNamesMixin, VarianceUncertainty):
    """This subclass is needed to allow CuPy operation names"""


class _InverseVarianceWrapper(_CupyOperationNamesMixin, InverseVariance):
    """This subclass is needed to allow CuPy operation names"""


def _wrap_uncertainty(unc):
    """
    Wrap the uncertainty for a ccd
    """
    match unc:
        case StdDevUncertainty():
            return _StdDevUncertaintyWrapper(unc)
        case VarianceUncertainty():
            return _VarianceUncertaintyWrapper(unc)
        case InverseVariance():
            return _InverseVarianceWrapper(unc)
        case _:
            raise TypeError("Unsupported uncertainty type")


def _unwrap_uncertainty(unc):
    """
    Unwrap the uncertainty for a ccd
    """
    # Unwrap using the uncertainty's .array so that the uncertainty
    # setter will automatically handle the units for the uncertainty.
    match unc:
        case _StdDevUncertaintyWrapper():
            return StdDevUncertainty(unc.array)
        case _VarianceUncertaintyWrapper():
            return VarianceUncertainty(unc.array)
        case _InverseVarianceWrapper():
            return InverseVariance(unc.array)
        case _:
            raise TypeError("Unsupported uncertainty type")


def _wrap_ccddata_for_array_api(ccd):
    """
    Wrap a CCDData object for use with array API backends.
    """
    if isinstance(ccd, _CCDDataWrapperForArrayAPI):
        if ccd.uncertainty is not None:
            ccd.uncertainty = _wrap_uncertainty(ccd.uncertainty)
        return ccd

    _ccd = _CCDDataWrapperForArrayAPI(ccd)
    if _ccd.uncertainty is not None:
        _ccd.uncertainty = _wrap_uncertainty(_ccd.uncertainty)
    return _ccd


def _unwrap_ccddata_for_array_api(ccd):
    """
    Unwrap a CCDData object from array API backends to the original CCDData.
    """

    if ccd.uncertainty is not None:
        ccd.uncertainty = _unwrap_uncertainty(ccd.uncertainty)

    if isinstance(ccd, CCDData):
        return ccd

    if not isinstance(ccd, _CCDDataWrapperForArrayAPI):
        raise TypeError(
            "Input must be a CCDData or _CCDDataWrapperForArrayAPI instance."
        )

    # Convert back to CCDData
    return CCDData(ccd)
