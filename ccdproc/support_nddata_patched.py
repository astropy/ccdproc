# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy
from itertools import islice
import warnings

from astropy.utils import wraps
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.compat.funcsigs import signature
from astropy.extern import six
from astropy.extern.six.moves import zip

from astropy.nddata import NDData

__all__ = ['support_nddata']


# All supported properties except "data" which is mandatory!
SUPPORTED_PROPERTIES = ['data', 'uncertainty', 'mask', 'meta', 'unit', 'wcs',
                        'flags']


def support_nddata(_func=None, accepts=NDData,
                   repack=False, returns=None, keeps=None,
                   **attribute_argument_mapping):
    """Decorator to wrap functions that could accept an unwrapped NDData
    instance.

    Parameters
    ----------
    _func : callable, None, optional
        The function to decorate or ``None`` if used as factory.
        Default is ``None``.

    accepts : cls, optional
        The class or subclass of ``NDData`` that should be unpacked before
        calling the function.
        Default is ``NDData``

    repack : bool, optional
        Should be ``True`` if the return should be converted to the input
        class again after the wrapped function call.
        Default is ``False``.

        .. note::
           Must be ``True`` if either one of ``returns`` or ``keeps``
           is specified.

    returns : iterable, None, optional
        An iterable containing strings which returned value should be set
        on the class. For example if a function returns data and mask, this
        should be ``['data', 'mask']``. If ``None`` assume the function only
        returns one argument: ``'data'``.
        Default is ``None``.

        .. note::
           Must be ``None`` if ``repack=False``.

    keeps : iterable. None, optional
        An iterable containing strings that indicate which values should be
        copied from the original input to the returned class. If ``None``
        assume that no attributes are copied.
        Default is ``None``.

        .. note::
           Must be ``None`` if ``repack=False``.

        .. warning::
           If the decorated function should work with `CCDData`, you *probably*
           need to specify ``keeps=['unit']``, unless the function returns a
           `~astropy.units.Quantity` or CCDData-like object with unit.

    attribute_argument_mapping :
        Keyword parameters that optionally indicate which function argument
        should be interpreted as which attribute on the input. By default
        it assumes the function takes a ``data`` argument as first argument,
        but if the first argument is called ``input`` one should pass
        ``support_nddata(..., data='input')`` to the function.

    Returns
    -------
    decorator_factory or decorated_function : callable
        If ``_func=None`` this returns a decorator, otherwise it returns the
        decorated ``_func``.

    Notes
    -----
    This is a slightly modified version of `~astropy.nddata.support_nddata`, so
    for more hints checkout their documentation or have a look at the
    ``ccdproc.core.py`` code.
    """
    if (returns is not None or keeps is not None) and not repack:
        raise ValueError('returns or keep should only be set if repack=True.')
    elif returns is None and repack:
        raise ValueError('returns should be set if repack=True.')
    else:
        returns = [] if returns is None else returns
        keeps = [] if keeps is None else keeps

    # Short version to avoid the long variable name later.
    attr_arg_map = attribute_argument_mapping
    if any(keep in returns for keep in keeps):
        raise ValueError("cannot specify the same attribute in `returns` and "
                         "`keeps`.")
    all_returns = returns + keeps

    def support_nddata_decorator(func):
        # Find out args and kwargs
        func_args = []
        func_kwargs = []
        for param_name, param in six.iteritems(signature(func).parameters):
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise ValueError("func may not have *args or **kwargs.")
            elif param.default == param.empty:
                func_args.append(param_name)
            else:
                func_kwargs.append(param_name)

        # First argument should be data
        if (not func_args or func_args[0] != attr_arg_map.get('data', 'data')):
            raise ValueError("Can only wrap functions whose first positional "
                             "argument is `{0}`."
                             "".format(attr_arg_map.get('data', 'data')))

        # Get all supported properties that match a parameter in the signature.
        supported_properties = [prop for prop in islice(SUPPORTED_PROPERTIES, 1, None)
                                if attr_arg_map.get(prop, prop) in func_kwargs]

        """
        # Create a Table to store the information about the wrapped function.
        # Can be used to create a Table that can be inserted in the docstring.
        # Note: Creating and writing an astropy Table takes very long so
        #       creating the docstring at import time may be a severe time
        #       consumer...
        #       Maybe worth investigating if some templating engine might
        #       generate them faster.
        from astropy.io.ascii import RST
        from astropy.table import Table

        _names = SUPPORTED_PROPERTIES
        _used, _calc, _copy = [], [], []
        for prop in _names:
            _used.append('X' if prop in supported_properties else '--')
            _calc.append('X' if prop in returns else '--')
            _copy.append('X' if prop in keeps else '--')
        # Data is always used!
        _used[0] = 'X'
        _tbl = Table([_names, _used, _calc, _copy],
                     names=('attribute', 'used', 'calculated', 'copied'))
        _tbl = ascii.RST().write(_tbl)
        _doc = '\n'.join(_tbl)
        print(_doc)  # print to get the nice output.
        # # End of Table creation.
        """

        @wraps(func)
        def wrapper(data, *args, **kwargs):
            unpack = isinstance(data, accepts)
            input_data = data
            if not unpack and isinstance(data, NDData):
                raise TypeError("Only NDData sub-classes that inherit from {0}"
                                " can be used by this function."
                                "".format(accepts.__name__))

            # If data is an NDData instance, we can try and find properties
            # that can be passed as kwargs.
            if unpack:
                # We loop over a list of pre-defined properties
                for prop in supported_properties:
                    # We only need to do something if the property exists on
                    # the NDData object
                    try:
                        value = getattr(data, prop)
                    except AttributeError:
                        continue
                    # Skip if the property exists but is None or empty.
                    if prop == 'meta' and not value:
                        continue
                    if prop != 'meta' and value is None:
                        continue
                    # Warn if the property is set and explicitly given
                    propmatch = attr_arg_map.get(prop, prop)
                    if propmatch in kwargs and kwargs[propmatch] is not None:
                        warnings.warn(
                            "Property {0} has been passed explicitly and as an"
                            " NDData property {1}, using explicitly specified "
                            "value.".format(propmatch, prop),
                            AstropyUserWarning)
                        continue
                    # Otherwise use the property as input for the function.
                    kwargs[propmatch] = value
                # Finally, replace data by the data itself
                data = data.data

            result = func(data, *args, **kwargs)

            if unpack and repack:
                # If there are multiple required returned arguments make sure
                # the result is a tuple (because we don't want to unpack
                # numpy arrays or compare to their length, never!) and has the
                # same length.
                if len(returns) > 1:
                    if (not isinstance(result, tuple) or
                            len(returns) != len(result)):
                        raise ValueError("Function did not return the "
                                         "expected number of arguments.")
                elif len(returns) == 1:
                    result = [result]
                if keeps is not None:
                    for keep in keeps:
                        result.append(deepcopy(getattr(input_data, keep)))
                resultdata = result[all_returns.index('data')]
                resultkwargs = {ret: res
                                for ret, res in zip(all_returns, result)
                                if ret != 'data'}
                return input_data.__class__(resultdata, **resultkwargs)
            else:
                return result
        return wrapper

    # If _func is set, this means that the decorator was used without
    # parameters so we have to return the result of the
    # support_nddata_decorator decorator rather than the decorator itself
    if _func is not None:
        return support_nddata_decorator(_func)
    else:
        return support_nddata_decorator
