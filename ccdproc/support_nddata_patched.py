# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy
import warnings

from astropy.utils import wraps
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.compat.funcsigs import signature
from astropy.extern.six.moves import zip

from astropy.nddata import NDData

__all__ = ['support_nddata']


# All supported properties except "data" which is mandatory!
SUPPORTED_PROPERTIES = ['uncertainty', 'mask', 'meta', 'unit', 'wcs']


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
    if (returns is not None or keeps is not None or
            attribute_argument_mapping) and not repack:
        raise ValueError('returns or keep should only be set if repack=True')
    if returns is None and repack:
        raise ValueError('returns should be set if repack=True')

    # Short version so the
    attr_arg_map = attribute_argument_mapping
    if keeps is not None:
        all_returns = returns + keeps
        if any(keep in returns for keep in keeps):
            raise TypeError("cannot specify the same attribute in `returns` "
                            "and `keeps`.")
    else:
        keeps = []

    def support_nddata_decorator(func):
        # Find out args and kwargs
        sig = signature(func)
        func_args = []
        func_kwargs = []
        for param in sig.parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise ValueError("func may not have *args or **kwargs")
            elif param.default == param.empty:
                func_args.append(param.name)
            else:
                func_kwargs.append(param.name)

        # First argument should be data
        if (len(func_args) == 0 or
                func_args[0] != attr_arg_map.get('data', 'data')):
            raise ValueError("Can only wrap functions whose first positional "
                             "argument is `{0}`"
                             "".format(attr_arg_map.get('data', 'data')))

        @wraps(func)
        def wrapper(data, *args, **kwargs):
            unpack = isinstance(data, accepts)
            input_data = data
            if not unpack and isinstance(data, NDData):
                raise TypeError("Only NDData sub-classes that inherit from {0}"
                                " can be used by this function"
                                "".format(accepts.__name__))

            # If data is an NDData instance, we can try and find properties
            # that can be passed as kwargs.
            if unpack:
                # We loop over a list of pre-defined properties
                for prop in SUPPORTED_PROPERTIES:
                    # We only need to do something if the property exists on
                    # the NDData object
                    if hasattr(data, prop):
                        value = getattr(data, prop)
                        if ((prop == 'meta' and len(value) > 0) or
                                (prop != 'meta' and value is not None)):
                            propmatch = attr_arg_map.get(prop, prop)
                            if propmatch in func_kwargs:
                                if prop in kwargs and kwargs[prop] is not None:
                                    warnings.warn(
                                        "Property {0} has been passed "
                                        "explicitly and as an NDData property "
                                        ", using explicitly specified value"
                                        "".format(prop), AstropyUserWarning)
                                else:
                                    kwargs[propmatch] = value
                # Finally, replace data by the data itself
                data = data.data

            result = func(data, *args, **kwargs)

            if unpack and repack:
                if len(returns) > 1 and len(returns) != len(result):
                    raise ValueError("Function did not return the expected "
                                     "number of arguments")
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
