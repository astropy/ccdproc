from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from functools import wraps
import inspect
from itertools import chain

import numpy as np

from astropy.extern import six
from astropy.nddata import NDData
from astropy import units as u

import ccdproc  # really only need Keyword from ccdproc

__all__ = []

_LOG_ARGUMENT = 'add_keyword'

_LOG_ARG_HELP = \
    """
    {arg} : str, `~ccdproc.Keyword` or dict-like, optional
        Item(s) to add to metadata of result. Set to False or None to
        completely  disable logging.
        Default is to add a dictionary with a single item:
        The key is the name of this function  and the value is a string
        containing the arguments the function was called with, except the
        value of this argument.
    """.format(arg=_LOG_ARGUMENT)


def log_to_metadata(func):
    """
    Decorator that adds logging to ccdproc functions.

    The decorator adds the optional argument _LOG_ARGUMENT to function
    signature and updates the function's docstring to reflect that.

    It also sets the default value of the argument to the name of the function
    and the arguments it was called with.
    """
    func.__doc__ = func.__doc__.format(log=_LOG_ARG_HELP)

    (original_args, varargs, keywords, defaults) = inspect.getargspec(func)

    # grab the names of positional arguments for use in automatic logging
    try:
        original_positional_args = original_args[:-len(defaults)]
    except TypeError:
        original_positional_args = original_args

    # Add logging keyword and its default value for docstring
    original_args.append(_LOG_ARGUMENT)
    try:
        defaults = list(defaults)
    except TypeError:
        defaults = []
    defaults.append(True)

    signature_with_arg_added = inspect.formatargspec(original_args, varargs,
                                                     keywords, defaults)
    signature_with_arg_added = "{0}{1}".format(func.__name__,
                                               signature_with_arg_added)
    func.__doc__ = "\n".join([signature_with_arg_added, func.__doc__])

    @wraps(func)
    def wrapper(*args, **kwd):
        # Grab the logging keyword, if it is present.
        log_result = kwd.pop(_LOG_ARGUMENT, True)
        result = func(*args, **kwd)

        if not log_result:
            # No need to add metadata....
            meta_dict = {}
        elif log_result is not True:
            meta_dict = _metadata_to_dict(log_result)
        else:
            # Logging is not turned off, but user did not provide a value
            # so construct one.
            key = func.__name__
            all_args = chain(zip(original_positional_args, args),
                             six.iteritems(kwd))
            all_args = ["{0}={1}".format(name,
                                         _replace_array_with_placeholder(val))
                        for name, val in all_args]
            log_val = ", ".join(all_args)
            log_val = log_val.replace("\n", "")
            meta_dict = {key: log_val}

        for k, v in six.iteritems(meta_dict):
            result._insert_in_metadata_fits_safe(k, v)
        return result

    return wrapper


def _metadata_to_dict(arg):
    if isinstance(arg, six.string_types):
        # add the key, no value
        return {arg: None}
    elif isinstance(arg, ccdproc.Keyword):
        return {arg.name: arg.value}
    else:
        return arg


def _replace_array_with_placeholder(value):
    return_type_not_value = False
    if isinstance(value, u.Quantity):
        return_type_not_value = not value.isscalar
    elif isinstance(value, (NDData, np.ndarray)):
        try:
            length = len(value)
        except TypeError:
            # value has no length...
            try:
                # ...but if it is NDData its .data will have a length
                length = len(value.data)
            except TypeError:
                # No idea what this data is, assume length is not 1
                length = 42
        return_type_not_value = length > 1

    if return_type_not_value:
        return "<{0}>".format(value.__class__.__name__)
    else:
        return value
