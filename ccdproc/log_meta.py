from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from functools import wraps
import inspect

from astropy.extern import six

import ccdproc.ccdproc  # really only need Keyword from ccdproc

_LOG_ARGUMENT = 'add_keyword'

_LOG_ARG_HELP = \
    """
    {arg} : str, `~ccdproc.ccdproc.Keyword` or dict-like, optional
        Item(s) to add to metadata of result. Set to None to completely
        disable logging. Default is to add a dictionary with a single item:
        the key is the name of this function  and the value is a string
        containing the arguments the function was called with, except the
        value of this argument. 
    """.format(arg=_LOG_ARGUMENT)


def log_to_metadata(func):
    """
    Decorator that adds logging to ccdproc functions

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
    defaults.append(None)

    signature_with_arg_added = inspect.formatargspec(original_args, varargs,
                                                     keywords, defaults)
    signature_with_arg_added = "{0}{1}".format(func.__name__,
                                               signature_with_arg_added)
    func.__doc__ = "\n".join([signature_with_arg_added, func.__doc__])

    @wraps(func)
    def wrapper(*args, **kwd):
        # Grab the logging keyword, if it is present.
        log_result = kwd.pop(_LOG_ARGUMENT, False)
        result = func(*args, **kwd)

        if log_result:
            _insert_in_metadata(result.meta, log_result)
        elif not log_result and log_result is not None:
            # Logging is not turned off, but user did not provide a value
            # so construct one.
            key = func.__name__
            pos_args = ["{}={}".format(arg_name, arg_value)
                        for arg_name, arg_value
                        in zip(original_positional_args, args)]
            kwd_args = ["{}={}".format(k, v) for k, v in six.iteritems(kwd)]
            pos_args.extend(kwd_args)
            log_val = " ".join(pos_args)
            log_val = log_val.replace("\n", "")
            to_log = {key: log_val}
            _insert_in_metadata(result.meta, to_log)
        return result
    return wrapper


def _insert_in_metadata(metadata, arg):
    if isinstance(arg, six.string_types):
        # add the key, no value
        metadata[arg] = None
    elif isinstance(arg, ccdproc.ccdproc.Keyword):
        metadata[arg.name] = arg.value
    else:
        try:
            for k, v in six.iteritems(arg):
                metadata[k] = v
        except AttributeError:
            raise
