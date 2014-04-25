from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from functools import wraps

#from decorator import decorator
from astropy.extern import six

import ccdproc.ccdproc  # really only need Keyword from ccdproc

_LOG_ARGUMENT = 'add_keyword'

_LOG_ARG_HELP = \
    """
    {arg} : str, ~ccdproc.ccdproc.Keyword or dict-like
        Item(s) to add to metadata of result.
    """.format(arg=_LOG_ARGUMENT)


def log_to_metadata(func):
    """
    Decorator that adds logging to ccdproc functions
    """
    func.__doc__ = func.__doc__.format(log=_LOG_ARG_HELP)

    @wraps(func)
    def wrapper(*args, **kwd):
        log_result = kwd.pop(_LOG_ARGUMENT, False)
        result = func(*args, **kwd)
        if log_result:
            _insert_in_metadata(result.meta, log_result)
        return result
    return wrapper


def _insert_in_metadata(metadata, arg):
    if isinstance(arg, basestring):
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
