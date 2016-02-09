"""
Useful objects based on collections
"""

from __future__ import (absolute_import, print_function)

from collections import OrderedDict


class CaseInsensitiveOrderedDict(OrderedDict):
    """
    docstring for CaseInsensitiveOrderedDict
    """

    def __init__(self, *arg, **kwd):
        super(CaseInsensitiveOrderedDict, self).__init__(*arg, **kwd)

    def _transform_key(self, key):
        return key.lower()

    def __setitem__(self, key, value):
        super(CaseInsensitiveOrderedDict,
              self).__setitem__(self._transform_key(key), value)

    def __getitem__(self, key):
        return super(CaseInsensitiveOrderedDict,
                     self).__getitem__(self._transform_key(key))

    def __delitem__(self, key):
        return super(CaseInsensitiveOrderedDict,
                     self).__delitem__(self._transform_key(key))

    def __contains__(self, key):
        return super(CaseInsensitiveOrderedDict,
                     self).__contains__(self._transform_key(key))
