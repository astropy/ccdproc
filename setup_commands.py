# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys

from setuptools import Command

build_docs_msg = """
`python setup.py build_docs` no longer builds the documentation.  Instead
you will need to run the following command:
    $ tox -e build_docs
If you don't already have tox installed, you can install it with:
    $ pip install tox
"""

test_msg = """
`python setup.py test` no longer runs the tests.  Instead you will need to
run the following command:
    $ tox -e test
If you don't already have tox installed, you can install it with:
    $ pip install tox
"""


class BuildDocs(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(build_docs_msg + '\n', file=sys.stderr)
        exit(1)


class Test(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(test_msg + '\n', file=sys.stderr)
        exit(1)


cmdclass = {'test': Test,
            'build_docs': BuildDocs}
