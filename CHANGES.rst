
0.3.0 (unreleased)
------------------

New Features
^^^^^^^^^^^^

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes
^^^^^^^^^


0.2.3 (unreleased)
------------------

New Features
^^^^^^^^^^^^

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes
^^^^^^^^^


0.2.2 (2014-11-05)
------------------

New Features
^^^^^^^^^^^^

- Add dtype argument to `ccdproc.Combiner` to help control memory use [#178]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Allow the unit string "adu" to be upper or lower case in a FIS header [#182]

Bug Fixes
^^^^^^^^^


0.2.1 (2014-09-09)
------------------

New Features
^^^^^^^^^^^^

- Add a unit directly from BUNIT if it is available in the FITS header [#169]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Relaxed the requirements on what the metadata must be. It can be anything dict-like, e.g. an astropy.io.fits.Header, a python dict, an OrderedDict or some custom object created by the user. [#167]

Bug Fixes
^^^^^^^^^

- Fixed a new-style formating issue in the logging [#170]


0.2 (2014-07-28)
----------------

- Initial release.
