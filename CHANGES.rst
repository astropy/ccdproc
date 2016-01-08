
0.4.0 (unreleased)
------------------

New Features
^^^^^^^^^^^^

- Add a new keyword to ImageFileCollection.files_filtered to return the full
  path to a file [#275]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes
^^^^^^^^^

0.3.2 (unreleased)
------------------

New Features
^^^^^^^^^^^^

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Opt in to new container-based builds on travis. [#227]

- Update astropy_helpers to 1.0.5. [#245]

Bug Fixes
^^^^^^^^^

- Ensure that creating a WCS from a header that contains list-like keywords
  (e.g. ``BLANK`` or ``HISTORY``) succeeds. [#229, #231]

0.3.1 (2015-05-12)
------------------

New Features
^^^^^^^^^^^^

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Add extensive tests to ensure ``ccdproc`` functions do not modify the input
  data. [#208]

- Remove red-box warning about API stability from docs. [#210]

- Support astropy 1.0.5, which made changes to ``NDData``. [#242]

Bug Fixes
^^^^^^^^^

- Make ``subtract_overscan`` act on a copy of the input data. [#206]

- Overscan subtraction failed on non-square images if the overscan axis was the
  first index, ``0``. [#240, #244]

0.3.0 (2015-03-17)
------------------

New Features
^^^^^^^^^^^^

- When reading in a FITS file, the extension to be used can be specified.  If
  it is not and there is no data in the primary extension, the first extension
  with data will be used.

- Set wcs attribute when reading from a FITS file that contains WCS keywords
  and write WCS keywords to header when converting to an HDU. [#195]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Updated CCDData to use the new version of NDDATA in astropy v1.0.   This
  breaks backward compatibility with earlier versions of astropy.

Bug Fixes
^^^^^^^^^

- Ensure ``dtype`` of combined images matches the ``dtype`` of the
  ``Combiner`` object. [#189]

0.2.2 (2014-11-05)
------------------

New Features
^^^^^^^^^^^^

- Add dtype argument to `ccdproc.Combiner` to help control memory use [#178]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Added Changes to the docs [#183]

Bug Fixes
^^^^^^^^^

- Allow the unit string "adu" to be upper or lower case in a FIS header [#182]

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
