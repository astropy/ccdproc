
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
