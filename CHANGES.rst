
0.3.0 (unreleased)
------------------

New Features
^^^^^^^^^^^^

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
- Updated CCDData to use the new version of NDDATA in astropy v1.0.   This
  breaks backward compatibility with earlier versions of astropy.  

Bug Fixes
^^^^^^^^^


0.2.2 (unreleased)
------------------

New Features
^^^^^^^^^^^^

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^


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
