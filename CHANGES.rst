1.3.0 (Unreleased)
------------------


New Features
^^^^^^^^^^^^

- Add representation for ImageFileCollection [#475, #515]

- Added ext parameter and property to ImageFileCollection to specify the FITS extension.[#463]

- Add keywords.deleter method to ImageFileCollection [#474]

- Added ``glob_include`` and ``glob_exclude`` parameter to
  ``ImageFileCollection`` [#484]

- Add ``bitfield_to_boolean_mask`` function to convert a ``bitfield`` to a
  boolean mask (following the numpy conventions). [#460]

- Added ``gain_corrected`` option in ccd_process so that calibration
  files do not need to previously been gain corrected [#491]

- Add a new ``wcs_relax`` argument to ``CCDData.to_header()`` that is passed
  through to the ``WCS`` method of the same name to allow more flexible
  handing of headers with SIP distortion. [#501]

- ``combine`` now accepts ``numpy.ndarray`` as the input ``img_list``. [#493, #503]

- Added ``sum`` option in method for ``combime``. [#500, #508]


Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- removed ability to set ccdata.unit to None [#451]

- deprecated ``summary_info`` property of ``ImageFileCollection`` now raises
  a deprecation warning. [#486]

- Logging will include the abbreviation even if the ``meta`` attribute of
  the processed ``CCDData`` isn't a ``fits.Header``. [#528]

- The ``CCDData`` class and the functions ``fits_ccddata_reader`` and
  ``fits_ccddata_writer`` will be imported from ``astropy.nddata`` if
  astropy >= 2.0 is installed (instead of the one defined in ``ccdproc``). [#528]

- Building the documentation requires astropy >= 2.0. [#528]


Bug Fixes
^^^^^^^^^

- ``ImageFileCollection`` now handles Headers with duplicated keywords
  (other than ``COMMENT`` and ``HISTORY``) by ignoring all but the first. [#467]

- The ``ccd`` method of ``ImageFileCollection`` will raise an
  ``NotImplementedError`` in case the parameter ``overwrite=True`` or
  ``clobber=True`` is used instead of silently ignoring the parameter. [#527]

- The ``sort`` method of ``ImageFileCollection`` now requires an explicitly
  given ``keys`` argument. [#534]

- Fixed a problem with ``CCDData.read`` when the extension wasn't given and the
  primary HDU contained no ``data`` but another HDU did. In that case the header
  were not correctly combined. [#541]


1.2.0 (2016-12-13)
------------------

ccdproc has now the following additional dependency:

  - scikit-image.


New Features
^^^^^^^^^^^^

- Add an optional attribute named ``filenames`` to ``ImageFileCollection``,
  so that users can pass a list of FITS files to the collection. [#374, #403]

- Added ``block_replicate``, ``block_reduce`` and ``block_average`` functions.
  [#402]

- Added ``median_filter`` function. [#420]

- ``combine`` now takes an additional ``combine_uncertainty_function`` argument
  which is passed as ``uncertainty_func`` parameter to
  ``Combiner.median_combine`` or ``Combiner.average_combine``. [#416]

- Added ``ccdmask`` function. [#414, #432]


Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ccdprocs core functions now explicitly add HIERARCH cards. [#359, #399, #413]

- ``combine`` now accepts a ``dtype`` argument which is passed to
  ``Combiner.__init__``. [#391, #392]

- Removed ``CaseInsensitiveOrderedDict`` because it is not used in the current
  code base. [#428]


Bug Fixes
^^^^^^^^^

- The default dtype of the ``combine``-result doesn't depend on the dtype
  of the first CCDData anymore. This also corrects the memory consumption
  calculation. [#391, #392]

- ``ccd_process`` now copies the meta of the input when subtracting the
  master bias. [#404]

- Fixed ``combine`` with ``CCDData`` objects using ``StdDevUncertainty`` as
  uncertainty. [#416, #424]

- ``ccds`` generator from ``ImageFileCollection`` now uses the full path to the
  file when calling ``fits_ccddata_reader``. [#421 #422]

1.1.0 (2016-08-01)
------------------

New Features
^^^^^^^^^^^^

- Add an additional combination method, ``clip_extrema``, that drops the highest
  and/or lowest pixels in an image stack. [#356, #358]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``cosmicray_lacosmic`` default ``satlevel`` changed from 65536 to 65535. [#347]

- Auto-identify files with extension ``fts`` as FITS files. [#355, #364]

- Raise more explicit exception if unit of uncalibrated image and master do
  not match in ``subtract_bias`` or ``subtract_dark``. [#361, #366]

- Updated the ``Combiner`` class so that it could process images with >2
  dimensions. [#340, #375]

Bug Fixes
^^^^^^^^^

- ``Combiner`` creates plain array uncertainties when using``average_combine``
  or ``median_combine``. [#351]

- ``flat_correct`` does not properly scale uncertainty in the flat. [#345, #363]

- Error message in weights setter fixed. [#376]


1.0.1 (2016-03-15)
------------------

The 1.0.1 release was a release to fix some minor packaging issues.


1.0.0 (2016-03-15)
------------------

General
^^^^^^^

- ccdproc has now the following requirements:

  - Python 2.7 or 3.4 or later.
  - astropy 1.0 or later
  - numpy 1.9 or later
  - scipy
  - astroscrappy
  - reproject

New Features
^^^^^^^^^^^^

- Add a WCS setter for ``CCDData``. [#256]
- Allow user to set the function used for uncertainty calculation in
  ``average_combine`` and ``median_combine``. [#258]
- Add a new keyword to ImageFileCollection.files_filtered to return the full
  path to a file [#275]
- Added ccd_process for handling multiple steps. [#211]
- CCDData.write now writes multi-extension-FITS files. The mask and uncertainty
  are saved as extensions if these attributes were set. The name of the
  extensions can be altered with the parameters ``hdu_mask`` (default extension
  name ``'MASK'``) and ``hdu_uncertainty`` (default ``'UNCERT'``).
  CCDData.read can read these files and has the same optional parameters. [#302]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Issue warning if there are no FITS images in an ``ImageFileCollection``. [#246]
- The overscan_axis argument in subtract_overscan can now be set to
  None, to let subtract_overscan provide a best guess for the axis. [#263]
- Add support for wildcard and reversed FITS style slicing. [#265]
- When reading a FITS file with CCDData.read, if no data exists in the
  primary hdu, the resultant header object is a combination of the
  header information in the primary hdu and the first hdu with data. [#271]
- Changed cosmicray_lacosmic to use astroscrappy for cleaning cosmic rays. [#272]
- CCDData arithmetic with number/Quantity now preserves any existing WCS. [#278]
- Update astropy_helpers to 1.1.1. [#287]
- Drop support for Python 2.6. [#300]
- The ``add_keyword`` parameter now has a default of ``True``, to be more
  explicit. [#310]
- Return name of file instead of full path in ``ImageFileCollection``
  generators. [#315]


Bug Fixes
^^^^^^^^^

- Adding/Subtracting a CCDData instance with a Quantity with a different unit
  produced wrong results. [#291]
- The uncertainty resulting when combining CCDData will be divided by the
  square root of the number of combined pixel [#309]
- Improve documentation for read/write methods on ``CCDData`` [#320]
- Add correct path separator when returning full path from
  ``ImageFileCollection.files_filtered``. [#325]


0.3.3 (2015-10-24)
------------------

New Features
^^^^^^^^^^^^

- add a ``sort`` method to ImageFileCollection [#274]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Opt in to new container-based builds on travis. [#227]

- Update astropy_helpers to 1.0.5. [#245]

Bug Fixes
^^^^^^^^^

- Ensure that creating a WCS from a header that contains list-like keywords
  (e.g. ``BLANK`` or ``HISTORY``) succeeds. [#229, #231]

0.3.2 (never released)
----------------------

There was no 0.3.2 release because of a packaging error.

0.3.1 (2015-05-12)
------------------

New Features
^^^^^^^^^^^^

- Add CCDData generator for ImageCollection [#405]

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

- Allow the unit string "adu" to be upper or lower case in a FITS header [#182]

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
