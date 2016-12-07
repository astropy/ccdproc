from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import fnmatch
from os import listdir, path
import logging

import numpy as np
import numpy.ma as ma

from astropy.table import Table
import astropy.io.fits as fits
from astropy.extern import six

import warnings
from astropy.utils.exceptions import AstropyUserWarning

from .ccddata import fits_ccddata_reader

logger = logging.getLogger(__name__)

__all__ = ['ImageFileCollection']
__doctest_skip__ = ['*']


class ImageFileCollection(object):
    """
    Representation of a collection of image files.

    The class offers a table summarizing values of
    keywords in the FITS headers of the files in the collection and offers
    convenient methods for iterating over the files in the collection. The
    generator methods use simple filtering syntax and can automate storage
    of any FITS files modified in the loop using the generator.

    Parameters
    ----------
    location : str or None, optional
        Path to directory containing FITS files.
        Default is ``None``.

    keywords : list of str, '*' or None, optional
        Keywords that should be used as column headings in the summary table.
        If the value is or includes '*' then all keywords that appear in any
        of the FITS headers of the files in the collection become table
        columns. Default value is '*' unless ``info_file`` is specified.
        Default is ``None``.

    info_file : str or None, optional
        Path to file that contains a table of information about FITS files.
        In this case the keywords are set to the names of the columns of the
        ``info_file`` unless ``keywords`` is explicitly set to a different
        list.
        Default is ``None``.

    filenames: str, list of str, or None, optional
        List of the names of FITS files which will be added to the collection.
        The filenames are assumed to be in ``location``.
        Default is ``None``.

    Raises
    ------
    ValueError
        Raised if keywords are set to a combination of '*' and any other
        value.
    """
    def __init__(self, location=None, keywords=None, info_file=None,
                 filenames=None):
        self._location = location
        self._filenames = filenames
        self._files = []
        if location:
            self._files = self._get_files()

        if self._files == []:
            warnings.warn("no FITS files in the collection.",
                          AstropyUserWarning)
        self._summary_info = {}
        if keywords is None:
            if info_file is not None:
                # Default to empty list so that keywords will be populated
                # from table columns names.
                keywords = []
            else:
                # Otherwise use all keywords.
                keywords = '*'
        if info_file is not None:
            try:
                info_path = path.join(self.location, info_file)
            except (AttributeError, TypeError):
                info_path = info_file
            try:
                self._summary_info = Table.read(info_path,
                                                format='ascii',
                                                delimiter=',')
                self._summary_info = Table(self._summary_info,
                                           masked=True)
            except IOError:
                if location:
                    logger.warning('unable to open table file %s, will try '
                                   'initializing from location instead.',
                                   info_path)
                else:
                    raise

        # Used internally to keep track of whether the user asked for all
        # keywords or a specific list. The keywords setter takes care of
        # actually setting the correct value, this just ensure that there
        # is always *some* value.
        self._all_keywords = False

        if keywords:
            self.keywords = keywords

    @property
    def summary(self):
        """
        `~astropy.table.Table` of values of FITS keywords for files in the
        collection.

        Each keyword is a column heading. In addition, there is a column
        called ``file`` that contains the name of the FITS file. The directory
        is not included as part of that name.

        The first column is always named ``file``.

        The order of the remaining columns depends on how the summary was
        constructed.

        If a wildcard, ``*`` was used then the order is the order in which
        the keywords appear in the FITS files from which the summary is
        constructed.

        If an explicit list of keywords was supplied in setting up the
        collection then the order of the columns is the order of the
        keywords.
        """
        return self._summary_info

    @property
    def summary_info(self):
        """
        Deprecated -- use `summary` instead -- `~astropy.table.Table` of values
        of FITS keywords for files in the collection.

        Each keyword is a column heading. In addition, there is a column
        called 'file' that contains the name of the FITS file. The directory
        is not included as part of that name.
        """
        return self._summary_info

    @property
    def location(self):
        """
        str, Path name to directory containing FITS files.
        """
        return self._location

    @property
    def keywords(self):
        """
        list of str, Keywords currently in the summary table.

        Setting the keywords causes the summary table to be regenerated unless
        the new keywords are a subset of the old.
        """
        if self.summary_info:
            return self.summary_info.keys()
        else:
            return []

    @keywords.setter
    def keywords(self, keywords=None):
        # since keywords are drawn from self.summary_info, setting
        # summary_info sets the keywords.
        if keywords is None:
            self._summary_info = []
            return

        if keywords == '*':
            self._all_keywords = True
        else:
            self._all_keywords = False

        logging.debug('keywords in setter before pruning: %s.', keywords)

        # remove duplicates and force a copy
        new_keys = list(set(keywords))

        logging.debug('keywords after pruning %s.', new_keys)

        full_new_keys = list(set(new_keys))
        full_new_keys.append('file')
        full_new_set = set(full_new_keys)
        current_set = set(self.keywords)
        if full_new_set.issubset(current_set):
            logging.debug('table columns before trimming: %s.',
                          ' '.join(current_set))
            cut_keys = current_set.difference(full_new_set)
            logging.debug('will try removing columns: %s.',
                          ' '.join(cut_keys))
            for key in cut_keys:
                self._summary_info.remove_column(key)
            logging.debug('after removal column names are: %s.',
                          ' '.join(self.keywords))
        else:
            logging.debug('should be building new table...')
            # Reorder the keywords to match the initial ordering.
            new_keys.sort(key=keywords.index)
            self._summary_info = self._fits_summary(header_keywords=new_keys)

    @property
    def files(self):
        """
        list of str, Unfiltered list of FITS files in location.
        """
        return self._files

    def values(self, keyword, unique=False):
        """
        List of values for a keyword.

        Parameters
        ----------
        keyword : str
            Keyword (i.e. table column) for which values are desired.

        unique : bool, optional
            If True, return only the unique values for the keyword.
            Default is ``False``.

        Returns
        -------
        list
            Values as a list.
        """
        if keyword not in self.keywords:
            raise ValueError(
                'keyword %s is not in the current summary' % keyword)

        if unique:
            return list(set(self.summary_info[keyword]))
        else:
            return list(self.summary_info[keyword])

    def files_filtered(self, **kwd):
        """Determine files whose keywords have listed values.

        ``**kwd`` is list of keywords and values the files must have.

        If the keyword ``include_path=True`` is set, the returned list
        contains not just the filename, but the full path to each file.

        The value '*' represents any value.
        A missing keyword is indicated by value ''.

        Example::

            >>> keys = ['imagetyp','filter']
            >>> collection = ImageFileCollection('test/data', keywords=keys)
            >>> collection.files_filtered(imagetyp='LIGHT', filter='R')
            >>> collection.files_filtered(imagetyp='*', filter='')

        NOTE: Value comparison is case *insensitive* for strings.
        """
        # force a copy by explicitly converting to a list
        current_file_mask = list(self.summary_info['file'].mask)

        include_path = kwd.pop('include_path', False)

        self._find_keywords_by_values(**kwd)
        filtered_files = self.summary_info['file'].compressed()
        self.summary_info['file'].mask = current_file_mask
        if include_path:
            filtered_files = [path.join(self._location, f)
                              for f in filtered_files]
        return filtered_files

    def refresh(self):
        """
        Refresh the collection by re-reading headers.
        """
        keywords = '*' if self._all_keywords else self.keywords
        # Re-load list of files
        self._files = self._get_files()
        self._summary_info = self._fits_summary(header_keywords=keywords)

    def sort(self, keys=None):
        """Sort the list of files to determine the order of iteration.

        Sort the table of files according to one or more keys. This does not
        create a new object, instead is sorts in place.

        Parameters
        ----------
        keys : str, list of str or None, optional
            The key(s) to order the table by.
            Default is ``None``.
        """
        if len(self._summary_info) > 0:
            self._summary_info.sort(keys)
            self._files = list(self.summary_info['file'])

    def _get_files(self):
        """ Helper method which checks whether ``files`` should be set
        to a subset of file names or to all file names in a directory.

        Returns
        -------
        files : list or str
            List of file names which will be added to the collection.
        """
        files = []
        if self._filenames:
            if isinstance(self._filenames, six.string_types):
                files.append(self._filenames)
            else:
                files = self._filenames
        else:
            files = self._fits_files_in_directory()

        return files

    def _dict_from_fits_header(self, file_name, input_summary=None,
                               missing_marker=None):
        """
        Construct an ordered dictionary whose keys are the header keywords
        and values are a list of the values from this file and the input
        dictionary. If the input dictionary is ordered then that order is
        preserved.

        Parameters
        ----------
        file_name : str
            Name of FITS file.

        input_summary : dict or None, optional
            Existing dictionary to which new values should be appended.
            Default is ``None``.

        missing_marker : any type, optional
            Fill value for missing header-keywords.
            Default is ``None``.

        Returns
        -------
        file_table : `~astropy.table.Table`
        """
        from collections import OrderedDict

        def _add_val_to_dict(key, value, tbl_dict, n_previous, missing_marker):
            key = key.lower()
            try:
                tbl_dict[key].append(value)
            except KeyError:
                tbl_dict[key] = [missing_marker] * n_previous
                tbl_dict[key].append(value)

        if input_summary is None:
            summary = OrderedDict()
            n_previous = 0
        else:
            summary = input_summary
            n_previous = len(summary['file'])

        h = fits.getheader(file_name)
        assert 'file' not in h

        # Try opening header before this so that file name is only added if
        # file is valid FITS
        try:
            summary['file'].append(path.basename(file_name))
        except KeyError:
            summary['file'] = [path.basename(file_name)]

        missing_in_this_file = [k for k in summary if (k not in h and
                                                       k != 'file')]

        multi_entry_keys = {'comment': [],
                            'history': []}

        for k, v in six.iteritems(h):
            if k == '':
                continue

            if k.lower() in ['comment', 'history']:
                multi_entry_keys[k.lower()].append(str(v))
                # Accumulate these in a separate dictionary until the
                # end to avoid adding multiple entries to summary.
                continue
            else:
                val = v

            _add_val_to_dict(k, val, summary, n_previous, missing_marker)

        for k, v in six.iteritems(multi_entry_keys):
            if v:
                joined = ','.join(v)
                _add_val_to_dict(k, joined, summary, n_previous,
                                 missing_marker)

        for missing in missing_in_this_file:
            summary[missing].append(missing_marker)

        return summary

    def _set_column_name_case_to_match_keywords(self, header_keys,
                                                summary_table):
        key_name_dict = {k.lower(): k for k in header_keys
                         if k != k.lower()}

        for lcase, user_case in six.iteritems(key_name_dict):
            try:
                summary_table.rename_column(lcase, user_case)
            except KeyError:
                pass

    def _fits_summary(self, header_keywords=None):
        """
        Generate a summary table of keywords from FITS headers.

        Parameters
        ----------
        header_keywords : list of str, '*' or None, optional
            Keywords whose value should be extracted from FITS headers.
            Default value is ``None``.
        """
        from astropy.table import MaskedColumn

        if not self.files:
            return None

        # Make sure we have a list...for example, in python 3, dict.keys()
        # is not a list.
        original_keywords = list(header_keywords)

        # Get rid of any duplicate keywords, also forces a copy.
        header_keys = set(original_keywords)
        header_keys.add('file')

        file_name_column = MaskedColumn(name='file', data=self.files)

        if not header_keys or (header_keys == set(['file'])):
            summary_table = Table(masked=True)
            summary_table.add_column(file_name_column)
            return summary_table

        summary_dict = None
        missing_marker = None

        for file_name in file_name_column:
            file_path = path.join(self.location, file_name)
            try:
                # Note: summary_dict is an OrderedDict, so should preserve
                # the order of the keywords in the FITS header.
                summary_dict = self._dict_from_fits_header(
                    file_path, input_summary=summary_dict,
                    missing_marker=missing_marker)
            except IOError as e:
                logger.warning('unable to get FITS header for file %s: %s.',
                               file_path, e)
                continue

        summary_table = Table(summary_dict, masked=True)

        for column in summary_table.colnames:
            summary_table[column].mask = [v is missing_marker
                                          for v in summary_table[column]]

        self._set_column_name_case_to_match_keywords(header_keys,
                                                     summary_table)
        missing_columns = header_keys - set(summary_table.colnames)
        missing_columns -= set(['*'])

        length = len(summary_table)
        for column in missing_columns:
            all_masked = MaskedColumn(name=column, data=np.zeros(length),
                                      mask=np.ones(length))
            summary_table.add_column(all_masked)

        if '*' not in header_keys:
            # Rearrange table columns to match order of keywords.
            # File always comes first.
            header_keys -= set(['file'])
            original_order = ['file'] + sorted(header_keys,
                                               key=original_keywords.index)
            summary_table = summary_table[original_order]

        if not summary_table.masked:
            summary_table = Table(summary_table, masked=True)

        return summary_table

    def _find_keywords_by_values(self, **kwd):
        """
        Find files whose keywords have given values.

        `**kwd` is list of keywords and values the files must have.

        The value '*' represents any value.
        A missing keyword is indicated by value ''

        Example::

            >>> keys = ['imagetyp','filter']
            >>> collection = ImageFileCollection('test/data', keywords=keys)
            >>> collection.files_filtered(imagetyp='LIGHT', filter='R')
            >>> collection.files_filtered(imagetyp='*', filter='')

        NOTE: Value comparison is case *insensitive* for strings.
        """
        keywords = kwd.keys()
        values = kwd.values()

        if (set(keywords).issubset(set(self.keywords))):
            # we already have the information in memory
            use_info = self.summary_info
        else:
            # we need to load information about these keywords.
            use_info = self._fits_summary(header_keywords=keywords)

        matches = np.array([True] * len(use_info))
        for key, value in zip(keywords, values):
            logger.debug('key %s, value %s', key, value)
            logger.debug('value in table %s', use_info[key])
            value_missing = use_info[key].mask
            logger.debug('value missing: %s', value_missing)
            value_not_missing = np.logical_not(value_missing)
            if value == '*':
                have_this_value = value_not_missing
            elif value is not None:
                if isinstance(value, six.string_types):
                    # need to loop explicitly over array rather than using
                    # where to correctly do string comparison.
                    have_this_value = np.array([False] * len(use_info))
                    for idx, file_key_value in enumerate(use_info[key]):
                        if value_not_missing[idx]:
                            value_matches = (file_key_value.lower() ==
                                             value.lower())
                        else:
                            value_matches = False

                        have_this_value[idx] = (value_not_missing[idx] &
                                                value_matches)
                else:
                    have_this_value = value_not_missing
                    tmp = (use_info[key][value_not_missing] == value)
                    have_this_value[value_not_missing] = tmp
                    have_this_value &= value_not_missing
            else:
                # this case--when value==None--is asking for the files which
                # are missing a value for this keyword
                have_this_value = value_missing

            matches &= have_this_value

        # the numpy convention is that the mask is True for values to
        # be omitted, hence use ~matches.
        logger.debug('Matches: %s', matches)
        self.summary_info['file'].mask = ma.nomask
        self.summary_info['file'][~matches] = ma.masked

    def _fits_files_in_directory(self, extensions=None,
                                 compressed=True):
        """
        Get names of FITS files in directory, based on filename extension.

        Parameters
        ----------
        extension : list of str or None, optional
            List of filename extensions that are FITS files. Default is
            ``['fit', 'fits', 'fts']``.
            Default is ``None``.

        compressed : bool, optional
            If ``True``, compressed files should be included in the list
            (e.g. `.fits.gz`).
            Default is ``True``.

        Returns
        -------
        list
            *Names* of the files (with extension), not the full pathname.
        """
        from .ccddata import _recognized_fits_file_extensions

        full_extensions = extensions or list(_recognized_fits_file_extensions)

        if compressed:
            with_gz = [extension + '.gz' for extension in full_extensions]
            full_extensions.extend(with_gz)

        all_files = listdir(self.location)
        files = []
        for extension in full_extensions:
            files.extend(fnmatch.filter(all_files, '*' + extension))

        files.sort()
        return files

    def _generator(self, return_type,
                   save_with_name="", save_location='',
                   clobber=False,
                   overwrite=False,
                   do_not_scale_image_data=True,
                   return_fname=False,
                   ccd_kwargs=None,
                   **kwd):
        """
        Generator that yields each {name} in the collection.

        If any of the parameters ``save_with_name``, ``save_location`` or
        ``overwrite`` evaluates to ``True`` the generator will write a copy of
        each FITS file it is iterating over. In other words, if
        ``save_with_name`` and/or ``save_location`` is a string with non-zero
        length, and/or ``overwrite`` is ``True``, a copy of each FITS file will
        be made.

        Parameters
        ----------
        save_with_name : str, optional
            string added to end of file name (before extension) if
            FITS file should be saved after iteration. Unless
            ``save_location`` is set, files will be saved to location of
            the source files ``self.location``.
            Default is ``''``.

        save_location : str, optional
            Directory in which to save FITS files; implies that FITS
            files will be saved. Note this provides an easy way to
            copy a directory of files--loop over the {name} with
            ``save_location`` set.
            Default is ``''``.

        overwrite : bool, optional
            If ``True``, overwrite input FITS files.
            Default is ``False``.

        clobber : bool, optional
            Alias for ``overwrite``.
            Default is ``False``.

        do_not_scale_image_data : bool, optional
            If ``True``, prevents fits from scaling images. Default is
            ``{default_scaling}``.
            Default is ``True``.

        return_fname : bool, optional
            If True, return the tuple (header, file_name) instead of just
            header. The file name returned is the name of the file only,
            not the full path to the file.
            Default is ``False``.

        ccd_kwargs : dict, optional
            Dict with parameters for `~ccdproc.fits_ccddata_reader`.
            For instance, the key ``'unit'`` can be used to specify the unit
            of the data. If ``'unit'`` is not given then ``'adu'`` is used as
            the default unit.
            See `~ccdproc.fits_ccddata_reader` for a complete list of
            parameters that can be passed through ``ccd_kwargs``.

        kwd :
            Any additional keywords are used to filter the items returned; see
            Examples for details.

        Returns
        -------
        {return_type}
            If ``return_fname`` is ``False``, yield the next {name} in the
            collection.

        ({return_type}, str)
            If ``return_fname`` is ``True``, yield a tuple of
            ({name}, ``file name``) for the next item in the collection.
        """
        # store mask so we can reset at end--must COPY, otherwise
        # current_mask just points to the mask of summary_info
        if not self.summary_info:
            return

        current_mask = {}
        for col in self.summary_info.columns:
            current_mask[col] = self.summary_info[col].mask

        if kwd:
            self._find_keywords_by_values(**kwd)

        ccd_kwargs = ccd_kwargs or {}

        for full_path in self._paths():
            no_scale = do_not_scale_image_data
            hdulist = fits.open(full_path,
                                do_not_scale_image_data=no_scale)

            file_name = path.basename(full_path)

            return_options = {
                    'header': lambda: hdulist[0].header,
                    'hdu': lambda: hdulist[0],
                    'data': lambda: hdulist[0].data,
                    'ccd': lambda: fits_ccddata_reader(full_path, **ccd_kwargs)
                    }
            try:
                yield (return_options[return_type]()  # pragma: no branch
                       if (not return_fname) else
                       (return_options[return_type](), file_name))
            except KeyError:
                raise ValueError('no generator for {}'.format(return_type))

            if save_location:
                destination_dir = save_location
            else:
                destination_dir = path.dirname(full_path)
            basename = path.basename(full_path)
            if save_with_name:
                base, ext = path.splitext(basename)
                basename = base + save_with_name + ext

            new_path = path.join(destination_dir, basename)

            # I really should have called the option overwrite from
            # the beginning. The hack below ensures old code works,
            # at least...
            nuke_existing = clobber or overwrite
            if (new_path != full_path) or nuke_existing:
                try:
                    hdulist.writeto(new_path, clobber=nuke_existing)
                except IOError:
                    logger.error('error writing file %s', new_path)
                    raise
            hdulist.close()

        # reset mask
        for col in self.summary_info.columns:
            self.summary_info[col].mask = current_mask[col]

    def _paths(self):
        """
        Full path to each file.
        """
        unmasked_files = self.summary_info['file'].compressed()
        return [path.join(self.location, file_) for file_ in unmasked_files]

    def headers(self, do_not_scale_image_data=True, **kwd):
        return self._generator('header',
                               do_not_scale_image_data=do_not_scale_image_data,
                               **kwd)
    headers.__doc__ = _generator.__doc__.format(
        name='header', default_scaling='True',
        return_type='astropy.io.fits.Header')

    def hdus(self, do_not_scale_image_data=False, **kwd):
        return self._generator('hdu',
                               do_not_scale_image_data=do_not_scale_image_data,
                               **kwd)
    hdus.__doc__ = _generator.__doc__.format(
        name='HDU', default_scaling='False', return_type='astropy.io.fits.HDU')

    def data(self, do_not_scale_image_data=False, **kwd):
        return self._generator('data',
                               do_not_scale_image_data=do_not_scale_image_data,
                               **kwd)
    data.__doc__ = _generator.__doc__.format(
        name='image', default_scaling='False', return_type='numpy.ndarray')

    def ccds(self, ccd_kwargs=None, **kwd):
        return self._generator('ccd', ccd_kwargs=ccd_kwargs, **kwd)
    ccds.__doc__ = _generator.__doc__.format(
        name='CCDData', default_scaling='True', return_type='ccdproc.CCDData')
