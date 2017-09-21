# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

from collections import OrderedDict
import fnmatch
from os import listdir, path
import logging

import numpy as np
import numpy.ma as ma

from astropy.table import Table, MaskedColumn
import astropy.io.fits as fits
from astropy.extern import six
from astropy.utils import minversion

import warnings
from astropy.utils.exceptions import AstropyUserWarning

from .ccddata import fits_ccddata_reader, _recognized_fits_file_extensions

logger = logging.getLogger(__name__)

__all__ = ['ImageFileCollection']
__doctest_skip__ = ['*']

_ASTROPY_LT_1_3 = not minversion("astropy", "1.3")


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

    glob_include: str or None, optional
        Unix-style filename pattern to select filenames to include in the file
        collection. Can be used in conjunction with ``glob_exclude`` to
        easily select subsets of files in the target directory.
        Default is ``None``.

    glob_exclude: str or None, optional
        Unix-style filename pattern to select filenames to exclude from the
        file collection. Can be used in conjunction with ``glob_include`` to
        easily select subsets of files in the target directory.
        Default is ``None``.

     ext: str or int, optional
         The extension from which the header and data will be read in all files.
         Default is ``0``.

    Raises
    ------
    ValueError
        Raised if keywords are set to a combination of '*' and any other
        value.
    """
    def __init__(self, location=None, keywords=None, info_file=None,
                 filenames=None, glob_include=None, glob_exclude=None, ext=0):

        # Include or exclude files from the collection based on glob pattern
        # matching - has to go above call to _get_files()
        if glob_exclude is not None:
            glob_exclude = str(glob_exclude)  # some minimal validation
        self._glob_exclude = glob_exclude

        if glob_include is not None:
            glob_include = str(glob_include)
        self._glob_include = glob_include

        self._location = location
        self._filenames = filenames
        self._files = []
        self._info_file = info_file
        if location:
            self._files = self._get_files()

        if self._files == []:
            warnings.warn("no FITS files in the collection.",
                          AstropyUserWarning)
        self._summary = {}
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
                self._summary = Table.read(info_path, format='ascii',
                                           delimiter=',')
                self._summary = Table(self._summary, masked=True)
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

        self._ext = ext

        if keywords:
            self.keywords = keywords

    def __repr__(self):
        if self.location is None:
            location = ""
        else:
            location = "location={!r}".format(self.location)

        if self._all_keywords:
            kw = ""
        else:
            kw = "keywords={!r}".format(self.keywords[1:])

        if self._info_file is None:
            infofile = ''
        else:
            infofile = "info_file={!r}".format(self._info_file)

        if self.glob_exclude is None:
            glob_exclude = ''
        else:
            glob_exclude = "glob_exclude={!r}".format(self.glob_exclude)

        if self.glob_include is None:
            glob_include = ''
        else:
            glob_include = "glob_include={!r}".format(self.glob_include)

        if self.ext == 0:
            ext = ""
        else:
            ext = "ext={}".format(self.ext)

        if self._filenames is None:
            filenames = ""
        else:
            filenames = "filenames={}".format(self._filenames)

        params = [location, kw, infofile, filenames, glob_include, glob_exclude, ext]
        params = ', '.join([p for p in params if p])

        str_repr = "{self.__class__.__name__}({params})".format(
            self=self, params=params)

        return str_repr

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
        return self._summary

    @property
    def summary_info(self):
        """
        `~astropy.table.Table` of values of FITS keywords for files in the
        collection.

        Each keyword is a column heading. In addition, there is a column
        called 'file' that contains the name of the FITS file. The directory
        is not included as part of that name.

        .. deprecated:: 0.4
        """
        warnings.warn('"summary_info" is deprecated and will be removed in '
                      'a future version. Use the "summary" attribute instead.',
                      AstropyUserWarning)
        return self._summary

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

        .. versionchanged:: 1.3
            Added ``deleter`` for ``keywords`` property.
        """
        if self.summary:
            return self.summary.keys()
        else:
            return []

    @keywords.setter
    def keywords(self, keywords):
        # since keywords are drawn from self.summary, setting
        # summary sets the keywords.
        if keywords is None:
            self._summary = []
            return

        if keywords == '*':
            self._all_keywords = True
        else:
            self._all_keywords = False

        logging.debug('keywords in setter before pruning: %s.', keywords)

        # remove duplicates and force a copy so we can sort the items later
        # by their given position.
        new_keys_set = set(keywords)
        new_keys_lst = list(new_keys_set)
        new_keys_set.add('file')

        logging.debug('keywords after pruning %s.', new_keys_lst)

        current_set = set(self.keywords)
        if new_keys_set.issubset(current_set):
            logging.debug('table columns before trimming: %s.',
                          ' '.join(current_set))
            cut_keys = current_set.difference(new_keys_set)
            logging.debug('will try removing columns: %s.',
                          ' '.join(cut_keys))
            for key in cut_keys:
                self._summary.remove_column(key)
            logging.debug('after removal column names are: %s.',
                          ' '.join(self.keywords))
        else:
            logging.debug('should be building new table...')
            # Reorder the keywords to match the initial ordering.
            new_keys_lst.sort(key=keywords.index)
            self._summary = self._fits_summary(new_keys_lst)

    @keywords.deleter
    def keywords(self):
        # since keywords are drawn from self._summary, setting
        # _summary = [] deletes the keywords.
        self._summary = []

    @property
    def files(self):
        """
        list of str, Unfiltered list of FITS files in location.
        """
        return self._files

    @property
    def glob_include(self):
        """
        str or None, Unix-style filename pattern to select filenames to include
        in the file collection.
        """
        return self._glob_include

    @property
    def glob_exclude(self):
        """
        str or None, Unix-style filename pattern to select filenames to exclude
        in the file collection.
        """
        return self._glob_exclude

    @property
    def ext(self):
        """
        str or int, The extension from which the header and data will
        be read in all files.
        """
        return self._ext

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
            return list(set(self.summary[keyword]))
        else:
            return list(self.summary[keyword])

    def _preprocess_kwargs_for_filtering(self, kwargs):
        """Method to find out if the kwargs dictionary that will be used to
        find matching files should be preprocessed.

        Currently this includes checking if any character should be replaced
        by a whitespace.
        """
        char_to_replace = kwargs.pop('replace_', None)
        if char_to_replace is not None:
            kwargs = {key.replace(char_to_replace, ' '): value
                      for key, value in six.iteritems(kwargs)}
            # It should be impossible to pass in not-strings so this function
            # doesn't have a check for non-string keys. The only functions that
            # call this method only accept it as `**kwargs` so these have to be
            # strings or it would raise "TypeError: func_name() keywords must
            # be strings". At least on Python 3.
        return kwargs

    def files_filtered(self, **kwd):
        """Determine files whose keywords have listed values.

        Parameters
        ----------
        include_path : bool, keyword-only
            If the keyword ``include_path=True`` is set, the returned list
            contains not just the filename, but the full path to each file.
            Default is ``False``.

        replace_ : str, optional, keyword-only
            If this parameter is given it should be a string of length 1 that
            indicates which character is replaced by a whitespace. This affects
            all keys passed in as ``**kwd``.

            .. versionadded:: 1.3

        **kwd :
            ``**kwd`` is dict of keywords and values the files must have.
            The value '*' represents any value.
            A missing keyword is indicated by value ''.

        Returns
        -------
        filenames : list
            The files that satisfy the keyword-value restrictions specified by
            the ``**kwd``.

        Examples
        --------
        Some examples for filtering::

            >>> keys = ['imagetyp','filter']
            >>> collection = ImageFileCollection('test/data', keywords=keys)
            >>> collection.files_filtered(imagetyp='LIGHT', filter='R')
            >>> collection.files_filtered(imagetyp='*', filter='')

        In case there is a keyword with whitespaces you can use::

            >>> collection.files_filtered(image_typ='LIGHT',
            ...                           replace_='_')

        This will look for the ``image typ`` keyword (the underscore was
        replaced by a whitespace). This could be useful in case the header
        contains keys like ``ESO TPL ID`` (or similar).

        Notes
        -----
        Value comparison is case *insensitive* for strings.
        """
        # force a copy by explicitly converting to a list
        current_file_mask = list(self.summary['file'].mask)

        include_path = kwd.pop('include_path', False)
        kwd = self._preprocess_kwargs_for_filtering(kwd)

        self._find_keywords_by_values(**kwd)
        filtered_files = self.summary['file'].compressed()
        self.summary['file'].mask = current_file_mask
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
        self._summary = self._fits_summary(header_keywords=keywords)

    def sort(self, keys):
        """Sort the list of files to determine the order of iteration.

        Sort the table of files according to one or more keys. This does not
        create a new object, instead is sorts in place.

        Parameters
        ----------
        keys : str, list of str
            The key(s) to order the table by.
        """
        if len(self._summary) > 0:
            self._summary.sort(keys)
            self._files = list(self.summary['file'])

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

            if self.glob_include is not None:
                files = fnmatch.filter(files, self.glob_include)
            if self.glob_exclude is not None:
                files = [file for file in files
                         if not fnmatch.fnmatch(file, self.glob_exclude)]

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

        def _add_val_to_dict(key, value, tbl_dict, n_previous, missing_marker):
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

        h = fits.getheader(file_name, self.ext)

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

        alreadyencountered = set()
        for k, v in six.iteritems(h):
            if k == '':
                continue

            k = k.lower()

            if k in ['comment', 'history']:
                multi_entry_keys[k].append(str(v))
                # Accumulate these in a separate dictionary until the
                # end to avoid adding multiple entries to summary.
                continue
            elif k in alreadyencountered:
                # The "normal" multi-entries HISTORY, COMMENT and BLANK are
                # already processed so any further duplication is probably
                # a mistake. It would lead to problems in ImageFileCollection
                # to add it as well, so simply ignore those.
                warnings.warn(
                    'Header from file "{f}" contains multiple entries for {k},'
                    ' the pair "{k}={v}" will be ignored.'
                    ''.format(k=k, v=v, f=file_name),
                    UserWarning)
                continue
            else:
                # Add the key to the already encountered keys so we don't add
                # it more than once.
                alreadyencountered.add(k)

            _add_val_to_dict(k, v, summary, n_previous, missing_marker)

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
        for k in header_keys:
            k_lower = k.lower()
            if k_lower != k:
                try:
                    summary_table.rename_column(k_lower, k)
                except KeyError:
                    pass

    def _fits_summary(self, header_keywords):
        """
        Generate a summary table of keywords from FITS headers.

        Parameters
        ----------
        header_keywords : list of str or '*'
            Keywords whose value should be extracted from FITS headers or '*'
            to extract all.
        """

        if not self.files:
            return None

        # Make sure we have a list...for example, in python 3, dict.keys()
        # is not a list.
        original_keywords = list(header_keywords)

        # Get rid of any duplicate keywords, also forces a copy.
        header_keys = set(original_keywords)
        header_keys.add('file')

        file_name_column = MaskedColumn(name='file', data=self.files)

        if not header_keys or (header_keys == {'file'}):
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
        missing_columns -= {'*'}

        length = len(summary_table)
        for column in missing_columns:
            all_masked = MaskedColumn(name=column, data=np.zeros(length),
                                      mask=np.ones(length))
            summary_table.add_column(all_masked)

        if '*' not in header_keys:
            # Rearrange table columns to match order of keywords.
            # File always comes first.
            header_keys -= {'file'}
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

        if set(keywords).issubset(self.keywords):
            # we already have the information in memory
            use_info = self.summary
        else:
            # we need to load information about these keywords.
            use_info = self._fits_summary(header_keywords=keywords)

        matches = np.ones(len(use_info), dtype=bool)
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
                    have_this_value = np.zeros(len(use_info), dtype=bool)
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
        self.summary['file'].mask = ma.nomask
        self.summary['file'].mask[~matches] = True

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
            Dict with parameters for `~astropy.nddata.fits_ccddata_reader`.
            For instance, the key ``'unit'`` can be used to specify the unit
            of the data. If ``'unit'`` is not given then ``'adu'`` is used as
            the default unit.
            See `~astropy.nddata.fits_ccddata_reader` for a complete list of
            parameters that can be passed through ``ccd_kwargs``.

        replace_ : str, optional, keyword-only
            If this parameter is given it should be a string of length 1 that
            indicates which character is replaced by a whitespace. This affects
            all keys passed in as ``**kwd``.

            .. versionadded:: 1.3

        **kwd :
            Any additional keywords are used to filter the items returned; see
            `files_filtered` examples for details.

        Returns
        -------
        `{return_type}`
            If ``return_fname`` is ``False``, yield the next {name} in the
            collection.

        (`{return_type}`, str)
            If ``return_fname`` is ``True``, yield a tuple of
            ({name}, ``file name``) for the next item in the collection.
        """
        # store mask so we can reset at end--must COPY, otherwise
        # current_mask just points to the mask of summary
        if not self.summary:
            return

        current_mask = {}
        for col in self.summary.columns:
            current_mask[col] = self.summary[col].mask

        if kwd:
            kwd = self._preprocess_kwargs_for_filtering(kwd)
            self._find_keywords_by_values(**kwd)

        ccd_kwargs = ccd_kwargs or {}

        for full_path in self._paths():
            no_scale = do_not_scale_image_data
            hdulist = fits.open(full_path,
                                do_not_scale_image_data=no_scale)

            file_name = path.basename(full_path)

            ext_index = hdulist.index_of(self.ext)

            return_options = {
                    'header': lambda: hdulist[ext_index].header,
                    'hdu': lambda: hdulist[ext_index],
                    'data': lambda: hdulist[ext_index].data,
                    'ccd': lambda: fits_ccddata_reader(full_path,
                                                       hdu=ext_index,
                                                       **ccd_kwargs)
                    }
            try:
                if return_fname:
                    yield return_options[return_type](), file_name
                else:
                    yield return_options[return_type]()
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
            if clobber or overwrite:
                if _ASTROPY_LT_1_3:
                    nuke_existing = {'clobber': True}
                else:
                    nuke_existing = {'overwrite': True}
            else:
                nuke_existing = {}
            if (new_path != full_path) or nuke_existing:
                try:
                    hdulist.writeto(new_path, **nuke_existing)
                except IOError:
                    logger.error('error writing file %s', new_path)
                    raise
            hdulist.close()

        # reset mask
        for col in self.summary.columns:
            self.summary[col].mask = current_mask[col]

    def _paths(self):
        """
        Full path to each file.
        """
        unmasked_files = self.summary['file'].compressed()
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
        name='HDUList', default_scaling='False',
        return_type='astropy.io.fits.HDUList')

    def data(self, do_not_scale_image_data=False, **kwd):
        return self._generator('data',
                               do_not_scale_image_data=do_not_scale_image_data,
                               **kwd)
    data.__doc__ = _generator.__doc__.format(
        name='image', default_scaling='False', return_type='numpy.ndarray')

    def ccds(self, ccd_kwargs=None, **kwd):
        if kwd.get('clobber') or kwd.get('overwrite'):
            raise NotImplementedError(
                "overwrite=True (or clobber=True) is not supported for CCDs.")
        return self._generator('ccd', ccd_kwargs=ccd_kwargs, **kwd)
    ccds.__doc__ = _generator.__doc__.format(
        name='CCDData', default_scaling='True', return_type='astropy.nddata.CCDData')
