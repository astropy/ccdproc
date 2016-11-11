from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import os

import numpy as np
import astropy.units as u
from astropy.nddata import StdDevUncertainty
from astropy.modeling import models, fitting

from .core import CCDData, subtract_bias
from .ccddata import fits_ccddata_reader, fits_ccddata_writer
from .combiner import combine
from .image_collection import ImageFileCollection


def fit_shutter_bias(combined_flats, exptimekey='EXPTIME',
                     normalizer=lambda flat: flat.data.max(),
                     verbose=False):
    '''
    Determine the shutter correction given a set of input bias subtracted flats.
    This algorithm assumes that the input light level for all files is constant,
    so this should be run on data such as dome flats and not twilight flats.

    Parameters
    ----------
    combined_flats : `list` containing `~ccdproc.CCDData` objects
        Set of bias subtracted flats of different exposure times taken under
        constant illumination.

    exptimekey : `string`, defaults to 'EXPTIME'
        The header keyword which contains the exposure time of the image in
        seconds.

    normalizer : function, optional
        The algorithm requires a measurement of the light levels in the images
        which represent the maximum exposure level.  Without noise, this would
        simply be the maximum pixel level, so this defaults to the maximum
        pixel value in the input image.  The user may specify a different
        function here (see Examples below).

    Returns
    -------
    bias : `float`
        The value (in seconds) of the bias between the fitted exposure time and
        the exposure time listed in the header.
    '''
    flux = [normalizer(f) for f in combined_flats]
    time = [f.header[exptimekey] for f in combined_flats]
    line0 = models.Linear1D(slope=1, intercept=0)
    fitter = fitting.LinearLSQFitter()
    line = fitter(line0, time, flux)
    bias = line.intercept.value/line.slope.value
    if verbose:
        print('Shutter Bias = {:.3f}'.format(bias))
    return bias




def Surma1993_algorithm(combined_flats, exptimekey='EXPTIME',
                        shutter_bias=0,
                        normalizer=lambda flat: flat.data.max(),
                        verbose=False, output=None):
    '''
    Determine the shutter correction map given a set of input bias subtracted
    flats.  This algorithm assumes that the header exposure time values are
    accurate -- that they represent the maximum typical exposure time in each
    image.  If this is not the case, then the resulting shutter correction map
    will be incorrect.  The `fit_shutter_bias` can be used to check this
    assumption and correct the files as neeed.
    
    Parameters
    ----------
    combined_flats : `list` containing `~ccdproc.CCDData` objects
        Set of bias subtracted flats of different exposure times taken under
        constant illumination.
    
    exptimekey : `string`, defaults to 'EXPTIME'
        The header keyword which contains the exposure time of the image in
        seconds.

    shutter_bias=0 : `float`, optional
        The shutter bias value in seconds.  This is the difference between the
        header exposure time and the actual exposure time for the pixels in the
        image which get the maximum exposure time.  This value can be measured
        using the `fit_shutter_bias` method.

    normalizer : function, optional
        The algorithm requires a measurement of the light levels in the images
        which represent the maximum exposure level.  Without noise, this would
        simply be the maximum pixel level, so this defaults to the maximum
        pixel value in the input image.  The user may specify a different
        function here (see Examples below).

    output : `str`, optional
        The filename to write the output shutter map to.  This will be a FITS
        file with two extensions.  The first contains the shutter map values
        (in seconds) and the second contains the uncertainty in those values.

    Notes
    -----
    This code follows the implementation described in Surma, P. (1993) 
    "Shutter-free flatfielding for CCD detectors" Astronomy and Astrophysics
    (ISSN 0004-6361), 278, 654–658.

    Returns
    -------
    shutter_map : `~ccdproc.CCDData`
        The shutter correction map in units of seconds.

    Examples
    --------
    Calling `Surma1993_algorithm` with normalizer that uses the maximum value
    in an image that has been median filtered by a 10x10 box:
    
    >>> result = Surma1993_algorithm(combined_flats,
                 normalizer=lambda f: median_filter(f.data, size=(10,10)).max())
    '''
    assert 'GAIN' in combined_flats[0].header.keys()
    gain = combined_flats[0].header['GAIN']

    normflats = [normalizer(f) for f in combined_flats]
    a = gain*np.sum(normflats)
    b = gain*np.sum([normflats[i]/float(f.header[exptimekey]+shutter_bias)
                    for i,f in enumerate(combined_flats)])
    c = gain*np.sum([normflats[i]/float(f.header[exptimekey]+shutter_bias)**2
                    for i,f in enumerate(combined_flats)])

    Dijcomb = combine(combined_flats, method='average')
    Dij = gain*Dijcomb.data*len(combined_flats)

    Eijm = [CCDData(data=f.data/float(f.header[exptimekey]+shutter_bias), unit=f.unit)
            for f in combined_flats]
    Eijcomb = combine(Eijm, method='average')
    Eij = gain*Eijcomb.data*len(combined_flats)

    alpha_ij = 1.0 / (a*c - b**2) * (c*Dij - b*Eij)
    delta_alpha = np.sqrt( c / (a*c - b**2) )
    beta_ij = 1.0 / (a*c - b**2) * (a*Eij - b*Dij)
    delta_beta = np.sqrt( a / (a*c - b**2) )

    SF = beta_ij / alpha_ij  # SF = -t_SH * (1-SH_ij) in the paper terminology
    delta_SF = np.sqrt( (alpha_ij**-1 * delta_beta)**2 + 
                        (beta_ij * alpha_ij**-2 * delta_alpha)**2 )
    SNR = np.mean(abs(SF)/delta_SF)

    shutter_map = CCDData(data=SF,
                          uncertainty=StdDevUncertainty(delta_SF),
                          unit=u.second,
                          meta={'SNR': (SNR, 'Mean signal to noise per pixel')})

    if verbose:
        print('Shutter Map (min, mean, median, max, stddev) = '\
              '{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(
              shutter_map.data.min(), shutter_map.data.mean(),
              np.median(shutter_map.data), shutter_map.data.max(),
              np.std(shutter_map.data) ))
        print('Fraction of Shutter Map Pixels > 0: {:d}/{:d} = {:.3f}'.format(
              np.sum(shutter_map.data > 0),
              shutter_map.data.size,
              np.sum(shutter_map.data > 0) / shutter_map.data.size))
    if output:
        if os.path.exists(output):
            os.remove(output)
        fits_ccddata_writer(shutter_map, output)

    return shutter_map


def apply_shutter_map(image, shutter_map, shutter_bias=0, exptimekey='EXPTIME'):
    exptime_map = shutter_map.add((image.header[exptimekey]+shutter_bias)*u.second)
    result = image.divide(exptime_map)
    return result


def get_shutter_map(files, biases=None, normalizer=lambda flat: flat.data.max(),
                    check_shutter_bias=True, exptimekey='EXPTIME',
                    min_files_to_sigma_clip=5, sigma_clip_low_thresh=5,
                    sigma_clip_high_thresh=5,
                    keywordguide = {'keyword': 'IMAGETYP',
                                    'flatvalue': 'Flat Field',
                                    'biasvalue': 'Bias Frame'},
                    verbose=False, output=None):
    '''
    Uses Surma1993_algorithm to determine the shutter correction map given
    input files.  The input files are assumed to contain flat fields taken under
    constant illumination (no twilight flats) and bias frames.
    
    The bias frames are median combined in to a master bias which is subtracted
    from each flat frame.
    
    Any flat frames with the same exposure time are averaged (possibly with
    sigma clipping, see min_files_to_sigma_clip) to make a master flat at each
    exposure time.
    
    The resulting list of bias subtracted master flats at various exposure times
    are passed to `Surma1993_algorithm` to determine the shutter map.

    Parameters
    ----------
    files : `ccdproc.ImageFilecollection` or `list` or `str`
        If input is a `ccdproc.ImageFilecollection` object, then this is used to
        select flat and bias files.  
        
        If input is a list ...
        
        If input is a string ...
    
    exptimekey : `string`, defaults to 'EXPTIME'
        The header keyword which contains the exposure time of the image in
        seconds.

    shutter_bias=0 : `float`, optional
        The shutter bias value in seconds.  This is the difference between the
        header exposure time and the actual exposure time for the pixels in the
        image which get the maximum exposure time.  This value can be measured
        using the `fit_shutter_bias` method.

    normalizer : function, optional
        The algorithm requires a measurement of the light levels in the images
        which represent the maximum exposure level.  Without noise, this would
        simply be the maximum pixel level, so this defaults to the maximum
        pixel value in the input image.  The user may specify a different
        function here (see Examples below).

    output : `str`, optional
        The filename to write the output shutter map to.  This will be a FITS
        file with two extensions.  The first contains the shutter map values
        (in seconds) and the second contains the uncertainty in those values.

    Notes
    -----
    This code follows the implementation described in Surma, P. (1993) 
    "Shutter-free flatfielding for CCD detectors" Astronomy and Astrophysics
    (ISSN 0004-6361), 278, 654–658.

    Returns
    -------
    shutter_map : `~ccdproc.CCDData`
        The shutter correction map in units of seconds.

    '''
    if type(files) is str:
        sys.exit(1)
    elif type(files) is list:
        sys.exit(1)
    elif type(files) is ImageFileCollection:
        pass
    else:
        sys.exit(1)

    bytype = files.summary.group_by(keywordguide['keyword'])
    flats = bytype.groups[bytype.groups.keys[keywordguide['keyword']]\
                          == keywordguide['flatvalue']]
    biases = bytype.groups[bytype.groups.keys[keywordguide['keyword']]\
                           == keywordguide['biasvalue']]

    bias_images = [fits_ccddata_reader(os.path.join(files.location, f))
                   for f in biases['file']]
    if len(bias_images) > 0:
        ## Generate master bias to subtract from each flat
        if len(bias_images) > 1:
            master_bias = combine(bias_images, combine='median')
        else:
            master_bias = bias_images[0]

    assert exptimekey in flats.keys()
    flats = flats.group_by(exptimekey)
    combined_flats = []
    for exptime in flats.groups.keys[exptimekey]:
        thisexptime = flats.groups[flats.groups.keys[exptimekey] == exptime]
        ## Combine each group of flats with same exposure times
        flat_images = [fits_ccddata_reader(os.path.join(files.location, f))
                       for f in thisexptime['file']]
        if len(flat_images) > 1:
            # Bias subtract flats
            if len(biases) > 0:
                flat_images = [subtract_bias(im, master_bias) for im in flat_images]
            sigma_clip = len(flat_images) > min_files_to_sigma_clip
            combined_flat = combine(flat_images, method='average',
                                    sigma_clip=sigma_clip,
                                    sigma_clip_low_thresh=sigma_clip_low_thresh,
                                    sigma_clip_high_thresh=sigma_clip_high_thresh)
            combined_flats.append(combined_flat)
            assert combined_flat.shape == combined_flats[0].shape
        else:
            combined_flats.append(flat_images[0])

    if check_shutter_bias:
        shutter_bias = fit_shutter_bias(combined_flats, verbose=verbose)
    else:
        shutter_bias = 0
    shutter_map = Surma1993_algorithm(combined_flats, normalizer=normalizer,
                              shutter_bias=shutter_bias, output=output,
                              verbose=verbose)
    return shutter_map

