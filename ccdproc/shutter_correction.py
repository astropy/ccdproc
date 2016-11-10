from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import os

import numpy as np
import astropy.units as u
from astropy.nddata import StdDevUncertainty

from .core import CCDData, subtract_bias
from .ccddata import fits_ccddata_reader, fits_ccddata_writer
from .combiner import combine
from .image_collection import ImageFileCollection


def shutter_correction_algorithm(combined_flats, exptimekey='EXPTIME',
                       normalizer=lambda flat: flat.data.max(),
                       output=None):
    '''
    Determine the shutter correction given a set of input bias subtracted flats.
    
    Parameters
    ----------
    combined_flats : `list` containing `~ccdproc.CCDData` objects
        Set of bias subtracted flats of different exposure times taken under
        constant illumination.
    
    exptimekey : `string`, defaults to 'EXPTIME'
        The header keyword which contains the exposure time of the image in
        seconds.

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
    assert 'GAIN' in combined_flats[0].header.keys()
    gain = combined_flats[0].header['GAIN']

#     def nflat(flat, size=(10,10)):
#         mflat = ndimage.filters.median_filter(flat.data, size=size)
#         print(flat.data.max(), np.percentile(flat.data, 99), mflat.max())
#         return mflat.max()


    a = gain*np.sum([normalizer(f) for f in combined_flats])
    b = gain*np.sum([normalizer(f)/float(f.header[exptimekey]) for f in combined_flats])
    c = gain*np.sum([normalizer(f)/float(f.header[exptimekey])**2 for f in combined_flats])

    Dijcomb = combine(combined_flats, method='average')
    Dij = gain*Dijcomb.data*len(combined_flats)

    Eijm = [CCDData(data=f.data/float(f.header[exptimekey]), unit=f.unit)
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

    if output:
        if os.path.exists(output):
            os.remove(output)
        fits_ccddata_writer(shutter_map, output)

    return shutter_map


def calculate_shutter_correction_map(files, biases=None,
                                     exptimekey='EXPTIME',
                                     normalizer=lambda flat: flat.data.max(),
                                     keywordguide = {'keyword': 'IMAGETYP',
                                                     'flatvalue': 'Flat Field',
                                                     'biasvalue': 'Bias Frame'},
                                     output=None):
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
            sigma_clip = len(flat_images) > 5  # perform sigma clipping if >5 files
            combined_flat = combine(flat_images, method='average',
                                        sigma_clip=sigma_clip,
                                        sigma_clip_low_thresh=5,
                                        sigma_clip_high_thresh=5)
            combined_flats.append(combined_flat)
            assert combined_flat.shape == combined_flats[0].shape
        else:
            combined_flats.append(flat_images[0])

    shutter_map = shutter_correction_algorithm(combined_flats, output=output)
    return shutter_map


def apply_shutter_correction(image, shutter_map, exptimekey='EXPTIME'):
    exptime_map = shutter_map.add(image.header[exptimekey]*u.second)
    result = image.divide(exptime_map)
    return result


if __name__ == '__main__':
    filepath = '/Users/jwalawender/Data/VYSOS/ShutterMap/V5/20161027UT'
    keywords = ['EXPTIME', 'SET-TEMP', 'CCD-TEMP', 'XBINNING', 'YBINNING', 
                'IMAGETYP', 'OBJECT', 'DATE-OBS']
    ifc = ImageFileCollection(location=filepath, keywords=keywords)
    bytype = ifc.summary.group_by('IMAGETYP')
    flats = bytype.groups[bytype.groups.keys['IMAGETYP'] == 'Flat Field']
    biases = bytype.groups[bytype.groups.keys['IMAGETYP'] == 'Bias Frame']
    print(flats)
    print(biases)
