from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import os

from .core import *
from .image_collection import ImageFileCollection


def calculate_shutter_correction_map(files, biases=None,
                                     exptimekey='EXPTIME',
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

    if len(biases) > 0:
        ## Generate master bias to subtract from each flat
        bias_images = [fits_ccddata_reader(os.path.join(files.location, f))
                       for f in biases['file']]
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
                flat_images = [ccd.subtract_bias(im, master_bias) for im in flat_images]
            sigma_clip = len(files) > 5  # perform sigma clipping if >5 files
            combined_flat = ccd.combine(images, method='average',
                                        sigma_clip=sigma_clip,
                                        sigma_clip_low_thresh=5,
                                        sigma_clip_high_thresh=5)
            combined_flats.append(combined_flat)
            assert combined_flat.shape == flats[0].shape
        else:
            combined_flats.append(flat_images[0])

    def nflat(flat, delta=25):
        nl, nc = flat.data.shape
        normalization_factor = np.median(flat.data[nl-delta:nl+delta,nc-delta:nc+delta])
        nflat = flat.data / normalization_factor
        return nflat

    a = gain*np.sum([nflat(f) for f in combined_flats])
    b = gain*np.sum([nflat(f)/float(f.header[exptimekey]) for f in combined_flats])
    c = gain*np.sum([nflat(f)/float(f.header[exptimekey])**2 for f in combined_flats])

    Dijcomb = ccd.combine(combined_flats, method='average')
    Dij = gain*Dijcomb.data*len(combined_flats)

    Eijm = [ccd.CCDData(data=f.data/float(f.header[exptimekey]), unit=f.unit)
            for f in combined_flats]
    Eijcomb = ccd.combine(Eijm, method='average')
    Eij = gain*Eijcomb.data*len(combined_flats)

    alpha_ij = 1.0 / (a*c - b**2) * (c*Dij - b*Eij)
    delta_alpha = np.sqrt( c / (a*c - b**2) )
    beta_ij = 1.0 / (a*c - b**2) * (a*Eij - b*Dij)
    delta_beta = np.sqrt( a / (a*c - b**2) )

    SF = beta_ij / alpha_ij  # SF = -t_SH * (1-SH_ij) in the paper terminology
    delta_SF = np.sqrt( (alpha_ij**-1 * delta_beta)**2 + 
                        (beta_ij * alpha_ij**-2 * delta_alpha)**2 )
    SNR = np.mean(abs(SF)/delta_SF)
    print('Typical pixel SNR of correction = {:.1f}'.format(SNR))
    ShutterMap = ccd.CCDData(data=SF, uncertainty=delta_SF, unit=u.second,
                     meta={'SNR': (SNR, 'Mean signal to noise per pixel')})

    if output:
        if os.path.exists(output):
            os.remove(output)
        ccd.fits_ccddata_writer(ShutterMap, output)

    return ShutterMap



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
