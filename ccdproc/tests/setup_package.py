# Licensed under a 3-clause BSD style license - see LICENSE.rst
def get_package_data():
    return {
        _ASTROPY_PACKAGE_NAME_ + '.tests': ['coveragerc',
                                            'data/a8280271.fits',
                                            'data/sip-wcs.fit',
                                            'data/expected_ifc_file_properties.csv',
                                            'data/science-mef.fits',
                                            'data/flat-mef.fits']}
