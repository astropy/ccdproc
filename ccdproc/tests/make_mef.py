import numpy as np

from astropy.utils.misc import NumpyRNGContext
from astropy.io import fits
from astropy.nddata import CCDData

from ccdproc import flat_correct


def make_sample_mef(science_name, flat_name, size=10, dtype='float32'):
    """
    Make a multi-extension FITS image with random data
    and a MEF flat.

    Parameters
    ----------

    science_name : str
        Name of the science image created by this function.

    flat_name : str
        Name of the flat image created by this function.

    size : int, optional
        Size of each dimension of the image; images created are square.

    dtype : str or numpy dtype, optional
        dtype of the generated images.
    """
    with NumpyRNGContext(1234):
        number_of_image_extensions = 3
        science_image = [fits.PrimaryHDU()]
        flat_image = [fits.PrimaryHDU()]
        for _ in range(number_of_image_extensions):
            # Simulate a cloudy night, average pixel
            # value of 100 with a read_noise of 1 electron.
            data = np.random.normal(100., 1.0, [size, size]).astype(dtype)
            hdu = fits.ImageHDU(data=data)
            # Make a header that is at least somewhat realistic
            hdu.header['unit'] = 'electron'
            hdu.header['object'] = 'clouds'
            hdu.header['exptime'] = 30.0
            hdu.header['date-obs'] = '1928-07-23T21:03:27'
            hdu.header['filter'] = 'B'
            hdu.header['imagetyp'] = 'LIGHT'
            science_image.append(hdu)

            # Make a perfect flat
            flat = np.ones_like(data, dtype=dtype)
            flat_hdu = fits.ImageHDU(data=flat)
            flat_hdu.header['unit'] = 'electron'
            flat_hdu.header['filter'] = 'B'
            flat_hdu.header['imagetyp'] = 'FLAT'
            flat_hdu.header['date-obs'] = '1928-07-23T21:03:27'
            flat_image.append(flat_hdu)

    science_image = fits.HDUList(science_image)
    science_image.writeto(science_name)

    flat_image = fits.HDUList(flat_image)
    flat_image.writeto(flat_name)


if __name__ == '__main__':
    make_sample_mef('data/science-mef.fits', 'data/flat-mef.fits')
