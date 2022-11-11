import os
import glob
import math
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation
from astropy.stats import sigma_clip, SigmaClip
from astropy.wcs import WCS
from astropy.table import Table, vstack

from photutils import Background2D, SExtractorBackground
from photutils import detect_sources, deblend_sources
from photutils import CircularAnnulus
from photutils.segmentation import SegmentationImage


DF_pixel_scale = 2.5

def query_atlas_catalog(field,
                        ra_range,
                        dec_range,
                        wsid,
                        password,
                        atalas_dir='./',
                        mag_limit=12):
    
    """
    Query the ATLAS Catalog using Casjob tool.
    The query result is saved under atalas_dir directory.
    
    Parameters
    ----------
    Field: str
        Field name (identifier).
    ra_range: tuple or list
        Range of RA
    dec_range: tuple or list
        Range of dec
    wsid: str
        casjob WSID
    password: str
        casjob password
    mag_limit: float
        Limiting magnitude in g
    
    Returns
    -------
    table_atlas: astropy.Table
        Crossmatched ATLAS table.
        
    """
    from .atlas import query_atlas
    
    fname_atlas = os.path.join(atalas_dir, f'{field}_atlas.csv')
    
    # Check if a queried catalog already existed
    if not os.path.exists(fname_atlas):
    
        # Check and delete previous queries
        catalog_match = os.path.join(atalas_dir, 'atlas_*.csv')
        for fn in glob.glob(catalog_match):
            os.remove(fn)

        # Query the ATLAS catalog, sleep until finished
        out = query_atlas(ra_range, dec_range,
                          wsid=wsid, password=password,
                          mag_limit=mag_limit)
        while len(glob.glob(catalog_match))==0:
            time.sleep(0.1)

        # Read and rename the queried table with field name
        fname_query = glob.glob(catalog_match)[0]
        
        # Sleep a few seconds for query
        time.sleep(5)
        
        # Rename and read the queried catalog
        shutil.copy(fname_query, fname_atlas)
    
    # Load table
    table_atlas = Table.read(fname_atlas, format='csv')

    return table_atlas
    
def make_atlas_catalog(ra_range,
                       dec_range,
                       catalog_dir,
                       mag_limit=12):

    """
    Make the ATLAS Catalog from the catalog csv files.

    Parameters
    ----------
    ra_range: tuple or list
        Range of RA
    dec_range: tuple or list
        Range of dec
    catalog_atals_dir: str
        Path to the local ATLAS catalog files.
        In the dir files are sorted by mag or dec (e.g. 00_m_16).
    mag_limit: float
        Limiting magnitude in g

    Returns
    -------
    table_atlas: astropy.Table
        ATLAS table
        
    """
    
    # Keep the coordinates and g and r mag
    use_columns_atlas = ['col1','col2','col22','col23','col26','col27']
    
    # RA and Dec integer
    ra_intgs = np.arange(np.floor(ra_range[0]), np.floor(ra_range[1])+1, 1, dtype=int)
    dec_intgs = np.arange(np.floor(dec_range[0]), np.floor(dec_range[1])+1, 1, dtype=int)
    
    # Read and join square degree catalogs
    table_atlas = Table()
    for ra_int in ra_intgs:
        for dec_int in dec_intgs:
            ra_str = ''.join([str(ra_int // 100 % 10), str(ra_int // 10 % 10), str(ra_int % 10)])
            dec_str= ''.join([str(abs(dec_int) // 10 % 10), str(abs(dec_int) % 10)])
            dec_str = '+' + dec_str if dec_int>0 else '-' + dec_str
            radec_str = '00_m_16/{0}{1}.rc2'.format(ra_str, dec_str)
            fn_cat_sqdeg = os.path.join(catalog_dir, radec_str)
            tab_sqdeg = Table.read(fn_cat_sqdeg, format='ascii',
                                   include_names=use_columns_atlas)
            table_atlas = vstack([table_atlas, tab_sqdeg])

    # Rename and assign unit
    # See readme on archive.stsci.edu/hlsps/atlas-refcat2
    table_atlas = Table(table_atlas, names=['RA', 'Dec',  'g', 'dg', 'r', 'dr'])
    table_atlas['RA'] = table_atlas['RA']/1e8
    table_atlas['Dec'] = table_atlas['Dec']/1e8
    table_atlas['g'] = table_atlas['g']/1e3
    table_atlas['dg'] = table_atlas['dg']/1e3
    table_atlas['r'] = table_atlas['r']/1e3
    table_atlas['dr'] = table_atlas['dr']/1e3
    
    # magnitude cut
    table_atlas = table_atlas[table_atlas['g']<=mag_limit]
    
    return table_atlas
    

def calculate_ra_dec_range(header):
    """ Calculate RA and Dec range from the wcs of header. """
    footprint = WCS(header).calc_footprint()
    ra_range = (footprint[:,0].min(), footprint[:,0].max())
    dec_range = (footprint[:,1].min(), footprint[:,1].max())
    return ra_range, dec_range

def coord_Im2Array(X_IMAGE, Y_IMAGE, origin=1):
    """ Convert image coordniate to numpy array coordinate """
    x_arr, y_arr = int(max(round(Y_IMAGE)-origin, 0)), int(max(round(X_IMAGE)-origin, 0))
    return x_arr, y_arr

def Intensity2SB(Intensity, BKG, ZP, pixel_scale=DF_pixel_scale):
    """ Convert intensity to surface brightness (mag/arcsec^2) given the background value, zero point and pixel scale """
    I = np.atleast_1d(np.copy(Intensity))
    I[np.isnan(I)] = BKG
    if np.any(I<=BKG):
        I[I<=BKG] = np.nan
    I_SB = -2.5*np.log10(I - BKG) + ZP + 2.5 * math.log10(pixel_scale**2)
    return I_SB
 
def measure_dist_to_edge(table, mask_area,
                        Xname="X_IMAGE", Yname="Y_IMAGE", origin=0):
    """
    Measure distance of each source to the edge of the mask.
    Note photutils is 0-based and SExtractor is 1-based.
    
    table: astropy Table
    mask_area: mask map (1: masked)

    """

    # Position at which mask map starts/ends
    dX_edge = np.diff(np.sum(~mask_area,axis=0))
    dY_edge = np.diff(np.sum(~mask_area,axis=1))

    X_data_min = np.argmax(dX_edge) + origin
    X_data_max = np.argmin(dX_edge) + origin
    Y_data_min = np.argmax(dY_edge) + origin
    Y_data_max = np.argmin(dY_edge) + origin

    # Compute distances
    dist_mask = np.empty(len(table))
    for i, obj in enumerate(table):
        X_obj, Y_obj = (obj[Xname], obj[Yname])
        X_edge = np.min([X_obj-X_data_min, X_data_max-X_obj])
        Y_edge = np.min([Y_obj-Y_data_min, Y_data_max-Y_obj])
        dist_mask[i] = np.min([X_edge, Y_edge])
    
    return dist_mask
    
    
def background_extraction(field, mask=None, return_rms=True,
                      b_size=64, f_size=3, n_iter=5, **kwargs):
    """ Extract background & rms image using SE estimator with mask """
    
    try:
        Bkg = Background2D(field, mask=mask,
                           bkg_estimator=SExtractorBackground(),
                           box_size=b_size, filter_size=f_size,
                           sigma_clip=SigmaClip(sigma=3., maxiters=n_iter),
                           **kwargs)
        back = Bkg.background
        back_rms = Bkg.background_rms
        
    except ValueError:
        img = field.copy()
        if mask is not None:
            img[mask] = np.nan
        back = np.nanmedian(field) * np.ones_like(field)
        back_rms = np.nanstd(field) * np.ones_like(field)
        
    if return_rms:
        return back, back_rms
    else:
        return back


class Thumb_Image:
    """
    A class for operation and info storing of a thumbnail image.
    Used for measuring scaling and stacking.

    row: astropy.table.row.Row
        Astropy table row.
    wcs: astropy.wcs.wcs
        WCS of image.
    """

    def __init__(self, row, wcs):
        self.wcs = wcs
        self.row = row
        
    def make_star_thumb(self,
                        image, seg_map=None,
                        n_win=20, seeing=2.5, max_size=200,
                        origin=1, verbose=False):
        """
        Crop the image and segmentation map into thumbnails.

        Parameters
        ----------
        image : 2d array
            Full image
        seg_map : 2d array
            Full segmentation map
        n_win : int, optional, default 20
            Enlarge factor (of fwhm) for the thumb size
        seeing : float, optional, default 2.5
            Estimate of seeing FWHM in pixel
        max_size : int, optional, default 200
            Max thumb size in pixel
        origin : 1 or 0, optional, default 1
            Position of the first pixel. origin=1 for SE convention.
            
        """

        # Centroid in the image from the SE measurement
        # Note SE convention is 1-based (differ from photutils)
        X_c, Y_c = self.row["X_IMAGE"], self.row["Y_IMAGE"]

        # Define thumbnail size
        fwhm =  max(self.row["FWHM_IMAGE"], seeing)
        win_size = min(int(n_win * max(fwhm, 2)), max_size)

        # Calculate boundary
        X_min, X_max = max(origin, X_c - win_size), min(image.shape[1], X_c + win_size)
        Y_min, Y_max = max(origin, Y_c - win_size), min(image.shape[0], Y_c + win_size)
        x_min, y_min = coord_Im2Array(X_min, Y_min, origin) # python convention
        x_max, y_max = coord_Im2Array(X_max, Y_max, origin)

        X_WORLD, Y_WORLD = self.row["X_WORLD"], self.row["Y_WORLD"]

        if verbose:
            print("NUMBER: ", self.row["NUMBER"])
            print("X_c, Y_c: ", (X_c, Y_c))
            print("RA, DEC: ", (X_WORLD, Y_WORLD))
            print("x_min, x_max, y_min, y_max: ", x_min, x_max, y_min, y_max)
            print("X_min, X_max, Y_min, Y_max: ", X_min, X_max, Y_min, Y_max)

        # Crop
        self.img_thumb = image[x_min:x_max, y_min:y_max].copy()
        if seg_map is None:
            self.seg_thumb = None
            self.mask_thumb = np.zeros_like(self.img_thumb, dtype=bool)
        else:
            self.seg_thumb = seg_map[x_min:x_max, y_min:y_max]
            self.mask_thumb = (self.seg_thumb!=0) # mask all sources

        # Centroid position in the cutout (0-based python convention)
        self.cen_star = np.array([X_c - y_min - origin, Y_c - x_min - origin])
        
    def extract_star(self, image,
                     seg_map=None,
                     sn_thre=2.5,
                     b_size=100,
                     n_dilation=3,
                     display_bkg=False,
                     display=False, **kwargs):
        
        """
        Local background and segmentation.
        If no segmentation map provided, do a local detection & deblend
        to remove faint undetected source.
        
        Parameters
        ----------
        image : 2d array
            Full image
        seg_map : 2d array
            Full segmentation map
        sn_thre : float, optional, default 2.5
            SNR threshold used for detection if seg_map is None
        b_size: float, optional, default 100
            Background size in pix for extract local background of thumbnail.
            If None, set around 1/5 of the thumbnail.
        n_dilation : int, optional, default 3
            Number of iterations to dilate the mask map.
        
        """
        # Make thumbnail image
        self.make_star_thumb(image, seg_map, **kwargs)
        
        img_thumb = self.img_thumb
        seg_thumb = self.seg_thumb
        mask_thumb = self.mask_thumb
        
        # Measure local background, use constant if the thumbnail is small
        shape = img_thumb.shape
        if b_size is None:
            b_size = round(min(shape)//5/25)*25
        
        if shape[0] >= b_size:
            back, back_rms = background_extraction(img_thumb, mask=mask_thumb, b_size=b_size)
        else:
            im_ = np.ones_like(img_thumb)
            img_thumb_ma = img_thumb[~mask_thumb]
            back, back_rms = (np.median(img_thumb_ma)*im_,
                              mad_std(img_thumb_ma)*im_)
        self.bkg = back
        self.bkg_rms = back_rms
                
        if seg_thumb is None:
            # do local source detection to remove faint stars using photutils
            threshold = back + (sn_thre * back_rms)
            segm = detect_sources(img_thumb, threshold, npixels=5)

            # deblending using photutils
            segm_deb = deblend_sources(img_thumb, segm, npixels=5,
                                           nlevels=64, contrast=0.005)
        else:
            segm_deb = SegmentationImage(seg_thumb)
            
        # star_ma mask other sources in the thumbnail
        star_label = segm_deb.data[round(self.cen_star[1]), round(self.cen_star[0])]
        star_ma = ~((segm_deb.data==star_label) | (segm_deb.data==0))
        self.star_ma = star_ma
        
        # Dilation on mask map
        if n_dilation is not None:
            self.mask_thumb = binary_dilation(mask_thumb, iterations=n_dilation)
            self.star_ma = binary_dilation(star_ma, iterations=n_dilation)
            
    def compute_Rnorm(self, R=12, **kwargs):
        """
        Compute the scaling factor at R using an annulus.
        Note the output values include the background level.
        
        Paramters
        ---------
        R : int, optional, default 12
            radius in pix at which the scaling factor is meausured
        kwargs : dict
            kwargs passed to compute_Rnorm
        
        """
        from .utils import compute_Rnorm
        I_mean, I_med, I_std, I_flag = compute_Rnorm(self.img_thumb,
                                                     self.star_ma,
                                                     self.cen_star,
                                                     R=R, **kwargs)
        self.I_mean = I_mean
        self.I_med = I_med
        self.I_std = I_std
        self.I_flag = I_flag
        
        # Use the median of background as the local background
        self.I_sky = np.median(self.bkg)


### Measuring scaling ###

def compute_Rnorm(image, mask_field, cen,
                  R=12, wid_ring=1, wid_cross=4,
                  mask_cross=True, display=False):
    """
    Compute the scaling factor using an annulus.
    Note the output values include the background level.
    
    Paramters
    ---------
    image : input image for measurement
    mask_field : mask map with masked pixels = 1.
    cen : center of the target in image coordiante
    R : radius of annulus in pix
    wid_ring : half-width of annulus in pix
    wid_cross : half-width of spike mask in pix
        
    Returns
    -------
    I_mean: mean value in the annulus
    I_med : median value in the annulus
    I_std : std value in the annulus
    I_flag : 0 good / 1 bad (available pixles < 5)
    
    """
    
    if image is None:
        return [np.nan] * 3 + [1]
    
    cen = (cen[0], cen[1])
    anl = CircularAnnulus([cen], R-wid_ring, R+wid_ring)
    anl_ma = anl.to_mask()[0].to_image(image.shape)
    in_ring = anl_ma > 0.5        # sky ring (R-wid, R+wid)
    mask = in_ring & (~mask_field) & (~np.isnan(image))
        # sky ring with other sources masked
    
    # Whether to mask the cross regions, important if R is small
    if mask_cross:
        yy, xx = np.indices(image.shape)
        rr = np.sqrt((xx-cen[0])**2+(yy-cen[1])**2)
        in_cross = ((abs(xx-cen[0])<wid_cross))|(abs(yy-cen[1])<wid_cross)
        mask = mask * (~in_cross)
    
    if len(image[mask]) < 5:
        return [np.nan] * 3 + [1]
    
    z_ = sigma_clip(image[mask], sigma=3, maxiters=5)
    z = z_.compressed()
        
    I_mean = np.average(z, weights=anl_ma[mask][~z_.mask])
    I_med, I_std = np.median(z), np.std(z)
    
    if display:
        L = min(100, int(mask.shape[0]))
        
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,4))
        ax1.imshow(mask, cmap="gray", alpha=0.7)
        ax1.imshow(mask_field, alpha=0.2)
        ax1.imshow(image, cmap='viridis', alpha=0.7,
                   norm=AsinhNorm(0.05, vmin=image.min(), vmax=I_med+50*I_std))
        ax1.plot(cen[0], cen[1], 'r+', ms=10)
        
        ax2.hist(z,alpha=0.7)
        
        # Label mean value
        plt.axvline(I_mean, color='k')
        plt.text(0.5, 0.9, "%.1f"%I_mean,
                 color='darkorange', ha='center', transform=ax2.transAxes)
        
        # Label 20% / 80% quantiles
        I_20 = np.quantile(z, 0.2)
        I_80 = np.quantile(z, 0.8)
        for I, x_txt in zip([I_20, I_80], [0.2, 0.8]):
            plt.axvline(I, color='k', ls="--")
            plt.text(x_txt, 0.9, "%.1f"%I, color='orange',
                     ha='center', transform=ax2.transAxes)
        
        ax1.set_xlim(cen[0]-L//4, cen[0]+L//4)
        ax1.set_ylim(cen[1]-L//4, cen[1]+L//4)
        
        plt.show()
        
    return I_mean, I_med, I_std, 0
