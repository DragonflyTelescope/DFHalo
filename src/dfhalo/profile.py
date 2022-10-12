import os
import glob
import time
import shutil

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import warnings

from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyUserWarning

def calculate_threshold_radius(r, I, threshold, I_satr):
    """ Calculate the radius at which I drop certain fraction of I_satr """
    I_thre = I_satr - 2.5*np.log10(threshold)
    return r[np.nanargmin(abs(I-I_thre))]

def compute_radial_profile(img, cen=None, mask=None, 
                          back=None, bins=None,
                          seeing=2.5, 
                          pixel_scale=2.5, 
                          sky_mean=0, dr=1,
                          core_undersample=False, 
                          use_annulus=False):
    
    """ Extract radial profiles from the image cutout. """
    
    if mask is None:
        mask = np.zeros_like(img, dtype=bool)
        
    if back is None:     
        back = np.ones_like(img) * sky_mean
    bkg_val = np.median(back)
    
    if cen is None:
        cen = (img.shape[1]-1)/2., (img.shape[0]-1)/2.
    
    if use_annulus:
        img[mask] = np.nan
    
    yy, xx = np.indices(img.shape)
    rr = np.sqrt((xx - cen[0])**2 + (yy - cen[1])**2)
    r = rr[~mask].ravel()  # radius in pix
    z = img[~mask].ravel()  # pixel intensity
    r_core = np.int32(2 * seeing) # core radius in pix

    # Decide the outermost radial bin r_max before going into the background
    bkg_cumsum = np.arange(1, len(z)+1, 1) * bkg_val
    z_diff =  abs(z.cumsum() - bkg_cumsum)
    n_pix_max = len(z) - np.argmin(abs(z_diff - 0.00005 * z_diff[-1]))
    r_max = np.min([img.shape[0]//2, np.sqrt(n_pix_max/np.pi)])
    
    r *= pixel_scale   # radius in arcsec
    r_core *= pixel_scale
    r_max *= pixel_scale
    d_r = dr * pixel_scale
    
    clip = lambda z: sigma_clip((z), sigma=5, maxiters=5)
    if bins is None:
        # Radial bins: discrete/linear within r_core + log beyond it
        if core_undersample:  
            # for undersampled core, bin at int pixels
            bins_inner = np.unique(r[r<r_core]) - 1e-3
        else:
            n_bin_inner = int(min((r_core/d_r*2), 6))
            bins_inner = np.linspace(0, r_core-d_r, n_bin_inner) - 1e-3

        n_bin_outer = np.max([6, np.min([np.int32(r_max/d_r/10), 50])])
        if r_max > (r_core+d_r):
            bins_outer = np.logspace(np.log10(r_core+d_r),
                                     np.log10(r_max+2*d_r), n_bin_outer)
        else:
            bins_outer = []
        bins = np.concatenate([bins_inner, bins_outer])
        _, bins = np.histogram(r, bins=bins)
    
    # Calculate binned 1d profile
    r_rbin, z_rbin = np.array([]), np.array([])
    
    for k, b in enumerate(bins[:-1]):
        r_in, r_out = bins[k], bins[k+1]
        in_bin = (r>=r_in) & (r<=r_out)
        if use_annulus:
            # Fractional ovelap w/ annulus
            annl = CircularAnnulus(cen, abs(r_in)/pixel_scale, r_out/pixel_scale)
            annl_ma = annl.to_mask()
            # Intensity by fractional mask
            z_ = annl_ma.multiply(img)
            
            zb = np.sum(z_[~np.isnan(z_)]) / annl.area
            rb = np.mean(r[in_bin])
            
        else:
            z_clip = clip(z[~np.isnan(z) & in_bin])
            if np.ma.is_masked(z_clip):
                z_clip = z_clip.compressed()
            if len(z_clip)==0:
                continue

            zb = np.mean(z_clip)
            zstd_b = np.std(z_clip) if len(z_clip) > 10 else 0
            rb = np.mean(r[in_bin])

        z_rbin = np.append(z_rbin, zb)
        r_rbin = np.append(r_rbin, rb)
    
    # Decide the approxiamte radius within which it saturates
    # roughly at where intersity drops by half
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dz_rbin = np.diff(np.log10(z_rbin))
        dz_cum = np.cumsum(dz_rbin)
    r_satr = r_rbin[np.argmax(dz_cum<-0.3)] + 1e-3
    
    return r_rbin, z_rbin, r_satr


def extract_threshold_profile(fn, fn_seg, fn_SEcat, tab_atlas, 
                              thresholds=np.logspace(-0.5,-3.5,16), 
                              mag_range=[8.5,10.5],
                              N_source_min=3000,
                              dist_mask_min=None,
                              pixel_scale=2.5,
                              sep=5*u.arcsec,
                              bkg_val=0):
    
    """ 
    Extract single curve of growth.
    
    Parameters
    ----------
    fn: str
        Path of frames.
    fn_seg: str
        Path of segementation maps.
    fn_SEcat: str
        Path of SExtractor catalogs.
    tab_atlas: astropy.Table
        Crossmatched ATLAS table.
    thresholds: np.array
        Thresholds at x% of the saturation brightness.
    mag_range: list
        Range of magnitude of bright stars for measurement
    N_source_min: int
        Minimum number of detected sources required in the frame.
    sep: astropy.units.Quantity, default 5*u.arcsec
        Crossmatch seperation.
    dist_mask_min: int, default None
        Minimum distance to the field edges mask.
        
    Returns
    -------
    r_norms: 3d np.array
        Curves of Growth (axis 0: frame, axis 1: star, axis 2: radius)
        
    """
    from .utils import Intensity2SB, Thumb_Image, measure_dist_to_edge
    
    # Read data
    hdu = fits.open(fn)
    data_full = hdu[0].data
    header = hdu[0].header
    hdu.close()
    
    try:
        ZP = header['FIDZP']
    except KeyError:
        ZP = header['REFZP']
    #pixel_scale = header['PIXSCALE']
    wcs = WCS(header)

    # Read segment map
    seg_map = fits.getdata(fn_seg)
    
    # Mask nan area
    mask_nan = (data_full==0)
    data_full[mask_nan] = np.nan
    seg_map[mask_nan] = np.max(seg_map)+1

    # Read SExtractor catalog
    tab_SE = Table.read(fn_SEcat, format='ascii.sextractor')
    
    # Skip if no enough detected sources
    cond = (tab_SE['MAG_AUTO']>=13) & (tab_SE['MAG_AUTO']<=18)
    if len(tab_SE[cond])< N_source_min:
        return None
    else:
        # Cross match SE catalog and ATLAS catalog
        coords_atlas = SkyCoord(tab_atlas['RA'], tab_atlas['Dec'], unit=u.deg)
        coords_SE = SkyCoord(tab_SE['X_WORLD'], tab_SE['Y_WORLD'])
        idx, d2d, _ = coords_SE.match_to_catalog_sky(coords_atlas)
        match = d2d < sep

        tab_SE_match = tab_SE[match]
        tab_atlas_match = tab_atlas[idx[match]]

        # Saturation surface brightness
        I_satr = np.median(tab_SE_match['MU_MAX'])

        # Add ATLAS mag to the SE table
        g_mag = tab_atlas_match['g']
        tab_SE_match['gmag_atlas'] = g_mag
        
        # Filter for target bright stars
        tab_SE_match = tab_SE_match[(g_mag>=mag_range[0])&(g_mag<=mag_range[1])]

        # Remove stars close to the nan area if needed
        if dist_mask_min is None:
            table_star = tab_SE_match
        else:
            # measure distance to masked area
            dist_mask = measure_dist_to_edge(tab_SE_match, mask_nan)
            table_star = tab_SE_match[dist_mask>dist_mask_min]
        
        # Sort by g mag
        table_star.sort('gmag_atlas')
        
        # Measure profiles for N target stars
        r_profiles = np.array([])
        for i in range(len(table_star)):

            # Make thumbnail image
            thumb = Thumb_Image(table_star[i], wcs)
            
            # Extract image and mask cutouts
            thumb.extract_star(data_full, seg_map=seg_map, n_win=30, max_size=300)
            
            # Skip if there is too many masked pixels around (>80%)
            if np.sum(thumb.star_ma) > 0.8 * np.size(thumb.img_thumb):
                return None
                
            # Compute profiles
            r_rbin, z_rbin, r_satr = compute_radial_profile(thumb.img_thumb, 
                                                            cen=thumb.cen_star,
                                                            back=0., #back=thumb.bkg,
                                                            dr=0.2, seeing=2.5, sky_mean=0, 
                                                            pixel_scale=pixel_scale, 
                                                            core_undersample=False)
            # Convert intensity to surface brightnesss
            I_rbin = Intensity2SB(z_rbin, BKG=bkg_val, ZP=ZP, pixel_scale=pixel_scale)
            
            # Calculate radii at a series of thresholds
            r_profile = np.array([calculate_threshold_radius(r_rbin, I_rbin, thre, I_satr) for thre in thresholds])
            
            # Append saturation radius
            r_profile = np.append(r_satr, r_profile)
            
            # Add threshold radii
            r_profiles = np.vstack([r_profiles, r_profile]) if i>0 else r_profile
        
        return r_profiles
    
def normalize_profiles(r_profiles, thresholds, threshold_range=[0.005,0.2], threshold_norm=0.5):
    """
    Normalize profiles by fitting 1D linear model
    
    Parameters
    ----------
    r_profiles: 2d np.array
        Curves of Growth (axis 0: star, axis 1: radius)
    thresholds: np.array
        Thresholds at x% of the saturation brightness.
    threshold_range: [float , float]
        Range of threshold for fitting 1D linear model.
    threshold_norm: float
        Threshold at which the curves are normalized.
        The value is interpolated from 1D linear model.
    
    Returns
    -------
    r_profiles_norm: 2d np.array
        Normalized curves of Growth (axis 0: star, axis 1: radius)
        
    """
    
    if r_profiles is None:
        return None
    
    N_star = r_profiles.shape[0]
    contrasts = 1/thresholds
    contrast_norm = 1/threshold_norm
    r_profiles_norm = np.empty_like(r_profiles)
    
    for j in range(N_star):
        r_ = r_profiles[j]  # first value is saturation radius

        # fir a 1D linear model between threshold_range
        fit_range = (thresholds>=threshold_range[0])&(thresholds<=threshold_range[1])
        slope, intcp =  np.polyfit(np.log10(contrasts[fit_range]), np.log10(r_[1:][fit_range]), deg=1)

        # normalize by value at contrast_norm
        norm = slope * np.log10(contrast_norm) + intcp
        r_profiles_norm[j] = r_/(10**norm)
        
    return r_profiles_norm
