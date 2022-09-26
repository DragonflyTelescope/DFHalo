import os
import glob
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

from .profile import *
from .plot import *
from .clustering import *


def extract_profile_pipe(hdu_list, segm_list, 
                         catalog_list, table_atlas, 
                         thresholds=np.logspace(-0.3,-3.,22), 
                         mag_range=[8.5,10.5],
                         pixel_scale=2.5,
                         N_source_min=3600,
                         dist_mask_min=None):
    
    """ 
    Extract curves of growth.
    
    Parameters
    ----------
    hdu_list: list
        Path list of frames.
    segm_list: list
        Path list of segementation maps.
    catalog_list: list
        Path list of SExtractor catalogs.
    table_atlas: astropy.Table
        Crossmatched ATLAS table.
    thresholds: np.array
        Thresholds at x% of the saturation brightness.
    mag_range: list
        Range of magnitude of bright stars for measurement
    dist_mask_min: int, default None
        Minimum distance to the field edges mask.
        
    Returns
    -------
    r_norms: 3d np.array
        Curves of Growth (axis 0: frame, axis 1: star, axis 3: radius)
    flags: 1d n.array
        1: Good measurements  0: Bad
    
    """
    
    # Get filter names
    filters = np.array([fits.getheader(fn)["FILTER"] for fn in hdu_list])   
    
    # Set minimum number of intermediate bright sources detected
    N_source = np.zeros(len(hdu_list), dtype=int)
    for i, (catalog_path) in enumerate(catalog_list):
        SE_catalog = Table.read(catalog_path, format='ascii.sextractor')
        N_source[i] = len(SE_catalog[(SE_catalog['MAG_AUTO']>=13) & (SE_catalog['MAG_AUTO']<18)])
    N_source_G = N_source[filters=='G']
    N_source_R = N_source[filters=='R']
    
    IQR_N_source_G = np.quantile(N_source_G, 0.84)-np.quantile(N_source_G, 0.16)
    IQR_N_source_R = np.quantile(N_source_R, 0.84)-np.quantile(N_source_R, 0.16)
    N_source_G_min = int(np.median(N_source_G)-IQR_N_source_G)
    N_source_R_min = int(np.median(N_source_R)-IQR_N_source_R)
    
    # Plot histograms of # of detected sources
    plt.figure()
    plt.hist(N_source_G, color='seagreen', bins=8, alpha=0.7)
    plt.hist(N_source_R, color='firebrick', bins=8, alpha=0.7)
    plt.axvline(N_source_G_min, color='lime', ls='--', alpha=0.9)
    plt.axvline(N_source_R_min, color='r', ls='--', alpha=0.9)
    plt.xlabel('# source detected')
    plt.show()
    
    # Measure profiles
    profiles = {}
    flags = np.zeros_like(hdu_list, dtype=int)
    
    for i, (filt, hdu_path, segm_path, catalog_path) in tqdm(enumerate(zip(filters, hdu_list, segm_list, catalog_list))):
        name = os.path.basename(hdu_path).split('_light')[0]
        
        if filt=='G':
            N_source_min = N_source_G_min
        elif filt=='R':
            N_source_min = N_source_R_min
    
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyUserWarning)
            r_profiles = extract_threshold_profile(hdu_path, segm_path,
                                                   catalog_path, table_atlas,
                                                   thresholds=thresholds, 
                                                   mag_range=mag_range,
                                                   dist_mask_min=dist_mask_min, 
                                                   pixel_scale=pixel_scale,
                                                   N_source_min=N_source_min)

            # Normalize profiles by 1D intercepts at threshold_norm
            r_norm = normalize_profiles(r_profiles, thresholds, 
                                        threshold_range=[0.005,0.2],
                                        threshold_norm=0.5)
            
        if r_norm is None:
            N_star = 0
            flags[i] = 0
        else:
            N_star = r_norm.shape[0]
            flags[i] = 1
        
        profiles[name] = dict(r_norm=r_norm, N_star=N_star)
        
    # Get max number of sources to set array dimensions
    N_star_max = np.max([item['N_star'] for (key,item) in profiles.items()])
    print("N star = ", N_star_max)
    
    # Stack profiles into one long array
    r_norms = np.array([])
    for i,name in enumerate(profiles.keys()):
        r_norm = profiles[name]['r_norm']
        N_star = profiles[name]['N_star']
        
        # Fill in all nan
        if N_star==0:
            r_norm = np.nan * np.zeros((N_star_max, len(thresholds)+1))
        else:    
            # Fill in nan to align the stacked array shape among frames
            N_fill = N_star_max - N_star

            if N_fill>0:
                r_norm = np.vstack([r_norm, np.nan * np.zeros((N_fill, len(thresholds)+1))])

        r_norms = np.vstack([r_norms, [r_norm]]) if i>0 else [r_norm]
        
    return r_norms, flags


def eval_halo_pipe(field,
                   hdu_list,
                   segm_list,
                   catalog_list,
                   ra_range, dec_range, 
                   wsid, password, 
                   thresholds=np.logspace(-0.3,-3.,22),
                   mag_range=[8.5,10.5],
                   pixel_scale=2.5,
                   dist_mask_min=100,
                   atalas_dir='./',
                   save_dir='./'):
    
    """ 
    Evaluate bright stellar halos.
    
    Parameters
    ----------
    hdu_list: list
        Path list of frames.
    segm_list: list
        Path list of segementation maps.
    catalog_list: list
        Path list of SExtractor catalogs.
    ra_range: tuple or list
        Range of RA
    dec_range: tuple or list
        Range of dec
    wsid: str
        casjob WSID
    password: str
        casjob password
    thresholds: np.array
        Thresholds at x% of the saturation brightness.
    mag_range: list
        Range of magnitude of bright stars for measurement
    pixel_scale : float
        pixel scale in arcsec/pixel
    dist_mask_min: int, default None
        Minimum distance to the field edges mask.
    
    """
    
    N_frame = len(hdu_list)
    
    # Get filter names
    filters_ = np.array([fits.getheader(fn)["FILTER"] for fn in hdu_list])   
    
    # Query ATLAS catalog and sleep a while for its finish
    fname_query = query_atlas_catalog(ra_range, dec_range, wsid, password, atalas_dir, mag_limit=12)
    time.sleep(5)
    
    # Rename and read the queried catalog
    fname_atlas = os.path.join(atalas_dir, f'{field}_atlas.csv')
    shutil.copy(fname_query, fname_atlas)
    table_atlas = Table.read(fname_atlas, format='csv')
    
    # Contrasts from thresholds
    contrasts = 1/thresholds
    
    # Extract curves of growth
    r_norms_, flags_ = extract_profile_pipe(hdu_list,
                                            segm_list, 
                                            catalog_list, 
                                            table_atlas, 
                                            thresholds=thresholds,
                                            mag_range=mag_range,
                                            dist_mask_min=dist_mask_min, 
                                            pixel_scale=pixel_scale)
    
    # Drop profiles bad flags, keep for the brigthest N_star
    # r_norms_ is MxNxK array, M:# of frames, N:max # of stars among frames, K:1 + # of thresholds
    r_norms = r_norms_[flags_==1][:,:,1:]   # the first value is saturated r0
    filters = filters_[flags_==1]
    
    # Plot profiles
    plot_profiles(r_norms, filters, contrasts, save_dir=save_dir, suffix='_'+field)
    
    # Clustering pofiles
#    labels = clustering_profiles_optimize(r_norms, filters, contrasts,
#                                          log=False, eps_grid = np.arange(0.2,0.65,0.05),
#                                          save_dir=save_dir, field=field)
                                      
    labels = clustering_profiles_optimize(r_norms, filters, contrasts,
                                            log=True, eps_grid = np.arange(0.1,0.3,0.02),
                                            save_dir=save_dir, field=field)
    
    table = Table({'frame':hdu_list, 'filter':filters_, 'flag':flags_})
    table['label'] = -1 * np.ones_like(flags_)
    table['label'][flags_==1] = labels
    
    table.write(os.path.join(save_dir, f'{field}_halo.txt'), format='ascii', overwrite=True)
    
    return table, r_norms, filters
