import os
import glob
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

from .profile import *
from .clustering import *
from .utils import *
from .plot import *

def eval_halo_pipe(field,
                   hdu_list,
                   segm_list,
                   catalog_list,
                   ra_range, dec_range,
                   wsid, password,
                   thresholds=np.logspace(-0.3,-3.3,22),
                   mag_range=[8.5,10.5],
                   pixel_scale=2.5,
                   contrast_range=[200, 1000],
                   do_clustering=True,
                   eps_grid=np.arange(0.1,0.3,0.02),
                   dist_mask_min=100,
                   atalas_dir='./',
                   catalog_atals_dir=None,
                   save_dir='./',
                   plot=True):
    
    """
    Evaluate bright stellar halos.
    
    A list of corresponding segementation maps and SE catalogs are required.
    
    Parameters
    ----------
    field: str
        Field name (identifier).
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
    pixel_scale: float
        Pixel scale in arcsec/pixel
    contrast_range: [float , float]
        Range of contrast for fitting 1D linear model.
    do_clustering: bool, optional
        If True, do clustering on the profiles to identify outliers.
    eps_grid: 1d array, optional
        Input grid of eps for parameter tuning.
    dist_mask_min: int, optional, default None
        Minimum distance in pix to the field edges mask.
    atalas_dir: str, optional
        Path to store the ATLAS query file.
    catalog_atals_dir: str, optional
        Path to the local ATLAS catalog files.
        If specified, ATALS catalog will be made locally.
        In the dir files are sorted by mag (e.g. 00_m_16) or dec.
    plot: bool, optional
        Whether to draw diagnostic plots.
    
    """
    
    print(f"Running halo evaluation for {field}.")
    
    N_frame = len(hdu_list)
    
    # Get filter names
    filters_ = np.array([fits.getheader(fn)["FILTER"] for fn in hdu_list])
    
    if catalog_atals_dir is None:
        # Query ATLAS catalog
        table_atlas = query_atlas_catalog(field, ra_range, dec_range, wsid, password, atalas_dir, mag_limit=12)
    else:
        # Build ATLAS catalog from local csv files
        table_atlas = make_atlas_catalog(ra_range, dec_range, mag_limit=12, catalog_dir=catalog_atals_dir)
    
    # Contrasts from thresholds
    contrasts = 1/thresholds
    
    # Extract curves of growth
    r_norms_, flags_ = extract_profile_pipe(hdu_list,
                                            segm_list,
                                            catalog_list,
                                            table_atlas,
                                            thresholds=thresholds,
                                            mag_range=mag_range,
                                            pixel_scale=pixel_scale,
                                            N_source_min=3000,
                                            dist_mask_min=dist_mask_min,
                                            plot=plot)
    
    # Drop profiles bad flags, keep for the brigthest N_star
    # r_norms_ is MxNxK array, M:# of frames, N:max # of stars among frames, K:1 + # of thresholds
    r_norms = r_norms_[flags_==1][:,:,1:]   # the first value is saturated r0
    filters = filters_[flags_==1]
    
    if plot:
        plot_profiles(r_norms, filters, contrasts,
                      save=True, save_dir=save_dir, suffix='_'+field)
                      
    if do_clustering:
        # Clustering in log space
        labels = clustering_profiles_optimize(r_norms, filters, contrasts,
                                              log=True, eps_grid=eps_grid,
                                              field=field, plot=plot,
                                              save_plot=True, save_dir=save_dir)

        # Clustering pofiles in linear space
#        labels = clustering_profiles_optimize(r_norms, filters, contrasts,
#                                              log=False, eps_grid = np.arange(0.2,0.65,0.05),
#                                              field=field, plot=plot,
#                                              save_plot=True, save_dir=save_dir)
    
    # Fit a 1D linear model to outskirts
    slopes_list = np.array([])
    slope_med_list = np.array([])
    for r_norm in r_norms:
        first_col = r_norm[:,0]
        r_ = r_norm[~np.isnan(first_col)]
        if len(r_) == 0:
            slope, slope_med = 99, 99
        else:
            slopes, slope_med = fit_profile_slopes(r_, contrasts, contrast_range)
        slopes_list = np.append(slopes_list, slopes)
        slope_med_list = np.append(slope_med_list, slope_med)
    
    # Show histogram of slopes
    plt.hist(slope_med_list,alpha=0.5)
    plt.xlabel("Slope med")
    plt.show()
    
    # Make the table with flags, labels, and slopes
    table = Table({'frame':hdu_list, 'filter':filters_, 'flag':flags_})
    table['slope'] = 99.0  # default value 99
    table['slope'][flags_==1] = slope_med_list
    if do_clustering:
        table['label'] = -1  # default value -1
        table['label'][flags_==1] = labels
    else:
        table['label'] = 99 * np.ones_like(hdu_list)
    
    # Append comments to the table
    str_contrs_range = ", ".join(map(str, contrast_range))
    comments = ["flags: 1: good. 0: bad profule measurement.",
                "slope measured between contrast [{:}].".format(str_contrs_range),
                "label: DBSCAN clustering labels. -1: bad/outlier. 99: no clustering."]
    table.meta['comments'] = comments
    
    # Write to disk
    table.write(os.path.join(save_dir, f'{field}_halo.txt'), format='ascii', overwrite=True)
    
    return table, r_norms, filters
    
    
def extract_profile_pipe(hdu_list, segm_list, 
                         catalog_list, table_atlas, 
                         thresholds=np.logspace(-0.3,-3.,22), 
                         mag_range=[8.5,10.5],
                         pixel_scale=2.5,
                         N_source_min=3000,
                         dist_mask_min=None,
                         plot=True):
    
    """ 
    Extract curves of growth in the list of frames.
    
    A list of corresponding segementation maps, SE catalogs, and
    a crossmatched catalog with ATALS are required.
    
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
        Range of magnitude of bright stars for measurement.
    pixel_scale : float
        Pixel scale in arcsec/pixel
    N_source_min: int, default 3000
        Minimum number of sources required in the frame.
        If set None, use a 2.5% quantile.
    dist_mask_min: int, optional, default None
        Minimum distance to the field edges mask.
    plot: bool, optional
        Whether to draw diagnostic plots.
    
    Returns
    -------
    r_norms: 3d np.array
        Curves of Growth (axis 0: frame, axis 1: star, axis 2: radius)
    flags: 1d n.array
        1: Good  0: Bad
    
    """
    
    # Get filter names
    filters = np.array([fits.getheader(fn)["FILTER"] for fn in hdu_list])   
    
    # Set minimum number of intermediate bright sources detected
    if N_source_min is None:
        N_source = np.zeros(len(hdu_list), dtype=int)
        for i, (catalog_path) in enumerate(catalog_list):
            SE_catalog = Table.read(catalog_path, format='ascii.sextractor')
            N_source[i] = len(SE_catalog[(SE_catalog['MAG_AUTO']>=13) & (SE_catalog['MAG_AUTO']<18)])
        N_source_G = N_source[filters=='G']
        N_source_R = N_source[filters=='R']
        
        N_source_G_min = np.quantile(N_source_G, 0.025)
        N_source_R_min = np.quantile(N_source_R, 0.025)
        
        # Plot histograms of # of detected sources
        if plot:
            plt.figure()
            plt.hist(N_source_G, color='seagreen', bins=8, alpha=0.7)
            plt.hist(N_source_R, color='firebrick', bins=8, alpha=0.7)
            plt.axvline(N_source_G_min, color='lime', ls='--', alpha=0.9)
            plt.axvline(N_source_R_min, color='r', ls='--', alpha=0.9)
            plt.xlabel('# of source detected')
            plt.show()
            
    else:
        N_source_G_min = N_source_R_min = 3000
    
    # Measure profiles
    profiles = {}
    flags = np.zeros_like(hdu_list, dtype=int)
    
    print("Remove sources < {:d} pixel to the edges".format(dist_mask_min))
    
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
    print("N star set to ", N_star_max)
    
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
