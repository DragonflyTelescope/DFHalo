import os
import glob
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.stats import mad_std

from .profile import *
from .clustering import *
from .utils import *
from .plot import *

def eval_halo_pipe(field,
                   hdu_list,
                   segm_list,
                   catalog_list,
                   ra_range, dec_range,
                   catalog_atlas_dir=None,
                   wsid=None, password=None,
                   thresholds=np.logspace(-0.3,-3.3,22),
                   mag_range=[8.5,10.5],
                   pixel_scale=2.5,
                   do_clustering=True,
                   eps_grid=np.arange(0.1,0.3,0.02),
                   threshold_range=[0.02,0.5],
                   threshold_norm=None,
                   fit_contrast_range=[200, 3000],
                   N_source_min=500,
                   x_on_y=False,
                   dist_mask_min=100,
                   atlas_dir='./',
                   save_dir='.',
                   verbose=True,
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
    wsid: str, optional
        casjob WSID
        If None, catalog_atlas_dir is required to build catalog.
    password: str, optional
        casjob password
        If None, catalog_atlas_dir is required to build catalog.
    catalog_atlas_dir: str, optional
        Path to the local ATLAS catalog files.
        If specified, ATLAS catalog will be made locally.
        In the directory, files are sorted by mag (e.g. 00_m_16) or dec.
    thresholds: np.array
        Thresholds at x% of the saturation brightness.
    mag_range: list
        Range of magnitude of bright stars for measurement
    pixel_scale: float
        Pixel scale in arcsec/pixel
    do_clustering: bool, optional
        If True, do clustering on the profiles to identify outliers.
    eps_grid: 1d array, optional
        Input grid of eps for parameter tuning.
    threshold_range: [float , float], optional
        Range of threshold for fitting 1D linear model.
        The fit is used for normalizing profiles among frames.
    threshold_norm: float, optional
        Threshold at which the curves are normalized.
        The value is interpolated from 1D linear model.
    fit_contrast_range: [float , float], optional
        Range of contrast for fitting 1D linear model.
        The fit is on individual stars per frame for consistency check.
    N_source_min: int, default 500
        Minimum number of sources required in the frame.
        If set as None, use a 2.5% quantile.
    x_on_y: bool, optional
        If True, the linear fitting will be threshold on radii.
    dist_mask_min: int, optional, default None
        Minimum distance in pix to the field edges mask.
    atlas_dir: str, optional
        Path to store the ATLAS query file.
    save_dir: str, optional
        Directory to save plots and table. If None, no save.
    verbose: bool, optional
        Verbose printout
    plot: bool, optional
        Whether to draw diagnostic plots.
    
    """
    
    if save_dir is None:
        save = False
    else:
        save = True
    
    N_frame = len(hdu_list)
    
    if verbose:
        print("Running halo evaluation on {:d} frames of field {:s}.".format(N_frame, field))
    
    # Get filter names
    filters_ = np.array([fits.getheader(fn)["FILTER"] for fn in hdu_list])
    
    if catalog_atlas_dir is None:
        # Query ATLAS catalog
        table_atlas = query_atlas_catalog(field, ra_range, dec_range,
                                          wsid, password, atlas_dir,
                                          mag_limit=12, verbose=verbose)
    else:
        # Build ATLAS catalog from local csv files
        table_atlas = make_atlas_catalog(ra_range, dec_range, mag_limit=12, catalog_dir=catalog_atlas_dir)
    
    # Contrasts from thresholds
    contrasts = 1/thresholds
    
    # Extract profile
    r_norms_, flags_ = extract_profile_pipe(hdu_list,
                                            segm_list,
                                            catalog_list,
                                            table_atlas,
                                            thresholds=thresholds,
                                            mag_range=mag_range,
                                            threshold_range=threshold_range,
                                            threshold_norm=threshold_norm,
                                            pixel_scale=pixel_scale,
                                            N_source_min=N_source_min,
                                            dist_mask_min=dist_mask_min,
                                            plot=plot, verbose=verbose)
    
    # Drop profiles bad flags, keep for the brigthest N_star
    # r_norms_ is MxNxK array, M:# of frames, N:max # of stars among frames, K:1 + # of thresholds
    r_norms = r_norms_[flags_==1][:,:,1:]   # the first value is saturated r0
    filters = filters_[flags_==1]
    
    if plot:
        plot_profiles(r_norms, filters, contrasts,
                      save=save, save_dir=save_dir, suffix='_'+field)
                      
    if do_clustering:
        # Clustering in log space
        labels = clustering_profiles_optimize(r_norms, filters, contrasts,
                                              log=True, eps_grid=eps_grid,
                                              field=field, plot=plot, verbose=verbose,
                                              save_plot=save, save_dir=save_dir)

        # Clustering pofiles in linear space
#        labels = clustering_profiles_optimize(r_norms, filters, contrasts,
#                                              log=False, eps_grid = np.arange(0.2,0.65,0.05),
#                                              field=field, plot=plot,
#                                              save_plot=True, save_dir=save_dir)
    
    # Fit a 1D linear model on outskirts
    slopes_list = np.array([])
    slope_med_list = np.array([])
    chi2_list = np.array([])
    
    # stddev of profiles, values evaluated at normalized radius
    std_profiles = np.nanmedian(mad_std(np.log10(r_norms), axis=1, ignore_nan=True),axis=0)
    std_y = np.nanmedian(std_profiles)
    
    for r_norm in r_norms:
        first_col = r_norm[:,0]
        r_ = r_norm[~np.isnan(first_col)]
        
        if len(r_) == 0:
            slope_med, chi2 = 99, 99
        else:
            slopes, slope_med, chi2 = fit_profile_slopes(np.log10(r_), contrasts, fit_contrast_range, x_on_y=x_on_y, std_y=std_y)
            slopes_list = np.append(slopes_list, slopes)
            
        chi2_list = np.append(chi2_list, chi2)
        slope_med_list = np.append(slope_med_list, slope_med)
    
    # Show histogram of slopes
    if plot:
        plt.hist(slope_med_list,alpha=0.5)
        plt.xlabel("Slope med")
        plt.show()
    
    # Make the table with flags, labels, and slopes
    table = Table({'frame':hdu_list, 'filter':filters_, 'flag':flags_})
    table['slope'] = 99.0  # default value 99
    table['chi2'] = 99.0   # default value 99
    table['slope'][flags_==1] = np.around(slope_med_list,4)
    table['chi2'][flags_==1] = np.around(chi2_list,3)
    
    if do_clustering:
        table['label'] = -1  # default value -1
        table['label'][flags_==1] = labels
    else:
        table['label'] = 99 * np.ones(len(hdu_list))
    
    # Append comments to the table
    str_contrs_range = ", ".join(map(str, fit_contrast_range))
    comments = ["flags: 1: good. 0: bad profule measurement.",
                "slope measured between contrast [{:}].".format(str_contrs_range),
                "label: DBSCAN clustering labels. -1: bad/outlier. 99: no clustering."]
    table.meta['comments'] = comments
    
    if save:
    # Write to disk
        table.write(os.path.join(save_dir, f'{field}_halo.txt'),
                    format='ascii', overwrite=True)
    
    return table, r_norms, filters
    
    
def extract_profile_pipe(hdu_list, segm_list, 
                         catalog_list, table_atlas, 
                         thresholds=np.logspace(-0.3,-3.,22), 
                         mag_range=[8.5,10.5],
                         threshold_range=[0.02,0.5],
                         threshold_norm=None,
                         pixel_scale=2.5,
                         N_source_min=500,
                         dist_mask_min=None,
                         plot=True, verbose=True):
    
    """ 
    Extract radii-threshold profile in the list of frames.
    
    A list of corresponding segementation maps, SE catalogs, and
    a crossmatched catalog with ATLAS are required.
    
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
    threshold_range: [float , float]
        Range of threshold for fitting 1D linear model.
    threshold_norm: float
        Threshold at which the curves are normalized.
        The value is interpolated from 1D linear model.
    pixel_scale : float
        Pixel scale in arcsec/pixel
    N_source_min: int, default 500
        Minimum number of sources required in the frame.
        If set None, use a 2.5% quantile.
    dist_mask_min: int, optional, default None
        Minimum distance to the field edges mask.
    plot: bool, optional
        Whether to draw diagnostic plots.
    verbose: bool, optional
        Verbose printout
        
    Returns
    -------
    r_norms: 3d np.array
        Radii-threshold profile (axis 0: frame, axis 1: star, axis 2: radius)
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
        
        if len(N_source_G)>=5:
            N_source_G_min = np.quantile(N_source_G, 0.025)
        else:
            N_source_G_min = 100
        if len(N_source_R)>=5:
            N_source_R_min = np.quantile(N_source_R, 0.025)
        else:
            N_source_R_min = 100
        
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
        N_source_G_min = N_source_R_min = N_source_min
    
    # Measure profiles
    profiles = {}
    
    if verbose:
        print("Remove sources < {:d} pixel to the edges".format(dist_mask_min))
    
    for idx, (filt, hdu_path, segm_path, catalog_path) in tqdm(enumerate(zip(filters, hdu_list, segm_list, catalog_list))):
        name = os.path.basename(hdu_path).split('_light')[0]
        
        if filt=='G':
            N_source_min = N_source_G_min
        elif filt=='R':
            N_source_min = N_source_R_min
    
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyUserWarning)
            try:
                r_profiles = extract_threshold_profile(hdu_path, segm_path,
                                                       catalog_path, table_atlas,
                                                       thresholds=thresholds,
                                                       mag_range=mag_range,
                                                       dist_mask_min=dist_mask_min,
                                                       pixel_scale=pixel_scale,
                                                       N_source_min=N_source_min)

                # Normalize profiles by 1D intercepts at threshold_norm
                r_norm = normalize_profiles(r_profiles, thresholds,
                                            threshold_range=threshold_range,
                                            threshold_norm=threshold_norm)
            except IndexError:
                print(f'{hdu_path} failed in profile extraction!')
                r_norm = None
        
        # Set flag=0 if no valid measurement
        if r_norm is None:
            N_star = 0
            flag = 0
        else:
            N_star = r_norm.shape[0]
            flag = 1
        
        profiles[idx] = dict(hdu_path=hdu_path, r_norm=r_norm, N_star=N_star, flag=flag)
        
    # Get max number of sources to set array dimensions
    N_star_max = np.max([item['N_star'] for (key,item) in profiles.items()])
    if verbose:
        print("N star set to ", N_star_max)
    
    # Stack profiles into one long array
    r_norms = np.array([])
    flags = np.array([])
    for idx in profiles.keys():
        r_norm = profiles[idx]['r_norm']
        N_star = profiles[idx]['N_star']
        flag = profiles[idx]['flag']
        
        # Fill in all nan
        if N_star==0:
            r_norm = np.nan * np.zeros((N_star_max, len(thresholds)+1))
        else:    
            # Fill in nan to align the stacked array shape among frames
            N_fill = N_star_max - N_star

            if N_fill>0:
                r_norm = np.vstack([r_norm, np.nan * np.zeros((N_fill, len(thresholds)+1))])

        r_norms = np.vstack([r_norms, [r_norm]]) if idx>0 else [r_norm]
        flags = np.append(flags, flag)
        
    return r_norms, flags
