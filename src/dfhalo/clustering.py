import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics

from .plot import plot_profile_clustering

def clustering_profiles_optimize(r_norms, filters, contrasts,
                                 eps_grid=np.arange(0.1,0.3,0.02),
                                 N_min_sample=5, log=True,
                                 field='', plot=True,
                                 save_plot=True, save_dir='.'):
    
    """
    Do clustering on profiles with optimized hyperparameter eps.
    
    Parameters
    ----------
    
    r_norms: 3d np.array
        Curves of Growth (axis 0: frame, axis 1: star, axis 2: radius)
    filters: str np.array
        Filters of frames.
    contrasts: 1d array
        Contrasts (1/threshold) corresponding at which r_norms are measured.
    eps_grid: 1d array
        Input grid of eps for parameter tuning.
    N_min_sample: int, default 5
        Minimum number of samples in a single cluster in DBSCAN clustering.
    log: bool, default True
        Whether to clustering profiles in log space.
    field: str
        Field name (identifier).
    plot: bool
        Whether to plot diagnostic figures.
    save_plot: bool
        Whether to save plot
        
    Returns
    -------
    labels: 1d array
        Labels of clustering.
        
    """
    
    scores = np.zeros(len(eps_grid))
    
    # Profiles for G and R
    r_norms_G = r_norms[filters=='G']
    r_norms_R = r_norms[filters=='R']
    
    # Median profile by stars (averaging frames)
    r_norms_star = np.nanmedian(r_norms, axis=0)
    r_norms_G_star = np.nanmedian(r_norms_G, axis=0)
    r_norms_R_star = np.nanmedian(r_norms_R, axis=0)
    
    # Median profile averaging stars and frames
    r_norm_med = np.nanmedian(r_norms_star, axis=0)
    
    # Median profiles by frames (corrected and averaging stars)
    r_norms_med = np.nanmedian(r_norms/r_norms_star, axis=1)*r_norm_med
    r_norms_G_med = np.nanmedian(r_norms_G/r_norms_G_star, axis=1)*r_norm_med
    r_norms_R_med = np.nanmedian(r_norms_R/r_norms_R_star, axis=1)*r_norm_med
    
    # Median profile of G and R
    r_norm_G_med = np.nanmedian(r_norms_G_med, axis=0)
    r_norm_R_med = np.nanmedian(r_norms_R_med, axis=0)
    
    # Median profile for each frame
    X_ = r_norms_med
    X = X_.copy()  # X is training data
    
    # Profiles color-corrected
    X[filters=='G'] = X_[filters=='G']/r_norm_G_med
    X[filters=='R'] = X_[filters=='R']/r_norm_R_med
    
    if log: X = np.log10(X)

    for k, eps in enumerate(eps_grid):
        
        labels = clustering_data(X, contrasts,
                                 eps=eps, N_min_sample=N_min_sample, log=log,
                                 save_dir=save_dir, field=field, plot=False)
        
        if len(set(labels)) > 1:
            score = metrics.silhouette_score(X, labels)
        else:
            score = np.nan
            
        scores[k] = score
    
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.plot(eps_grid, scores)
        ax.set_xlabel('eps')
        ax.set_ylabel('silhouette')
        plt.show()
    
    eps_opt = eps_grid[np.nanargmax(scores)]
    
    labels = clustering_data(X, contrasts, eps=eps_opt,
                             N_min_sample=N_min_sample,
                             log=log, plot=plot,
                             save_plot=save_plot)
    
    if plot:
        plot_profile_clustering(X_, labels, contrasts,
                                norm=False, log=False,
                                save=save_plot,
                                save_dir=save_dir,
                                suffix='_'+field)
                                
    return labels


def clustering_data(X, contrasts,
                    N_min_sample=5,
                    eps=0.2, log=True,
                    field='', plot=True,
                    save_plot=True, save_dir='.'):
    
    """
    Do DBSCAN clustering on normalized profiles.
    
    Parameters
    ----------
    
    X: 2d np.array
        Training data
    contrasts: 1d array
        Contrasts (1/threshold) corresponding at which r_norms are measured.
    eps: float, default 0.5
        eps in DBSCAN clustering.
    N_min_sample: int, default 5
        Minimum number of samples in a single cluster in DBSCAN clustering.
    log: bool, default True
        Whether to clustering profiles in log space.
    field: str
        Field name (identifier).
    save_plot: bool
        Whether to save plot
        
    Returns
    -------
    labels: 1d array
        Labels of clustering.
    
    """
    # Clustering by curve of growth
    db = DBSCAN(eps=eps, min_samples=N_min_sample, algorithm='auto').fit(X)

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("eps = %.3f,  "%eps, "N of cluster = %d,  "%n_clusters_, "N of noise = %d"%n_noise_)
    
    if plot:
        plot_profile_clustering(X, labels, contrasts,
                                norm=True, log=log,
                                save=save_plot,
                                save_dir=save_dir,
                                suffix=f'_normed_{field}')
    return labels
