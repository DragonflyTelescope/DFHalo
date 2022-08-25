import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics

from .plot import plot_profile_clustering

def clustering_profiles(r_norms, filters, contrasts, 
                        save_dir='.', field=''):
    
    # Clustering by curve of growth
    print('Clustering profiles...')
    N_min_sample = 10

    X = np.nanmedian(r_norms,axis=1)
    db = DBSCAN(eps=3, min_samples=N_min_sample, algorithm='auto').fit(X)

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("N cluster = ", n_clusters_, ",  N noise = ", n_noise_)

    plot_profile_clustering(X, labels, contrasts, 
                            norm=False, save_dir=save_dir, suffix='_'+field)

    # Median profile for G and R
    r_norms_G = r_norms[filters=='G']
    r_norms_R = r_norms[filters=='R']
    r_norm_G_med = np.nanmedian(np.nanmedian(r_norms_G,axis=1), axis=0)
    r_norm_R_med = np.nanmedian(np.nanmedian(r_norms_R,axis=1), axis=0)

    # Clustering by color-corrected curve of growth
    X_ = np.nanmedian(r_norms, axis=1)
    X = X_.copy()
    X[filters=='G'] = X_[filters=='G']/r_norm_G_med
    X[filters=='R'] = X_[filters=='R']/r_norm_R_med

    db = DBSCAN(eps=0.5, min_samples=N_min_sample, algorithm='auto').fit(X)

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("N cluster = ", n_clusters_, ",  N noise = ", n_noise_)

    plot_profile_clustering(X, labels, contrasts, 
                            norm=True, save_dir=save_dir, suffix=f'_normed_{field}')
    return labels