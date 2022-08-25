import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import mad_std


def plot_profiles(r_norms, bands, contrasts, save_dir='.', suffix=''):
    plt.figure(figsize=(10,8))
    colors = plt.cm.jet(np.linspace(0.1, 0.9, len(r_norms)))
    
    for i, band in enumerate(bands):
        color = colors[i]
        
        # i-th frame
        r_norm_i = r_norms[i]

        # Slightly offset the profiles
        dx = 1 + np.random.random(1) * 0.1

        if band=='R':
            color = 'firebrick' 
        elif band=='G':
            color = 'seagreen'

        for j in range(len(r_norm_i)):
            # j-th star
            plt.plot(contrasts * dx, r_norm_i[j], 'o', alpha=0.05, color=color)
        
        yerr = np.abs(np.nanmedian(r_norm_i, axis=0) - np.nanquantile(r_norm_i, [0.16,0.84], axis=0))
        plt.plot(contrasts * dx, np.nanmedian(r_norm_i, axis=0), '-s', 
                 ms=6, mec='k', lw=2, alpha=0.3, color=color, zorder=2)
        plt.errorbar(contrasts * dx, np.nanmedian(r_norm_i, axis=0), 
                     yerr=yerr, alpha=0.1, color=color, zorder=1)
        plt.text(contrasts[-1] * dx * 1.2, np.nanmedian(r_norm_i[:,-1]), i, fontsize=8)
    
    r_norms_G = r_norms[bands=='G']
    r_norms_R = r_norms[bands=='R']
    
    r_norm = np.nanmedian(r_norms, axis=0)
    r_norm_G = np.nanmedian(r_norms_G, axis=0)
    r_norm_R = np.nanmedian(r_norms_R, axis=0)
    
    # Median profiles for G and R among frames
    r_norm_G_med = np.nanmedian(r_norm_G, axis=0)
    r_norm_R_med = np.nanmedian(r_norm_R, axis=0)
    
    plt.plot(contrasts, r_norm_G_med, '-', ms=5, lw=3, alpha=0.8, label='Median', color='lime', zorder=3)
    plt.plot(contrasts, r_norm_R_med, '-', ms=5, lw=3, alpha=0.8, label='Median', color='r', zorder=3)
    
    ## median profile among frames (averaging stars)
    plt.plot(contrasts, np.nanmedian(np.nanmedian(r_norms_G,axis=1), axis=0), '--', ms=5, lw=3, alpha=0.8, label='Median', color='lime', zorder=3)
    plt.plot(contrasts, np.nanmedian(np.nanmedian(r_norms_R,axis=1), axis=0), '--', ms=5, lw=3, alpha=0.8, label='Median', color='r', zorder=3)
    ##

    plt.ylim(0.9,22)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Contrasts')
    plt.ylabel('R$_c$ / R$_0$')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'profiles{suffix}.png'))
    plt.show()
    
    plt.figure()
    plt.plot(contrasts, mad_std(r_norm, axis=0), color='k')
    plt.plot(contrasts, mad_std(r_norm_G, axis=0), color='lime')
    plt.plot(contrasts, mad_std(r_norm_R, axis=0), color='r')
    
    ## sigma among frames (averaging stars)
    plt.plot(contrasts, mad_std(np.nanmedian(r_norms, axis=1), axis=0), '--', color='k')
    plt.plot(contrasts, mad_std(np.nanmedian(r_norms_G, axis=1), axis=0), '--', color='lime')
    plt.plot(contrasts, mad_std(np.nanmedian(r_norms_R, axis=1), axis=0), '--', color='r')
    ##
    
    plt.xlabel('Contrasts')
    plt.ylabel('$\sigma$')
    plt.xscale('log')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'dispersion{suffix}.png'))
    plt.show()
    
    
def plot_profile_clustering(X, labels, contrasts, 
                            norm=True, save_dir='.', suffix=''):
    
    unique_labels = set(labels)
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    colors = {label:plt.cm.plasma_r(each) for label, each in zip(unique_labels, np.linspace(0.2, 0.8, n_clusters_))}
    colors[-1] = (0.1,0.1,0.1)

    plt.figure()
    for i in range(len(X)):
        plt.plot(contrasts, X[i], color=colors[labels[i]], alpha=0.2)
    
    if norm==False:
        plt.ylim(0.85,25)
        plt.yscale('log')
    else:
        plt.ylim(0.,3)
    plt.xscale('log')
    plt.xlabel('Contrasts')
    plt.ylabel('R$_c$ / R$_0$')
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'clustering_profiles{suffix}.png'))
#     plt.show()
    
