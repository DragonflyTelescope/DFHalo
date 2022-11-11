import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import mad_std


def plot_profiles(r_norms, filters, contrasts, flags=None,
                  save=False, save_dir='.', suffix=''):
    """ Display profiles """
    
    # Profiles for G and R
    r_norms_G = r_norms[filters=='G']
    r_norms_R = r_norms[filters=='R']
    
    # Median profiles by stars (averaging frames)
    r_norms_G_star = np.nanmedian(r_norms_G, axis=0)
    r_norms_R_star = np.nanmedian(r_norms_R, axis=0)
    
    # Median profile averaging stars and frames for display norm
    r_norm_med = np.nanmedian(np.nanmedian(r_norms, axis=1), axis=0)
    
    # Remove star-star variations caused by systematics
    r_norms_ = r_norms.copy()
    r_norms_[filters=='G']  = r_norms_G/ r_norms_G_star * r_norm_med
    r_norms_[filters=='R']  = r_norms_R/ r_norms_R_star * r_norm_med
    
    plt.figure(figsize=(10,8))
    colors = plt.cm.jet(np.linspace(0.1, 0.9, len(r_norms)))
    
    for i, band in enumerate(filters):
        color = colors[i]
        
        # i-th frame
        r_norm_i = r_norms_[i]

        # Slightly offset the profiles
        dx = 1 + np.random.random(1) * 0.05
        
        if flags is None:
            if band=='R':
                color = 'firebrick'
            elif band=='G':
                color = 'seagreen'
        else:
            if flags[i] == 0:
                color = 'orange' # bad halos
            else:
                color = 'steelblue' # good halos

        for j in range(len(r_norm_i)):
            # j-th star
            plt.plot(contrasts * dx, r_norm_i[j], 'o', ms=1, alpha=0.05, color=color)
        
        # stars averaged
        r_norm_i_med = np.nanmedian(r_norm_i, axis=0)
        
        #yerr = np.abs(r_norm_i_med - np.nanquantile(r_norm_i, [0.16,0.84], axis=0))
        plt.plot(contrasts * dx, r_norm_i_med, '-s',
                 ms=3, mec='k', lw=2, alpha=0.4, color=color, zorder=2)
#        plt.errorbar(contrasts * dx, r_norm_i_med,
#                     yerr=yerr, alpha=0.1, color=color, zorder=1)
        plt.text(contrasts[-1] * dx * 1.2, np.nanmedian(r_norm_i[:,-1]), i, fontsize=8)
    
    # Median corrected profile among frames (averaging stars)
    r_norms_med = np.nanmedian(r_norms_, axis=1)
    
    plt.plot(contrasts, np.nanmedian(r_norms_med[filters=='G'], axis=0), '-', ms=5, lw=3, alpha=0.8, label='Median', color='lime', zorder=3)
    plt.plot(contrasts, np.nanmedian(r_norms_med[filters=='R'], axis=0), '-', ms=5, lw=3, alpha=0.8, label='Median', color='r', zorder=3)

    plt.ylim(0.85,23)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Contrasts', fontsize=18)
    plt.ylabel('R$_c$ / R$_0$', fontsize=18)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_dir, f'profiles{suffix}.png'))
    plt.show()
    
    plt.figure()
    # sigma among frames (averaging stars)
    plt.plot(contrasts, mad_std(r_norms_med, axis=0), '-', color='k')
    plt.plot(contrasts, mad_std(r_norms_med[filters=='G'], axis=0), '-', color='lime')
    plt.plot(contrasts, mad_std(r_norms_med[filters=='R'], axis=0), '-', color='r')
    
    plt.xlabel('Contrasts')
    plt.ylabel('$\sigma$')
    plt.xscale('log')
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_dir, f'dispersion{suffix}.png'))
    plt.show()
    
    
def plot_profile_clustering(X, labels, contrasts, 
                            norm=True, log=False,
                            save=True, save_dir='.',
                            suffix=''):
    """ Display clustering result of profiles. """
    
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
        
    if log:
        plt.ylim(-0.3,1)
        plt.ylabel('log (R$_c$ / R$_0$)')
    else:
        plt.ylabel('R$_c$ / R$_0$')
        
    plt.xscale('log')
    plt.xlabel('Contrasts')
    
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_dir, f'clustering_profiles{suffix}.png'))
    plt.show()
    
