import matplotlib.pyplot as plt 
from PIL import Image
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
prefix = '/rds/general/user/emuller/home'

sns.set_style("whitegrid")
#sns.set(style="white")
sns.despine()
def pca_2d(X, y=None, data='se', fitted=''):
    ax = sns.scatterplot(x = X[:,0], y=X[:,1], hue=y, palette="Set2", edgecolor="none")
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('%s dataset PCA + %s clusters' % (data, fitted))
    plt.savefig(prefix + '/emily/phd/003_image_matching/clustering/outputs/%s_pca_%s.png' % (data, fitted))
    return print ('Principal component scatterplot saved')

def heatmap(X, fitted_ = None, k=None, data = 'se'):
    if fitted_ is None:
        ax = sns.heatmap(X)
        ax.set_title('Feature Extracted HeatMap')
        plt.savefig(prefix + '/emily/phd/003_image_matching/clustering/outputs/%s_heatmap.png' % data)
        plt.clf()
    else:
        new = np.zeros(shape=(X.shape[0], X.shape[1] + 1))
        new[:,:X.shape[1]]  = X
        new[:,X.shape[1]] = fitted_
        #np.tile(fitted_, [15,1 ]).T
        sort = new[new[:,512].argsort()] 
        fig, (ax, ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [6, 1]})
        fig.subplots_adjust(wspace=0.01)
        #sns.heatmap(sort[:,:511], ax=ax, cbar = False, xticklabels='auto')
        sns.heatmap(sort[:,:X.shape[1]-1], ax=ax, cbar = False, xticklabels='auto', center=True)
        ax.set(yticklabels=[])
        fig.colorbar(ax.collections[0], ax=ax, location = 'left', use_gridspec=False)
        sns.heatmap(np.reshape(fitted_[fitted_.argsort()], (fitted_.shape[0], 1)), ax=ax2, cmap="YlGnBu", cbar = False)
        sns.heatmap(np.reshape(fitted_[fitted_.argsort()], (fitted_.shape[0], 1)), ax=ax2, cmap="YlGnBu", cbar = False)
        ax2.set(yticklabels=[])
        ax2.set(xticklabels=['Clusters'])
        ax.set_title('Feature Extraction HeatMap for k=%s' % str(k))
        plt.savefig(prefix + '/emily/phd/003_image_matching/clustering/outputs/%s_heatmap_fitted_%s.png' % (data, str(k)))
    return print ('Plotted heatmap')

def nn(X, n_neighbors=1000, data = 'se'):
    nbrs = NearestNeighbors(n_neighbors = n_neighbors).fit(X) 
    distances_, indices_ = nbrs.kneighbors(X) 
    for i in range(distances_.shape[0]): 
        plt.plot(np.arange(0,n_neighbors,1), distances_[i])
    plt.xlabel('kth Neighbour')
    plt.ylabel('Euclidean Distance')
    plt.savefig(prefix + '/emily/phd/003_image_matching/clustering/outputs/k_distance_graph_%s_%s.png' % (data, str(n_neighbors) ) )
    plt.clf()
    return print ('Plotted nearest neighbour graph for %s' % data)

# def pca_3d(X, y=None, data='iris', fitted='true'):
#     ax = sns.scatterplot(x = X[:,0], y=X[:,1], hue=y, palette="Set2")
#     plt.savefig('003_image_matching/python_outputs/clustering/%s/pca_%s.png' % (data, fitted))
#     return print ('Principal component scatterplot saved')

