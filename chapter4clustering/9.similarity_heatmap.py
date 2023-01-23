# heatmap of average distance between clusters
import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
import sys

dim = 512
feature = '0'
clusters = 20

prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home/emily/phd/003_image_matching/clustering/output/'
prefix2 = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/project/pathways/live/Transferability/emily_phd_images/keras-rmac-clustering_output/2018_original/'

df_2011 = pd.read_csv(prefix + 'census2011_zoom_clusters_20.csv', error_bad_lines=False, engine="python")
df_2011['0'] = df_2011['0'].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))

X_11 = np.array(df_2011[feature].values.tolist())
print (X_11.shape)
# n_clusters = df_2011[clusters].max() + 1

# df_2021 = pd.read_csv(prefix + 'census2021_zoom_clusters_20.csv', error_bad_lines=False, engine="python")
# df_2021['0'] = df_2021['0'].apply(lambda x: 
#                            np.fromstring(
#                                x.replace('\n','')
#                                 .replace('[','')
#                                 .replace(']','')
#                                 .replace('  ',' '), sep=' '))

# X_21 = np.array(df_2021[feature].values.tolist())
# print (X_21.shape)
# n_clusters = 20

df_2018 = pd.read_csv(prefix2 + '2018_rmac_feature_vector_clusters_20.csv', error_bad_lines=False, engine="python")
df_2018['0'] = df_2018['0'].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))

X_18 = np.array(df_2018[feature].values.tolist())
print (X_18.shape)
n_clusters = 20

def get_cluster_centers(df):
    total = df[[feature, clusters]].groupby('clusters').sum()
    count = df[[feature, clusters]].groupby('clusters').count()
    centroid = total/count
    return list(count.index), list(centroid[feature])

def get_cluster_center(K, df, X):
    sub = df[df['clusters'] == K]['0'].index
    centroid = X[sub].mean(axis=0)
    return centroid

def k_mean_distance(centroid_1, centroid_2):
    # Calculate Euclidean distance for each data point assigned to centroid 
    distance = [(centroid_1[i]-centroid_2[i])**2 for i in range(dim)]
    dist = np.sqrt(sum(distance))
    # return the mean value
    return dist

# plot k-means centroid subspace
# print ('Getting cluster centers')
# clusters_, X_center = get_cluster_centers(df) 
# centroids = X_center

distances = np.zeros((n_clusters, n_clusters))
for i in range(n_clusters):
    for j in range(n_clusters):
        centroid_1 = get_cluster_center(i, df_2018, X_18)
        centroid_2 = get_cluster_center(j, df_2011, X_11)
        dist = k_mean_distance(centroid_1, centroid_2)
        distances[i][j] = dist


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pylab import figure, cm
plt.clf()
a = np.random.random((16, 16))
plt.imshow(distances, cmap=cm.Reds_r, interpolation='nearest')
plt.colorbar()
plt.xlabel('2011')
plt.ylabel('2018')
plt.savefig("/home/emily/phd/2_interpretability/distances/heatmap_2018_2011.png")


fig = plt.figure(1)
ax = fig.add_subplot(111, autoscale_on=False)
plt.imshow(distances, cmap=cm.Reds_r, interpolation='nearest')
ax.annotate('x', fontsize=20, xy=(1, 9),
            xycoords='data'
            )


from pylab import figure, cm
# from matplotlib.colors import LogNorm

# C = 1-distances
# f = figure(figsize=(6.2, 5.6))
# ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
# axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])

# im = ax.matshow(C, cmap=cm.gray_r, norm=LogNorm(vmin=0.01, vmax=1))

# t = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
# f.colorbar(im, cax=axcolor, ticks=t, format="$%.2f$")

# f.show()



import scipy
import pylab
import scipy.cluster.hierarchy as sch

# Generate features and distance matrix.
D = distances

# Compute and plot dendrogram.
fig = pylab.figure()
axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
Y = sch.linkage(D, method='single')
Z = sch.dendrogram(Y, orientation='right', color_threshold=1)
# axdendro.set_xlim([0.8, 0.4])
# axdendro.set_yticks([])

# Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
index = Z['leaves']
D = D[index,:]
D = D[:,index]
im = axmatrix.matshow(D, aspect='auto', origin='lower')
axmatrix.set_xticks([])
axmatrix.set_yticks([])

# Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
pylab.colorbar(im, cax=axcolor)

# Display and save figure.
fig.show()
fig.savefig('dendrogram.png')