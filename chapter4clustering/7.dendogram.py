# heatmap of average distance between clusters
import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
import sys

path = '/media/emily/south/rmac_feature_vectors/2018_rmac_feature_vector_clusters_20.csv'
dim = 512
feature = '0'
clusters = 20

df = pd.read_csv(path, error_bad_lines=False, engine="python")
print (df.shape)
# df['img'] = df['img_id'].apply(lambda x: x.split('/')[-1])
df[feature] = df[feature].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))

X = np.array(df[feature].values.tolist())
print (X.shape)
n_clusters = 20

def get_cluster_centers(df):
    total = df[[feature, clusters]].groupby('clusters').sum()
    count = df[[feature, clusters]].groupby('clusters').count()
    centroid = total/count
    return list(count.index), list(centroid[feature])

def get_cluster_center(K, df):
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
        centroid_1 = get_cluster_center(i, df)
        centroid_2 = get_cluster_center(j, df)
        dist = k_mean_distance(centroid_1, centroid_2)
        distances[i][j] = dist


# import matplotlib.pyplot as plt
# import numpy as np

# a = np.random.random((16, 16))
# plt.imshow(distances, cmap=cm.gray_r, interpolation='nearest')
# plt.colorbar()
# plt.show()

# from pylab import figure, cm
# from matplotlib.colors import LogNorm

# C = 1-distances
# f = figure(figsize=(6.2, 5.6))
# ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
# axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])

# im = ax.matshow(C, cmap=cm.gray_r, norm=LogNorm(vmin=0.01, vmax=1))

# t = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
# f.colorbar(im, cax=axcolor, ticks=t, format="$%.2f$")

# f.show()


################ SANKEY
import pandas as pd
from pySankey import sankey
import matplotlib.pyplot as plt 

c = pd.read_csv('/home/emily/phd/2_interpretability/sankey/rmac_clusters.csv')

colors = {
    0: "#d62728",
    1: "#1f77b4", # high-density
    2: "#8c564b",
    3: "#8c564b",
    4: "#8c564b", # green = veg
    5: "#8c564b", # commercial
    6: "#1f77b4", 
    7: "#1f77b4", # terraced 
    8: "#9467bd", # disregarded
    9: "#9467bd", 
    10: "#d62728", 
    11: "#1f77b4", # low-density
    12: "#8c564b", 
    13: "#1f77b4", 
    14: "#9467bd", 
    15: "#1f77b4", 
    16: "#ff7f0e", 
    17: "#2ca02c", 
    18: "#ff7f0e", 
    19: "#2ca02c"
}

colors = {
    0: "#1f77b4",
    1: "#2ca02c", 
    2: "#2ca02c",
    3: "#8c564b",
    4: "#ff7f0e", # disregarded
    5: "#8c564b", # 
    6: "#ff7f0e", 
    7: "#8c564b", # 
    8: "#9467bd", # 
    9: "#9467bd", 
    10: "#ff7f0e", # commercial
    11: "#2ca02c", #
    12: "#8c564b", 
    13: "#d62728", 
    14: "#8c564b", 
    15: "#d62728", 
    16: "#8c564b", 
    17: "#ff7f0e", 
    18: "#8c564b", 
    19: "#8c564b"
}

a = [11,1,12,4,3,2,5,14,8,9,10,0,13,7,19,17,6,18,16,15]
b = [a[19 - i] for i in range(20)]
sankey.sankey(c['cluster_20'], c['cluster_10'], aspect=20, leftLabels = b, fontsize=12)
plt.savefig("/home/emily/phd/2_interpretability/sankey/outputs/rmac_20_10.svg")

import scipy
import pylab
import scipy.cluster.hierarchy as sch

# Generate features and distance matrix.
D = distances
# Compute and plot dendrogram.
# fig = pylab.figure()
# # axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
# axdendro = fig.add_axes([0.73,0.1,0.2,0.6])
fig, ax = plt.subplots()
Y = sch.linkage(D, method='single')
Z = sch.dendrogram(Y, orientation='right', color_threshold=0.52)
# axdendro.set_xlim([0.8, 0.4])
# axdendro.set_yticks([])
plt.ylabel('Cluster')
plt.xlabel('Distance')
plt.savefig("/home/emily/phd/2_interpretability/distances/dendrogram_only.png")
#plt.show()

# Plot distance matrix.
# axmatrix = fig.add_axes([0.2,0.1,0.5,0.6])
# index = Z['leaves']
# D = D[index,:]
# D = D[:,index]
# im = axmatrix.matshow(D, aspect='auto', origin='lower')
# axmatrix.set_xticks([])
# axmatrix.set_yticks([])

# # Plot colorbar.
# axcolor = fig.add_axes([0.15,0.1,0.02,0.6])
# pylab.colorbar(im, cax=axcolor)
# axcolor.yaxis.tick_left()

# # axcolor = fig.add_axes([0.35,0.1,0.2,0.35])

# # Display and save figure.
# # fig.show()
# plt.savefig("/home/emily/phd/2_interpretability/distances/dendrogram_right_sf.svg")