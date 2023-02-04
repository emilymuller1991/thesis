import numpy as np
import pandas as pd 

df_2018 = pd.read_csv('/media/emily/south/keras_rmac/output/2018_rmac_feature_vector_clusters_20.csv')
# get distance from each cluster to center and plot
feature = '0'
clusters = 'clusters'
dim = 512
df_2018[feature] = df_2018[feature].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))

X = np.array(df_2018[feature].values.tolist())
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

def distance_centroid(centroid, X_):
    dist = np.sqrt(((X_ - centroid)**2).sum())
    return dist

evals = {}
for c in range(n_clusters):
    # get subset of data for this cluster 
    sub = df_2018[df_2018['clusters'] == c]['0'].index
    n = len(sub)
    X_ = X[sub]
    # calculate centroid
    centroid = X[sub].mean(axis=0)
    # calculate distance of each point to centroid
    d = distance_centroid(centroid, X_)
    evals[c] = [d, n]
df = pd.DataFrame(evals).T
df.columns = ['distance','count']
df['distance'].sum()
df.to_csv('/media/emily/south/phd/chapter4clustering/outputs/2018_rmac_cluster_evals.csv')
# 870269.8