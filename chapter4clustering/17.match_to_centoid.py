import pandas as pd
import os
import numpy as np

path = '/media/emily/south/rmac_feature_vectors/2018_rmac_feature_vector_clusters_20.csv'
dim = 512
feature = '0'

df = pd.read_csv(path, error_bad_lines=False, engine="python")
print (df.shape)
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
    total = df[[feature, 'clusters']].groupby('clusters').sum()
    count = df[[feature, 'clusters']].groupby('clusters').count()
    centroid = total/count
    return list(count.index), list(centroid[feature])

clusters, centers = get_cluster_centers(df) 
del X 
del df 

# GET RMAC FEATURE VECTORS FROM 2011 and 2021 
prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home/'
year = 'both_years_zoom'
df_years = pd.read_csv(prefix + '/emily/phd/003_image_matching/keras_rmac-master/census2021outputs/census_2011_and_census_2021_zoom.csv')
df_years['0'] = df_years['0'].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))

def minimise_distance(point, centroids, clusters):
    distance = [np.linalg.norm(point - i) for i in centroids]
    return distance, clusters[np.argmin(distance)]

import time 
# start clustering 
start = time.time()
df_years['matched'] = df_years['0'].apply(lambda x: minimise_distance(x, centers, clusters))
finish = time.time()
print ('matching algorithm finished in %s seconds' % str(finish-start))

df_years.to_csv(prefix + '/emily/phd/003_image_matching/keras_rmac-master/census2021outputs/census_2011_and_census_2021_zoom_matched.csv')




