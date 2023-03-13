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


# calculate the 90th percentile of the radius of cluster centroid as r
def get_radius(df, centroid, clusters):
    df['point_centroid'] = df.apply(lambda x: np.linalg.norm(x['0'] - centroid[x.clusters]), axis=1 )
    radii_max = []
    radii_min = []
    for c in clusters:
        sub = df[df['clusters'] == c]
        r_max = sub['point_centroid'].quantile(0.75) + (sub['point_centroid'].quantile(0.75) - sub['point_centroid'].quantile(0.25))*1.5
        r_min = sub['point_centroid'].quantile(0.25) - (sub['point_centroid'].quantile(0.75) - sub['point_centroid'].quantile(0.25))*1.5
        radii_max.append(r_max)
        radii_min.append(r_min)
    return df, radii_min, radii_max

df, radii_min, radii_max = get_radius(df, centers, clusters)

rdf = pd.DataFrame([clusters, radii_min,radii_max]).T

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# rdf[1].plot(linestyle=' ', marker='x', color='red', ax=ax1)
df[['point_centroid', 'clusters']].boxplot('point_centroid', by='clusters', ax=ax2, showfliers=False)
ax1.plot(ax1.get_xticks(), rdf[1],linestyle=' ', marker='x', color='red')
ax1.plot(ax1.get_xticks(), rdf[2],linestyle=' ', marker='x', color='red')
ax2.set_ylim(0.5,1.1)
ax1.set_ylim(0.5,1.1)
rdf.to_csv('chapter4clustering/outputs/R/2018_radii.csv')
df[['point_centroid', 'clusters']].to_csv('chapter4clustering/outputs/R/2018_distances_to_centroid.csv')
plt.show()
plt.savefig('chapter4clustering/outputs/boxplot_clusters_radii_markers_0.9_no_outliers.png',bbox_inches="tight", dpi=150)

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
    # this calculates the distance between this point and all cluster centroids
    distance = [np.linalg.norm(point - i) for i in centroids]
    return distance, clusters[np.argmin(distance)]

def minimise_distance(point, centroids, clusters, radii_max, radii_min):
    # this calculates the distance between this point and all cluster centroids
    distance = [np.linalg.norm(point - i) for i in centroids]
    d = min(distance)
    c_min = clusters[np.argmin(distance)]
    r_max = radii_max[c_min]
    r_min = radii_min[c_min]
    if d > r_max:
        p = 0.001
    elif d < r_min:
        p = 1
    else:
        p = 1 - ((d-r_min)/(r_max - r_min))
    return c_min, p, d

import time 
# start clustering 
start = time.time()
df_years['matched'] = df_years['0'].apply(lambda x: minimise_distance(x, centers, clusters, radii_max, radii_min))
finish = time.time()
print ('matching algorithm finished in %s seconds' % str(finish-start))

df_years['p'] = df_years.apply(lambda x: x.matched[1], axis=1)
df_years['clusters'] = df_years.apply(lambda x: x.matched[0], axis=1)
df_years['distance'] = df_years.apply(lambda x: x.matched[2], axis=1)

fig, ax1 = plt.subplots()
df_years[['distance', 'clusters']].boxplot('distance', by='clusters', ax=ax1)
plt.savefig('chapter4clustering/outputs/boxplot_mathed_years_distance.png',bbox_inches="tight", dpi=150)

fig, ax1 = plt.subplots()
df_years[['p', 'clusters']].boxplot('p', by='clusters', ax=ax1)
plt.savefig('chapter4clustering/outputs/boxplot_mathed_years_p.png',bbox_inches="tight", dpi=150)

df_years[['p', 'distance', 'clusters']].to_csv('chapter4clustering/outputs/R/both_years_distances_to_centroid.csv')

df_years = df_years.drop(['0'], axis=1)
prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home'
df_years.to_csv(prefix + '/emily/phd/003_image_matching/keras_rmac-master/census2021outputs/census_2011_and_census_2021_zoom_matched.csv')

df_years.to_csv('chapter4clustering/outputs/census_2011_and_census_2021_zoom_matched.csv')


