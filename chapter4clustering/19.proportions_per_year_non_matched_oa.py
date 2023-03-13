import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib 

prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home'
prefix2 = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/project/pathways/live/Transferability/emily_phd_images/keras-rmac-clustering_output/2018_original/'

# df_2021 = pd.read_csv('chapter4clustering/outputs/df_2021_matched_by_centroid.csv')
# df_2011 = pd.read_csv('chapter4clustering/outputs/df_2011_matched_by_centroid.csv')
df_both_years = pd.read_csv(prefix + '/emily/phd/003_image_matching/keras_rmac-master/census2021outputs/census_2011_and_census_2021_zoom_matched.csv')
df_2021 = df_both_years[df_both_years['year'] == 2021][['Unnamed: 0', 'year', 'p', 'clusters', 'distance']]
df_2021['matched'] = df_2021['clusters']
df_2011 = df_both_years[df_both_years['year'] == 2011][['Unnamed: 0', 'year', 'p', 'clusters', 'distance']]
df_2011['matched'] = df_2011['clusters']
df_2018 = pd.read_csv(prefix2 + '2018_rmac_feature_vector_clusters_20.csv', error_bad_lines=False, engine="python")

def cleanup(x):
    if x in [3,5,17]:
        return np.nan
    elif x in [4,8,9,14]:
        return 'Low-density'
    elif x in [1, 11]:
        return 'Open green space'
    elif x == 0:
        return 'Terraced'
    elif x == 15:
        return 'Commercial'
    elif x in [2,6]:
        return 'Other green'
    elif x in [7,16,18]:
        return np.nan
    elif x == 10:
        return 'High-density'
    elif x == 12:
        return np.nan
    elif x == 13:
        return 'Estates'
    elif x == 19:
        return np.nan
    else:
        return x

# df_2018 = df_2018.loc[:,~df_2018.columns.duplicated()].copy()
df_2021['clusters_2021_cleanup'] = df_2021['matched'].apply(lambda x: cleanup(x))
df_2011['clusters_2011_cleanup'] = df_2011['matched'].apply(lambda x: cleanup(x))

# clean dataframe, keep single img identifier for merge to oa
df_2011 = df_2011[['Unnamed: 0', 'matched','p', 'clusters_2011_cleanup']]
df_2011['idx'] = df_2011['Unnamed: 0'].apply(lambda x : int(x[:-2]))
df_2021 = df_2021[['Unnamed: 0', 'matched','p', 'clusters_2021_cleanup']]
df_2021['idx'] = df_2021['Unnamed: 0'].apply(lambda x : int(x[:-2]))

##################################################################################### CREATE OUTPUT FILE FOR SPATIAL MERGE
df_2011_ = df_2011.dropna()
df_2021_ = df_2021.dropna()
################################################ PERCENTAGES MERGED TO OA
# get merged oa counts
df_2011_oa = pd.read_csv('chapter3data/outputs/2011_panoids_merged_to_oa.csv')
df_2018_oa = pd.read_csv('chapter3data/outputs/2018_panoids_merged_to_oa.csv')
df_2021_oa = pd.read_csv('chapter3data/outputs/2021_panoids_merged_to_oa.csv')

df_2011_ = df_2011_.merge(df_2011_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2011_ = df_2011_.drop_duplicates()
df_2021_ = df_2021_.merge(df_2021_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2021_ = df_2021_.drop_duplicates()
df_2018_ = df_2018.merge(df_2018_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2018_ = df_2018_.drop_duplicates()

# one hot encode classes 2011
df_2011_class = pd.get_dummies(df_2011_[['clusters_2011_cleanup']])
df_2011_class['oa_name'] = df_2011_[['oa_name']]
df_2011_oa = df_2011_class.groupby('oa_name').sum()
#df_2011_oa = df_2011_oa.div(df_2011_oa.sum(axis=1), axis=0)
df_2011_oa = df_2011_oa.div(df_2011_class.groupby('oa_name').count(), axis=0)
df_2011_oa['count'] = df_2011_class.groupby('oa_name').count()['clusters_2011_cleanup_Commercial']

# one hot encode classes 2018
df_2021_class = pd.get_dummies(df_2021_[['clusters_2021_cleanup']])
df_2021_class['oa_name'] = df_2021_[['oa_name']]
df_2021_oa = df_2021_class.groupby('oa_name').sum()
#df_2021_oa = df_2021_oa.div(df_2021_oa.sum(axis=1), axis=0)
df_2021_oa = df_2021_oa.div(df_2021_class.groupby('oa_name').count(), axis=0)
df_2021_oa['count'] = df_2021_class.groupby('oa_name').count()['clusters_2021_cleanup_Commercial']

# one hot encode classes 2018
df_2018_class = pd.get_dummies(df_2018_[['clusters_2018_cleanup']])
df_2018_class['oa_name'] = df_2018_[['oa_name']]
df_2018_oa = df_2018_class.groupby('oa_name').sum()
#df_2021_oa = df_2021_oa.div(df_2021_oa.sum(axis=1), axis=0)
df_2018_oa = df_2018_oa.div(df_2018_class.groupby('oa_name').count(), axis=0)
df_2018_oa['count'] = df_2018_class.groupby('oa_name').count()['clusters_2018_cleanup_Commercial']

# will have to merge to oa's.
df_2011_oa['OA'] = df_2011_oa.index 
df_2021_oa['OA'] = df_2021_oa.index 
df_2018_oa['OA'] = df_2018_oa.index 

df_2011_oa.to_csv('chapter4clustering/outputs/2011_proportions_all_nonna_oa.csv')
df_2021_oa.to_csv('chapter4clustering/outputs/2021_proportions_all_nonna_oa.csv')
df_2018_oa.to_csv('chapter4clustering/outputs/2018_proportions_all_nonna_oa.csv')

merge_all = df_2011_oa.merge(df_2021_oa, on='OA')

################################################ PERCENTAGES
prop = df_2011_['clusters_2011_cleanup'].value_counts().sort_index().index
features = list(prop)
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    merge_all[new_column] = merge_all[t1_column] - merge_all[t0_column]

merge_all.to_csv('chapter4clustering/outputs/cluster_proportions_2011_2021_sql_absolute_OA_keep_interesting_matched_centroid.csv')

########################################################## SPOT CHECKS

################################################ SAVE 'E01004392' (lat,lon)
example_2021 = df_2021_[df_2021_['oa_name'] == 'E01004392'].merge(meta_2021_keep, left_on='Unnamed: 0', right_on='idx_y')
example_2011 = df_2011_[df_2011_['oa_name'] == 'E01004392'].merge(meta_2011_keep, left_on='Unnamed: 0', right_on='idx_y')

example_2011['clusters_2011_cleanup'].value_counts()
example_2021['clusters_2021_cleanup'].value_counts()

example_2021_b = example_2021[example_2021['keep'] == 0]
example_2021_d = example_2021[example_2021['keep'] == 1]
example_2011_b = example_2011[example_2011['keep'] == 0]
example_2011_d = example_2011[example_2011['keep'] == 1] 

example_2021_b.to_csv('chapter4clustering/outputs/spots/point_2_point_2021_walthamstow_b_matched_centroid.csv')
example_2021_d.to_csv('chapter4clustering/outputs/spots/point_2_point_2021_walthamstow_d_matched_centroid.csv')
example_2011_b.to_csv('chapter4clustering/outputs/spots/point_2_point_2011_walthamstow_b_matched_centroid.csv')
example_2011_d.to_csv('chapter4clustering/outputs/spots/point_2_point_2011_walthamstow_d_matched_centroid.csv')

################################################## GET UNIQUE CHANGES
joins_d = example_2021_d[['Unnamed: 0_x', 'matched','p', 'clusters_2021_cleanup']].merge(buffered_d, left_on='Unnamed: 0_x', right_on='idx_2021_d')
joins_d = joins_d.merge(example_2011_d[['Unnamed: 0_x', 'matched','p', 'clusters_2011_cleanup']], left_on='idx_2011_d', right_on='Unnamed: 0_x')
joins_b = example_2021_b[['Unnamed: 0_x', 'matched','p', 'clusters_2021_cleanup']].merge(buffered_b, left_on='Unnamed: 0_x', right_on='idx_2021_b')
joins_b = joins_b.merge(example_2011_b[['Unnamed: 0_x', 'matched','p', 'clusters_2011_cleanup']], left_on='idx_2011_b', right_on='Unnamed: 0_x')

# joins = pd.concat([joins_d[list(joins_d.columns[:-2])],  joins_b[list(joins_b.columns[:-2])]])
joins = pd.concat([joins_b, joins_d])

joins['change'] = joins.apply(lambda x: 1 if x.clusters_2021_cleanup != x.clusters_2011_cleanup else 0, axis=1)
change = joins[joins['change'] == 1]

changes = change[['clusters_2011_cleanup','clusters_2021_cleanup']]
changes['one'] = np.arange(23)
pivot = pd.pivot_table(changes, values='one', index='clusters_2011_cleanup', columns='clusters_2021_cleanup',
               aggfunc='count').reset_index()

pivot['Green space'] = np.zeros(6)
pivot = pivot.fillna(0)
pivot.to_csv('chapter4clustering/outputs/R/walthamstow_heatmap.csv')

import seaborn as sns
plt.clf()
f = plt.figure(figsize=(10, 10))

sns.heatmap(pivot[pivot.columns[1:]], annot=True, cbar=False)
plt.yticks(ticks=np.arange(0.5,6,1),labels=list( pivot[pivot.columns[0]]))
plt.xlabel('2021')
plt.ylabel('2011')
plt.savefig('chapter4clustering/outputs/spots/walthamstow_heatmap_change_matched_centroid.png',bbox_inches="tight", dpi=150)

#################################################################
delta = change[['clusters_2011_cleanup','clusters_2021_cleanup', 'Unnamed: 0_x_y','Unnamed: 0_x_x','p_y', 'p_x' ]]
np.savetxt(r'chapter4clustering/outputs/spots/walthamstow_matched_centroid.txt', delta.values, fmt='%s')
####################################################################
delta_p = joins[['matched_y', 'matched_x', 'p_y', 'p_x', 'change']]
delta_p.boxplot(['p_y','p_x'], by='change')
delta_p['cutoff'] = delta_p.apply(lambda x: 0 if (x.p_y <0.3 or x.p_x < 0.3) else 1, axis =1)
delta_p['both'] = delta_p.apply(lambda x: 1 if (x.change ==1 and x.cutoff == 1) else 0, axis=1)

delta_ = delta_p[delta_p['change'] == 1]
delta_['sum'] = delta_['p_y'] + delta_['p_x']
delta.sortby('sum')

delta_p.to_csv('chapter4clustering/outputs/R/walthamstow_changes_boxplots.csv')