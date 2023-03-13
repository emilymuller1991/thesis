import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib 

prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home/emily/phd/003_image_matching/clustering/output/'
df_both_years = pd.read_csv(prefix + '/emily/phd/003_image_matching/keras_rmac-master/census2021outputs/census_2011_and_census_2021_zoom_matched_p.csv')
df_2021 = df_both_years[df_both_years['year'] == 2021][['Unnamed: 0', 'year', 'p', 'clusters', 'distance']]
df_2021['matched'] = df_2021['clusters']
df_2011 = df_both_years[df_both_years['year'] == 2011][['Unnamed: 0', 'year', 'p', 'clusters', 'distance']]
df_2011['matched'] = df_2011['clusters']

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

df_2021['clusters_2021_cleanup'] = df_2021['matched'].apply(lambda x: cleanup(x))
df_2011['clusters_2011_cleanup'] = df_2011['matched'].apply(lambda x: cleanup(x))

# clean dataframe, keep single img identifier for merge to oa
df_2011 = df_2011[['Unnamed: 0', 'matched','p', 'clusters_2011_cleanup']]
df_2011['idx'] = df_2011['Unnamed: 0'].apply(lambda x : int(x[:-2]))
df_2021 = df_2021[['Unnamed: 0', 'matched','p', 'clusters_2021_cleanup']]
df_2021['idx'] = df_2021['Unnamed: 0'].apply(lambda x : int(x[:-2]))

##################################################################################### CREATE OUTPUT FILE FOR SPATIAL MERGE
### DROP NA images and merge to Image metadata
df_2011 = df_2011.dropna()
df_2021 = df_2021.dropna()
# 2011 - 906,042 # 2021 - 857,162
meta_2011 = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2011/census_2011_image_meta_double.csv')
df_2011_keep = list(df_2011['Unnamed: 0'])
meta_2011_keep = meta_2011[meta_2011['idx_y'].isin(df_2011_keep)]
##################################################### SPLIT ANGLES B & D 
meta_2011_keep['keep'] = meta_2011_keep.apply(lambda x: 1 if x.idx_y[-1] == 'd' else 0, axis=1)
meta_2011_keep_d = meta_2011_keep[meta_2011_keep['keep'] == 1]
# 451,876
meta_2011_keep_b = meta_2011_keep[meta_2011_keep['keep'] == 0]
# 454,165
meta_2021 = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2021/census_2021_image_meta_double.csv')
df_2021_keep = list(df_2021['Unnamed: 0'])
meta_2021_keep = meta_2021[meta_2021['idx_y'].isin(df_2021_keep)]
meta_2021_keep = meta_2021_keep.drop(['idx_y.1'], axis=1)
# #################################################### SPLIT ANGLES B & D 
meta_2021_keep['keep'] = meta_2021_keep.apply(lambda x: 1 if x.idx_y[-1] == 'd' else 0, axis=1)
meta_2021_keep_d = meta_2021_keep[meta_2021_keep['keep'] == 1]
# 425,373
meta_2021_keep_b = meta_2021_keep[meta_2021_keep['keep'] == 0]
# 431,787
################################################ IMAGES MATCHED BY BUFFER
buffered_b = pd.read_csv('chapter3data/outputs/2011_buffered_merge_to_2021_dropna_b_matched_centroid.csv') # 280,961
buffered_d = pd.read_csv('chapter3data/outputs/2011_buffered_merge_to_2021_dropna_d_matched_centroid.csv') # 274,765

buffered_b = buffered_b.drop_duplicates(subset='panoid_2021')
buffered_b = buffered_b.drop_duplicates(subset='panoid_2011') # 232,873
buffered_d = buffered_d.drop_duplicates(subset='panoid_2021')
buffered_d = buffered_d.drop_duplicates(subset='panoid_2011') # 227,517

buffered_b['idx_2011_b'] = buffered_b.apply(lambda x: str(x.idx_2011) + '_b', axis=1)
buffered_b['idx_2021_b'] = buffered_b.apply(lambda x: str(x.idx_2021) + '_b', axis=1)

buffered_d['idx_2011_d'] = buffered_d.apply(lambda x: str(x.idx_2011) + '_d', axis=1)
buffered_d['idx_2021_d'] = buffered_d.apply(lambda x: str(x.idx_2021) + '_d', axis=1) # 592023

# remove everything from df2011 and df2021 that is not in this list
df_2011_keep = list(buffered_d['idx_2011_d']) + list(buffered_b['idx_2011_b'])
df_2011_ = df_2011[df_2011['Unnamed: 0'].isin(df_2011_keep)]
# original shape is 1,471,463 # new shape matched to angle is 502,428 # drop duplicates 460,391

df_2021_keep = list(buffered_d['idx_2021_d']) + list(buffered_b['idx_2021_b'])
df_2021_ = df_2021[df_2021['Unnamed: 0'].isin(df_2021_keep)]
# original shape is 1,522,701 # new shape matched to angle is 495,203 # drop duplicates is 460,392

################################################ PERCENTAGES MERGED TO LSOA
# get merged oa counts
df_2011_oa = pd.read_csv('chapter3data/outputs/2011_panoids_merged_to_oa.csv')
df_2018_oa = pd.read_csv('chapter3data/outputs/2018_panoids_merged_to_oa.csv')
df_2021_oa = pd.read_csv('chapter3data/outputs/2021_panoids_merged_to_oa.csv')

df_2011_ = df_2011_.merge(df_2011_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2011_ = df_2011_.drop_duplicates()
df_2021_ = df_2021_.merge(df_2021_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2021_ = df_2021_.drop_duplicates()

# merge to lsoa
oa = pd.read_csv('/home/emily/phd/002_validation/source/oa/OA_2011_London_gen_MHW_4326_all_fields.csv')
df_2011_ = df_2011_.merge(oa[['OA11CD', 'LSOA11CD']], left_on='oa_name', right_on='OA11CD' )
df_2021_ = df_2021_.merge(oa[['OA11CD', 'LSOA11CD']], left_on='oa_name', right_on='OA11CD' )

df_2011_ = df_2011_.dropna()
df_2021_ = df_2021_.dropna()

df_2021_['keep'] = df_2021_['Unnamed: 0'].apply(lambda x: x.split('_')[-1])
df_2011_['keep'] = df_2011_['Unnamed: 0'].apply(lambda x: x.split('_')[-1])

df_2021_b = df_2021_[df_2021_['keep'] == 'b']
df_2021_d = df_2021_[df_2021_['keep'] == 'd']
df_2011_b = df_2011_[df_2011_['keep'] == 'b']
df_2011_d = df_2011_[df_2011_['keep'] == 'd'] 

################################################## GET UNIQUE CHANGES
joins_d = df_2021_d[['Unnamed: 0', 'matched','p', 'clusters_2021_cleanup', 'LSOA11CD']].merge(buffered_d, left_on='Unnamed: 0', right_on='idx_2021_d')
joins_d = joins_d.merge(df_2011_d[['Unnamed: 0', 'matched','p', 'clusters_2011_cleanup','LSOA11CD']], left_on='idx_2011_d', right_on='Unnamed: 0')
joins_b = df_2021_b[['Unnamed: 0', 'matched','p', 'clusters_2021_cleanup','LSOA11CD']].merge(buffered_b, left_on='Unnamed: 0', right_on='idx_2021_b')
joins_b = joins_b.merge(df_2011_b[['Unnamed: 0', 'matched','p', 'clusters_2011_cleanup','LSOA11CD']], left_on='idx_2011_b', right_on='Unnamed: 0')

joins = pd.concat([joins_b, joins_d])

rel = joins[['matched_y', 'clusters_2011_cleanup', 'p_y','matched_x','clusters_2021_cleanup', 'p_x','LSOA11CD_x']]
total = rel.groupby('LSOA11CD_x').count()['p_x']

thresholds = {       
0:0.239582,
1:0.195365,
2:0.156947,
3:0.256677,
4:0.201400,
5:0.186998,
6:0.176905,
7:0.276535,
8:0.214907,
9:0.213063,
10:0.264902,
11:0.193873,
12:0.192501,
13:0.255420,
14:0.190364,
15:0.384051,
16:0.254950,
17:0.241076,
18:0.190629,
19:0.165401
}

def sig_rule(x, c_x, y, c_y):
    mu_x = thresholds[c_x]
    mu_y = thresholds[c_y]
    if x >= mu_x and y >= mu_y:
        return 1
    else:
        return 0

rel['sig'] = rel.apply(lambda x: sig_rule(x.p_y, x.matched_y, x.p_x, x.matched_x), axis = 1)
rel['change'] = rel.apply(lambda x: 0 if x.clusters_2011_cleanup == x.clusters_2021_cleanup else 1, axis = 1)
# 459,014

rel['both'] = rel.apply(lambda x: 1 if (x.sig == 1 and x.change == 1) else 0, axis = 1)
rel['sig'].sum() # 151,332
rel['change'].sum() # 89,473
rel['both'].sum() # 20,539

prop = {}
features = list(rel['clusters_2011_cleanup'].value_counts().sort_index().index)
changes = rel[rel['both'] == 1]
for lsoa in list(total.index):
    area = changes[changes['LSOA11CD_x'] == lsoa]
    if area.shape[0] != 0:
        pivot = pd.pivot_table(area, values='both', index='clusters_2011_cleanup', columns='clusters_2021_cleanup',
                aggfunc='count').reset_index()
        pivot = pivot.fillna(0)
        prop[lsoa] = []
        for feature in features:
            # get gains
            try:
                gain = pivot[feature].sum()
            except:
                gain = 0
            try:
                loss = pivot[pivot['clusters_2011_cleanup'] == feature][pivot.columns[1:]].values.sum()
            except:
                loss = 0
            prop[lsoa].append(gain-loss)

prop_df = pd.DataFrame(prop).T
prop_df.columns = features
prop_df['lsoa'] = prop_df.index
prop_df.to_csv('chapter4clustering/outputs/absolute_counts_of_net_changes_in_cluster_lsoa.csv')

joins['both'] = rel['both']
joins_b['both'] = rel['sig'][:232196]
joins_d['both'] = rel['sig'][232196:]
joins_b.to_csv('chapter4clustering/outputs/image_pairs_significant_angle_b.csv')
joins_b.to_csv('chapter4clustering/outputs/image_pairs_significant_angle_d.csv')

# changes_ = pd.pivot_table(changes, values='both', index='clusters_2011_cleanup', columns='clusters_2021_cleanup',
#                    aggfunc='count').reset_index()
# sig = pd.pivot_table(rel[rel['sig'] ==1], values='both', index='clusters_2011_cleanup', columns='clusters_2021_cleanup',
#                    aggfunc='count').reset_index()
# all = pd.pivot_table(rel, values='both', index='clusters_2011_cleanup', columns='clusters_2021_cleanup',
#                    aggfunc='count').reset_index()

# all = all.fillna(0)
# sig = sig.fillna(0)
# changes_ = changes_.fillna(0)

# all_ = all[all.columns[1:]].div(all[all.columns[1:]].sum(axis=1), axis=0)
# all_['clusters_2011_cleanup'] = all['clusters_2011_cleanup']
# sig_ = sig[sig.columns[1:]].div(sig[sig.columns[1:]].sum(axis=1), axis=0)
# sig_['clusters_2011_cleanup'] = sig['clusters_2011_cleanup']
# all_ = all_[all.columns]
# sig_ = sig_[sig.columns]

# all_.to_csv('chapter4clustering/outputs/R/_all_london_changes_heatmap.csv')
# sig_.to_csv('chapter4clustering/outputs/R/_sig_london_changes_heatmap.csv')
# changes_.to_csv('chapter4clustering/outputs/R/_changes_london_changes_heatmap.csv')

import seaborn as sns
plt.clf()
f = plt.figure(figsize=(10, 10))
sns.heatmap(q[q.columns[1:]], annot=True)
plt.yticks(ticks=np.arange(0.5,7,1),labels=list( q[q.columns[0]]))
plt.xlabel('2021')
plt.ylabel('2011')

plt.clf()
f = plt.figure(figsize=(10, 10))
sns.heatmap(r[r.columns[1:]], annot=True)
plt.yticks(ticks=np.arange(0.5,7,1),labels=list(r[r.columns[0]]))
plt.xlabel('2021')
plt.ylabel('2011')
#### 2011 no change
# all_2011 = rel[['clusters_2011_cleanup', 'LSOA11CD_x', 'change']]
# no_change_2011 = all_2011[all_2011['change'] == 0]
# # one hot encode classes 2011
# df_2011_class = pd.get_dummies(no_change_2011[['clusters_2011_cleanup']])
# df_2011_class['LSOA11CD'] = no_change_2011[['LSOA11CD_x']]
# df_2011_oa = df_2011_class.groupby('LSOA11CD').sum()
# #### 2011 change
# all_2011 = rel[['clusters_2011_cleanup', 'LSOA11CD_x', 'change']]
# no_change_2011 = all_2011[all_2011['change'] == 0]
# # one hot encode classes 2011
# df_2011_class = pd.get_dummies(no_change_2011[['clusters_2011_cleanup']])
# df_2011_class['LSOA11CD'] = no_change_2011[['LSOA11CD_x']]
# df_2011_oa = df_2011_class.groupby('LSOA11CD').sum()