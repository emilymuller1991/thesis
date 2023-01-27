import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib 

prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home/emily/phd/003_image_matching/clustering/output/'
prefix2 = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/project/pathways/live/Transferability/emily_phd_images/keras-rmac-clustering_output/2018_original/'

df_2021 = pd.read_csv(prefix + 'census2021_zoom_clusters_20.csv', error_bad_lines=False, engine="python")
df_2011 = pd.read_csv(prefix + 'census2011_zoom_clusters_20.csv', error_bad_lines=False, engine="python")
df_2018 = pd.read_csv(prefix2 + '2018_rmac_feature_vector_clusters_20.csv', error_bad_lines=False, engine="python")

relabel_2011 = {
18:0,
10:1,
14:1,
8:2,
7:3,
4:4,
16:5,
13:6,
12:7,
9:8,
5:8,
17:9,
6:10,
3:11,
0:12,
11:13,
1:15,
15:16,
2:17,
12:18,
19:19
}

relabel_2021 = {
18:0,
9:1,
3:2,
13:3,
11:4,
5:5,
14:6,
2:7,
16:8,
17:9,
7:10,
15:11,
6:11,
8:12,
19:13,
1:18,
10:15,
0:16,
12:17,
4:19
}

def relabel_2021_(x):
    return relabel_2021[x]

def relabel_2011_(x):
    return relabel_2011[x]

df_2021['clusters_2021_edited'] = df_2021['clusters'].apply(lambda x: relabel_2021_(x))

df_2011['clusters_2011_edited'] = df_2011['clusters'].apply(lambda x: relabel_2011_(x))

# def cleanup(x):
#     if x in [3,5,17]:
#         return np.nan
#     elif x in [8,9]:
#         return 'Low-density'
#     elif x == 4:
#         return 'Leafy green residential'
#     elif x in [1, 11]:
#         return 'Open green space'
#     elif x == 0:
#         return 'Terraced'
#     elif x == 15:
#         return 'Commercial'
#     elif x in [2,6]:
#         return 'Other green'
#     elif x in [7,14,16,18]:
#         return 'Vehicles'
#     elif x == 10:
#         return 'High-density'
#     elif x == 12:
#         return 'Sheds'
#     elif x == 13:
#         return 'Estates'
#     elif x == 19:
#         return 'Fences'
#     else:
#         return x

def cleanup(x):
    if x in [3,5,17]:
        return np.nan
    elif x in [4,8,9,14]:
        return 'Low-density'
    elif x in [1,2,6, 11]:
        return 'Green space'
    elif x in [0,10]:
        return 'High-density'
    elif x == 15:
        return 'Commercial'
    elif x in [7,16,18]:
        return np.nan
    elif x == 13:
        return 'Estates'
    elif x == 12:
        return np.nan
    elif x == 19:
        return np.nan

# df_2018 = df_2018.loc[:,~df_2018.columns.duplicated()].copy()
df_2021['clusters_2021_cleanup'] = df_2021['clusters_2021_edited'].apply(lambda x: cleanup(x))
df_2011['clusters_2011_cleanup'] = df_2011['clusters_2011_edited'].apply(lambda x: cleanup(x))
df_2018['clusters_2018_cleanup'] = df_2018['clusters'].apply(lambda x: cleanup(x))

# clean dataframe, keep single img identifier for merge to oa
df_2011 = df_2011[['Unnamed: 0.1', 'clusters', 'clusters_2011_edited', 'clusters_2011_cleanup']]
df_2011['idx'] = df_2011['Unnamed: 0.1'].apply(lambda x : int(x[:-2]))
df_2018 = df_2018[['Unnamed: 0', 'clusters', 'clusters_2018_cleanup']]
df_2018['idx'] = df_2018['Unnamed: 0'].apply(lambda x : int(x[:-2]))
df_2021 = df_2021[['Unnamed: 0.1', 'clusters', 'clusters_2021_edited', 'clusters_2021_cleanup']]
df_2021['idx'] = df_2021['Unnamed: 0.1'].apply(lambda x : int(x[:-2]))


################################################ PERCENTAGES MERGED TO LSOA
# get merged oa counts
df_2011_oa = pd.read_csv('chapter3data/outputs/2011_panoids_merged_to_oa.csv')
df_2018_oa = pd.read_csv('chapter3data/outputs/2018_panoids_merged_to_oa.csv')
df_2021_oa = pd.read_csv('chapter3data/outputs/2021_panoids_merged_to_oa.csv')

df_2011_ = df_2011.merge(df_2011_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2011_ = df_2011_.drop_duplicates()
df_2018_ = df_2018.merge(df_2018_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2018_ = df_2018_.drop_duplicates()
df_2021_ = df_2021.merge(df_2021_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2021_ = df_2021_.drop_duplicates()

# merge to lsoa
oa = pd.read_csv('/home/emily/phd/002_validation/source/oa/OA_2011_London_gen_MHW_4326_all_fields.csv')
df_2011_ = df_2011_.merge(oa[['OA11CD', 'LSOA11CD']], left_on='oa_name', right_on='OA11CD' )
df_2018_ = df_2018_.merge(oa[['OA11CD', 'LSOA11CD']], left_on='oa_name', right_on='OA11CD' )
df_2021_ = df_2021_.merge(oa[['OA11CD', 'LSOA11CD']], left_on='oa_name', right_on='OA11CD' )

df_2011_ = df_2011_.dropna()
df_2018_ = df_2018_.dropna()
df_2021_ = df_2021_.dropna()

# one hot encode classes 2011
df_2011_class = pd.get_dummies(df_2011_[['clusters_2011_cleanup']])
df_2011_class['LSOA11CD'] = df_2011_[['LSOA11CD']]
df_2011_oa = df_2011_class.groupby('LSOA11CD').sum()
#df_2011_oa = df_2011_oa.div(df_2011_oa.sum(axis=1), axis=0)
df_2011_oa = df_2011_oa.div(df_2011_class.groupby('LSOA11CD').count(), axis=0)
df_2011_oa['count'] = df_2011_class.groupby('LSOA11CD').count()['clusters_2011_cleanup_Commercial']

# one hot encode classes 2018
df_2018_class = pd.get_dummies(df_2018_[['clusters_2018_cleanup']])
df_2018_class['LSOA11CD'] = df_2018_[['LSOA11CD']]
df_2018_oa = df_2018_class.groupby('LSOA11CD').sum()
#df_2018_oa = df_2018_oa.div(df_2018_oa.sum(axis=1), axis=0)
df_2018_oa = df_2018_oa.div(df_2018_class.groupby('LSOA11CD').count(), axis=0)

# one hot encode classes 2018
df_2021_class = pd.get_dummies(df_2021_[['clusters_2021_cleanup']])
df_2021_class['LSOA11CD'] = df_2021_[['LSOA11CD']]
df_2021_oa = df_2021_class.groupby('LSOA11CD').sum()
#df_2021_oa = df_2021_oa.div(df_2021_oa.sum(axis=1), axis=0)
df_2021_oa = df_2021_oa.div(df_2021_class.groupby('LSOA11CD').count(), axis=0)
df_2021_oa['count'] = df_2021_class.groupby('LSOA11CD').count()['clusters_2021_cleanup_Commercial']

# will have to merge to oa's.
df_2011_oa['lsoa'] = df_2011_oa.index 
df_2018_oa['lsoa'] = df_2018_oa.index 
df_2021_oa['lsoa'] = df_2021_oa.index 

df_2011_oa.to_csv('chapter4clustering/outputs/2011_reduced_set_proportions.csv')
df_2021_oa.to_csv('chapter4clustering/outputs/2021_reduced_set_proportions.csv')

df_2011_oa.to_csv('chapter4clustering/outputs/2011_reduced_set_proportions_dropna.csv')
df_2021_oa.to_csv('chapter4clustering/outputs/2021_reduced_set_proportions_dropna.csv')

merge_all = df_2011_oa.merge(df_2021_oa, on='lsoa')

################################################ PERCENTAGES
prop = df_2011['clusters_2011_cleanup'].value_counts().sort_index().index
features = list(prop)
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    merge_all[new_column] = merge_all[t1_column] - merge_all[t0_column]

merge_all.to_csv('outputs/cluster_proportions_2011_2021_sql_absolute_lsoa_keep_interesting.csv')


################################################ SCATTER MATRICES
prop = merge_all[['proportion_%s' % i for i in features] + ['lsoa']].dropna(axis=1)

certainty = pd.read_csv('chapter3data/outputs/sampling_rate_lsoa_all_years_sql.csv')

mask = prop.merge(certainty, left_on='lsoa', right_on='lsoa11cd')
masked = mask[(mask['2021'] > 0.5) & (mask['2011'] > 0.5)]
df = masked.drop(['lsoa11cd', '2011', '2018', '2021', 'roads', 'lsoa11cd.1', 'lsoa'], axis=1)
df.columns = features

from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
plt.yticks(rotation=-45)
plt.xticks(rotation=-45)

plt.clf()
f = plt.figure(figsize=(10, 10))
ax = plt.gca()
im = ax.matshow(df.corr(),cmap='seismic', vmin=-1, vmax=1)
plt.colorbar(im,fraction=0.046, pad=0.04)
ax.set_xticks(np.arange(5))
ax.set_xticklabels(features)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(features)
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=90,
          ha="right", va="center", rotation_mode="anchor")
f.tight_layout()
plt.savefig('chapter4clustering/outputs/corr_matrix_lsoa_reduced_group.png',bbox_inches="tight", dpi=150)

df.to_csv
################################################ SPOT CHECKS