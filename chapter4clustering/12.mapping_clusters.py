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
    elif x in [8,9,14]:
        return 'Low-density'
    elif x == 4:
        return 'Leafy green residential'
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

# get merged oa counts
df_2011_oa = pd.read_csv('outputs/2011_panoids_merged_to_oa.csv')
df_2018_oa = pd.read_csv('outputs/2018_panoids_merged_to_oa.csv')
df_2021_oa = pd.read_csv('outputs/2021_panoids_merged_to_oa.csv')

df_2011_ = df_2011.merge(df_2011_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2011_ = df_2011_.drop_duplicates()
df_2018_ = df_2018.merge(df_2018_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2018_ = df_2018_.drop_duplicates()
df_2021_ = df_2021.merge(df_2021_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2021_ = df_2021_.drop_duplicates()

df_2011_ = df_2011_.dropna()
df_2018_ = df_2018_.dropna()
df_2021_ = df_2021_.dropna()

# one hot encode classes 2011
df_2011_class = pd.get_dummies(df_2011_[['clusters_2011_cleanup']])
df_2011_class['oa_name'] = df_2011_[['oa_name']]
df_2011_oa = df_2011_class.groupby('oa_name').sum()
#df_2011_oa = df_2011_oa.div(df_2011_oa.sum(axis=1), axis=0)
df_2011_oa = df_2011_oa.div(df_2011_class.groupby('oa_name').count(), axis=0)

# one hot encode classes 2018
df_2018_class = pd.get_dummies(df_2018_[['clusters_2018_cleanup']])
df_2018_class['oa_name'] = df_2018_[['oa_name']]
df_2018_oa = df_2018_class.groupby('oa_name').sum()
#df_2018_oa = df_2018_oa.div(df_2018_oa.sum(axis=1), axis=0)
df_2018_oa = df_2018_oa.div(df_2018_class.groupby('oa_name').count(), axis=0)

# one hot encode classes 2018
df_2021_class = pd.get_dummies(df_2021_[['clusters_2021_cleanup']])
df_2021_class['oa_name'] = df_2021_[['oa_name']]
df_2021_oa = df_2021_class.groupby('oa_name').sum()
#df_2021_oa = df_2021_oa.div(df_2021_oa.sum(axis=1), axis=0)
df_2021_oa = df_2021_oa.div(df_2021_class.groupby('oa_name').count(), axis=0)

# will have to merge to oa's.
df_2011_oa['oa_name_'] = df_2011_oa.index 
df_2018_oa['oa_name_'] = df_2018_oa.index 
df_2021_oa['oa_name_'] = df_2021_oa.index 

merge_all = df_2011_oa.merge(df_2021_oa, on='oa_name_')

################################################ PERCENTAGES
prop = df_2011['clusters_2011_cleanup'].value_counts().sort_index().index
features = list(prop)
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    merge_all[new_column] = merge_all[t1_column] - merge_all[t0_column]

merge_all.to_csv('outputs/cluster_proportions_2011_2021_sql_absolute_keep_interesting.csv')

################################################ PERCENTAGES MERGED TO LSOA
# get merged oa counts
df_2011_oa = pd.read_csv('outputs/2011_panoids_merged_to_oa.csv')
df_2018_oa = pd.read_csv('outputs/2018_panoids_merged_to_oa.csv')
df_2021_oa = pd.read_csv('outputs/2021_panoids_merged_to_oa.csv')

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

# df_2011_ = df_2011_.dropna()
# df_2018_ = df_2018_.dropna()
# df_2021_ = df_2021_.dropna()

# one hot encode classes 2011
df_2011_class = pd.get_dummies(df_2011_[['clusters_2011_cleanup']])
df_2011_class['LSOA11CD'] = df_2011_[['LSOA11CD']]
df_2011_oa = df_2011_class.groupby('LSOA11CD').sum()
#df_2011_oa = df_2011_oa.div(df_2011_oa.sum(axis=1), axis=0)
df_2011_oa = df_2011_oa.div(df_2011_class.groupby('LSOA11CD').count(), axis=0)

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

# will have to merge to oa's.
df_2011_oa['lsoa'] = df_2011_oa.index 
df_2018_oa['lsoa'] = df_2018_oa.index 
df_2021_oa['lsoa'] = df_2021_oa.index 

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

################################################# QUARTILES
df_2021_oa_quart = df_2021_oa.copy()
for column in list(df_2021_oa.columns[:-1]):
    df_2021_oa_quart[column] = pd.cut(df_2021_oa[column], [0, 0.01,.25, .5, .75, 1.], labels=[0,1,2,3,4])

df_2011_oa_quart = df_2011_oa.copy()
for column in list(df_2011_oa.columns[:-1]):
    df_2011_oa_quart[column] = pd.cut(df_2011_oa[column], [0, 0.01,.25, .5, .75, 1.], labels=[0,1,2,3,4])

merge_all = df_2011_oa_quart.merge(df_2021_oa_quart, on='oa_name_')

features = list(prop.index)
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    merge_all[new_column] = merge_all[t1_column].astype(float) - merge_all[t0_column].astype(float)

merge_all.to_csv('outputs/cluster_quartiles_2011_2021_sql.csv')

################################################# DECILES
df_2021_oa_dec = df_2021_oa.copy()
for column in list(df_2021_oa.columns[:-1]):
    df_2021_oa_dec[column] = pd.cut(df_2021_oa[column], [-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=[0,1,2,3,4,5,6,7,8,9,10])

df_2011_oa_dec = df_2011_oa.copy()
for column in list(df_2011_oa.columns[:-1]):
    df_2011_oa_dec[column] = pd.cut(df_2011_oa[column], [-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=[0,1,2,3,4,5,6,7,8,9,10])

merge_all = df_2011_oa_dec.merge(df_2021_oa_dec, on='oa_name_')

features = list(prop.index)
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    merge_all[new_column] = merge_all[t1_column].astype(float) - merge_all[t0_column].astype(float)

merge_all.to_csv('outputs/cluster_deciles_2011_2021.csv')

################################################ ADD ERROR TO MEASUREMENT
# proportions dataframe
df_2021_oa 
df_2011_oa 

# get associated errors  - there are no output areas with zero counts!
df_2021_oa_counts = df_2021_class.groupby('oa_name').count()
df_2021_err = {}
for column in list(df_2021_oa.columns[:-1]):
    eps = 0.01
    p = df_2021_oa[column] + eps
    n = df_2021_oa_counts[column]
    perr = np.sqrt(np.abs(p*(1-p))/n)
    df_2021_err[column] = perr
df_2021_err = pd.DataFrame(df_2021_err)

# get associated errors  - there are no output areas with zero counts!
df_2011_oa_counts = df_2011_class.groupby('oa_name').count()
df_2011_err = {}
for column in list(df_2011_oa.columns[:-1]):
    eps = 0.01
    p = df_2011_oa[column] + eps
    n = df_2011_oa_counts[column]
    perr = np.sqrt(np.abs(p*(1-p))/n)
    df_2011_err[column] = perr
df_2011_err = pd.DataFrame(df_2011_err)

# save mask dataframe for certainty estimates in 2011 + 202
for mu in [0.5,0.75,0.9]:
    mask = {}
    for column in list(df_2021_oa.columns[:-1]):
        f = column.split('_')[-1]
        f_2011 = 'clusters_2011_cleanup_%s' %f
        f_2021 = 'clusters_2021_cleanup_%s' %f
        threshold_11 = pd.qcut(df_2011_err[f_2011],[mu], retbins=True)[1][0]
        threshold_21 = pd.qcut(df_2021_err[f_2021],[mu], retbins=True)[1][0]
        mask[f] = df_2011_err[f_2011].apply(lambda x: 0 if x <= threshold_11 else 1) + df_2021_err[f_2021].apply(lambda x: 0 if x <= threshold_21 else 1)
    df_mask = pd.DataFrame(mask) 
    df_mask.to_csv('outputs/uncertainty_mask_%s.csv' % str(mu) )    

for mu in [0.5,0.75,0.9]:
    mask = {}
    for column in list(df_2021_oa.columns[:-1]):
        threshold = pd.qcut(df_2021_err[column],[mu], retbins=True)[1][0]
        mask[column] = df_2021_err[column].apply(lambda x: 0 if x <= threshold else 1)
    df_mask = pd.DataFrame(mask) 
    df_mask.to_csv('outputs/2021_uncertainty_mask_%s.csv' % str(mu) )   

############################################################################################### LSOA
df_2011_oa = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2011/census_2011_image_meta_single_merged_oa.csv')
df_2018_oa = pd.read_csv('/home/emily/phd/2_interpretability/2018_all_perception_scores_merged_oa.csv')
df_2021_oa = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2021/census_2021_image_meta_single_merged_oa.csv')

df_2011_ = df_2011.merge(df_2011_oa[['LSoa_name', 'idx']], left_on='idx', right_on='idx' )
df_2011_ = df_2011_.drop_duplicates()
df_2018_ = df_2018.merge(df_2018_oa[['LSoa_name', 'idx']], left_on='idx', right_on='idx' )
df_2018_ = df_2018_.drop_duplicates()
df_2021_ = df_2021.merge(df_2021_oa[['LSoa_name', 'idx']], left_on='idx', right_on='idx' )
df_2021_ = df_2021_.drop_duplicates()

# one hot encode classes 2011
df_2011_class = pd.get_dummies(df_2011_[['clusters_2011_cleanup']])
df_2011_class['LSoa_name'] = df_2011_[['LSoa_name']]
df_2011_oa = df_2011_class.groupby('LSoa_name').sum()
#df_2011_oa = df_2011_oa.div(df_2011_oa.sum(axis=1), axis=0)
df_2011_oa = df_2011_oa.div(df_2011_class.groupby('LSoa_name').count(), axis=0)

# one hot encode classes 2018
df_2018_class = pd.get_dummies(df_2018_[['clusters_2018_cleanup']])
df_2018_class['LSoa_name'] = df_2018_[['LSoa_name']]
df_2018_oa = df_2018_class.groupby('LSoa_name').sum()
#df_2018_oa = df_2018_oa.div(df_2018_oa.sum(axis=1), axis=0)
df_2018_oa = df_2018_oa.div(df_2018_class.groupby('LSoa_name').count(), axis=0)

# one hot encode classes 2018
df_2021_class = pd.get_dummies(df_2021_[['clusters_2021_cleanup']])
df_2021_class['LSoa_name'] = df_2021_[['LSoa_name']]
df_2021_oa = df_2021_class.groupby('LSoa_name').sum()
#df_2021_oa = df_2021_oa.div(df_2021_oa.sum(axis=1), axis=0)
df_2021_oa = df_2021_oa.div(df_2021_class.groupby('LSoa_name').count(), axis=0)

# will have to merge to oa's.
df_2011_oa['LSoa_name_'] = df_2011_oa.index 
df_2018_oa['LSoa_name_'] = df_2018_oa.index 
df_2021_oa['LSoa_name_'] = df_2021_oa.index 

merge_all = df_2011_oa.merge(df_2021_oa, on='LSoa_name_')

################################################ PERCENTAGES
features = list(prop.index)
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    merge_all[new_column] = (merge_all[t1_column] - merge_all[t0_column]) / merge_all[t0_column]

merge_all.to_csv('outputs/cluster_proportions_2011_2021_lsoa.csv')

################################################# QUARTILES
df_2021_oa_quart = df_2021_oa.copy()
for column in list(df_2021_oa.columns[:-1]):
    df_2021_oa_quart[column] = pd.cut(df_2021_oa[column], [0, 0.01,.25, .5, .75, 1.], labels=[0,1,2,3,4])

df_2011_oa_quart = df_2011_oa.copy()
for column in list(df_2011_oa.columns[:-1]):
    df_2011_oa_quart[column] = pd.cut(df_2011_oa[column], [0, 0.01,.25, .5, .75, 1.], labels=[0,1,2,3,4])

merge_all = df_2011_oa_quart.merge(df_2021_oa_quart, on='LSoa_name_')

features = list(prop.index)
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    merge_all[new_column] = merge_all[t1_column].astype(float) - merge_all[t0_column].astype(float)

merge_all.to_csv('outputs/cluster_quartiles_2011_2021_lsoa.csv')

################################################# DECILES
df_2021_oa_dec = df_2021_oa.copy()
for column in list(df_2021_oa.columns[:-1]):
    df_2021_oa_dec[column] = pd.cut(df_2021_oa[column], [-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=[0,1,2,3,4,5,6,7,8,9,10])

df_2011_oa_dec = df_2011_oa.copy()
for column in list(df_2011_oa.columns[:-1]):
    df_2011_oa_dec[column] = pd.cut(df_2011_oa[column], [-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=[0,1,2,3,4,5,6,7,8,9,10])

merge_all = df_2011_oa_dec.merge(df_2021_oa_dec, on='LSoa_name_')

features = list(prop.index)
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    merge_all[new_column] = merge_all[t1_column].astype(float) - merge_all[t0_column].astype(float)

merge_all.to_csv('outputs/cluster_deciles_2011_2021_lsoa.csv')


################################################ SPOT CHECKS
df_2011_[df_2011_['LSOA11CD'] == 'E01002898'][df_2011_['clusters_2011_cleanup'] == 'Terraced']
df_2011_[df_2011_['LSOA11CD'] == 'E01002898'][df_2011_['clusters_2011_cleanup'] == 'High-density']

df_2021_[df_2021_['LSOA11CD'] == 'E01002898'][df_2021_['clusters_2021_cleanup'] == 'Terraced']
df_2021_[df_2021_['LSOA11CD'] == 'E01002898'][df_2021_['clusters_2021_cleanup'] == 'High-density']

################################################ SCATTER MATRICES
prop = merge_all[['proportion_Commercial', 'proportion_Estates',
       'proportion_High-density', 'proportion_Leafy green residential',
       'proportion_Low-density', 'proportion_Open green space',
       'proportion_Other green', 'proportion_Terraced', 'lsoa']]

certainty = pd.read_csv('/media/emily/south/chapter3data/outputs/sampling_rate_lsoa_all_years_sql.csv')

mask = prop.merge(certainty, left_on='lsoa', right_on='lsoa11cd')
masked = mask[(mask['2021'] > 0.5) & (mask['2011'] > 0.5)]
df = masked.drop(['lsoa11cd', '2011', '2018', '2021', 'roads', 'lsoa11cd.1', 'lsoa'], axis=1)
df.columns = features

from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
plt.yticks(rotation=-45)
plt.xticks(rotation=-45)

################################################ ADD ERROR TO MEASUREMENT
# proportions dataframe
# df_2021_oa 
# df_2011_oa 

# # get associated errors  - there are no output areas with zero counts!
# df_2021_oa_counts = df_2021_class.groupby('LSoa_name').count()
# df_2021_err = {}
# for column in list(df_2021_oa.columns[:-1]):
#     eps = 0.01
#     p = df_2021_oa[column] + eps
#     n = df_2021_oa_counts[column]
#     perr = np.sqrt(np.abs(p*(1-p))/n)
#     df_2021_err[column] = perr
# df_2021_err = pd.DataFrame(df_2021_err)

# # get associated errors  - there are no output areas with zero counts!
# df_2011_oa_counts = df_2011_class.groupby('LSoa_name').count()
# df_2011_err = {}
# for column in list(df_2011_oa.columns[:-1]):
#     eps = 0.01
#     p = df_2011_oa[column] + eps
#     n = df_2011_oa_counts[column]
#     perr = np.sqrt(np.abs(p*(1-p))/n)
#     df_2011_err[column] = perr
# df_2011_err = pd.DataFrame(df_2011_err)

# # save mask dataframe for certainty estimates in 2011 + 202
# for mu in [0.5,0.75,0.9]:
#     mask = {}
#     for column in list(df_2021_oa.columns[:-1]):
#         f = column.split('_')[-1]
#         f_2011 = 'clusters_2011_cleanup_%s' %f
#         f_2021 = 'clusters_2021_cleanup_%s' %f
#         threshold_11 = pd.qcut(df_2011_err[f_2011],[mu], retbins=True)[1][0]
#         threshold_21 = pd.qcut(df_2021_err[f_2021],[mu], retbins=True)[1][0]
#         mask[f] = df_2011_err[f_2011].apply(lambda x: 0 if x <= threshold_11 else 1) + df_2021_err[f_2021].apply(lambda x: 0 if x <= threshold_21 else 1)
#     df_mask = pd.DataFrame(mask) 
#     df_mask.to_csv('outputs/uncertainty_mask_%s_lsoa.csv' % str(mu) )    

# for mu in [0.5,0.75,0.9]:
#     mask = {}
#     for column in list(df_2021_oa.columns[:-1]):
#         threshold = pd.qcut(df_2021_err[column],[mu], retbins=True)[1][0]
#         mask[column] = df_2021_err[column].apply(lambda x: 0 if x <= threshold else 1)
#     df_mask = pd.DataFrame(mask) 
#     df_mask.to_csv('outputs/2021_uncertainty_mask_%s_lsoa.csv' % str(mu) )   


# ################################################### PLOTTING
# import matplotlib as mpl

# mpl.rcParams['font.family'] = 'Avenir'
# plt.rcParams['font.size'] = 18
# plt.rcParams['axes.linewidth'] = 2

# mpl.rcParams['axes.spines.left'] = True
# mpl.rcParams['axes.spines.right'] = False
# mpl.rcParams['axes.spines.top'] = True
# mpl.rcParams['axes.spines.bottom'] = False

# # plot ordered proportions per feature with err 
# for column in list(df_2021_oa.columns[:-1]):
#     p = df_2021_oa[column]
#     perr = df_2021_err[column]
#     concat = pd.concat([p,perr], axis=1)
#     concat.columns = [0,1]
#     concat = concat.sort_values(0).astype(float).dropna()
#     plot_errors(concat, column + 'lsoa')
#     plot_scatter(concat, column + 'lsoa')
#     plot_bars(concat, column + 'lsoa', [0.5,0.75,0.9])

# def plot_errors(concat, col):
#     mpl.rcParams['axes.spines.left'] = True
#     mpl.rcParams['axes.spines.right'] = False
#     mpl.rcParams['axes.spines.top'] = True
#     mpl.rcParams['axes.spines.bottom'] = False
#     plt.clf()
#     fig, ax = plt.subplots()
#     ax.margins(0.05) # Optional, just adds 5% padding to the autscaling
#     # plot means and std stacked in order
#     concat = concat.sort_values(0).astype(float).dropna()
#     concat.plot(y=0, yerr=1, marker='.', color='red', alpha = 0.7, ax=ax, legend=False, markerfacecolor='red', markeredgewidth=0, linewidth=0.01)
#     plt.yticks(rotation=-90)
#     plt.xticks([])
#     plt.xlabel('')
#     plt.ylim(-0.15,1.15)
#     plt.gca().invert_yaxis()
#     plt.setp(ax.spines.values(), linewidth=2)
#     plt.savefig('outputs/%s_prop_errors.png' % col)

# def plot_scatter(concat, col):
#     mpl.rcParams['axes.spines.left'] = True
#     mpl.rcParams['axes.spines.right'] = False
#     mpl.rcParams['axes.spines.top'] = False
#     mpl.rcParams['axes.spines.bottom'] = True
#     plt.clf()
#     fig, ax = plt.subplots()
#     ax.margins(0.05) # Optional, just adds 5% padding to the autscaling
#     # plot means and std stacked in order
#     ax.scatter(concat[0], concat[1])  
#     ax.set_xlabel('Proportion')
#     ax.set_ylabel('SE')
#     plt.savefig('outputs/%s_scatter_prop_errors.png' % col)

# def plot_bars(concat, col, mu):
#     mpl.rcParams['axes.spines.left'] = True
#     mpl.rcParams['axes.spines.right'] = False
#     mpl.rcParams['axes.spines.top'] = False
#     mpl.rcParams['axes.spines.bottom'] = True
#     mu = [0.5, 0.75, 0.9]
#     plt.clf()
#     fig, ax = plt.subplots()
#     ax.margins(0.05) # Optional, just adds 5% padding to the autscaling
#     # plot means and std stacked in order
#     concat = concat.sort_values(1).astype(float).dropna()
#     threshold = pd.qcut(concat[1],mu, retbins=True)[1]
#     # create colour map
#     cmap = matplotlib.cm.get_cmap('tab20c')
#     concat['keep'] = concat[1].apply(lambda x: cmap(0) if x <= threshold[0] else (cmap(1) if x <= threshold[1] else (cmap(2) if x<= threshold[2] else cmap(3))) )
#     plt.bar(x=np.arange(0,concat.shape[0]), height=concat[1], color=concat['keep'])
#     # horizontal line indicating the threshold
#     ax.plot([0., int(0.5*concat.shape[0])], [threshold[0], threshold[0]], "k--")
#     ax.plot([0., int(0.75*concat.shape[0])], [threshold[1], threshold[1]], "k--")
#     ax.plot([0., int(0.9*concat.shape[0])], [threshold[2], threshold[2]], "k--")
#     ax.set_ylabel('SE')
#     plt.savefig('outputs/%s_bar_errors.png' % col)

