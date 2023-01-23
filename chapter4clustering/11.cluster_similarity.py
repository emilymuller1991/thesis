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

def cleanup(x):
    if x in [3,5,17]:
        return np.nan
    elif x in [8,9]:
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
    elif x in [7,14,16,18]:
        return 'Vehicles'
    elif x == 10:
        return 'High-density'
    elif x == 12:
        return 'Sheds'
    elif x == 13:
        return 'Estates'
    elif x == 19:
        return 'Fences'
    else:
        return x

# def cleanup(x):
#     if x in [3,5,17]:
#         return np.nan
#     elif x in [8,9,14]:
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
#     elif x in [7,16,18]:
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

df_2021['clusters_2021_cleanup'] = df_2021['clusters_2021_edited'].apply(lambda x: cleanup(x))
df_2011['clusters_2011_cleanup'] = df_2011['clusters_2011_edited'].apply(lambda x: cleanup(x))
df_2018['clusters_2018_cleanup'] = df_2018['clusters'].apply(lambda x: cleanup(x))

# color_dict = {0:'green',
#                 1:'green',
#                 2:'black',
#                 3:'black',
#                 4:'green',
#                 5:'black',
#                 6:'black',
#                 7:'black',
#                 8:'green',
#                 9:'green',
#                 10:'black',
#                 11:'green',
#                 12:'black',
#                 13:'black',
#                 14:'green',
#                 15:'green',
#                 16:'black',
#                 17:'black',
#                 18:'black',
#                 19:'black'
# }

# df_2021['clusters_2021_edited'].value_counts().plot(kind='bar') #, color=[color_dict[i] for i in df_2021['clusters_2021_edited'].value_counts().index])
# plt.xlabel('Clusters')
# plt.ylabel('Frequency')
# plt.title('2021 Cluster Frequencies')
# plt.ylim(0,180000)
# plt.savefig('outputs/2021_cluster_frequencies.png',bbox_inches="tight", dpi=150)
# plt.clf()

# df_2011['clusters_2011_edited'].value_counts().plot(kind='bar') #,  color=[color_dict[i] for i in df_2011['clusters_2011_edited'].value_counts().index])
# plt.xlabel('Clusters')
# plt.ylabel('Frequency')
# plt.title('2011 Cluster Frequencies')
# plt.ylim(0,180000)
# plt.savefig('outputs/2011_cluster_frequencies.png',bbox_inches="tight", dpi=150)
# plt.clf()

# df_2018['clusters'].value_counts().plot(kind='bar') #,  color=[color_dict[i] for i in df_2018['clusters'].value_counts().index])
# plt.xlabel('Clusters')
# plt.ylabel('Frequency')
# plt.title('2018 Cluster Frequencies')
# plt.ylim(0,180000)
# plt.savefig('outputs/2018_cluster_frequencies.png',bbox_inches="tight", dpi=150)
# plt.clf()


#################### GROUPED
# df_2021['clusters_2021_cleanup'].value_counts().plot(kind='bar') #, color=[color_dict[i] for i in df_2021['clusters_2021_edited'].value_counts().index])
# plt.xlabel('Clusters')
# plt.ylabel('Frequency')
# plt.title('2021 Cluster Frequencies')
# plt.ylim(0,275000)
# plt.savefig('outputs/2021_cluster_frequencies_cleanup.png',bbox_inches="tight", dpi=150)
# plt.clf()

# df_2011['clusters_2011_cleanup'].value_counts().plot(kind='bar') #,  color=[color_dict[i] for i in df_2011['clusters_2011_edited'].value_counts().index])
# plt.xlabel('Clusters')
# plt.ylabel('Frequency')
# plt.title('2011 Cluster Frequencies')
# plt.ylim(0,275000)
# plt.savefig('outputs/2011_cluster_frequencies_cleanup.png',bbox_inches="tight", dpi=150)
# plt.clf()

# df_2018['clusters_2018_cleanup'].value_counts().plot(kind='bar') #,  color=[color_dict[i] for i in df_2018['clusters'].value_counts().index])
# plt.xlabel('Clusters')
# plt.ylabel('Frequency')
# plt.title('2018 Cluster Frequencies')
# plt.ylim(0,275000)
# plt.savefig('outputs/2018_cluster_frequencies_cleanup.png',bbox_inches="tight", dpi=150)
# plt.clf()

all_cleanup = pd.DataFrame({'2011': list(df_2011['clusters_2011_cleanup'].value_counts().sort_index()),
                            '2018': list(df_2018['clusters_2018_cleanup'].value_counts().sort_index()),
                            '2021': list(df_2021['clusters_2021_cleanup'].value_counts().sort_index())})

prop = all_cleanup/all_cleanup.sum(axis=0)
prop.index = df_2011['clusters_2011_cleanup'].value_counts().sort_index().index

plt.clf()
prop.plot(kind='bar')
plt.xticks(rotation=45, ha='right')
plt.savefig('outputs/all_years_frequencies_cleanup_14_as_lowdense.png',bbox_inches="tight", dpi=150)
############################ CLUSTER 14 

import matplotlib.pyplot as plt


labels = ['2011', '2018', '2021']
vehicles1 = [61190/1236929, 52274/1071186, 52578/1261229]
vehicles2 = [49529/1236929, 68631/1071186, 63319/1261229]
vehicles3 = [0, 34275/1071186, 89464/1261229]
vehicles4 = [0, 55980/1071186, 0]
width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, vehicles1, width)
ax.bar(labels, vehicles2, width, bottom=vehicles1)
ax.bar(labels, vehicles2, width, bottom=vehicles2)
ax.bar(labels, vehicles2, width, bottom=vehicles3)

ax.set_ylabel('Proportions')
plt.show()

#############################################################

edit_2011 = list(df_2011['clusters_2011_edited'].value_counts().sort_index())
edit_2011.insert(7,0)
edit_2011.insert(14,0)

edit_2021 = list(df_2021['clusters_2021_edited'].value_counts().sort_index())
edit_2021.insert(14,0)

all_cleanup = pd.DataFrame({'2011': edit_2011,
                            '2018': list(df_2018['clusters'].value_counts().sort_index()),
                            '2021': edit_2021})

prop = all_cleanup/all_cleanup.sum(axis=0)
# prop.index = df_2018['clusters'].value_counts().index

plt.clf()
prop.plot(kind='bar')
plt.xticks(rotation=45, ha='right')
ax.set_ylabel('Proportions')
plt.xlabel('Clusters')
plt.savefig('outputs/all_years_frequencies_ungrouped.png',bbox_inches="tight", dpi=150)

# clean dataframe, keep single img identifier for merge to oa
df_2011 = df_2011[['Unnamed: 0.1', 'clusters', 'clusters_2011_edited', 'clusters_2011_cleanup']]
df_2011['idx'] = df_2011['Unnamed: 0.1'].apply(lambda x : int(x[:-2]))
df_2018 = df_2018[['Unnamed: 0', 'clusters', 'clusters_2018_cleanup']]
df_2018['idx'] = df_2018['Unnamed: 0'].apply(lambda x : int(x[:-2]))
df_2021 = df_2021[['Unnamed: 0.1', 'clusters', 'clusters_2021_edited', 'clusters_2021_cleanup']]
df_2021['idx'] = df_2021['Unnamed: 0.1'].apply(lambda x : int(x[:-2]))

# get merged oa counts
df_2011_oa = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2011/census_2011_image_meta_single_merged_oa.csv')
df_2018_oa = pd.read_csv('/home/emily/phd/2_interpretability/2018_all_perception_scores_merged_oa.csv')
df_2021_oa = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2021/census_2021_image_meta_single_merged_oa.csv')

df_2011_ = df_2011.merge(df_2011_oa[['OA11CD', 'idx']], left_on='idx', right_on='idx' )
df_2011_ = df_2011_.drop_duplicates()
df_2018_ = df_2018.merge(df_2018_oa[['OA11CD', 'idx']], left_on='idx', right_on='idx' )
df_2018_ = df_2018_.drop_duplicates()
df_2021_ = df_2021.merge(df_2021_oa[['OA11CD', 'idx']], left_on='idx', right_on='idx' )
df_2021_ = df_2021_.drop_duplicates()

# one hot encode classes 2011
df_2011_class = pd.get_dummies(df_2011_[['clusters_2011_cleanup']])
df_2011_class['OA11CD'] = df_2011_[['OA11CD']]
df_2011_oa = df_2011_class.groupby('OA11CD').sum()
#df_2011_oa = df_2011_oa.div(df_2011_oa.sum(axis=1), axis=0)
df_2011_oa = df_2011_oa.div(df_2011_class.groupby('OA11CD').count(), axis=0)

# one hot encode classes 2018
df_2018_class = pd.get_dummies(df_2018_[['clusters_2018_cleanup']])
df_2018_class['OA11CD'] = df_2018_[['OA11CD']]
df_2018_oa = df_2018_class.groupby('OA11CD').sum()
#df_2018_oa = df_2018_oa.div(df_2018_oa.sum(axis=1), axis=0)
df_2018_oa = df_2018_oa.div(df_2018_class.groupby('OA11CD').count(), axis=0)

# one hot encode classes 2018
df_2021_class = pd.get_dummies(df_2021_[['clusters_2021_cleanup']])
df_2021_class['OA11CD'] = df_2021_[['OA11CD']]
df_2021_oa = df_2021_class.groupby('OA11CD').sum()
#df_2021_oa = df_2021_oa.div(df_2021_oa.sum(axis=1), axis=0)
df_2021_oa = df_2021_oa.div(df_2021_class.groupby('OA11CD').count(), axis=0)

# will have to merge to oa's.
df_2011_oa['OA11CD_'] = df_2011_oa.index 
df_2018_oa['OA11CD_'] = df_2018_oa.index 
df_2021_oa['OA11CD_'] = df_2021_oa.index 

merge_all = df_2011_oa.merge(df_2021_oa, on='OA11CD_')

################################################ PERCENTAGES
features = list(prop.index)
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    merge_all[new_column] = (merge_all[t1_column] - merge_all[t0_column]) / merge_all[t0_column]

merge_all.to_csv('outputs/cluster_proportions_2011_2021.csv')

################################################# QUARTILES
df_2021_oa_quart = df_2021_oa.copy()
for column in list(df_2021_oa.columns[:-1]):
    df_2021_oa_quart[column] = pd.cut(df_2021_oa[column], [0, 0.01,.25, .5, .75, 1.], labels=[0,1,2,3,4])

df_2011_oa_quart = df_2011_oa.copy()
for column in list(df_2011_oa.columns[:-1]):
    df_2011_oa_quart[column] = pd.cut(df_2011_oa[column], [0, 0.01,.25, .5, .75, 1.], labels=[0,1,2,3,4])

merge_all = df_2011_oa_quart.merge(df_2021_oa_quart, on='OA11CD_')

features = list(prop.index)
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    merge_all[new_column] = merge_all[t1_column].astype(float) - merge_all[t0_column].astype(float)

merge_all.to_csv('outputs/cluster_quartiles_2011_2021.csv')

################################################# DECILES
df_2021_oa_dec = df_2021_oa.copy()
for column in list(df_2021_oa.columns[:-1]):
    df_2021_oa_dec[column] = pd.cut(df_2021_oa[column], [-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=[0,1,2,3,4,5,6,7,8,9,10])

df_2011_oa_dec = df_2011_oa.copy()
for column in list(df_2011_oa.columns[:-1]):
    df_2011_oa_dec[column] = pd.cut(df_2011_oa[column], [-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=[0,1,2,3,4,5,6,7,8,9,10])

merge_all = df_2011_oa_dec.merge(df_2021_oa_dec, on='OA11CD_')

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
df_2021_oa_counts = df_2021_class.groupby('OA11CD').count()
df_2021_err = {}
for column in list(df_2021_oa.columns[:-1]):
    eps = 0.01
    p = df_2021_oa[column] + eps
    n = df_2021_oa_counts[column]
    perr = np.sqrt(np.abs(p*(1-p))/n)
    df_2021_err[column] = perr
df_2021_err = pd.DataFrame(df_2021_err)

# get associated errors  - there are no output areas with zero counts!
df_2011_oa_counts = df_2011_class.groupby('OA11CD').count()
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

df_2011_ = df_2011.merge(df_2011_oa[['LSOA11CD', 'idx']], left_on='idx', right_on='idx' )
df_2011_ = df_2011_.drop_duplicates()
df_2018_ = df_2018.merge(df_2018_oa[['LSOA11CD', 'idx']], left_on='idx', right_on='idx' )
df_2018_ = df_2018_.drop_duplicates()
df_2021_ = df_2021.merge(df_2021_oa[['LSOA11CD', 'idx']], left_on='idx', right_on='idx' )
df_2021_ = df_2021_.drop_duplicates()

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
df_2011_oa['LSOA11CD_'] = df_2011_oa.index 
df_2018_oa['LSOA11CD_'] = df_2018_oa.index 
df_2021_oa['LSOA11CD_'] = df_2021_oa.index 

merge_all = df_2011_oa.merge(df_2021_oa, on='LSOA11CD_')

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

merge_all = df_2011_oa_quart.merge(df_2021_oa_quart, on='LSOA11CD_')

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

merge_all = df_2011_oa_dec.merge(df_2021_oa_dec, on='LSOA11CD_')

features = list(prop.index)
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    merge_all[new_column] = merge_all[t1_column].astype(float) - merge_all[t0_column].astype(float)

merge_all.to_csv('outputs/cluster_deciles_2011_2021_lsoa.csv')

################################################ ADD ERROR TO MEASUREMENT
# proportions dataframe
df_2021_oa 
df_2011_oa 

# get associated errors  - there are no output areas with zero counts!
df_2021_oa_counts = df_2021_class.groupby('LSOA11CD').count()
df_2021_err = {}
for column in list(df_2021_oa.columns[:-1]):
    eps = 0.01
    p = df_2021_oa[column] + eps
    n = df_2021_oa_counts[column]
    perr = np.sqrt(np.abs(p*(1-p))/n)
    df_2021_err[column] = perr
df_2021_err = pd.DataFrame(df_2021_err)

# get associated errors  - there are no output areas with zero counts!
df_2011_oa_counts = df_2011_class.groupby('LSOA11CD').count()
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
    df_mask.to_csv('outputs/uncertainty_mask_%s_lsoa.csv' % str(mu) )    

for mu in [0.5,0.75,0.9]:
    mask = {}
    for column in list(df_2021_oa.columns[:-1]):
        threshold = pd.qcut(df_2021_err[column],[mu], retbins=True)[1][0]
        mask[column] = df_2021_err[column].apply(lambda x: 0 if x <= threshold else 1)
    df_mask = pd.DataFrame(mask) 
    df_mask.to_csv('outputs/2021_uncertainty_mask_%s_lsoa.csv' % str(mu) )   


################################################### PLOTTING
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.spines.bottom'] = False

# plot ordered proportions per feature with err 
for column in list(df_2021_oa.columns[:-1]):
    p = df_2021_oa[column]
    perr = df_2021_err[column]
    concat = pd.concat([p,perr], axis=1)
    concat.columns = [0,1]
    concat = concat.sort_values(0).astype(float).dropna()
    plot_errors(concat, column + 'lsoa')
    plot_scatter(concat, column + 'lsoa')
    plot_bars(concat, column + 'lsoa', [0.5,0.75,0.9])

def plot_errors(concat, col):
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = True
    mpl.rcParams['axes.spines.bottom'] = False
    plt.clf()
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autscaling
    # plot means and std stacked in order
    concat = concat.sort_values(0).astype(float).dropna()
    concat.plot(y=0, yerr=1, marker='.', color='red', alpha = 0.7, ax=ax, legend=False, markerfacecolor='red', markeredgewidth=0, linewidth=0.01)
    plt.yticks(rotation=-90)
    plt.xticks([])
    plt.xlabel('')
    plt.ylim(-0.15,1.15)
    plt.gca().invert_yaxis()
    plt.setp(ax.spines.values(), linewidth=2)
    plt.savefig('outputs/%s_prop_errors.png' % col)

def plot_scatter(concat, col):
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.bottom'] = True
    plt.clf()
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autscaling
    # plot means and std stacked in order
    ax.scatter(concat[0], concat[1])  
    ax.set_xlabel('Proportion')
    ax.set_ylabel('SE')
    plt.savefig('outputs/%s_scatter_prop_errors.png' % col)

def plot_bars(concat, col, mu):
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.bottom'] = True
    mu = [0.5, 0.75, 0.9]
    plt.clf()
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autscaling
    # plot means and std stacked in order
    concat = concat.sort_values(1).astype(float).dropna()
    threshold = pd.qcut(concat[1],mu, retbins=True)[1]
    # create colour map
    cmap = matplotlib.cm.get_cmap('tab20c')
    concat['keep'] = concat[1].apply(lambda x: cmap(0) if x <= threshold[0] else (cmap(1) if x <= threshold[1] else (cmap(2) if x<= threshold[2] else cmap(3))) )
    plt.bar(x=np.arange(0,concat.shape[0]), height=concat[1], color=concat['keep'])
    # horizontal line indicating the threshold
    ax.plot([0., int(0.5*concat.shape[0])], [threshold[0], threshold[0]], "k--")
    ax.plot([0., int(0.75*concat.shape[0])], [threshold[1], threshold[1]], "k--")
    ax.plot([0., int(0.9*concat.shape[0])], [threshold[2], threshold[2]], "k--")
    ax.set_ylabel('SE')
    plt.savefig('outputs/%s_bar_errors.png' % col)

