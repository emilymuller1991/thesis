# this file serves to analyse sets of individual OA's for how they have changes in proportions over the years 
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

# df_2011 = df_2011_.dropna()
# df_2018 = df_2018_.dropna()
# df_2021 = df_2021_.dropna()

features = ['Commercial', 'Estates', 'Green space', 'High-density', 'Low-density']

def expand_list(df, features):
    new_list = []
    for f in features:
        try:
            new_list.append(df[f])
        except:
            new_list.append(0)
    return new_list

spot_comm = {}
lsoa_commercial = ['E01004392', 'E01001718', 'E01000956', 'E01003958']
for lsoa in lsoa_commercial:
    counts_2011 = df_2011_[df_2011_['LSOA11CD'] == lsoa]['clusters_2011_cleanup'].value_counts().sort_index()
    counts_2021 = df_2021_[df_2021_['LSOA11CD'] == lsoa]['clusters_2021_cleanup'].value_counts().sort_index()
    expand_2011 = expand_list(counts_2011, features)
    expand_2021 = expand_list(counts_2021, features)
    both = [(expand_2011[i], expand_2021[i]) for i in range(5)]
    both.append((sum(expand_2011),sum(expand_2021)))
    spot_comm[lsoa] = both
commercial = pd.DataFrame(spot_comm).T 
commercial.columns = features + ['total']

for lsoa in lsoa_commercial[:1]:
    counts_2011 = df_2011_[df_2011_['LSOA11CD'] == lsoa]
    counts_2021 = df_2021_[df_2021_['LSOA11CD'] == lsoa]

# merge to lat lon
df_2011_oa = pd.read_csv('chapter3data/outputs/2011_panoids_merged_to_oa.csv')
df_2018_oa = pd.read_csv('chapter3data/outputs/2018_panoids_merged_to_oa.csv')
df_2021_oa = pd.read_csv('chapter3data/outputs/2021_panoids_merged_to_oa.csv')

counts_2011 = counts_2011.merge(df_2011_oa, left_on='idx', right_on='idx')
counts_2021 = counts_2021.merge(df_2021_oa, left_on='idx', right_on='idx')

counts_2011.to_csv('chapter4clustering/outputs/spots/E01004392_2011.csv')
counts_2021.to_csv('chapter4clustering/outputs/spots/E01004392_2021.csv')