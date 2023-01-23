import pandas as pd 
import numpy as np
from pySankey import sankey
from sklearn.metrics import normalized_mutual_info_score as nmi 

# find how many of the 2018 images also appear in the 2021 image clustering task 
meta_2018 = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/2018/greater_london_2018_panoids_to_download.csv')
meta_2021 = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2021/census_2021_image_meta_single.csv')
meta_2021 = meta_2021.drop(columns=['idx_y', 'idx_y.1'])

# get panoids appearing in both sets
merged = meta_2018.merge(meta_2021, on='panoid', how='inner', indicator=True)

# get just idx 2018 and idx 2021
df = merged[['idx_x', 'idx_y']]
rdf = pd.DataFrame(np.repeat(df.values, 2, axis=0), columns=df.columns)

# duplicate angles of images
rdf['repeat'] = 1
rdf['repeat'] = rdf.groupby('idx_x').repeat.cumsum() - 1
rdf['angle'] = rdf['repeat'].apply(lambda x: '_b' if x == 1 else '_d')
rdf['angle_2018'] = rdf.apply(lambda x: str(x.idx_x) + x.angle, axis=1)
rdf['angle_2021'] = rdf.apply(lambda x: str(x.idx_y) + x.angle, axis=1)
df = rdf[['angle_2018', 'angle_2021']]

# merge to cluster outputs but first read in clusters
prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home/emily/phd/003_image_matching/clustering/output/'
prefix2 = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/project/pathways/live/Transferability/emily_phd_images/keras-rmac-clustering_output/2018_original/'
df_2021_zoom = pd.read_csv(prefix + 'census2021_zoom_clusters_20.csv', error_bad_lines=False, engine="python")
df_2018 = pd.read_csv(prefix2 + '2018_rmac_feature_vector_clusters_20.csv', error_bad_lines=False, engine="python")

# merge
merged = df.merge(df_2018[['Unnamed: 0', 'clusters']], left_on='angle_2018', right_on='Unnamed: 0', how='left')
merged.columns = ['angle_2018', 'angle_2021', 'Unnamed: 0', 'clusters_2018']
merged = merged.merge(df_2021_zoom[['Unnamed: 0.1', 'clusters']], left_on='angle_2021', right_on='Unnamed: 0.1', how='left')
merged.columns = ['angle_2018', 'angle_2021', 'Unnamed: 0', 'clusters_2018', 'Unnamed: 0.1', 'clusters_2021']
c = merged.dropna()

# relabel_2021_ = {
# 18:0,
# 9:1,
# 3:2,
# 13:3,
# 11:4,
# 5:5,
# 14:6,
# 2:7,
# 16:8,
# 17:9,
# 7:10,
# 15:11,
# 6:11,
# 8:12,
# 19:13,
# 1:14,
# 10:15,
# 0:16,
# 12:17,
# 1:18,
# 4:19
# }

relabel_2021_ = {
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

def relabel_2021(x):
    return relabel_2021_[x]

c['clusters_2021_matched'] = c['clusters_2021'].apply(lambda x: relabel_2021(x))
keep = c.dropna()

nmi_all = nmi(keep['clusters_2018'], keep['clusters_2021_matched'])
# 0.5553174735088422

sankey.sankey(keep['clusters_2018'], keep['clusters_2021_matched'], aspect=20, 
    fontsize=12)

# keep only the important clusters but don't aggregate
def drop_disregarded(x):
    if x in [3,5,17]:
        return np.nan
    else:
        return x

#c = merged.dropna()
c['clusters_2018_drop_disregarded'] = c['clusters_2018'].apply(lambda x: drop_disregarded(x))
c['clusters_2021_drop_disregarded'] = c['clusters_2021_matched'].apply(lambda x: drop_disregarded(x))
keep2 = c.dropna()
# shape = 84 172

nmi_drop_disregarded = nmi(keep2['clusters_2018_drop_disregarded'], keep2['clusters_2021_drop_disregarded'])
# 0.5716068586510747

sankey.sankey(keep2['clusters_2018_drop_disregarded'], keep2['clusters_2021_drop_disregarded'], aspect=20, 
    fontsize=12)

######## aggregate and drop clusters not interested in
def group_similar(x):
    if x in [3,5,17]:
        return np.nan
    elif x in [8,9,14]:
        return 8
    elif x == 4:
        return 4
    elif x in [1, 11]:
        return 1
    elif x == 0:
        return 0
    elif x == 15:
        return 15
    elif x in [2,6]:
        return 2
    elif x in [7,16,18]:
        return 7
    elif x == 10:
        return 10
    elif x == 12:
        return 12
    elif x == 13:
        return 13
    elif x == 19:
        return 19
    else:
        return x

# def cleanup_categories(x):
#     if x in [2,3,5,7,13,17,6,16,18,12,19]:
#         return np.nan
#     elif x in [4,8,9,14]:
#         return 'Low-density'
#     elif x in [1, 11]:
#         return 'Green'
#     elif x == 0:
#         return 'Terraced'
#     elif x == 15:
#         return 'Commercial'
#     else:
#         return x

c = merged.dropna()
c['clusters_2021_matched'] = c['clusters_2021'].apply(lambda x: relabel_2021(x))
c['clusters_2018_group_similar'] = c['clusters_2018'].apply(lambda x: group_similar(x))
c['clusters_2021_group_similar'] = c['clusters_2021_matched'].apply(lambda x: group_similar(x))
keep2 = c.dropna()

nmi_group_similar = nmi(keep2['clusters_2018_group_similar'], keep2['clusters_2021_group_similar'])
# 0.5791757733768758

# 0.5620412840495684

######## aggregate and drop vehicles
def group_similar_no_vehicle(x):
    if x in [3,5,17]:
        return np.nan
    elif x in [8,9,14]:
        return 8
    elif x == 4:
        return 4
    elif x in [1, 11]:
        return 1
    elif x == 0:
        return 0
    elif x == 15:
        return 15
    elif x in [2,6]:
        return 2
    elif x in [7,16,18]:
        return np.nan
    elif x == 10:
        return 10
    elif x == 12:
        return 12
    elif x == 13:
        return 13
    elif x == 19:
        return 19
    else:
        return x


c['clusters_2018_group_similar_no_veh'] = c['clusters_2018'].apply(lambda x: group_similar_no_vehicle(x))
c['clusters_2021_group_similar_no_veh'] = c['clusters_2021_matched'].apply(lambda x: group_similar_no_vehicle(x))
keep2 = c.dropna()

nmi_group_similar = nmi(keep2['clusters_2018_group_similar_no_veh'], keep2['clusters_2021_group_similar_no_veh'])
# 0.6317430197298827
# 0.6169430738265519, 149,636 - no vehicles

c = merged.dropna()
c['clusters_2021_matched'] = c['clusters_2021'].apply(lambda x: relabel_2021(x))

def keep_interesting(x):
    if x in [3,5,17]:
        return np.nan
    elif x in [8,9,14]:
        return np.nan
    elif x == 4:
        return np.nan
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
colorDict = {
    'Low-density':'#00fffffd',
    'Medium-density':'#00fffffd',
    'Leafy green residential':'#00fffffd',
    'Open green space':'#00ff00fd',
    'Green':'#00ff00fd',
    'Terraced':'#0000fffd',
    'Commercial':'#ffcc00fd',
    'Other green': '#008000fd',
    'Vehicles': '#ff0000fd',
    'High-density': '#ff6600fd' ,
    'Sheds': '#c87137fd', 
    'Estates': '#ff6600fd',
    'Fences': '#c87137fd'
}

c['clusters_2018_keep_interesting'] = c['clusters_2018'].apply(lambda x: keep_interesting(x))
c['clusters_2021_keep_interesting'] = c['clusters_2021_matched'].apply(lambda x: keep_interesting(x))
keep3 = c.dropna()

nmi_keep_interesting = nmi(keep3['clusters_2018_keep_interesting'], keep3['clusters_2021_keep_interesting'])
# 0.8007415235362085
# 0.7425666977342138, 99,674 - 14 as vehi

sankey.sankey(keep3['clusters_2018_keep_interesting'], keep3['clusters_2021_keep_interesting'], colorDict=colorDict, aspect=20, 
    fontsize=12)

fig = plt.gcf()

# Set size in inches
fig.set_size_inches(6, 10)

# Set the color of the background to white
fig.set_facecolor("w")

# Save the figure
fig.savefig("outputs/sankey_intersection_keep_interesting_14_as_ld.png", bbox_inches="tight", dpi=150)