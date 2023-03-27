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
df_2021 = pd.read_csv('chapter4clustering/outputs/df_2021_matched_by_centroid.csv')
df_2018 = pd.read_csv(prefix2 + '2018_rmac_feature_vector_clusters_20.csv', error_bad_lines=False, engine="python")

# merge
merged = df.merge(df_2018[['Unnamed: 0', 'clusters']], left_on='angle_2018', right_on='Unnamed: 0', how='left')
merged.columns = ['angle_2018', 'angle_2021', 'Unnamed: 0', 'clusters_2018']
merged = merged.merge(df_2021[['Unnamed: 0', 'matched']], left_on='angle_2021', right_on='Unnamed: 0', how='left')
merged.columns = ['angle_2018', 'angle_2021', 'Unnamed: 0', 'clusters_2018', 'Unnamed: 0.1', 'clusters_2021']
c = merged.dropna()

keep = c.dropna()

nmi_all = nmi(keep['clusters_2018'], keep['clusters_2021'])
# 0.6110089980476983

sankey.sankey(keep['clusters_2018'], keep['clusters_2021'], aspect=20, 
    fontsize=12)

# keep only the important clusters but don't aggregate
def drop_disregarded(x):
    if x in [3,5,17]:
        return np.nan
    else:
        return x

#c = merged.dropna()
c['clusters_2018_drop_disregarded'] = c['clusters_2018'].apply(lambda x: drop_disregarded(x))
c['clusters_2021_drop_disregarded'] = c['clusters_2021'].apply(lambda x: drop_disregarded(x))
keep2 = c.dropna()
# shape = 179,484

nmi_drop_disregarded = nmi(keep2['clusters_2018_drop_disregarded'], keep2['clusters_2021_drop_disregarded'])
# 0.6383992790875631

#sankey.sankey(keep2['clusters_2018_drop_disregarded'], keep2['clusters_2021_drop_disregarded'], aspect=20, 
    #fontsize=12)

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
c['clusters_2018_group_similar'] = c['clusters_2018'].apply(lambda x: group_similar(x))
c['clusters_2021_group_similar'] = c['clusters_2021'].apply(lambda x: group_similar(x))
keep2 = c.dropna()
# 179,484

nmi_group_similar = nmi(keep2['clusters_2018_group_similar'], keep2['clusters_2021_group_similar'])
# 0.6314186053687324

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
c['clusters_2021_group_similar_no_veh'] = c['clusters_2021'].apply(lambda x: group_similar_no_vehicle(x))
keep2 = c.dropna()
# 154238

nmi_group_similar = nmi(keep2['clusters_2018_group_similar_no_veh'], keep2['clusters_2021_group_similar_no_veh'])
# 0.6500909665683218 - no vehicles

def keep_interesting(x):
    if x in [3,5,17]:
        return np.nan
    elif x in [8,9,14]:
        return "Low-density"
    elif x == 4:
        return 'Leafy green'
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
c['clusters_2021_keep_interesting'] = c['clusters_2021'].apply(lambda x: keep_interesting(x))
keep3 = c.dropna()
# 120736

sankey = keep3[['clusters_2018_keep_interesting','clusters_2021_keep_interesting']]
sankey['tuple'] = sankey.apply(lambda x: (x.clusters_2018_keep_interesting, x.clusters_2021_keep_interesting), axis=1)
sankey_grouped = sankey.groupby('tuple').count()
sankey_grouped['source'] = [sankey_grouped.index[i][0] + '_2018' for i in range(sankey_grouped.shape[0])]
sankey_grouped['target'] = [sankey_grouped.index[i][1] + '_2021' for i in range(sankey_grouped.shape[0])]
sankey_grouped = sankey_grouped[['clusters_2018_keep_interesting', 'source', 'target']]
sankey_grouped.to_csv('chapter4clustering/outputs/intersection_sankey.csv')

nmi_keep_interesting = nmi(keep3['clusters_2018_keep_interesting'], keep3['clusters_2021_keep_interesting'])
# 0.72

sankey.sankey(keep3['clusters_2018_keep_interesting'], keep3['clusters_2021_keep_interesting'], colorDict=colorDict, aspect=20, 
    fontsize=12)

fig = plt.gcf()

# Set size in inches
fig.set_size_inches(6, 10)

# Set the color of the background to white
fig.set_facecolor("w")

# Save the figure
fig.savefig("outputs/sankey_intersection_keep_interesting_14_as_ld.png", bbox_inches="tight", dpi=150)

keep3['same'] = keep3.apply(lambda x: 1 if x.clusters_2018_keep_interesting == x.clusters_2021_keep_interesting else 0, axis =1)
pivot = pd.pivot_table(keep3, values='same', index='clusters_2018_keep_interesting', columns='clusters_2021_keep_interesting',
               aggfunc='count').reset_index()
pivot.to_csv('chapter4clustering/outputs/R/sankey_intersection_heatmap_change.csv')


pivot_ = pivot[pivot.columns[1:]].div(pivot[pivot.columns[1:]].sum(axis=1), axis=0)
pivot_['clusters_2018_keep_interesting'] = pivot['clusters_2018_keep_interesting']
pivot_ = pivot_[pivot.columns]

pivot_.to_csv('chapter4clustering/outputs/R/_sankey_intersection_heatmap_change.csv')

# pivot_ = pivot[pivot.columns[1:]].div(pivot[pivot.columns[1:]].sum(axis=1), axis=0)
# pivot_.to_csv('chapter4clustering/outputs/R/_lsoa_hierarchical_cluster_change.csv')