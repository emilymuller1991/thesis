import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib 
import seaborn as sns

meta_2011 = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2011/census_2011_image_meta_double.csv')
meta_2018 = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/2018/greater_london_2018_panoids_to_download.csv')
meta_2021 = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2021/census_2021_image_meta_double.csv')

# get angles for 2018 metadata
rdf = pd.DataFrame(np.repeat(meta_2018.values, 2, axis=0), columns=meta_2018.columns)
rdf['repeat'] = 1.0
rdf['repeat'] = rdf.groupby('idx').repeat.cumsum() - 1
rdf['angle'] = rdf['repeat'].apply(lambda x: '_b' if x == 1 else '_d')
rdf['idx_y'] = rdf.apply(lambda x: str(x.idx) + x.angle, axis=1)
meta_2018 = rdf.drop(columns=['repeat', 'angle'])

# merge all to clusters
prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home/emily/phd/003_image_matching/clustering/output/'
prefix2 = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/project/pathways/live/Transferability/emily_phd_images/keras-rmac-clustering_output/2018_original/'
df_2021 = pd.read_csv(prefix + 'census2021_zoom_clusters_20.csv', error_bad_lines=False, engine="python")
df_2011 = pd.read_csv(prefix + 'census2011_zoom_clusters_20.csv', error_bad_lines=False, engine="python")
df_2018 = pd.read_csv(prefix2 + '2018_rmac_feature_vector_clusters_20.csv', error_bad_lines=False, engine="python")

# merge
merged_2011 = meta_2011.merge(df_2011[['Unnamed: 0.1', 'clusters']], left_on='idx_y', right_on='Unnamed: 0.1', how='left')
merged_2018 = meta_2018.merge(df_2018[['Unnamed: 0', 'clusters']], left_on='idx_y', right_on='Unnamed: 0', how='left')
merged_2021 = meta_2021.merge(df_2021[['Unnamed: 0.1', 'clusters']], left_on='idx_y', right_on='Unnamed: 0.1', how='left')
merged_2021 = merged_2021.drop(columns=['idx_y.1'])
merged_2011 = merged_2011.dropna()
merged_2018 = merged_2018.dropna()
merged_2021 = merged_2021.dropna()

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
12:14,
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
1:14,
10:15,
0:16,
12:17,
1:18,
4:19
}

def relabel_2021_(x):
    return relabel_2021[x]

def relabel_2011_(x):
    return relabel_2011[x]

merged_2021['clusters_2021_edited'] = merged_2021['clusters'].apply(lambda x: relabel_2021_(x))
merged_2011['clusters_2011_edited'] = merged_2011['clusters'].apply(lambda x: relabel_2011_(x))

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.grid'] = False

# 2011 plots
plt.clf()
group_2011 = merged_2011[['month', 'clusters_2011_edited', 'idx']].groupby(['month', 'clusters_2011_edited']).count()
group_2011 = group_2011.reset_index().pivot(columns='clusters_2011_edited',index='month',values='idx').fillna(0)
group_2011 = group_2011/group_2011.sum(axis=0)

sns.heatmap(group_2011)
plt.xlabel('Cluster')
plt.ylabel('Month')
plt.savefig('outputs/2011_monthly.png',bbox_inches="tight", dpi=150)

# 2018 plots
plt.clf()
group_2018 = merged_2018[['month', 'clusters', 'idx']].groupby(['month', 'clusters']).count()
group_2018 = group_2018.reset_index().pivot(columns='clusters',index='month',values='idx').fillna(0)
group_2018 = group_2018/group_2018.sum(axis=0)

T = group_2018.T
T['12'] = np.zeros(20)
group_2018 = T.T

sns.heatmap(group_2018)
plt.xticks(np.arange(20), np.arange(20))
plt.xlabel('Cluster')
plt.ylabel('Month')
plt.savefig('outputs/2018_monthly.png',bbox_inches="tight", dpi=150)

# 2021 plots
plt.clf()
group_2021 = merged_2021[['month', 'clusters_2021_edited', 'idx']].groupby(['month', 'clusters_2021_edited']).count()
group_2021 = group_2021.reset_index().pivot(columns='clusters_2021_edited',index='month',values='idx').fillna(0)
group_2021 = group_2021/group_2021.sum(axis=0)

sns.heatmap(group_2021)
plt.xlabel('Cluster')
plt.ylabel('Month')
plt.savefig('outputs/2021_monthly.png',bbox_inches="tight", dpi=150)
