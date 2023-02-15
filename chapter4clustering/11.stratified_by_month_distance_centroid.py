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
df_2021 = pd.read_csv('chapter4clustering/outputs/df_2021_matched_by_centroid.csv')
df_2011 = pd.read_csv('chapter4clustering/outputs/df_2011_matched_by_centroid.csv')
df_2018 = pd.read_csv(prefix2 + '2018_rmac_feature_vector_clusters_20.csv', error_bad_lines=False, engine="python")

# merge
merged_2011 = meta_2011.merge(df_2011[['Unnamed: 0', 'matched']], left_on='idx_y', right_on='Unnamed: 0', how='left')
merged_2018 = meta_2018.merge(df_2018[['Unnamed: 0', 'clusters']], left_on='idx_y', right_on='Unnamed: 0', how='left')
merged_2021 = meta_2021.merge(df_2021[['Unnamed: 0', 'matched']], left_on='idx_y', right_on='Unnamed: 0', how='left')
merged_2021 = merged_2021.drop(columns=['idx_y.1'])
merged_2011 = merged_2011.dropna()
merged_2018 = merged_2018.dropna()
merged_2021 = merged_2021.dropna()

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.grid'] = False

# 2011 plots
plt.clf()
group_2011 = merged_2011[['month', 'matched', 'idx']].groupby(['month', 'matched']).count()
group_2011 = group_2011.reset_index().pivot(columns='matched',index='month',values='idx').fillna(0)
group_2011 = group_2011/group_2011.sum(axis=0)
group_2011.to_csv('chapter4clustering/outputs/R/2011_clusters_by_month.csv')
sns.heatmap(group_2011)
plt.xlabel('Cluster')
plt.ylabel('Month')
plt.savefig('outputs/2011_monthly.png',bbox_inches="tight", dpi=150)

# 2018 plots
plt.clf()
group_2018 = merged_2018[['month', 'clusters', 'idx']].groupby(['month', 'clusters']).count()
group_2018 = group_2018.reset_index().pivot(columns='clusters',index='month',values='idx').fillna(0)
group_2018 = group_2018/group_2018.sum(axis=0)
group_2011.to_csv('chapter4clustering/outputs/R/2018_clusters_by_month.csv')

T = group_2018.T
T['12'] = np.zeros(20)
group_2018 = T.T
group_2018.to_csv('chapter4clustering/outputs/R/2018_clusters_by_month.csv')

sns.heatmap(group_2018)
plt.xticks(np.arange(20), np.arange(20))
plt.xlabel('Cluster')
plt.ylabel('Month')
plt.savefig('outputs/2018_monthly.png',bbox_inches="tight", dpi=150)

# 2021 plots
plt.clf()
group_2021 = merged_2021[['month', 'matched', 'idx']].groupby(['month', 'matched']).count()
group_2021 = group_2021.reset_index().pivot(columns='matched',index='month',values='idx').fillna(0)
group_2021 = group_2021/group_2021.sum(axis=0)
group_2021.to_csv('chapter4clustering/outputs/R/2021_clusters_by_month.csv')

sns.heatmap(group_2021)
plt.xlabel('Cluster')
plt.ylabel('Month')
plt.savefig('outputs/2021_monthly.png',bbox_inches="tight", dpi=150)
