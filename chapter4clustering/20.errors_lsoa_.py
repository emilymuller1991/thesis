# the goal of this notebook is to perform welch's t-test on proportion of clusters in each dataset.
import pandas as pd 
import numpy as np
import interval
from sklearn.metrics import normalized_mutual_info_score as nmi 

df_2011 = pd.read_csv('chapter4clustering/outputs/2011_proportions_all_nonna.csv')
df_2021 = pd.read_csv('chapter4clustering/outputs/2021_proportions_all_nonna.csv')

# remove those with low spatial sampling 
certainty = pd.read_csv('chapter3data/outputs/sampling_rate_lsoa_all_years_sql.csv')
merge_2011 = certainty.merge(df_2011, left_on='lsoa11cd', right_on='lsoa')
merge_2011['p_2011'] = merge_2011['2011']
merge_2011['counts'] = merge_2011['count']

def get_sample(c, p, size):
    if p > 1:
        p = 1
    N = np.ceil(c/p)
    print (c, N)
    sim = np.random.binomial(n=N,p=p,size=size)
    return np.std(100*sim/N)

merge_2011['err'] = merge_2011.apply(lambda x: get_sample(x.counts,x.p_2011,10000), axis=1 )

##############################################################################
merge_2021 = certainty.merge(df_2021, left_on='lsoa11cd', right_on='lsoa')
merge_2021['p_2021'] = merge_2021['2021']
merge_2021['counts'] = merge_2021['count']

def get_sample(c, p, size):
    if p > 1:
        p = 1
    N = np.ceil(c/p)
    print (c, N)
    sim = np.random.binomial(n=N,p=p,size=size)
    return np.std(100*sim/N)

merge_2021['err'] = merge_2021.apply(lambda x: get_sample(x.counts,x.p_2021,10000), axis=1 )

###################################################################################
clusters_2011 = pd.read_csv("/media/emily/south/phd/chapter4clustering/outputs/2011_proportions_all_nonna_lsoa_hierarchical8.csv")
clusters_2021 = pd.read_csv("/media/emily/south/phd/chapter4clustering/outputs/2021_proportions_all_nonna_lsoa_hierarchical8.csv")

clusters_2011 = clusters_2011.merge(merge_2011[['lsoa', 'err']], left_on='lsoa', right_on='lsoa')
clusters_2021 = clusters_2021.merge(merge_2021[['lsoa', 'err']], left_on='lsoa', right_on='lsoa')

clusters_2011.to_csv("/media/emily/south/phd/chapter4clustering/outputs/2011_proportions_all_nonna_lsoa_hierarchical8_err.csv")
clusters_2021.to_csv("/media/emily/south/phd/chapter4clustering/outputs/2021_proportions_all_nonna_lsoa_hierarchical8_err.csv")

###################################################################################
c2011 = clusters_2011[['lsoa','hierarchical8','err']]
c2021 = clusters_2021[['lsoa','hierarchical8','err']]

merged = c2011.merge(c2021, left_on='lsoa', right_on='lsoa')
merged['same'] = merged.apply(lambda x: 1 if x.hierarchical8_x == x.hierarchical8_y else 0, axis = 1)

merged.to_csv("/media/emily/south/phd/chapter4clustering/outputs/lsoa_hierarchical8_changes.csv")


pivot = pd.pivot_table(merged, values='same', index='hierarchical8_x', columns='hierarchical8_y',
               aggfunc='count').reset_index()

pivot.to_csv('chapter4clustering/outputs/R/lsoa_hierarchical_cluster_change.csv')


pivot_ = pivot[pivot.columns[1:]].div(pivot[pivot.columns[1:]].sum(axis=1), axis=0)
pivot_.to_csv('chapter4clustering/outputs/R/_lsoa_hierarchical_cluster_change.csv')

# pivot_['hierarchical8_y'] = pivot['hierarchical8_y']

# import seaborn as sns
# plt.clf()
# f = plt.figure(figsize=(10, 10))

# sns.heatmap(pivot[pivot.columns[1:]], annot=True, cbar=False)#
clusters_2011 = pd.read_csv("/media/emily/south/phd/chapter4clustering/outputs/2011_proportions_all_nonna_lsoa_hierarchical8.csv")
clusters_2021 = pd.read_csv("/media/emily/south/phd/chapter4clustering/outputs/2021_proportions_all_nonna_lsoa_hierarchical8.csv")


###################################################################################
c2011 = clusters_2011[['lsoa','hierarchical8']]
c2021 = clusters_2021[['lsoa','hierarchical8']]

merged = c2011.merge(c2021, left_on='lsoa', right_on='lsoa')
merged['same'] = merged.apply(lambda x: 1 if x.hierarchical8_x == x.hierarchical8_y else 0, axis = 1)

merged.to_csv("/media/emily/south/phd/chapter4clustering/outputs/both_years_proportions_all_nonna_lsoa_hierarchical8.csv")