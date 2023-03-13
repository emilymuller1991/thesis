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

import seaborn as sns
plt.clf()
f = plt.figure(figsize=(10, 10))

sns.heatmap(pivot[pivot.columns[1:]], annot=True, cbar=False)

# from pySankey import sankey

# sankey.sankey(merged['hierarchical8_x'], merged['hierarchical8_y'],aspect=20, fontsize=12)
# plt.show()

# nmi = nmi(merged['hierarchical8_x'], merged['hierarchical8_y'])

# merge_2021.plot.scatter(x = 'p_2021', y='err')
# plt.xlim(0,1)
# # std_err = {}
# # size = 10000

#     n = df_2011['count']
#     N = 
#     df = df_2011[[feature, 'count']]
#     df.columns = ['features', 'counts']
#     err = df.apply(lambda x: get_sample(x.counts,x.features,10000), axis=1 )
#     std_err[feature] = list(err)
# err_df_2011 = pd.DataFrame(std_err)

# std_err = {}
# size = 10000
# for feature in df_2021.columns[1:-1]:
#     p = df_2021[feature]
#     n = df_2021['count']
#     df = df_2021[[feature, 'count']]
#     df.columns = ['features', 'counts']
#     err = df.apply(lambda x: get_sample(x.counts,x.features,size), axis=1 )
#     std_err[feature] = list(err)
# err_df_2021 = pd.DataFrame(std_err)

# df_2011.reset_index(drop=True, inplace=True)
# df_2021.reset_index(drop=True, inplace=True)

# ############################################################ CALCULATING WELCH's t-test significance
# from scipy import stats
# def get_sig(mu_1, mu_2, sd_1, sd_2, n_1, n_2):
#     sd_1_corrected = sd_1/np.sqrt(n_1)
#     sd_2_corrected = sd_2/np.sqrt(n_2)
#     t = (mu_1 - mu_2) / np.sqrt(sd_1_corrected**2 + sd_2_corrected**2)
#     df = ((sd_1**2/n_1) + (sd_2**2/n_2))**2/((sd_1**2/(n_1*(n_1-1))) + (sd_2**4/(n_2*(n_2-1))))
#     p = stats.t.sf(np.abs(t), df=df)
#     return t, p, df

# welch = {}
# col = [i.split('_')[-1] for i in list(err_df_2021.columns)]
# for feature in col:
#     t0_column = 'clusters_2011_cleanup_%s' % feature
#     t1_column = 'clusters_2021_cleanup_%s' % feature
#     mu_0 = df_2011[t0_column]
#     sd_0 = err_df_2011[t0_column]
#     n_0 = df_2011['count']
#     mu_1 = df_2021[t1_column]
#     sd_1 = err_df_2021[t1_column]
#     n_1 = df_2021['count']
#     df = pd.DataFrame({'mu_0': mu_0*100, 'mu_1': mu_1*100, 'sd_0': sd_0, 'sd_1': sd_1, 'n_0': n_0, 'n_1': n_1})
#     df['t,p,d'] = df.apply(lambda x: get_sig(x.mu_0, x.mu_1, x.sd_0, x.sd_1, x.n_0, x.n_1), axis=1)
#     welch[feature] = list(df['t,p,d'].apply(lambda x: x[1]))

# welch_df = pd.DataFrame(welch)
# ############################################################CALCULATING OVERLAPPING INTERVALS
# import portion as P

# std_int_2011 = {}
# for feature in df_2011.columns[1:-1]:
#     p = df_2011[feature]*100
#     err = err_df_2011[feature]
#     concat = pd.concat([p,err], axis=1)
#     concat.columns = ['p', 'err']
#     concat['int'] = concat.apply(lambda x: P.closed(x.p-1.96*x.err, x.p+1.96*x.err), axis=1 )
#     std_int_2011[feature] = concat['int']
# int_2011 = pd.DataFrame(std_int_2011)
# int_2011.columns = [i.split('_')[-1] for i in list(err_df_2021.columns)]

# std_int_2021 = {}
# for feature in df_2021.columns[1:-1]:
#     p = df_2021[feature]*100
#     err = err_df_2021[feature]
#     concat = pd.concat([p,err], axis=1)
#     concat.columns = ['p', 'err']
#     concat['int'] = concat.apply(lambda x: P.closed(x.p-1.96*x.err, x.p+1.96*x.err), axis=1 )
#     std_int_2021[feature] = concat['int']
# int_2021 = pd.DataFrame(std_int_2021)
# int_2021.columns = [i.split('_')[-1] for i in list(err_df_2021.columns)]

# sig = {}
# for feature in int_2021.columns:
#     x = int_2011[feature]
#     y = int_2021[feature]
#     concat = pd.concat([x,y], axis=1)
#     concat.columns = ['x','y']
#     concat['overlap'] = concat.apply(lambda x: x.x.overlaps(x.y), axis=1 )
#     sig[feature] = concat['overlap']
# sig_df = 1 - pd.DataFrame(sig)

# sig_df['lsoa'] = df_2011['LSOA11CD']
# sig_df.to_csv('chapter4clustering/outputs/significant_temporal_changes_reduced_set.csv')

# # Commercial                                                     96
# # Estates                                                       339
# # Green space                                                   254
# # High-density                                                  112
# # Low-density                                                   762

# change = {}
# features = list(sig_df.columns[:-1])
# for feature in features:
#     new_column = 'proportion_%s' % feature
#     t0_column = 'clusters_2011_cleanup_%s' % feature
#     t1_column = 'clusters_2021_cleanup_%s' % feature
#     change[new_column] = df_2021[t1_column]*100 -df_2011[t0_column]*100
# change_df = pd.DataFrame(change)
# change_df['lsoa'] = df_2011['LSOA11CD']

# change_df.to_csv('chapter4clustering/outputs/cluster_proportions_reduced_set.csv')

# masked = change_df[sig_df]
# masked['lsoa'] = df_2011['LSOA11CD']
# masked.to_csv('chapter4clustering/outputs/cluster_masked_reduced_set.csv')
