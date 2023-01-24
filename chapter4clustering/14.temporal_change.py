# the goal of this notebook is to perform welch's t-test on proportion of clusters in each dataset.
import pandas as pd 
import numpy as np
import interval

df_2011 = pd.read_csv('chapter4clustering/outputs/2011_reduced_set_proportions.csv')
df_2021 = pd.read_csv('chapter4clustering/outputs/2021_reduced_set_proportions.csv')

# remove those with low spatial sampling 
certainty = pd.read_csv('chapter3data/outputs/sampling_rate_lsoa_all_years_sql.csv')
mask = df_2011.merge(certainty, left_on='lsoa', right_on='lsoa11cd')
masked = mask[(mask['2021'] > 0.5) & (mask['2011'] > 0.5)] # 4834 - 4632
df_2011 = masked.drop(['lsoa11cd', '2011', '2018', '2021', 'roads', 'lsoa11cd.1', 'lsoa'], axis=1)

mask = df_2021.merge(certainty, left_on='lsoa', right_on='lsoa11cd')
masked = mask[(mask['2021'] > 0.5) & (mask['2011'] > 0.5)] # 4834 - 4632
df_2021 = masked.drop(['lsoa11cd', '2011', '2018', '2021', 'roads', 'lsoa11cd.1', 'lsoa'], axis=1)

def get_sample(n,p,size):
    sim = np.random.binomial(n=n,p=p,size=size)
    return np.std(100*sim/n)

std_err = {}
size = 10000
for feature in df_2011.columns[1:-1]:
    p = df_2011[feature]
    n = df_2011['count']
    df = df_2011[[feature, 'count']]
    df.columns = ['features', 'counts']
    err = df.apply(lambda x: get_sample(x.counts,x.features,size), axis=1 )
    std_err[feature] = list(err)
err_df_2011 = pd.DataFrame(std_err)

std_err = {}
size = 10000
for feature in df_2021.columns[1:-1]:
    p = df_2021[feature]
    n = df_2021['count']
    df = df_2021[[feature, 'count']]
    df.columns = ['features', 'counts']
    err = df.apply(lambda x: get_sample(x.counts,x.features,size), axis=1 )
    std_err[feature] = list(err)
err_df_2021 = pd.DataFrame(std_err)

df_2011.reset_index(drop=True, inplace=True)
df_2021.reset_index(drop=True, inplace=True)

import portion as P

std_int_2011 = {}
for feature in df_2011.columns[1:-1]:
    p = df_2011[feature]*100
    err = err_df_2011[feature]
    concat = pd.concat([p,err], axis=1)
    concat.columns = ['p', 'err']
    concat['int'] = concat.apply(lambda x: P.closed(x.p-1.96*x.err, x.p+1.96*x.err), axis=1 )
    std_int_2011[feature] = concat['int']
int_2011 = pd.DataFrame(std_int_2011)
int_2011.columns = [i.split('_')[-1] for i in list(err_df_2021.columns)]

std_int_2021 = {}
for feature in df_2021.columns[1:-1]:
    p = df_2021[feature]*100
    err = err_df_2021[feature]
    concat = pd.concat([p,err], axis=1)
    concat.columns = ['p', 'err']
    concat['int'] = concat.apply(lambda x: P.closed(x.p-1.96*x.err, x.p+1.96*x.err), axis=1 )
    std_int_2021[feature] = concat['int']
int_2021 = pd.DataFrame(std_int_2021)
int_2021.columns = [i.split('_')[-1] for i in list(err_df_2021.columns)]

sig = {}
for feature in int_2021.columns:
    x = int_2011[feature]
    y = int_2021[feature]
    concat = pd.concat([x,y], axis=1)
    concat.columns = ['x','y']
    concat['overlap'] = concat.apply(lambda x: x.x.overlaps(x.y), axis=1 )
    sig[feature] = concat['overlap']
sig_df = 1 - pd.DataFrame(sig)

sig_df['lsoa'] = df_2011['LSOA11CD']

sig_df.to_csv('chapter4clustering/outputs/significant_temporal_changes_reduced_set.csv')

# Commercial                                                     96
# Estates                                                       339
# Green space                                                   254
# High-density                                                  112
# Low-density                                                   762

change = {}
features = list(sig_df.columns[:-1])
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    change[new_column] = df_2021[t1_column]*100 -df_2011[t0_column]*100
change_df = pd.DataFrame(change)
change_df['lsoa'] = df_2011['LSOA11CD']

change_df.to_csv('chapter4clustering/outputs/cluster_proportions_reduced_set.csv')

