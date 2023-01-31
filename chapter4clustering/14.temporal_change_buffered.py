# the goal of this notebook is to perform welch's t-test on proportion of clusters in each dataset.
import pandas as pd 
import numpy as np
import interval

df_2011 = pd.read_csv('chapter4clustering/outputs/2011_reduced_set_proportions_merged_buffer_dropna.csv')
df_2021 = pd.read_csv('chapter4clustering/outputs/2021_reduced_set_proportions_merged_buffer_dropna.csv')

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
    std_err[feature] = list(err)100100
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

############################################################ CALCULATING WELCH's t-test significance
from scipy import stats
def get_sig(mu_1, mu_2, sd_1, sd_2, n_1, n_2):
    # sd_1_corrected = sd_1/np.sqrt(n_1)
    # sd_2_corrected = sd_2/np.sqrt(n_2)
    t = (mu_1 - mu_2) / np.sqrt((sd_1**2/n_1) + (sd_2**2/n_2))
    df_num = ((sd_1**2/n_1) + (sd_2**2/n_2))**2
    df_denom_1 = sd_1**4/(n_1**2*(n_1-1)) 
    df_denom_2 = sd_2**4/(n_2**2*(n_2-1))
    df = df_num/(df_denom_1+df_denom_2)
    p = stats.t.sf(np.abs(t), df=df)
    return t, p, df

welch = {}
col = [i.split('_')[-1] for i in list(err_df_2021.columns)]
for feature in col:
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    mu_0 = df_2011[t0_column]
    sd_0 = err_df_2011[t0_column]
    n_0 = df_2011['count']
    mu_1 = df_2021[t1_column]
    sd_1 = err_df_2021[t1_column]
    n_1 = df_2021['count']
    df = pd.DataFrame({'mu_0': mu_0*100, 'mu_1': mu_1*100, 'sd_0': sd_0, 'sd_1': sd_1, 'n_0': n_0, 'n_1': n_1})
    df['t,p,d'] = df.apply(lambda x: get_sig(x.mu_0, x.mu_1, x.sd_0, x.sd_1, x.mu_0*x.n_0/100, x.mu_1*x.n_1/100), axis=1)
    welch[feature] = list(df['t,p,d'].apply(lambda x: x[1]))

welch_df = pd.DataFrame(welch)


from scipy import stats

t_score = stats.ttest_ind_from_stats(mean1=df.iloc[4112]['mu_0'], std1=df.iloc[4112]['sd_0'], nobs1=10000, \
                               mean2=df.iloc[4112]['mu_1'], std2=df.iloc[4112]['sd_1'], nobs2=10000, \
                               equal_var=False)
t_score


############################################################CALCULATING OVERLAPPING INTERVALS
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
sig_df.to_csv('chapter4clustering/outputs/significant_temporal_changes_reduced_set_merged_buffer.csv')

# Commercial                                                     96
# Estates                                                       339
# Green space                                                   254
# High-density                                                  112
# Low-density                                                   762

change = {}
features = list(sig_df.columns)
for feature in features:
    new_column = 'proportion_%s' % feature
    t0_column = 'clusters_2011_cleanup_%s' % feature
    t1_column = 'clusters_2021_cleanup_%s' % feature
    change[new_column] = df_2021[t1_column]*100 -df_2011[t0_column]*100
change_df = pd.DataFrame(change)
change_df['lsoa'] = df_2011['LSOA11CD']

change_df.to_csv('chapter4clustering/outputs/cluster_proportions_reduced_set.csv')

masked = change_df[sig_df]
masked['lsoa'] = df_2011['LSOA11CD']
masked.to_csv('chapter4clustering/outputs/cluster_masked_reduced_set.csv')
