import pandas as pd 
import numpy as np

df_2018 = pd.read_csv('/home/emily/phd/2_interpretability/2018_all_variables.csv')

prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home/emily/phd/'
perceptions = ['beauty', 'boring', 'lively', 'depressing', 'walk', 'safety', 'wealth']
df_2011 = pd.DataFrame()
df_2021 = pd.DataFrame()
for perception in perceptions:
    df_2011_ = pd.read_csv(prefix + '006_place_pulse/place-pulse-2.0/outputs/classification/census/2011/pretrained_%s_new_plots_predictions.csv' % perception)
    df_2021_ = pd.read_csv(prefix + '006_place_pulse/place-pulse-2.0/outputs/classification/census/2021/pretrained_%s_new_plots_predictions.csv' % perception)
    df_2011_.columns = [perception, 'id']
    df_2021_.columns = [perception, 'id']
    df_2011 = pd.concat([df_2011, df_2011_], axis=1)
    df_2021 = pd.concat([df_2021, df_2021_], axis=1)

# pandas drop duplicate columns
df_2011 = df_2011.loc[:,~df_2011.columns.duplicated()].copy()
df_2021 = df_2021.loc[:,~df_2021.columns.duplicated()].copy()

# dropna values
df_2011 = df_2011[~df_2011.isna().any(axis=1)]
df_2018 = df_2018[~df_2018.isna().any(axis=1)]
df_2021 = df_2021[~df_2021.isna().any(axis=1)]

# get images merged to output areas
df_2011_oa = pd.read_csv('chapter3data/outputs/2011_panoids_merged_to_oa.csv')
df_2018_oa = pd.read_csv('chapter3data/outputs/2018_panoids_merged_to_oa.csv')
df_2021_oa = pd.read_csv('chapter3data/outputs/2021_panoids_merged_to_oa.csv')

# format idx column
df_2011['idx'] = df_2011['id'].apply(lambda x: int(x[:-6]))
df_2021['idx'] = df_2021['id'].apply(lambda x: int(x[:-6]))
# df_2018 = df_2018[['idx', perception]]

# merge perception scores to output areas
df_2011_oa = df_2011.merge(df_2011_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2021_oa = df_2021.merge(df_2021_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )

# merge to lsoa
oa = pd.read_csv('/home/emily/phd/002_validation/source/oa/OA_2011_London_gen_MHW_4326_all_fields.csv')
df_2011_oa = df_2011_oa.merge(oa[['OA11CD', 'LSOA11CD']], left_on='oa_name', right_on='OA11CD' )
df_2021_oa = df_2021_oa.merge(oa[['OA11CD', 'LSOA11CD']], left_on='oa_name', right_on='OA11CD' )

df_2011_oa = df_2011_oa.dropna()
df_2021_oa = df_2021_oa.dropna()
# df_2018_ = df_2018_.dropna()

df_2011_mean = df_2011_oa.groupby('LSOA11CD').mean()
# df_2018_mean = df_2018_oa[[perception, 'oa_name']].groupby('oa_name').mean()
df_2021_mean = df_2021_oa.groupby('LSOA11CD').mean()

# merge all values together into one dataframe
for perception in perceptions:
    name_2011 = 'decile_2011_%s' % perception
    name_2021 = 'decile_2021_%s' % perception
    df_2011_mean[name_2011] = pd.qcut(df_2011_mean[perception], 10, labels=np.arange(10)).astype(int)
    # df_2018_mean['deciles_2018'] = pd.qcut(df_2018_mean[perception], 10, labels=np.arange(10))
    df_2021_mean[name_2021] = pd.qcut(df_2021_mean[perception], 10, labels=np.arange(10)).astype(int)

    df_2011_mean['oa'] = df_2011_mean.index
    # df_2018_mean['oa'] = df_2018_mean.index
    df_2021_mean['oa'] = df_2021_mean.index

    # merge by oa 
    merged = df_2011_mean.merge(df_2021_mean, on='oa')
    # merged = merged.merge(df_2021_mean, on='oa')
    merged['count'] = np.ones(merged.shape[0])
    # heatmap
    pivot = pd.pivot_table(merged[['oa',name_2011,name_2021,'count']], values='count', index=name_2011, columns=name_2021,
                   aggfunc='count').reset_index()
    
    # get totals
    x = np.array(pivot[pivot.columns[1:]])
    equal = x.diagonal().sum()/np.nansum(x)
    pm1 = (x.diagonal(offset=-1).sum() + x.diagonal(offset=1).sum() + x.diagonal().sum())/np.nansum(x)
    pm2 = (x.diagonal(offset=-2).sum() + x.diagonal(offset=2).sum() + x.diagonal(offset=-1).sum() + x.diagonal(offset=1).sum() + x.diagonal().sum())/np.nansum(x)

    ratio = np.nansum(np.triu(x,1))/np.nansum(np.tril(x,-1))*100

    print (perception, equal, pm1, pm2, 100 -ratio)


    #pivot.to_csv('chapter5perceptions/outputs/R/lsa_change_in_deciles_%s_perception.csv' % perception)

for perception in perceptions:
    name = 'decile_change_%s' % perception
    name_2011 = 'decile_2011_%s' % perception
    name_2021 = 'decile_2021_%s' % perception
    merged[name] = merged[name_2021].astype(int) - merged[name_2011].astype(int)

merged.to_csv('chapter5perceptions/outputs/lsoa_change_in_deciles.csv')

#### DECILE BOTH YEARS TOGETHER
df_both_mean = pd.concat([df_2011_mean[perceptions], df_2021_mean[perceptions]])
# merge all values together into one dataframe
for perception in perceptions:
    name = 'decile_%s' % perception
    df_both_mean[name] = pd.qcut(df_both_mean[perception], 10, labels=np.arange(10)).astype(int)

    df_both_mean['oa'] = df_both_mean.index

    df_2011_mean = df_both_mean.iloc[0:df_2011_mean.shape[0]]
    df_2021_mean = df_both_mean.iloc[df_2011_mean.shape[0]:df_2011_mean.shape[0]*2]
    # merge by oa 
    merged = df_2011_mean.merge(df_2021_mean, on='oa')
    # merged = merged.merge(df_2021_mean, on='oa')
    merged['count'] = np.ones(merged.shape[0])
    # heatmap
    name_2011 = name + '_x'
    name_2021 = name + '_y'
    pivot = pd.pivot_table(merged[['oa',name_2011,name_2021,'count']], values='count', index=name_2011, columns=name_2021,
                   aggfunc='count').reset_index()
    pivot.to_csv('chapter5perceptions/outputs/R/lsoa_change_in_deciles_BOTH_%s_perception.csv' % perception)