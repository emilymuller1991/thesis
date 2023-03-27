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

# merge to metadata and stratify by month
meta_2011 = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2011/census_2011_image_meta_double.csv')
meta_2018 = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/2018/greater_london_2018_panoids_to_download.csv')
meta_2021 = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2021/census_2021_image_meta_double.csv')

df_2011['idx_y'] = df_2011['id'].apply(lambda x: x[:-4])
df_2021['idx_y'] = df_2021['id'].apply(lambda x: x[:-4])
meta_merge_2011 = meta_2011[['month', 'idx_y']].merge(df_2011, on='idx_y')
meta_merge_2021 = meta_2021[['month', 'idx_y']].merge(df_2021, on='idx_y')

for perception in perceptions:
    meta_merge_2011['decile'] = pd.qcut(meta_merge_2011[perception], 10, labels=np.arange(10))
    month_2011 = meta_merge_2011[['month', 'decile', 'idx_y']].groupby(['month', 'decile']).count()
    month_2011 = month_2011.reset_index().pivot(columns='decile', index='month', values='idx_y')
    month_2011 = month_2011/month_2011.sum(axis=0)
    month_2011.to_csv('chapter5perceptions/outputs/R/2011_%s_perception_by_month.csv' % perception)

    meta_merge_2021['decile'] = pd.qcut(meta_merge_2021[perception], 10, labels=np.arange(10))
    month_2021 = meta_merge_2021[['month', 'decile', 'idx_y']].groupby(['month', 'decile']).count()
    month_2021 = month_2021.reset_index().pivot(columns='decile', index='month', values='idx_y')
    month_2021 = month_2021/month_2021.sum(axis=0)
    month_2021.to_csv('chapter5perceptions/outputs/R/2021_%s_perception_by_month.csv' % perception)

meta_merge_2011['year'] = np.repeat('2011', meta_merge_2011.shape[0])
meta_merge_2021['year'] = np.repeat('2021', meta_merge_2021.shape[0])

meta_merge = pd.concat([meta_merge_2011, meta_merge_2021])
for perception in perceptions:
    meta_merge['decile'] = pd.qcut(meta_merge[perception], 2, labels=np.arange(2))
    month = meta_merge[['decile', 'idx_y','year']].groupby(['decile','year']).count()
    month = month.reset_index().pivot(columns='decile', index=[ 'year'], values='idx_y')
    month = month/month.sum(axis=0)
    print (perception, month)
    #month.to_csv('chapter5perceptions/outputs/R/both_years_%s_perception_by_month.csv' % perception)
