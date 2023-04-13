import pandas as pd 

# read in perceptions of wealth for 2021
prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home/emily/phd/'
perceptions = ['safety', 'wealth', 'boring', 'depressing', 'lively', 'walk', 'beauty']

dfs = []
for perception in perceptions:
    perceptions_oa = pd.read_csv(prefix + '006_place_pulse/place-pulse-2.0/outputs/classification/census/2021/pretrained_%s_new_plots_predictions.csv' % perception)
    perceptions_oa['idx'] = perceptions_oa['0'].apply(lambda x: int(x[:-6]))
    # merge perception scores to LS output areas
    df_2021_oa = pd.read_csv('chapter3data/outputs/2021_panoids_merged_to_oa.csv')
    merged = perceptions_oa.merge(df_2021_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
    oa = pd.read_csv('/home/emily/phd/002_validation/source/oa/OA_2011_London_gen_MHW_4326_all_fields.csv')
    merged = merged.merge(oa[['OA11CD', 'LSOA11CD']], left_on='oa_name', right_on='OA11CD' )
    wealth_oa = merged.groupby('LSOA11CD').mean()
    #wealth_oa['lsoa'] = wealth_oa.index
    dfs.append(wealth_oa['Unnamed: 0'])

df = pd.concat(dfs, axis=1)
df.columns = perceptions 
df['lsoa'] = df.index

df.to_csv('chapter6substantive/outputs/2021_all_perceptions_lsoa.csv')

