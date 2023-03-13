import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib 

prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home'
prefix2 = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/project/pathways/live/Transferability/emily_phd_images/keras-rmac-clustering_output/2018_original/'

# df_2021 = pd.read_csv('chapter4clustering/outputs/df_2021_matched_by_centroid.csv')
# df_2011 = pd.read_csv('chapter4clustering/outputs/df_2011_matched_by_centroid.csv')
df_both_years = pd.read_csv(prefix + '/emily/phd/003_image_matching/keras_rmac-master/census2021outputs/census_2011_and_census_2021_zoom_matched.csv')
df_2021 = df_both_years[df_both_years['year'] == 2021][['Unnamed: 0', 'year', 'p', 'clusters', 'distance']]
df_2021['matched'] = df_2021['clusters']
df_2011 = df_both_years[df_both_years['year'] == 2011][['Unnamed: 0', 'year', 'p', 'clusters', 'distance']]
df_2011['matched'] = df_2011['clusters']
#df_2018 = pd.read_csv(prefix2 + '2018_rmac_feature_vector_clusters_20.csv', error_bad_lines=False, engine="python")


def cleanup(x):
    if x in [3,5,17]:
        return np.nan
    elif x in [4,8,9,14]:
        return 'Low-density'
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

# df_2018 = df_2018.loc[:,~df_2018.columns.duplicated()].copy()
df_2021['clusters_2021_cleanup'] = df_2021['matched'].apply(lambda x: cleanup(x))
df_2011['clusters_2011_cleanup'] = df_2011['matched'].apply(lambda x: cleanup(x))

# clean dataframe, keep single img identifier for merge to oa
df_2011 = df_2011[['Unnamed: 0', 'matched','p', 'clusters_2011_cleanup']]
df_2011['idx'] = df_2011['Unnamed: 0'].apply(lambda x : int(x[:-2]))
df_2021 = df_2021[['Unnamed: 0', 'matched','p', 'clusters_2021_cleanup']]
df_2021['idx'] = df_2021['Unnamed: 0'].apply(lambda x : int(x[:-2]))

##################################################################################### CREATE OUTPUT FILE FOR SPATIAL MERGE
df_2011_ = df_2011.dropna()
df_2021_ = df_2021.dropna()
################################################ PERCENTAGES MERGED TO LSOA
# get merged oa counts
df_2011_oa = pd.read_csv('chapter3data/outputs/2011_panoids_merged_to_oa.csv')
df_2018_oa = pd.read_csv('chapter3data/outputs/2018_panoids_merged_to_oa.csv')
df_2021_oa = pd.read_csv('chapter3data/outputs/2021_panoids_merged_to_oa.csv')

df_2011_ = df_2011_.merge(df_2011_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2011_ = df_2011_.drop_duplicates()
df_2021_ = df_2021_.merge(df_2021_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
df_2021_ = df_2021_.drop_duplicates()

# merge to lsoa
oa = pd.read_csv('/home/emily/phd/002_validation/source/oa/OA_2011_London_gen_MHW_4326_all_fields.csv')
df_2011_ = df_2011_.merge(oa[['OA11CD', 'LSOA11CD']], left_on='oa_name', right_on='OA11CD' )
df_2021_ = df_2021_.merge(oa[['OA11CD', 'LSOA11CD']], left_on='oa_name', right_on='OA11CD' )

df_2011_ = df_2011_.dropna()
df_2021_ = df_2021_.dropna()

################################################ METADATA
meta_2011 = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2011/census_2011_image_meta_double.csv')
meta_2021 = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2021/census_2021_image_meta_double.csv')

lsoas = ['E01033599', 'E01004737', 'E01003015', 'E01003940', 'E01003975', 'E01004280', 'E01004013', 'E01003184', 'E01003093', 'E01003051', 'E01004050', 'E01003336']

example_2021 = df_2021_[df_2021_['LSOA11CD'].isin(lsoas)].merge(meta_2021, left_on='Unnamed: 0', right_on='idx_y')
example_2011 = df_2011_[df_2011_['LSOA11CD'].isin(lsoas)].merge(meta_2011, left_on='Unnamed: 0', right_on='idx_y')

example_2011['angle'] = example_2011['Unnamed: 0_x'].apply(lambda x: x[-1:])
example_2021['angle'] = example_2021['Unnamed: 0_x'].apply(lambda x: x[-1:])

example_2021_b = example_2021[example_2021['angle'] == 'b']
example_2021_d = example_2021[example_2021['angle'] == 'd']
example_2011_b = example_2011[example_2011['angle'] == 'b']
example_2011_d = example_2011[example_2011['angle'] == 'd']

example_2021_b.to_csv('chapter4clustering/outputs/spots/estates_2021_b.csv')
example_2021_d.to_csv('chapter4clustering/outputs/spots/estates_2021_d.csv')
example_2011_b.to_csv('chapter4clustering/outputs/spots/estates_2011_b.csv')
example_2011_d.to_csv('chapter4clustering/outputs/spots/estates_2011_d.csv')

# 2948  E01003015  0.630252  0.487395  0.932773    1.0  E01003015
# 2984  E01003051  1.097561  0.902439  0.902439    1.0  E01003051
# 3026  E01003093  0.727273  0.704545  0.818182    1.0  E01003093
# 3116  E01003184  0.423077  0.423077  0.576923    1.0  E01003184
# 3262  E01003336  0.801653  0.801653  0.909091    1.0  E01003336
# 3852  E01003940  0.793651  0.492063  0.825397    1.0  E01003940
# 3883  E01003975  0.697479  0.596639  0.882353    1.0  E01003975
# 3917  E01004013  0.677419  0.685484  0.782258    1.0  E01004013
# 3952  E01004050  0.636905  0.636905  0.738095    1.0  E01004050
# 4174  E01004280  0.682692  0.653846  0.807692    1.0  E01004280
# 4616  E01004737  0.669903  0.407767  0.737864    1.0  E01004737
# 4786  E01033599  0.225806  0.161290  0.258065    1.0  E01033599