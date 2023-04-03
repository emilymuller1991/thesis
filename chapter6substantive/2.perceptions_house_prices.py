import pandas as pd 
import numpy as np

house_prices = pd.read_csv('/home/emily/phd/drives/phd/chapter6substantive/source/house_prices.csv')
house_prices['price'] = house_prices['Price (central estimate, natural log scale)'].apply(lambda x: np.exp(x))
house_prices = house_prices[house_prices['Year'] == 2019]

# read in perceptions of wealth for 2021
prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home/emily/phd/'
perception = 'wealth'
perceptions_oa = pd.read_csv(prefix + '006_place_pulse/place-pulse-2.0/outputs/classification/census/2021/pretrained_%s_new_plots_predictions.csv' % perception)
perceptions_oa['idx'] = perceptions_oa['0'].apply(lambda x: int(x[:-6]))

# merge perception scores to output areas
df_2021_oa = pd.read_csv('chapter3data/outputs/2021_panoids_merged_to_oa.csv')
merged = perceptions_oa.merge(df_2021_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )

wealth_oa = merged.groupby('oa_name').mean()
wealth_oa['oa'] = wealth_oa.index

merged = house_prices[['OA code','price','Price (central estimate, natural log scale)']].merge(wealth_oa[['oa', 'Unnamed: 0']], left_on='OA code', right_on='oa')
merged.to_csv('/home/emily/phd/drives/phd/chapter6substantive/outputs/house_prices_wealth_perception_oa.csv')

# merge perception scores to LS output areas
df_2021_oa = pd.read_csv('chapter3data/outputs/2021_panoids_merged_to_oa.csv')
merged = perceptions_oa.merge(df_2021_oa[['oa_name', 'idx']], left_on='idx', right_on='idx' )
oa = pd.read_csv('/home/emily/phd/002_validation/source/oa/OA_2011_London_gen_MHW_4326_all_fields.csv')
merged = merged.merge(oa[['OA11CD', 'LSOA11CD']], left_on='oa_name', right_on='OA11CD' )

wealth_oa = merged.groupby('LSOA11CD').mean()
wealth_oa['lsoa'] = wealth_oa.index

merged = house_prices[['LSOA code','price','Price (central estimate, natural log scale)']].merge(wealth_oa[['lsoa', 'Unnamed: 0']], left_on='LSOA code', right_on='lsoa')
merged_lsoa = merged[['LSOA code', 'price', 'Unnamed: 0']].groupby('LSOA code').mean()
merged_lsoa['Price (central estimate, natural log scale)'] = merged_lsoa['price'].apply(lambda x: np.log(x))
merged_lsoa.plot.scatter('Price (central estimate, natural log scale)', 'Unnamed: 0')

merged_lsoa.to_csv('/home/emily/phd/drives/phd/chapter6substantive/outputs/house_prices_wealth_perception_lsoa.csv')