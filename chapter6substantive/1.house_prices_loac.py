# environmental exposure across urban typologies
import pandas as pd 

# read in urban classifications
df_8 =  pd.read_csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/both_years_proportions_all_nonna_lsoa_hierarchical8.csv')
df_13 = pd.read_csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/both_years_proportions_all_nonna_lsoa_hierarchical13.csv')
df_24 = pd.read_csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/both_years_proportions_all_nonna_lsoa_hierarchical24.csv')

df_8 = df_8[df_8['same'] == 1]
df_13 = df_13[df_13['same'] == 1]
df_24 = df_24[df_24['same'] == 1]

df_8.columns = ['Unnamed:0', 'lsoa', 'hierarchical8x', 'hierarchical8y', 'same']
df_13.columns = ['Unnamed:0', 'lsoa', 'hierarchical13x', 'hierarchical13y', 'same']
df_24.columns = ['Unnamed:0', 'lsoa', 'hierarchical24x', 'hierarchical24y', 'same']

# house prices
import numpy as np
house_prices = pd.read_csv('/home/emily/phd/drives/phd/chapter6substantive/source/house_prices.csv')
house_prices['price'] = house_prices['Price (central estimate, natural log scale)'].apply(lambda x: np.exp(x))
house_prices = house_prices[house_prices['Year'] == 2019]

exp = house_prices.groupby('LSOA code').mean()
exp['lsoa'] = exp.index

merged = exp[['lsoa','price']].merge(df_8[['lsoa', 'hierarchical8x']], left_on='lsoa', right_on='lsoa', how='left')
merged = merged.merge(df_13[['lsoa', 'hierarchical13x']], left_on='lsoa', right_on='lsoa', how='left')
merged = merged.merge(df_24[['lsoa', 'hierarchical24x']], left_on='lsoa', right_on='lsoa', how='left')

merged.to_csv('/home/emily/phd/drives/phd/chapter3data/outputs/house_prices_loac_hierarchicalx.csv')
