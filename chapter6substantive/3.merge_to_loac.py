import pandas as pd 

file_name = '2019_deprivation_scores_london'
file_to_merge = pd.read_csv('chapter6substantive/outputs/%s.csv' % file_name)
# IMAGE BASED VALUES
df_8 =  pd.read_csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/both_years_proportions_all_nonna_lsoa_hierarchical8.csv')
df_13 = pd.read_csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/both_years_proportions_all_nonna_lsoa_hierarchical13.csv')
df_24 = pd.read_csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/both_years_proportions_all_nonna_lsoa_hierarchical24.csv')

df_8 = df_8[df_8['same'] == 1]
df_13 = df_13[df_13['same'] == 1]
df_24 = df_24[df_24['same'] == 1]

df_8.columns = ['Unnamed:0', 'lsoa', 'hierarchical8x', 'hierarchical8y', 'same']
df_13.columns = ['Unnamed:0', 'lsoa', 'hierarchical13x', 'hierarchical13y', 'same']
df_24.columns = ['Unnamed:0', 'lsoa', 'hierarchical24x', 'hierarchical24y', 'same']

merged = df_dep_london_2019.merge(df_8[['lsoa', 'hierarchical8x']], left_on='LSOA code (2011)', right_on='lsoa', how='left')
merged = merged.merge(df_13[['lsoa', 'hierarchical13x']], left_on='LSOA code (2011)', right_on='lsoa', how='left')
merged = merged.merge(df_24[['lsoa', 'hierarchical24x']], left_on='LSOA code (2011)', right_on='lsoa', how='left')

file_to_merge.to_csv('chapter6substantive/outputs/%s_london_hierarchicalx.csv' % file_name)