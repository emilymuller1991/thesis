import pandas as pd 

# WEALTH PERCEPTION TO INCOME DEPRIVATION SCORE
file_name = '2019_deprivation_scores_london'
file_to_merge = pd.read_csv('chapter6substantive/outputs/%s.csv' % file_name)

# read perception file
perceptions = pd.read_csv('chapter6substantive/outputs/2021_all_perceptions_lsoa.csv')

merged = file_to_merge.merge(perceptions, left_on='LSOA code (2011)', right_on='lsoa')
merged = merged.drop(['Unnamed: 0', 'LSOA code (2011)', 'LSOA11CD', 'lsoa'], axis=1)
merged.to_csv('chapter6substantive/outputs/deprivation_merge_perceptions.csv')

normalized_df=(merged-merged.mean())/merged.std()
# normalized_df=(df-df.min())/(df.max()-df.min())
normalized_df.to_csv('chapter6substantive/outputs/deprivation_merge_perceptions_norm.csv')

cor = merged.corr()[0:9][['safety', 'wealth', 'boring', 'depressing', 'lively', 'walk', 'beauty']]

cor.to_csv('chapter6substantive/outputs/deprivation_merge_perceptions_corr.csv')