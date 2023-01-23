# this file calculates the number of sampled points in an output area as a percentage.

import pandas as pd

# import all street points merged to output areas
points_merged = pd.read_csv('/home/emily/phd/0_get_images/outputs/roads/gla_roads_points_20m_merge_to_oa.csv')

# original_points 
points_grouped = points_merged[['fid','OA11CD']].groupby('OA11CD').count()

# get merged panoid oa counts
df_2011_oa = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2011/census_2011_image_meta_single_merged_oa.csv')
df_2018_oa = pd.read_csv('/home/emily/phd/2_interpretability/2018_all_perception_scores_merged_oa.csv')
df_2021_oa = pd.read_csv('/home/emily/phd/0_get_images/outputs/psql/census_2021/census_2021_image_meta_single_merged_oa.csv')

df_2018_oa['idx'] = df_2018_oa['Unnamed: 0.1.1'].apply(lambda x: x[:-6])
df_2018_oa = df_2018_oa[['OA11CD', 'idx']].drop_duplicates()

df_2011_oa_grouped = df_2011_oa[['OA11CD', 'id']].groupby('OA11CD').count()
df_2018_oa_grouped = df_2018_oa[['OA11CD', 'idx']].groupby('OA11CD').count()
df_2021_oa_grouped = df_2021_oa[['OA11CD', 'id']].groupby('OA11CD').count()

merged = points_grouped.merge(df_2011_oa_grouped, on='OA11CD')
merged = merged.merge(df_2018_oa_grouped, on='OA11CD')
merged = merged.merge(df_2021_oa_grouped, on='OA11CD')

merged_proportions = merged.apply(lambda x: x/x.fid, axis=1)
merged_proportions.columns = ['points', '2011', '2018', '2021']
merged_proportions.to_csv('outputs/sampling_rate_oa_all_years.csv')

import matplotlib.pyplot as plt
merged_proportions.boxplot(column=['2011', '2018', '2021'], showfliers=False)  
plt.ylim(0,10)
plt.xlabel('Year')
plt.ylabel('Sampling Multiplier')
plt.savefig('outputs/sampling_rate_boxplots.png',bbox_inches="tight", dpi=150)

plt.clf()
merged_proportions.boxplot(column=['2011', '2018', '2021'])  
plt.ylim(0,2)
plt.xlabel('Year')
plt.ylabel('Sampling Multiplier')
plt.savefig('outputs/sampling_rate_boxplots_zoom.png',bbox_inches="tight", dpi=150)