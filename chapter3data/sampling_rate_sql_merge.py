# this file calculates the number of sampled points in an output area as a percentage.
import pandas as pd

# import all street points merged to output areas
points = pd.read_csv('outputs/road_points_merged_to_oa.csv')

# original_points 
points_grouped = points[['idx','oa_name']].groupby('oa_name').count()

# get merged panoid oa counts
df_2011_oa = pd.read_csv('outputs/2011_panoids_merged_to_oa.csv')
df_2018_oa = pd.read_csv('outputs/2018_panoids_merged_to_oa.csv')
df_2021_oa = pd.read_csv('outputs/2021_panoids_merged_to_oa.csv')

# get grouped counts
df_2011_oa_grouped = df_2011_oa[['idx','oa_name']].groupby('oa_name').count()
df_2018_oa_grouped = df_2018_oa[['idx','oa_name']].groupby('oa_name').count()
df_2021_oa_grouped = df_2021_oa[['idx','oa_name']].groupby('oa_name').count()

# GROUND TRUTH OF ALL OUPUT AREAS
oa = pd.read_csv('/home/emily/phd/002_validation/source/oa/OA_2011_London_gen_MHW_4326.csv')

merged = oa.merge(df_2011_oa_grouped, left_on='oa11cd', right_on='oa_name', how='left')
merged = merged.merge(df_2018_oa_grouped, left_on='oa11cd', right_on='oa_name',how='left')
merged = merged.merge(df_2021_oa_grouped, left_on='oa11cd', right_on='oa_name',how='left')
merged = merged.merge(points_grouped, left_on='oa11cd', right_on='oa_name',how='left')
merged.columns = ['objectid','oa11cd','lad11cd','st_areasha','st_lengths','2011','2018','2021','roads']

merged_proportions = merged[['2011','2018','2021','roads']].apply(lambda x: x/x.roads, axis=1)
merged_proportions['oa11cd'] = merged['oa11cd']
merged_proportions.to_csv('outputs/sampling_rate_oa_all_years_sql.csv')

import matplotlib.pyplot as plt 

################################## OUTPUT AREA PLOTS
plt.clf()
merged.boxplot(column=['2011', '2018', '2021'])
plt.ylim(merged[['2011', '2018', '2021']].min()[0],merged[['2011', '2018', '2021']].max()[0])
plt.xlabel('Year')
plt.ylabel('Images per OA')
plt.savefig('outputs/images_per_pa_boxplots_sql.png',bbox_inches="tight", dpi=150)

plt.clf()
merged.boxplot(column=['2011', '2018', '2021'], showfliers=False)  
plt.ylim(0,80)
plt.xlabel('Year')
plt.ylabel('Images per OA')
plt.savefig('outputs/images_per_pa_boxplots_zoom_sql.png',bbox_inches="tight", dpi=150)

plt.clf()
merged_proportions.boxplot(column=['2011', '2018', '2021'])
plt.ylim(merged_proportions.min()[0],merged_proportions.max()[0])
plt.xlabel('Year')
plt.ylabel('Sampling Multiplier')
plt.savefig('outputs/sampling_rate_boxplots_sql.png',bbox_inches="tight", dpi=150)

plt.clf()
merged_proportions.boxplot(column=['2011', '2018', '2021'], showfliers=False)  
plt.ylim(0,2)
plt.xlabel('Year')
plt.ylabel('Sampling Multiplier')
plt.savefig('outputs/sampling_rate_boxplots_zoom_sql.png',bbox_inches="tight", dpi=150)

# counts in each category 
pd.cut(merged_proportions['2011'],[-1,0,0.5,0.75,1,200]).value_counts()
merged_proportions.hist(bins=80)

################################################### PER LSOA
oa = pd.read_csv('/home/emily/phd/002_validation/source/oa/OA_2011_London_gen_MHW_4326_all_fields.csv')

merged = oa[['OA11CD', 'LSOA11CD']].merge(df_2011_oa_grouped, left_on='OA11CD', right_on='oa_name', how='left')
merged = merged.merge(df_2018_oa_grouped, left_on='OA11CD', right_on='oa_name',how='left')
merged = merged.merge(df_2021_oa_grouped, left_on='OA11CD', right_on='oa_name',how='left')
merged = merged.merge(points_grouped, left_on='OA11CD', right_on='oa_name',how='left')
merged.columns = ['oa11cd','lsoa11cd','2011','2018','2021','roads']
merged = merged.groupby('lsoa11cd').sum()

merged_proportions = merged[['2011','2018','2021','roads']].apply(lambda x: x/x.roads, axis=1)
merged_proportions['lsoa11cd'] = merged.index
merged_proportions.to_csv('outputs/sampling_rate_lsoa_all_years_sql.csv')

################################## LOWER SUPER OUTPUT AREA PLOTS
plt.clf()
merged.boxplot(column=['2011', '2018', '2021'])
plt.ylim(merged[['2011', '2018', '2021']].min()[0],merged[['2011', '2018', '2021']].max()[0])
plt.xlabel('Year')
plt.ylabel('Images per LSOA')
plt.savefig('outputs/images_per_lsoa_boxplots_sql.png',bbox_inches="tight", dpi=150)

plt.clf()
merged.boxplot(column=['2011', '2018', '2021'], showfliers=False)  
plt.ylim(0,350)
plt.xlabel('Year')
plt.ylabel('Images per LSOA')
plt.savefig('outputs/images_per_lsoa_boxplots_zoom_sql.png',bbox_inches="tight", dpi=150)

plt.clf()
merged_proportions.boxplot(column=['2011', '2018', '2021'])
plt.ylim(merged_proportions.min()[0],merged_proportions.max()[0])
plt.xlabel('Year')
plt.ylabel('Sampling Multiplier')
plt.savefig('outputs/sampling_rate_lsoa_boxplots_sql.png',bbox_inches="tight", dpi=150)

plt.clf()
merged_proportions.boxplot(column=['2011', '2018', '2021'], showfliers=False)  
plt.ylim(0,1.5)
plt.xlabel('Year')
plt.ylabel('Sampling Multiplier')
plt.savefig('outputs/sampling_rate_lsoa_boxplots_zoom_sql.png',bbox_inches="tight", dpi=150)