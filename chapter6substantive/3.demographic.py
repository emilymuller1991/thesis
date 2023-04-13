import pandas as pd 

london_lsoa = pd.read_csv('chapter6substantive/london_lsoa.csv')

# population density 2011
df = pd.read_csv('chapter6substantive/source/2011_population.csv')
df_london = df[df['LSOA Code'].isin(list(london_lsoa['lsoa']))]
df_london_2011 = df_london[['LSOA Code', 'Persons per hectare']]

df = pd.read_csv('chapter6substantive/source/2021_population.csv')
df_london = df[df['Lower Layer Super Output Areas Code'].isin(list(london_lsoa['lsoa']))]
df_london_2021 = df_london[['Lower Layer Super Output Areas Code', 'Population Density']]

merged = df_london_2011.merge(df_london_2021, left_on='LSOA Code', right_on='Lower Layer Super Output Areas Code')
merged.columns = ['lsoa', 'popden2011', 'lsoa_x', 'popden2021']
merged = merged.drop(['lsoa_x'], axis=1)
merged['popden2021'] = merged['popden2021'].apply( lambda x: x/100)

merged.to_csv('chapter6substantive/pop_density_years.csv')

# share of the population 
df = pd.read_csv('chapter6substantive/source/2011_share_pop.csv')
df['0-14'] = df['0-4'] + df['5-9'] + df['10-14']
df['15-29'] = df['15-19'] + df['20-24'] + df['25-29']
df['30-69'] = df['30-34'] + df['35-39'] + df['40-44'] + df['45-49'] + df['50-54'] + df['55-59'] + df['60-64'] + df['65-69']
df['70+'] = df['70-74'] + df['75-79'] + df['80-84'] + df['85-89'] + df['90+']
df_share_2011 = df[['Area Codes', 'All Ages', '0-14', '15-29', '30-69', '70+']]
df_london_share_2011 = df_share_2011[df_share_2011['Area Codes'].isin(list(london_lsoa['lsoa']))]
df_london_share_2011.columns = ['lsoa', 'all_2011', '0-14_2011', '15-29_2011', '30-69_2011', '70+_2011']

df = pd.read_csv('chapter6substantive/source/2020_share_pop.csv', encoding='latin1')
df['0-14'] = sum([df[str(i)] for i in range(15)])
df['15-29'] = sum([df[str(i)] for i in range(15,30,1)])
df['30-69'] = sum([df[str(i)] for i in range(30,70,1)])
df['70+'] = sum([df[str(i)] for i in range(70,90,1)]) + df['90+']
df_share_2020 = df[['LSOA Code', 'All Ages', '0-14', '15-29', '30-69', '70+']]
df_london_share_2020 = df_share_2020[df_share_2020['LSOA Code'].isin(list(london_lsoa['lsoa']))]
df_london_share_2020.columns = ['lsoa', 'all_2020', '0-14_2020', '15-29_2020', '30-69_2020', '70+_2020']
merged = df_london_share_2011.merge(df_london_share_2020, on='lsoa')

merged.to_csv('chapter6substantive/share_pop_years.csv')

# deprivation
df = pd.read_csv('chapter6substantive/source/2010_deprivation.csv')
df_dep_2010 = df[['LSOA CODE', 'LIVING ENVIRONMENT SCORE', 'CRIME AND DISORDER SCORE', 'HEALTH DEPRIVATION AND DISABILITY SCORE', 'BARRIERS TO HOUSING AND SERVICES SCORE', 'EDUCATION SKILLS AND TRAINING SCORE', 'EMPLOYMENT SCORE', 'INCOME SCORE', 'IMD SCORE', 'Outdoors Sub-domain Score']]
df_dep_london_2010 = df_dep_2010[df_dep_2010['LSOA CODE'].isin(list(london_lsoa['lsoa']))]
df_dep_london_2010.to_csv('chapter6substantive/2010_deprivation_scores_london.csv')

df = pd.read_csv('chapter6substantive/source/2019_deprivation_scores.csv')
df_dep_2019 = df[['LSOA code (2011)', 'Living Environment Score', 'Crime Score', 'Health Deprivation and Disability Score', 'Barriers to Housing and Services Score', 'Education, Skills and Training Score', 'Employment Score (rate)', 'Income Score (rate)', 'Index of Multiple Deprivation (IMD) Score', 'Outdoors Sub-domain Score']]
df_dep_london_2019 = df_dep_2019[df_dep_2019['LSOA code (2011)'].isin(list(london_lsoa['lsoa']))]
df_dep_london_2019.to_csv('chapter6substantive/2019_deprivation_scores_london.csv')

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
df_dep_london_2019.to_csv('chapter6substantive/2019_deprivation_scores_london_hierarchicalx.csv')

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
