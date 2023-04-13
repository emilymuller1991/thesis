import pandas as pd 

# uncertainty mask 
mask = pd.read_csv('chapter3data/outputs/sampling_rate_lsoa_all_years_sql.csv')
masked = mask[mask['2021'] > 0.75]

res_wealth = pd.read_csv('chapter6substantive/outputs/wealth_perception_income_deprivation_residuals.csv')
res_boring = pd.read_csv('chapter6substantive/outputs/boring_perception_living_env_dep_residuals.csv')
res_depressing = pd.read_csv('chapter6substantive/outputs/depressing_perception_education_deprivation_residuals.csv')

res_wealth['res_dec'] = pd.qcut(res_wealth['resid'], 10, labels=range(10))
res_boring['res_dec'] = pd.qcut(res_boring['resid'], 10, labels=range(10))
res_depressing['res_dec'] = pd.qcut(res_depressing['resid'], 10, labels=range(10))

############################### ALL RESIDUALS
res_wealth = pd.read_csv('chapter6substantive/outputs/wealth_perception_income_deprivation_residuals.csv')
res_boring = pd.read_csv('chapter6substantive/outputs/boring_perception_living_env_dep_residuals.csv')
res_depressing = pd.read_csv('chapter6substantive/outputs/depressing_perception_education_deprivation_residuals.csv')

res_wealth['res_dec'] = pd.qcut(res_wealth['resid'], 10, labels=range(10))
res_boring['res_dec'] = pd.qcut(res_boring['resid'], 10, labels=range(10))
res_depressing['res_dec'] = pd.qcut(res_depressing['resid'], 10, labels=range(10))

res_wealth_bottom = res_wealth
res_boring_top = res_boring
res_depressing_top = res_depressing

intersection = set(res_wealth_bottom['LSOA.code..2011.']) & set(res_boring_top['LSOA.code..2011.']) & set(res_depressing_top['lsoa'])
intersection = list(intersection)
len(intersection)

res_wealth_sub = res_wealth[res_wealth['LSOA.code..2011.'].isin(intersection)]
res_boring_sub = res_boring[res_boring['LSOA.code..2011.'].isin(intersection)]
res_depressing_sub = res_depressing[res_depressing['lsoa'].isin(intersection)]
res_wealth_sub['Outdoors.Sub.domain.Score'] = res_boring_sub['Outdoors.Sub.domain.Score']
res_wealth_sub['outdoor_resid'] = res_boring_sub['resid']
res_wealth_sub['Education..Skills.and.Training.Score'] = res_depressing_sub['Education..Skills.and.Training.Score']

res_wealth_sub['wealth_resid'] = res_wealth_sub['resid']
res_wealth_sub['outdoor_resid'] = res_boring_sub['resid']
res_wealth_sub['education_resid'] = res_depressing_sub['resid']

res_wealth_sub_masked = res_wealth_sub[res_wealth_sub['LSOA.code..2011.'].isin(masked['lsoa11cd.1'])]
res_wealth_sub_masked.to_csv('chapter6substantive/outputs/intersecting_residuals_all_masked.csv')
##############################################################################

res_wealth_bottom = res_wealth[res_wealth['res_dec'] == 0] # 483
res_boring_top = res_boring[res_boring['res_dec'] == 9] #483
res_depressing_top = res_depressing[res_depressing['res_dec'] == 9]

intersection = set(res_wealth_bottom['LSOA.code..2011.']) & set(res_boring_top['LSOA.code..2011.']) & set(res_depressing_top['lsoa'])
intersection = list(intersection)
len(intersection)

res_wealth_sub = res_wealth[res_wealth['LSOA.code..2011.'].isin(intersection)]
res_boring_sub = res_boring[res_boring['LSOA.code..2011.'].isin(intersection)]
res_depressing_sub = res_depressing[res_depressing['lsoa'].isin(intersection)]
res_wealth_sub['Outdoors.Sub.domain.Score'] = res_boring_sub['Outdoors.Sub.domain.Score']
res_wealth_sub['outdoor_resid'] = res_boring_sub['resid']
res_wealth_sub['Education..Skills.and.Training.Score'] = res_depressing_sub['Education..Skills.and.Training.Score']

res_wealth_sub['wealth_resid'] = res_wealth_sub['resid']
res_wealth_sub['outdoor_resid'] = res_boring_sub['resid']
res_wealth_sub['education_resid'] = res_depressing_sub['resid']

res_wealth_sub_masked = res_wealth_sub[res_wealth_sub['LSOA.code..2011.'].isin(masked['lsoa11cd.1'])]
res_wealth_sub_masked.to_csv('chapter6substantive/outputs/intersecting_residuals_bottom_masked.csv')

########################################### top residuals
res_wealth_bottom = res_wealth[res_wealth['res_dec'] == 9] # 483
res_boring_top = res_boring[res_boring['res_dec'] == 0] #483
res_depressing_top = res_depressing[res_depressing['res_dec'] == 0]

intersection = set(res_wealth_bottom['LSOA.code..2011.']) & set(res_boring_top['LSOA.code..2011.']) & set(res_depressing_top['lsoa'])
intersection = list(intersection)
len(intersection)

res_wealth_sub = res_wealth[res_wealth['LSOA.code..2011.'].isin(intersection)]
res_boring_sub = res_boring[res_boring['LSOA.code..2011.'].isin(intersection)]
res_depressing_sub = res_depressing[res_depressing['lsoa'].isin(intersection)]
res_wealth_sub['Outdoors.Sub.domain.Score'] = res_boring_sub['Outdoors.Sub.domain.Score']
res_wealth_sub['Education..Skills.and.Training.Score'] = res_depressing_sub['Education..Skills.and.Training.Score']

res_wealth_sub['wealth_resid'] = res_wealth_sub['resid']
res_wealth_sub['outdoor_resid'] = res_boring_sub['resid']
res_wealth_sub['education_resid'] = res_depressing_sub['resid']

res_wealth_sub_masked = res_wealth_sub[res_wealth_sub['LSOA.code..2011.'].isin(masked['lsoa11cd.1'])]
res_wealth_sub_masked.to_csv('chapter6substantive/outputs/intersecting_residuals_top_masked.csv')

########################################### mid residuals
res_wealth_bottom = res_wealth[res_wealth['res_dec'].isin([4,5,6])] # 483
res_boring_top = res_boring[res_boring['res_dec'].isin([4,5,6])] #483
res_depressing_top = res_depressing[res_depressing['res_dec'].isin([4,5,6])]

intersection = set(res_wealth_bottom['LSOA.code..2011.']) & set(res_boring_top['LSOA.code..2011.']) & set(res_depressing_top['lsoa'])
intersection = list(intersection)
len(intersection)

res_wealth_sub = res_wealth[res_wealth['LSOA.code..2011.'].isin(intersection)]
res_boring_sub = res_boring[res_boring['LSOA.code..2011.'].isin(intersection)]
res_depressing_sub = res_depressing[res_depressing['lsoa'].isin(intersection)]
res_wealth_sub['Outdoors.Sub.domain.Score'] = res_boring_sub['Outdoors.Sub.domain.Score']
res_wealth_sub['Education..Skills.and.Training.Score'] = res_depressing_sub['Education..Skills.and.Training.Score']

res_wealth_sub['wealth_resid'] = res_wealth_sub['resid']
res_wealth_sub['outdoor_resid'] = res_boring_sub['resid']
res_wealth_sub['education_resid'] = res_depressing_sub['resid']

res_wealth_sub_masked = res_wealth_sub[res_wealth_sub['LSOA.code..2011.'].isin(masked['lsoa11cd.1'])]
res_wealth_sub_masked.to_csv('chapter6substantive/outputs/intersecting_residuals_mid_masked.csv')

