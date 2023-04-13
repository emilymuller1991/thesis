import pandas as pd 

df_bottom = pd.read_csv('chapter6substantive/outputs/intersecting_residuals_all_masked.csv') #77

# life_expectancy
le_f = pd.read_csv('/home/emily/phd/drives/phd/chapter6substantive/source/life_expectancy_females.csv')
le_f = le_f[le_f['Year'] == 2019]
le_m = pd.read_csv('/home/emily/phd/drives/phd/chapter6substantive/source/life_expectancy_males.csv')
le_m = le_m[le_m['Year'] == 2019]

merged_f = le_f.merge(df_bottom, left_on='LSOA code', right_on='LSOA.code..2011.')
#merged_f.boxplot('Life expectancy, Female (central estimate)', by='class')#    

merged = le_m.merge(df_bottom, left_on='LSOA code', right_on='LSOA.code..2011.')

# normalise all scores to be between 
merged['income_normal'] = (merged['Income.Score..rate.']- merged['Income.Score..rate.'].mean())/merged['Income.Score..rate.'].std()
merged['education_normal'] = (merged['Education..Skills.and.Training.Score']- merged['Education..Skills.and.Training.Score'].mean())/merged['Education..Skills.and.Training.Score'].std()
merged['outdoor_normal'] = (merged['Outdoors.Sub.domain.Score']- merged['Outdoors.Sub.domain.Score'].mean())/merged['Outdoors.Sub.domain.Score'].std()
merged['sum_score'] = merged['income_normal'] + merged['education_normal'] + merged['outdoor_normal'] 

merged['income'] = merged['Income.Score..rate.']
merged['education'] = merged['Education..Skills.and.Training.Score']
merged['outdoor'] = merged['Outdoors.Sub.domain.Score']

lm_variables = merged[['income_normal', 'education_normal', 'outdoor_normal', 'income', 'education', 'outdoor', 'wealth_resid', 'education_resid', 'outdoor_resid', 'Life expectancy, Male (central estimate)']]
lm_variables['Life expectancy, Female (central estimate)'] = merged_f[ 'Life expectancy, Female (central estimate)']
lm_variables.to_csv('chapter6substantive/outputs/all_scores_all_residuals_le.csv')


merged.boxplot('Life expectancy, Male (central estimate)', by='class')


fig, ax = plt.subplots()
merged[['Life expectancy, Male (central estimate)', 'class']].plot(kind='box', by='class', ax=ax)
ax2 = ax.twinx()
merged[['sum_score', 'class']].plot(kind='box', by='class', ax=ax2, positions=[3,4])
