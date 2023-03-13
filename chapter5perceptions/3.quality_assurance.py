import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

############################################################################################################ READ DATA

perceptions = pd.read_csv('/home/emily/phd/008_web_app/database/data_from_droplet/perceptions_30_6_2022.csv')
perceptions['date'] = pd.to_datetime(perceptions['time'])  

# new_data = perceptions[(perceptions['date'] > '2022-04-21 14:27:09')] 
new_data = perceptions[4899:]
new_data = new_data.drop_duplicates()

users = pd.read_csv('/home/emily/phd/008_web_app/database/data_from_droplet/demographics_new_30_6_2022.csv')
new_users = users[4:]
new_users['date'] = pd.to_datetime(new_users['time'])

# number of errors
image_not_shown = new_data[new_data['choice'] == '2' ]
ns = image_not_shown.shape[0]
print ('Number of Google no show: %s' % str(ns))

# number of not comparibles
not_comparible = new_data[new_data['choice'] == '0' ]
nc = not_comparible.shape[0]
print ('Number of not comparible: %s' % str(nc))

############################################################################################################# DATA PRO-PROCESSING
# usable ratings
new_data_usable = new_data[(new_data['choice'] != '0') & (new_data['choice'] != '2')]
nu = new_data_usable.shape[0]
print ('Number of usable data: %s' % str(nu))

# check user isn't repeatedly choosing one image
new_data_usable['left'] = new_data_usable.apply(lambda x: 0 if x.img_1 == x.choice else (1 if x.img_2 == x.choice else 2), axis=1)
nu = new_data_usable.shape[0]
print ('Number of usable data: %s' % str(nu))

one_side = new_data_usable.groupby('user_id').sum()['left']
totals = new_data_usable.groupby('user_id').count()['left']
prop = one_side/totals
to_drop = prop[prop < 0.1] + prop[prop > 0.9]

new_data = new_data_usable[~new_data_usable['user_id'].isin(list(to_drop.index))]
new_data['idx'] = np.arange(0, new_data.shape[0])

new_users = new_users[~new_users['user_id'].isin(list(to_drop.index))]

new_data = new_data.drop_duplicates(subset=['img_1', 'img_2', 'perception', 'choice', 'user_id'])
nu = new_data.shape[0]
print ('Number of usable data: %s' % str(nu))
############################################################################################################# USER AGREEABILITY STRATIFIED BY groups

# stratify raters between citizen versus AMT.
amt_1 = new_data[(new_data['date'] > '2022-06-12 14:27:09')] 
amt_2 = new_data[(new_data['date'] > '2022-06-09 14:27:09') & (new_data['date'] < '2022-06-12 14:27:09')] 
non_amt = new_data[(new_data['date'] <= '2022-06-09 14:27:09')] 

# add a tag for which group the user is in
def group(x):
    if x in list(amt_1['idx']):
        return 1
    if x in list(amt_2['idx']):
        return 1
    if x in list(non_amt['idx']):
        return 0
    else:
        return np.nan

# merged_ = new_data.copy()
# merged = new_data.copy()

# merged_['couple'] = merged_['img_2'] + merged['img_1'] 
# merged['couple'] = merged['img_1'] + merged['img_2'] 

both = new_data.copy()
both['group'] = both['idx'].apply(lambda x: group(x) )

# create new stratifications bases on london, non-london, m/f, exercise ratio
london_users = list(new_users[new_users['city'] == 'London']['user_id'])
nan_users = list(new_users[new_users['city'].isna()]['user_id'])
non_london_users = list(new_users[(new_users['city'].notna()) & (new_users['city'] != 'London') ]['user_id'])

def group_london(x):
    if x in london_users:
        return 1
    if x in nan_users:
        return 2
    if x in non_london_users:
        return 0
    else:
        return np.nan

male = list(new_users[new_users['gender'] == 'Male']['user_id'])
female = list(new_users[new_users['gender'] == 'Female']['user_id'])
other = list(new_users[(new_users['gender'] != 'Male') & (new_users['gender'] != 'Female') ]['user_id'])

def group_gender(x):
    if x in male:
        return 1
    if x in other:
        return 2
    if x in female:
        return 0
    else:
        return np.nan

high_activity = list(new_users[new_users['activity'].isin(['Four Plus Times a Week', 'Every Day']) ]['user_id'])
low_activity = list(new_users[(new_users['activity'] != 'Every Day') & (new_users['activity'] != 'Four Plus Times a Week') ]['user_id'])

def group_activity(x):
    if x in high_activity:
        return 1
    if x in low_activity:
        return 0
    else:
        return np.nan

both['london'] = both['user_id'].apply(lambda x: group_london(x) )
both['gender'] = both['user_id'].apply(lambda x: group_gender(x) )
both['activity'] = both['user_id'].apply(lambda x: group_activity(x) )

############################################################################################################# GET GAMES

df = both.copy()

df['tuple'] = df.apply(lambda x: ((x.img_1, x.img_2), (x.img_2, x.img_1)), axis=1)

unique_sets = list(set(list(df['tuple'])))

def get_index(x):
    try:
        return unique_sets.index(x)
    except:
        return unique_sets.index(tuple(reversed(x)))

df['tuple_loc'] = df['tuple'].apply(lambda x: get_index(x) )

def get_reversed(x):
    try:
        a = unique_sets.index(x)
        return 'keep'
    except:
        b = unique_sets.index(tuple(reversed(x)))
        return 'reversed'

df['tuple_reverse'] = df['tuple'].apply(lambda x: get_reversed(x) )

# remove all games which are played by the same player
df = df.drop_duplicates(subset=['img_1', 'img_2', 'perception', 'choice', 'user_id'])

top_19 = list(df['tuple_loc'].value_counts().head(14).index)
df_top = df[df['tuple_loc'].isin(top_19)]
# 1503 pairs in the dataset
# each games has 10 or more games
df_top['tuple_loc'] = df_top['tuple_loc'].astype('category')
df_top['game'] = df_top['tuple_loc']

# subselect dataframe groups
df_top = df_top[['left', 'game', 'london', 'gender', 'activity', 'group']]
# get images that equal wasn't chosen
df_top = df_top[df_top['left']!=2]

for col in list(df_top.columns):
    df_top[col] = df_top[col].astype('string')
    df_top[col] = df_top[col].astype('category')
df_top['left'] = df_top['left'].astype('int32')

# modelling without factors
from pymer4.models import Lmer
model = Lmer("left  ~ (1|game)",
             data=df_top, family = 'binomial')
print(model.fit())
model.plot_summary()
plt.savefig('plot/model0_game_re.svg', bbox='tight', pad_inches=3)

# random effects from all 19 image pairs
# ranefs = model.ranef.sort_values(by='X.Intercept.')
# ranefs['idx'] = np.arange(14)
# ranefs['baseline'] = np.zeros(14)

# import matplotlib as mpl

# mpl.rcParams['font.family'] = 'Avenir'
# plt.rcParams['font.size'] = 18
# plt.rcParams['axes.linewidth'] = 2

# mpl.rcParams['axes.spines.left'] = True
# mpl.rcParams['axes.spines.right'] = False
# mpl.rcParams['axes.spines.top'] = False
# mpl.rcParams['axes.spines.bottom'] = True

# fig, ax = plt.subplots()
# ranefs.plot(x = 'idx', y= 'X.Intercept.', legend=False, marker='x', color='black', markersize='2', linestyle=' ', ax=ax)
# ranefs.plot(x = 'idx', y='baseline', linestyle='--',  color='black', legend=False, ax=ax)

# ax.set_xticks(np.arange(19))
# ax.set_xticklabels(list(ranefs.index), rotation=90)
# plt.yticks(rotation=90)
# plt.show()

# model_activity = Lmer("left  ~ (1|game) + activity",
#              data=df_top, family = 'binomial')

model_activity = Lmer("left  ~ (activity|game)",
             data=df_top, family = 'binomial')

print(model_activity.fit())
# model_activity.plot_summary()
# plt.savefig('plot/model1_activity_re.svg', bbox='tight', pad_inches=3)

df_base = pd.DataFrame( {'game': np.tile(top_19,2), 'activity': np.concatenate((np.zeros(14),np.ones(14)) )})
df_base = pd.DataFrame( {'game': top_19, 'activity': np.zeros(14) })
def simulate_prediction(model, df_base):
    model.predict(df_base, use_rfx=True, pred_type='response', skip_data_checks=True, verify_predictions=True, verbose=False)


sub_london = df_top[df_top['london'] != '2.0']

model_london = Lmer("left  ~ (london|game) + (1|game)",
             data=sub_london, family = 'binomial')
print(model_london.fit())
model_london.plot_summary()
plt.savefig('plot/model2_london_re.svg', bbox='tight', pad_inches=3)

sub_gender = df_top[df_top['gender'] != '2.0']

model_gender = Lmer("left  ~ (gender|game)",
             data=sub_gender, family = 'binomial')
print(model_gender.fit())
model_gender.plot_summary()
plt.savefig('plot/model3_gender_re.svg', bbox='tight', pad_inches=3)

model_amt = Lmer("left  ~ (group|game)",
             data=df_top, family = 'binomial')
print(model_amt.fit())
model_amt.plot_summary()
plt.savefig('plot/model4_gender_re.svg', bbox='tight', pad_inches=3)