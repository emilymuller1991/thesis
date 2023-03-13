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
# nu = new_data_usable.shape[0]
# print ('Number of usable data: %s' % str(nu))

one_side = new_data_usable.groupby('user_id').sum()['left']
totals = new_data_usable.groupby('user_id').count()['left']
prop = one_side/totals
to_drop = prop[prop < 0.1] + prop[prop > 0.9]

new_data = new_data_usable[~new_data_usable['user_id'].isin(list(to_drop.index))]
new_data['idx'] = np.arange(0, new_data.shape[0])

new_users = new_users[~new_users['user_id'].isin(list(to_drop.index))]

new_data = new_data.drop_duplicates(subset=['img_1', 'img_2', 'perception', 'choice', 'user_id'])


nu = new_data.shape[0]
print ('Number of usable data: %s' % str(nu)) # 27228
############################################################################################################# BASIC DEMO

# number of unique raters
ur = new_data['user_id'].unique().shape[0]
print ('number of unique raters: %s' % str(ur))

# number of demographics entries
ud = new_users.shape[0]
print ('number of demographic entries: %s' % str(ud))

# distribution of ratings per user
mean_r = new_data['user_id'].value_counts().mean()
median_r = new_data['user_id'].value_counts().median()
print ('Mean number of ratings per user: %s' % str(mean_r))
print ('Median number of ratings per user: %s' % str(median_r))

# number of unique perception ratings
r = new_data.shape[0]
print ('number of unique perception ratings: %s' % str(r))

print ('number of LEFT perception ratings: %s' % str(new_data[new_data['left'] == 0].count().iloc[0]))
print ('number of RIGHT perception ratings: %s' % str(new_data[new_data['left'] == 1].count().iloc[0]))
print ('number of EQUAL perception ratings: %s' % str(new_data[new_data['left'] == 2].count().iloc[0]))

# distribution of ratings per user
mean_r = new_data['user_id'].value_counts().mean()
median_r = new_data['user_id'].value_counts().median()
print ('Mean number of ratings per user: %s' % str(mean_r))
print ('Median number of ratings per user: %s' % str(median_r))

# get average # of games per image
games_1 = new_data['img_1']
games_2 = new_data['img_2']
games = pd.DataFrame(pd.concat([games_1, games_2]))
games.value_counts().mean()

print ('Mean number of games per image: %s' % str(games.value_counts().mean()))
print ('Median number of games per image: %s' % str(games.value_counts().median()))

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
        return 2
    if x in list(non_amt['idx']):
        return 0
    else:
        return np.nan

# merged_ = new_data.copy()
# merged = new_data.copy()

# merged_['couple'] = merged_['img_2'] + merged['img_1'] 
# merged['couple'] = merged['img_1'] + merged['img_2'] 

# both = pd.concat([merged, merged_], axis=0)
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

########################################################################
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

# get counts
non_amt_o = df_top[df_top['group'] == 0]['user_id'].unique().shape[0]
amt_o = df_top[df_top['group'] != 0]['user_id'].unique().shape[0]

london_o = df_top[df_top['london'] == 1]['user_id'].unique().shape[0]
non_london_o = df_top[df_top['london'] == 0]['user_id'].unique().shape[0]
nan_london_o = df_top[df_top['london'] == 2]['user_id'].unique().shape[0]

female_o = df_top[df_top['gender'] == 0]['user_id'].unique().shape[0]
male_o = df_top[df_top['gender'] == 1]['user_id'].unique().shape[0]
neither_o = df_top[df_top['gender'] == 2]['user_id'].unique().shape[0]

high_active_o = df_top[df_top['activity'] == 1]['user_id'].unique().shape[0]
low_active_o = df_top[df_top['activity'] == 0]['user_id'].unique().shape[0]

print ('Number of individuals in non_amt: %s vs amt: %s' % (str(non_amt_o), str(amt_o)))
print ('Number of individuals in london: %s vs non_london: %s vs nan_london: %s' % (str(london_o), str(non_london_o), str(nan_london_o)))
print ('Number of individuals in female: %s vs male: %s vs other: %s' % (str(female_o), str(male_o), str(neither_o)))
print ('Number of individuals in high active: %s vs low active: %s' % (str(high_active_o), str(low_active_o)))

# subselect dataframe groups
df_top = df_top[['left', 'game', 'london', 'gender', 'activity', 'group']]
# get images that equal wasn't chosen
df_top = df_top[df_top['left']!=2]

########################################################################

general_r = []
amt_r = []
non_amt_r = []
ldn_r = []
non_ldn_r = []
nan_ldn_r = []
female_r = []
male_r = []
other_r = []
high_activity_r = []
low_activity_r = []

general_c = []
amt_c = []
non_amt_c = []
ldn_c = []
non_ldn_c = []
nan_ldn_c = []
female_c = []
male_c = []
other_c = []
high_activity_c = []
low_activity_c = []

for game in top_19:
    games = df_top[df_top['game'] == game]

    # general user-agreeability
    general_r_ = games['left'].value_counts().max()
    general_c_ = games['left'].count()
    general_r.append(general_r_)
    general_c.append(general_c_)

    # london/non-london agreeability
    london = games[games['london'] == 1]
    non_london = games[games['london'] == 0]
    nan_london = games[games['london'] == 2]

    london_r_ = london['left'].value_counts().max()
    london_c_ = london['left'].count()
    non_london_r_ = non_london['left'].value_counts().max()
    non_london_c_ = non_london['left'].count()
    nan_london_r_ = nan_london['left'].value_counts().max()
    nan_london_c_ = nan_london['left'].count()
    ldn_r.append(london_r_)
    non_ldn_r.append(non_london_r_)
    nan_ldn_r.append(nan_london_r_)
    ldn_c.append(london_c_)
    non_ldn_c.append(non_london_c_)
    nan_ldn_c.append(nan_london_c_)

    # amt/non-amt agreeability
    non_amt = games[games['group'] == 0]
    amt = games[games['group'].isin([1,2])]

    amt_r_ = non_amt['left'].value_counts().max()
    amt_c_ = non_amt['left'].count()
    non_amt_r_ = amt['left'].value_counts().max()
    non_amt_c_ = amt['left'].count()
    amt_r.append(amt_r_)
    non_amt_r.append(non_amt_r_)
    amt_c.append(amt_c_)
    non_amt_c.append(non_amt_c_)

    # female/male
    female = games[games['gender'] == 0]
    male = games[games['gender'] == 1]
    other = games[games['gender'] == 2]

    female_r_ = female['left'].value_counts().max()
    female_c_ = female['left'].count()
    male_r_ = male['left'].value_counts().max()
    male_c_ = male['left'].count()
    other_r_ = other['left'].sum()
    other_c_ = other['left'].count()
    female_r.append(female_r_)
    male_r.append(male_r_)
    other_r.append(other_r_)
    female_c.append(female_c_)
    male_c.append(male_c_)
    other_c.append(other_c_)

    # activity/non-activity
    high_active = games[games['activity'] == 1]
    low_active = games[games['activity'] == 0]

    high_active_r_ = high_active['left'].value_counts().max()
    high_active_c_ = high_active['left'].count()
    low_active_r_ = low_active['left'].value_counts().max()
    low_active_c_= low_active['left'].count()

    high_activity_r.append(high_active_r_)
    low_activity_r.append(low_active_r_)
    high_activity_c.append(high_active_c_)
    low_activity_c.append(low_active_c_)

agree = pd.DataFrame({'games': top_19, 
                            'general': general_r,
                            'amt': amt_r, 
                            'non_amt': non_amt_r,
                            'london': ldn_r,
                            'non_london': non_ldn_r,
                            'other_london': nan_ldn_r,
                            'female': female_r,
                            'male': male_r,
                            'neither': other_r,
                            'higha': high_activity_r,
                            'lowa': low_activity_r
                            })
counts = pd.DataFrame({'games': top_19, 
                            'general': general_c,
                            'amt': amt_c, 
                            'non_amt': non_amt_c,
                            'london': ldn_c,
                            'non_london': non_ldn_c,
                            'other_london': nan_ldn_c,
                            'female': female_c,
                            'male': male_c,
                            'neither': other_c,
                            'higha': high_activity_c,
                            'lowa': low_activity_c
                            }) 

prop = agree/counts
weights = counts/counts.sum(axis=0)
weighted = (prop*weights)

# # get counts
# non_amt_o = df_top[df_top['group'] == 0]['user_id'].unique().shape[0]
# amt_o = df_top[df_top['group'] != 0]['user_id'].unique().shape[0]

# london_o = df_top[df_top['london'] == 1]['user_id'].unique().shape[0]
# non_london_o = df_top[df_top['london'] == 0]['user_id'].unique().shape[0]
# nan_london_o = df_top[df_top['london'] == 2]['user_id'].unique().shape[0]

# female_o = df_top[df_top['gender'] == 0]['user_id'].unique().shape[0]
# male_o = df_top[df_top['gender'] == 1]['user_id'].unique().shape[0]
# neither_o = df_top[df_top['gender'] == 2]['user_id'].unique().shape[0]

# high_active_o = df_top[df_top['activity'] == 1]['user_id'].unique().shape[0]
# low_active_o = df_top[df_top['activity'] == 0]['user_id'].unique().shape[0]

# print ('Number of individuals in non_amt: %s vs amt: %s' % (str(non_amt_o), str(amt_o)))
# print ('Number of individuals in london: %s vs non_london: %s vs nan_london: %s' % (str(london_o), str(non_london_o), str(nan_london_o)))
# print ('Number of individuals in female: %s vs male: %s vs other: %s' % (str(female_o), str(male_o), str(neither_o)))
# print ('Number of individuals in high active: %s vs low active: %s' % (str(high_active_o), str(low_active_o)))

# unique_idx = [0,2,4,5,8,10,11,14,16,17,20,22,23]
# subset = agreeability_top_50.iloc[unique_idx].sort_values('general')
# subset['baseline'] = [0.5 for i in range(len(unique_idx))]
# subset.index = np.arange(0, len(unique_idx))
# groups = [0,1,1,2,2,2,3,3,3,4,4, 5]
# cmap = plt.get_cmap('tab10')
# colors = [cmap(i) for i in groups]
# mark_groups = [-1,0,1,2,3,4,5,6,7,8,9, 10, 11]
# markers = ['.', 'x', 'o', 'x', 'o', '^', 'x', 'o', '^', 'x', 'o', '']

# import matplotlib as mpl

# mpl.rcParams['axes.spines.left'] = True
# mpl.rcParams['axes.spines.right'] = False
# mpl.rcParams['axes.spines.top'] = True
# mpl.rcParams['axes.spines.bottom'] = False
# import matplotlib.colors as cor

# # Plot
# fig, ax = plt.subplots()
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# for i, col in zip(mark_groups, subset):
#     if col == 'games':
#         pass
#     elif col == 'general':
#         subset[col].plot(color=colors[i], marker=markers[i])
#     elif col == 'baseline':
#         subset[col].plot(color=colors[i], marker=markers[i], linestyle='--')
#     else:
#         subset[col].plot(color=colors[i], marker=markers[i], linestyle=' ')
#     # ax.plot(group.x, group.y, marker=marker, linestyle='', ms=12, label=name)
# legend=plt.legend( bbox_to_anchor=(0.95, 1),loc=1, borderaxespad=0.5, frameon=False, title='Group')
# ax.set_yticks([0,0.5,1])
# plt.yticks(rotation=-90)
# ax.set_xticks([])
# plt.setp(ax.spines.values(), linewidth=4)
# plt.gca().invert_yaxis()
# plt.show()
# plt.savefig('plot/group_stratification_choice.png', bbox='tight', pad_inches=3)

# subset[subset.columns[1:]].values
################################################################################### TIMELINE


# get time plot of data
# new_data['idx'] = np.arange(0, new_data.shape[0])
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True

# new_data_usable['idx'] = np.arange(0, new_data_usable.shape[0])
new_data.index = new_data['date']

fig, ax = plt.subplots()
new_data[['idx']].plot(legend=False, marker='.', color='grey', markersize='2', linestyle=' ', ax=ax)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Counts')
plt.tight_layout()
plt.savefig('plot/timeline.png',  bbox='tight', pad_inches=3)

new_data[['idx', 'date']].to_csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/timeline.csv')
#### Correlation Plots

# # merge dataframe to 7 clusters
# df = pd.read_csv('/home/emily/phd/008_web_app/database/metadata/image_meta_small_sample.csv')

# def cleanup(x):
#     if x in [2,3,5,7,16,18, 12]:
#         return np.nan
#     elif x in [4, 8, 9, 14]:
#         return 8 
#     elif x in [1, 6, 11, 19]:
#         return 1 
#     else:
#         return x

# # 1: vegetation, 8: low-density residential, 10: commercial, 13: estates, 15: high-density res, 17: industrial.

# df['clusters_edited'] = df['clusters'].apply(lambda x: cleanup(x))

# merged = new_data_usable.merge(df[['idx', 'clusters', 'pp']], left_on = 'img_1', right_on = 'idx', how='left')
# merged = merged.merge(df[['idx', 'clusters', 'pp']], left_on = 'img_2', right_on = 'idx', how='left')
# merged = merged.dropna()

# cluster_corr = np.zeros(shape = (20, 20))
# pp_corr = np.zeros(shape = (10,10))
# counts = np.zeros(shape = (20, 20))

# for index, row in merged.iterrows():
#     i = int(row['clusters_x'])
#     j = int(row['clusters_y'])
#     if row['choice'] == row['img_1'] and i < j:
#         cluster_corr[i][j] += 1
#     elif row['choice'] == row['img_2'] and i < j:
#         cluster_corr[j][i] += 1
#     elif row['choice'] == row['img_1'] and i > j:
#         cluster_corr[j][i] += 1
#     elif row['choice'] == row['img_2'] and i > j:
#         cluster_corr[i][j] += 1

#     counts[i][j] += 1
#     counts[j][i] += 1

#     x = int(row['pp_x'])
#     y = int(row['pp_y'])
#     if row['choice'] == row['img_1'] and x < y:
#         pp_corr[x][y] += 1
#     elif row['choice'] == row['img_2'] and x < y:
#         pp_corr[y][x] += 1
#     elif row['choice'] == row['img_1'] and x > y:
#         pp_corr[y][x] += 1
#     elif row['choice'] == row['img_2'] and x > y:
#         pp_corr[x][y] += 1

# import matplotlib.patches as mpatches

# im = plt.imshow(np.around(counts, 0))  
# values = np.unique(np.around(counts, 0))
# colors = [ im.cmap(im.norm(value)) for value in values]
# patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=values[i]) , edgecolor='b' ) for i in range(len(values)) ]
# legend=plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.5, frameon=True)
# frame = legend.get_frame() # to add a frame
# frame.set_facecolor('grey')
# plt.show()

# im = plt.imshow(np.around(cluster_corr, 0))  
# values = np.unique(np.around(cluster_corr, 0))
# colors = [ im.cmap(im.norm(value)) for value in values]
# patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=values[i]) , edgecolor='b' ) for i in range(len(values)) ]
# legend=plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.5, frameon=True)
# frame = legend.get_frame() # to add a frame
# frame.set_facecolor('grey')
# plt.show()

# av = cluster_corr/counts*100
# av_u = np.triu(av)
# av_l = np.tril(av)
# av_ul = av_u + av_u.T - np.diag(np.diag(av_u))
# ratio = av_l/av_ul

# ratio[ratio == 0] = np.nan
# im = plt.imshow(np.around(ratio, 1))  
# values = np.unique(np.around(ratio, 1))
# colors = [ im.cmap(im.norm(value)) for value in values]
# patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=values[i]) , edgecolor='b' ) for i in range(len(values)) ]
# legend=plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.5, frameon=True)
# frame = legend.get_frame() # to add a frame
# frame.set_facecolor('grey')
# plt.show()