import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# read in perceptions
perceptions = pd.read_csv('/home/emily/phd/008_web_app/database/data_from_droplet/old/perceptions_27_7_2021.csv')
perceptions['date'] = pd.to_datetime(perceptions['time'])  
perceptions = perceptions.dropna()
not_comparible = perceptions[perceptions['choice'] == '0' ]
image_not_shown = perceptions[perceptions['choice'] == '2' ]
equal = perceptions[perceptions['choice'] == '1' ]
usable_data = perceptions[(perceptions['choice'] != '0') & (perceptions['choice'] != '2') & (perceptions['img_1'] != '2')]

users = pd.read_csv('/home/emily/phd/008_web_app/database/data_from_droplet/old/demographics_27_7_2021.csv')
users = users[5:]
n_people = len(usable_data['user_id'].value_counts().index)

# images per user
usable_data.groupby('user_id').count().mean()
# repeated games

# repeated games agreement

# import seaborn as sns
# sns.set_style("whitegrid", {'axes.grid' : False})
# df.set_index(df.index).T.plot(kind='barh', stacked=True)

# read in metadata information for images
## for the pilot we only need the 1500 df

df = pd.read_csv('/home/emily/phd/008_web_app/database/metadata/pilot_meta.csv')

perceptions = usable_data
perceptions.columns = ['img_1', 'img_2', 'perception', 'choice', 'userid', 'date', 'date']
# first merge to clusters and pp
merged = perceptions.merge(df[['idx', 'clusters', 'pp', 'pp_float']], left_on = 'img_1', right_on='idx', how='left')
# rename columns 
merged = merged.merge(df[['idx', 'clusters', 'pp', 'pp_float']], left_on = 'img_2', right_on='idx', how='left')
# merged = merged.dropna()
# clusters = [0,1,2,4,6,7,8,9,10,11,13,14,15,16,18,19]
compare = np.zeros(shape = (20,20))
pp = np.zeros(shape = (10, 10))
wins = np.zeros(shape = (10,10))
loses = np.zeros(shape = (10,10))

for index, row in merged.iterrows():
    i = int(row['clusters_x'])
    j = int(row['clusters_y'])
    compare[i][j] += 1
    x = int(row['pp_x'])
    y = int(row['pp_y'])
    pp[x][y] += 1
    # if row['choice'] == row['img_2'] and x > y:
    #     wins[x][y] += 1
    # elif row['choice'] == row['img_1'] and x < y :
    #     wins[x][y] += 1
    if row['choice'] == row['img_1'] and x < y :
        loses[x][y] += 1
    elif row['choice'] == row['img_2'] and x > y :
        loses[x][y] += 1
    elif x == y:
        loses[x][y] += 1


fig, ax = plt.subplots(figsize=(10,8))
im = plt.imshow(np.around(compare, 0).astype(int))  
values = np.unique(np.around(compare, 0).astype(int))
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=values[i]) , edgecolor='b' ) for i in range(len(values)) ]
legend=plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.5, frameon=True)
frame = legend.get_frame() # to add a frame
frame.set_facecolor('white')
frame.set_edgecolor("white")
legend.set_title("Counts")
plt.xlabel('Cluster')
plt.xticks(np.arange(0,20,2))
plt.yticks(np.arange(0,20,2))
plt.savefig('/home/emily/phd/008_web_app/database/analysis/plots/pilot_cluster_heatmap.png', bbox='tight') 
df_compare = pd.DataFrame(compare)
df_compare.to_csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/pilot_cluster_heatmap.csv')

fig, ax = plt.subplots(figsize=(10,8))
im = plt.imshow(np.around(pp, 0).astype(int))  
values = np.unique(np.around(pp, 0).astype(int))
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=values[i]) , edgecolor='b' ) for i in range(len(values)) ]
legend=plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.5, frameon=True)
frame = legend.get_frame() # to add a frame
frame.set_facecolor('white')
frame.set_edgecolor("white")
legend.set_title("Counts")
plt.xlabel('PP Decile')
plt.xticks(np.arange(0,10,2))
plt.yticks(np.arange(0,10,2))
plt.savefig('/home/emily/phd/008_web_app/database/analysis/plots/pilot_pp_heatmap_1s.png' , bbox='tight') 
df_pp = pd.DataFrame(pp)
df_pp.to_csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/pilot_pp_heatmap.csv')

fig, ax = plt.subplots(figsize=(10,8))
im = plt.imshow(np.around(wins, 0))  
values = np.unique(np.around(wins, 0))
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=values[i]) , edgecolor='b' ) for i in range(len(values)) ]
legend=plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.5, frameon=True)
frame = legend.get_frame() # to add a frame
frame.set_facecolor('white')
frame.set_edgecolor("white")
legend.set_title("Counts")
plt.xlabel('PP Decile')
plt.xticks(np.arange(0,10,2))
plt.yticks(np.arange(0,10,2))
plt.savefig('/home/emily/phd/008_web_app/database/analysis/plots/pilot_wins.png' , bbox='tight') 

av = loses/pp
fig, ax = plt.subplots(figsize=(10,8))
im = plt.imshow(np.around(av, 1))  
values = np.unique(np.around(av, 1))
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=values[i]) , edgecolor='b' ) for i in range(len(values)) ]
legend=plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.5, frameon=True)
frame = legend.get_frame() # to add a frame
frame.set_facecolor('white')
frame.set_edgecolor("white")
legend.set_title("Proportion")
plt.xlabel('PP Decile')
plt.xticks(np.arange(0,10,2))
plt.yticks(np.arange(0,10,2))
plt.savefig('/home/emily/phd/008_web_app/database/analysis/plots/pilot_loses.png' , bbox='tight')

df_loses = pd.DataFrame(av)
df_loses.to_csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/pilot_loses_heatmap.csv')

# totals = compare.sum(axis=1) + compare.sum(axis=0)
# n_samples = np.zeros(shape=(20,1))

# for i in range(totals.shape[0]):
#     if i in [3, 5, 12, 17]:
#         n_samples[i][0] == 0
#     else:
#         n_samples[i][0] = totals[i]/df['clusters'].value_counts().iloc[np.where(df['clusters'].value_counts().index == i)[0][0] ]


# im = plt.imshow(np.round(n_samples,0)) 
# values = np.unique(np.round(n_samples,0))
# colors = [ im.cmap(im.norm(value)) for value in values]
# patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=values[i]) , edgecolor='b' ) for i in range(len(values)) ]
# legend=plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.5, frameon=True)
# frame = legend.get_frame() # to add a frame
# frame.set_facecolor('grey')
# # plt.savefig('/home/emily/Dropbox/PhD/figures/sampling/n_samples_%s_%s_%s.png' % (str(m), str(n), str(split_decile))) 


# print ('Average number of samples %s' % str(n_samples.mean()))

# # get average # of games per image
# games_1 = merged['img_1']
# games_2 = merged['img_2']
# games = pd.concat([games_1, games_2])

# games.value_counts()
# games.value_counts().quartile(0.25)
# games.value_counts().quartile(0.75)


# # diagonal winners
# t = pp.sum()
# d = np.diagonal(pp, offset = 0).sum()
# d_1 = np.diagonal(pp, offset = -1).sum() + np.diagonal(pp, offset = 1).sum() + d
# d_2 = np.diagonal(pp, offset = -2).sum() + np.diagonal(pp, offset = 2).sum() + d_1

# l = loses.sum()
# m = np.diagonal(loses, offset = 0).sum()
# m_1 = np.diagonal(loses, offset = -1).sum() + np.diagonal(loses, offset = 1).sum() + m
# m_2 = np.diagonal(loses, offset = -2).sum() + np.diagonal(loses, offset = 2).sum() + m_

# # inter-use agreeability

# merged_ = merged.copy()

# merged_['couple'] = merged_['img_2'] + merged['img_1'] 
# merged['couple'] = merged['img_1'] + merged['img_2'] 

# both = pd.concat([merged, merged_], axis=0)

# duplicates = both[both[['couple']].duplicated(keep=False)]

# unique_pairs = {}
# for index, row in duplicates.iterrows():
#     x = row['img_1']
#     y = row['img_2']
#     z = row['choice']
#     u = row['time']
#     if (x,y) in unique_pairs.keys():
#         unique_pairs[(x,y)].append((z,u))
#     elif (y,x) in unique_pairs.keys():
#         unique_pairs[(y,x)].append((z,u))
#     elif (x,y) not in unique_pairs.keys():
#         unique_pairs[(x,y)] = [(z,u)]
#     elif (y,x) not in unique_pairs.keys():
#         unique_pairs[(y,x)] = [(z,u)]



# both.duplicated(subset='couple', keep=False)

# false users 
#7d78c674-cabf-4ef4-bac7-31e8cc41caa5
#863e3ded-94ba-40cd-a3f5-b2c998370d45
#60443596-fb51-45a1-b2a9-a4b46a048725
#16fc8a75-b5e5-4afc-b75c-1f9975ee83c9
#8bb0d090-bd65-4b31-a5fe-dd4304ab2ed6
#e97fa310-d839-405c-ae85-ada8ff49bf52
# false = ['7d78c674-cabf-4ef4-bac7-31e8cc41caa5', '863e3ded-94ba-40cd-a3f5-b2c998370d45', '60443596-fb51-45a1-b2a9-a4b46a048725', '16fc8a75-b5e5-4afc-b75c-1f9975ee83c9', 
#                                                         '8bb0d090-bd65-4b31-a5fe-dd4304ab2ed6', 'e97fa310-d839-405c-ae85-ada8ff49bf52']


# RANK USING TRUESKILL
from trueskill import Rating, quality_1vs1, rate_1vs1

# assign all ratings with default
new_data = usable_data
teams = set(new_data['img_1'].tolist() + new_data['img_2'].tolist() ) 
#teams = set( new_data['choice'].value_counts().head(6864).index )

rating = {}
for team in teams:
    rating[team] = Rating()
    
games = 0
for idx, game in new_data.iterrows():
    try:
        if game['choice'] == game['img_1']:
            rating[game['img_1']], rating[game['img_2']] = rate_1vs1(rating[game['img_1']], rating[game['img_2']])
        if game['choice'] == game['img_2']:
            rating[game['img_2']], rating[game['img_1']] = rate_1vs1(rating[game['img_2']], rating[game['img_1']])
        if game['choice'] == '1':
            rating[game['img_2']], rating[game['img_1']] = rate_1vs1(rating[game['img_2']], rating[game['img_1']], drawn = True)
        games += 1
    except:
        pass

ratings = pd.DataFrame(rating).T
ratings['idx'] = ratings.index

sorted_ranks = ratings.sort_values(0)
############################################################################################################ VIS-RANKING
import matplotlib as mpl
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})

ratings['scaled'] = 1 + (ratings[0] - ratings[0].min()) * 9 / (ratings[0].max() - ratings[0].min())
# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.grid'] = False

mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.spines.bottom'] = False

fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# plot means and std stacked in order
sorted_ranks.plot(y =0, yerr=1, marker='.', color='grey', alpha = 0.7, ax=ax, legend=False, markerfacecolor='grey', markeredgewidth=0, linewidth=0.5)
plt.yticks(rotation=-90)
plt.xticks([])
plt.gca().invert_yaxis()
plt.setp(ax.spines.values(), linewidth=2)

ax.set_ylabel('Mean', rotation=270)
ax.yaxis.set_label_coords(-0.12, 0.5)
plt.title('Pilot Walkability', rotation = 270, x=1.05, y=0.3)
# ax.title.set_position((1, 0.5))
plt.savefig('/home/emily/phd/008_web_app/database/analysis/plots/pilot_trueskill_ranking.png' , bbox='tight')
sorted_ranks.to_csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/pilot_walk_ranks.csv')

###### Sample high and low rated images


# Ranking
# from elosports.elo import Elo

# teams = set(merged['img_1'].tolist() + merged['img_2'].tolist() ) 
# eloLeague = Elo(k=32)

# for team in teams:
#     eloLeague.addPlayer(team)

# for key in eloLeague.ratingDict.keys():
#     eloLeague.ratingDict[key] = eloLeague.ratingDict[key] - ((eloLeague.ratingDict[key] - 1500) * (1/3.))

# for idx, game in merged.iterrows():
#     if game['choice'] == game['img_1']:
#         eloLeague.gameOver(game['img_1'], game['img_2'], 0)

#     if game['choice'] == game['img_2']:
#         eloLeague.gameOver(game['img_2'], game['img_1'], 0)

# for team in eloLeague.ratingDict.keys():
# 	print (team, eloLeague.ratingDict[team])

# ratings = pd.DataFrame(eloLeague.ratingDict, index = np.arange(0, 1) ).T
# ratings['idx'] = ratings.index
# ratings['rank'] = ratings[0].rank()
# ratings_ = ratings.merge(df[['idx', 'clusters', 'pp']] )
# ratings_['clusters'] = ratings_['clusters'].astype("category")
# ratings_['pp'] = ratings_['pp'].astype("category")

# sort_by = ratings_.groupby(['clusters']).median().sort_values(by='rank')
# # sort_by = ratings_.groupby(['pp']).median().sort_values(by='rank')

# sns.boxplot(y = "clusters",
#             x = "rank",
#            data = ratings_, 
#            boxprops=dict(alpha=.2),
#            order=sort_by.index)
# # horizontal stripplot Python
# sns.stripplot(y = "clusters",
#             x = "rank",
#               color = 'darkred',
#                 alpha=0.3,
#               data = ratings_, 
#               jitter=0,
#               marker='|', 
#               linewidth=3,
#               s=25,
#               order=sort_by.index)
# plt.ylabel("Cluster", size=18)
# plt.xlabel("Rank", size=18)

# sort_by = ratings_.groupby(['pp']).median().sort_values(by='rank')
# sns.boxplot(y = "pp",
#             x = "rank",
#            data = ratings_, 
#            boxprops=dict(alpha=.2),
#            order=sort_by.index)
# # horizontal stripplot Python
# sns.stripplot(y = "pp",
#             x = "rank",
#               color = 'darkred',
#                 alpha=0.3,
#               data = ratings_, 
#               jitter=0,
#               marker='|', 
#               linewidth=3,
#               s=25,
#               order=sort_by.index)
# plt.ylabel("PP", size=18)
# plt.xlabel("Rank", size=18)