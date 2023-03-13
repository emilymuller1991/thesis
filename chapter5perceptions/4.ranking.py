import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

perceptions = pd.read_csv('/home/emily/phd/008_web_app/database/data_from_droplet/perceptions_30_6_2022.csv')
perceptions['date'] = pd.to_datetime(perceptions['time'])  

# new_data = perceptions[(perceptions['date'] > '2022-04-21 14:27:09')] 
new_data = perceptions[4899:]
new_data = new_data.drop_duplicates()

users = pd.read_csv('/home/emily/phd/008_web_app/database/data_from_droplet/demographics_new_30_6_2022.csv')
new_users = users[4:]
new_users['date'] = pd.to_datetime(new_users['time'])

############################################################################################################# DATA PRO-PROCESSING
# usable ratings
new_data_usable = new_data[(new_data['choice'] != '0') & (new_data['choice'] != '2')]
nu = new_data_usable.shape[0]
print ('Number of usable data: %s' % str(nu))

# check user isn't repeatedly choosing one image
new_data_usable['left'] = new_data_usable.apply(lambda x: 0 if x.img_1 == x.choice else 1, axis=1)
nu = new_data_usable.shape[0]
print ('Number of usable data: %s' % str(nu))

one_side = new_data_usable.groupby('user_id').sum()['left']
totals = new_data_usable.groupby('user_id').count()['left']
prop = one_side/totals
to_drop = prop[prop < 0.1] + prop[prop > 0.9]

new_data = new_data_usable[~new_data_usable['user_id'].isin(list(to_drop.index))]
new_data['idx'] = np.arange(0, new_data.shape[0])

new_users = new_users[~new_users['user_id'].isin(list(to_drop.index))]

############################################################################################################# METADATA
# read in metadata
df = pd.read_csv('/home/emily/phd/008_web_app/database/metadata/image_meta_small_sample.csv')

def cleanup(x):
    if x in [2,3,5,7,16,18, 12]:
        return np.nan
    elif x in [4, 8, 9, 14]:
        return 8
    elif x in [1, 6, 11, 19]:
        return 1
    else:
        return x

df['clusters_edited'] = df['clusters'].apply(lambda x: cleanup(x))

############################################################################################################ RANKING

from trueskill import Rating, quality_1vs1, rate_1vs1

# assign all ratings with default

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
sorted_ranks.to_csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/walk_ranks.csv')
############################################################################################################ VIS-RANKING
import matplotlib as mpl
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
# Edit the font, font size, and axes width
# mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
# plt.rcParams['axes.linewidth'] = 1

mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.spines.bottom'] = False

fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# plot means and std stacked in order
sorted_ranks.plot(y =0, yerr=1, marker='.', color=colors[0], alpha = 0.7, ax=ax, legend=False, markerfacecolor=colors[0], markeredgewidth=0, linewidth=0.03)
plt.yticks(rotation=-90)
plt.xticks([])
plt.gca().invert_yaxis()
plt.setp(ax.spines.values(), linewidth=2)

ax.set_ylabel('Mean', rotation=270)
ax.yaxis.set_label_coords(-0.12, 0.5)
plt.title('')
# ax.title.set_position((1, 0.5))
plt.savefig('plot/walk_ranks_right_font.png', bbox='tight', pad_inches=3)

############################################################################################################ PP RANKS
###### std dev of safety pp score

"""Read in metadata to add qscore label to image dataframe"""
meta_path = '/home/emily/phd/006_place_pulse/place-pulse-2.0/place_pulse_meta' + "/qscores.tsv"
meta = pd.read_csv(meta_path, sep="\t")

studies = ['50a68a51fdc9f05596000002',
'50f62c41a84ea7c5fdd2e454',
'50f62c68a84ea7c5fdd2e456',
'50f62cb7a84ea7c5fdd2e458',
'50f62ccfa84ea7c5fdd2e459',
'5217c351ad93a7d3e7b07a64'
]

names = [
    'Safer',
'Livelier',
'Boring',
'Wealthier',
'More depressing',
'More beautiful'
]

cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(6)]

for i, study in enumerate(studies):
    perception_meta = meta[meta["study_id"] == study]
    sorted = perception_meta.sort_values('trueskill.score')
    sorted.index = np.arange(0, sorted.shape[0])

    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    # plot means and std stacked in order
    sorted.to_csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/%s_ranks.csv' % names[i])
    # sorted.plot(y ='trueskill.score', yerr='trueskill.stds.-1', marker='.', color=colors[0], alpha = 0.7, ax=ax, legend=False, markerfacecolor=colors[0], markeredgewidth=0, linewidth=0.01)
    # plt.yticks(rotation=-90)
    # plt.xticks([])
    # plt.gca().invert_yaxis()
    # plt.setp(ax.spines.values(), linewidth=2)

    # ax.set_ylabel('Mean', rotation=270)
    # ax.yaxis.set_label_coords(-0.12, 0.5)
    # plt.title('')
    # # ax.title.set_position((1, 0.5))
    # plt.savefig('plot/%s_ranks_right_font.png' % names[i], bbox='tight', pad_inches=3)

##########################################################################################################################

ratings_ = ratings.merge(df[['idx', 'clusters', 'pp']] )
ratings_['clusters'] = ratings_['clusters'].astype("category")

# names = { 10.0: 'Commercial',
#             15.0: 'High-density', 
#             1.0: 'Vegetation', 
#             13.0: 'Estate',
#             0.0: 'Terraced Residential',
#             8.0: 'Low-density Residential',
# }

# def get_names(cluster_idx):
#     print (cluster_idx)
#     if cluster_idx not in names.keys():
#         return np.nan
#     else:
#         return names[cluster_idx]

# ratings_['clusters'] = ratings_['clusters'].apply( lambda x: get_names(x) )

ratings_['pp'] = ratings_['pp'].astype("category")

# scale values into the range 0 and 10
# ratings_['scaled'] = 1 + ratings_[0] / ratings[0].max() * 9

# ratings_['scaled'] = 1 + (ratings_[0] - ratings_[0].min()) * 9 / (ratings_[0].max() - ratings_[0].min())
sort_by = ratings_.groupby(['clusters']).median().sort_values(by=0)

mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False

fig = plt.figure(figsize=(10, 6))
sns.boxplot(y = "clusters",
            x = 0,
           data = ratings_, 
           boxprops=dict(alpha=.2),
           order=sort_by.index)

# horizontal stripplot Python
# sns.stripplot(y = "clusters",
#             x = 0,
#               color = 'darkred',
#                 alpha=0.3,
#               data = ratings_, 
#               jitter=0,
#               marker='|', 
#               linewidth=3,
#               s=25,
#               order=sort_by.index)
plt.ylabel("Cluster", size=18)
plt.xlabel("Mean", size=18)
plt.tight_layout()
plt.savefig('plot/cluster_rank_scaled.png', bbox='tight', pad_inches=3)

fig = plt.figure(figsize=(10, 6))
sort_by = ratings_.groupby(['pp']).median().sort_values(by=0)
sns.boxplot(y = "pp",
            x = 0,
           data = ratings_, 
           boxprops=dict(alpha=.2),
           order=sort_by.index)

# horizontal stripplot Python
# sns.stripplot(y = "pp",
#             x = 0,
#               color = 'darkred',
#                 alpha=0.3,
#               data = ratings_, 
#               jitter=0,
#               marker='|', 
#               linewidth=3,
#               s=25,
#               order=sort_by.index)
plt.ylabel("PP", size=18)
plt.xlabel("Q-score", size=18)
plt.tight_layout()
plt.savefig('plot/pp_rank.png', bbox='tight', pad_inches=3)

# get average std deviation across pp ratings and walkability
ratings_[1].mean()


########################################################################################################################## PLOT SAMPLES

# import matplotlib.image as mpimg

# def plot_album(image_paths, i, high):
#     fig = plt.figure(figsize = (8,8))
#     fig.subplots_adjust(wspace=0, hspace=0)
#     ax = [fig.add_subplot(2,2,i+1) for i in range(4)]
#     #fig, axes = plt.subplots(nrows=2, ncols=2, gridspec_kw = {'wspace':0, 'hspace':0})
#     # this assumes the images are in images_dir/album_name/<name>.jpg
#     #image_paths = glob(images_dir + album_name + '/*.jpg')
    
#     for imp, ax in zip(image_paths, ax):
#         img = mpimg.imread(imp)
#         ax.imshow(img)
#         ax.axis('off')
#     fig.tight_layout()
#     #plt.show()
#     plt.savefig('plot/qscores/%s.png' % (str(i) + high))
#     plt.clf()

# for i in range(0, 30, 1):
#     df_high = ratings_[(ratings_['scaled'] > 8) & (ratings_['scaled'] < 10)].sample(25)
#     df_low = ratings_[(ratings_['scaled'] > 0) & (ratings_['scaled'] < 3)].sample(25)
#     images_dir_high = list(df_high['idx'].apply( lambda x: 
#                                 '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/project/pathways/live/Transferability/emily_phd_images/2018/2018/2018/2018/2018/' + x + '.png'))
#     images_dir_low = list(df_low['idx'].apply( lambda x: 
#                                 '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/project/pathways/live/Transferability/emily_phd_images/2018/2018/2018/2018/2018/' + x + '.png'))
#     plot_album(images_dir_low, i, 'low')
#     plot_album(images_dir_high, i, 'high')


###### std dev of safety pp score

"""Read in metadata to add qscore label to image dataframe"""
meta_path = '/home/emily/phd/006_place_pulse/place-pulse-2.0/place_pulse_meta' + "/qscores.tsv"
meta = pd.read_csv(meta_path, sep="\t")

studies = ['50a68a51fdc9f05596000002',
'50f62c41a84ea7c5fdd2e454',
'50f62c68a84ea7c5fdd2e456',
'50f62cb7a84ea7c5fdd2e458',
'50f62ccfa84ea7c5fdd2e459',
'5217c351ad93a7d3e7b07a64'
]

names = [
    'safer',
'livelier',
'more boring',
'wealthier',
'more depressing',
'more beautiful'
]

for i, study in enumerate(studies):
    perception_meta = meta[meta["study_id"] == study]
    m = perception_meta['trueskill.stds.-1'].mean()
    print ('Average sd for study %s is: %s' % (str(names[i]), str(m)) )

# 
g = [511037, 367475, 174784, 144068, 220656, 149361, 31045]


########################################################################################################################## WALKING vs. PP CORR

pp = pd.read_csv('/home/emily/phd/006_place_pulse/place-pulse-2.0/outputs/2018_all_perception_scores.csv')
pp['Unnamed: 0.1'] = pp['Unnamed: 0.1'].apply( lambda x: x[:-4]) 

# pp['pp_lively'] = pd.qcut(pp['lively'], 10, labels=False)
# pp['pp_boring'] = pd.qcut(pp['boring'], 10, labels=False)
# pp['pp_wealth'] = pd.qcut(pp['wealth'], 10, labels=False)
# pp['pp_depressing'] = pd.qcut(pp['depressing'], 10, labels=False)
# pp['pp_safety'] = pd.qcut(pp['safety'], 10, labels=False)
# pp['pp_beauty'] = pd.qcut(pp['beauty'], 10, labels=False)

merged = ratings_.merge(pp, left_on = 'idx', right_on = 'Unnamed: 0.1', how='left')

# get correlation across pp scores:

corr_l = merged['scaled'].corr(merged['lively'])
corr_bo = merged['scaled'].corr(merged['boring'])
corr_w = merged['scaled'].corr(merged['wealth'])
corr_d = merged['scaled'].corr(merged['depressing'])
corr_be = merged['scaled'].corr(merged['beauty'])
corr_s = merged['scaled'].corr(merged['safety'])

print ('Safety corr is %s: ' % str(corr_s), 
        'Lively corr is %s: ' % str(corr_l),
        'Wealth corr is %s: ' % str(corr_w),
        'Boring corr is %s: ' % str(corr_bo),
        'Depressing corr is %s: ' % str(corr_d),
        'Beauty corr is %s: ' % str(corr_be),
)