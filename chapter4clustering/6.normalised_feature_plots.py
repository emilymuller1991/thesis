# normalised feature plots
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
from sklearn import preprocessing


everything = pd.read_csv('/home/emily/phd/2_interpretability/2018_all_variables.csv')
fi = pd.read_csv('/home/emily/phd/2_interpretability/features/outputs/RF_shap_class_fi.csv')
fi = fi.drop(columns='0')
sign = pd.read_csv('/home/emily/phd/2_interpretability/features/outputs/RF_shap_class_sign.csv')
sign = sign.drop(columns='0')

features = ['building_seg',
 'vegetation_seg',
 'car_seg',
 'road_seg',
 'sky_seg',
 'sidewalk_seg',
 'fence_seg',
 'wall_seg',
 'pole_seg',
 'traffic sign_seg',
 'terrain_seg',
 'car',
 'person_seg',
 'traffic light_seg',
 'truck_seg',
 'bicycle_seg',
 'bus_seg',
 'train_seg',
 'motorcycle_seg',
 'person',
 'truck']
colors = ['grey', 'green', 'red', 'black', 'white', 'grey', 'black', 'grey', 'black', 'orange', 'green', 'red', 'yellow', 'orange', 'red', 'black', 'red', 'blue', 'black', 'yellow', 'black']
a = everything[features]
x = a[features].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.columns = features
df['clusters'] = everything['clusters'].values
for K in list(everything['clusters'].unique()):
    #plt.figure(figsize=(8, 12))
    fig, ax = plt.subplots(figsize=(8,12))
    bp_seg = df[df['clusters'] == K].boxplot(column=features, by=None, ax=ax, showfliers=False, grid=False, boxprops= dict(linewidth=1.0, color='black'), 
                                                                                                                                        whiskerprops=dict(linewidth=1.0, color='black'), return_type='both', patch_artist = True, widths =0.9, 
                                                                                                                                        medianprops=dict(linewidth=1.0, color='black'))
    #colors = ['grey', 'green', 'red', 'black', 'white', 'purple', 'brown', 'red', 'yellow']
    for i, box in enumerate(bp_seg[1]['boxes']):
        box.set_facecolor(colors[i])
    bp_seg[0].set_xlim(0.2,21.5)
    bp_seg[0].set_ylim(-0.01,1.25)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_ylabel('Normalised Features, Cluster %s' %str(K) )

    ax1 = ax.twinx()
    ax1.bar(np.arange(1,22), fi.loc[K]*sign.loc[K]*5 , alpha=0.65, width=0.9, color='lavender', edgecolor='grey', hatch='//')
    ax1.invert_yaxis()
    ax1.set_yticks([-0.04, 0, 0.07])
    ax1.set_yticklabels(['-ve', 0, '+ve'], rotation=90)
    ax1.tick_params(axis='y',length=0)
    ax1.set_ylim(-1,0.25)
    ax1.set_xlim(0.2,21.5)
    ax1.set_ylabel('SHAP value')
    ax1.yaxis.set_label_coords(1.05,0.81)
    plt.show()
    break
    #plt.savefig(prefix + '2_interpretability/features/outputs/xtra_normalized_features_per_cluster_%s.png' % str(K), bbox='tight')
    plt.clf()