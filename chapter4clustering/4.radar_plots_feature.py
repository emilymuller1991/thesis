import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import pandas as pd
from math import pi
import numpy as np
from sklearn import preprocessing

everything = pd.read_csv('/home/emily/phd/2_interpretability/2018_all_variables.csv')
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
# colors = ['grey', 'green', 'red']
# colors = {
#     'building_seg': 'grey',
#     'vegetation_seg': 'green',
#     'car': 'red'
# }
a = everything[features]
#a = a[a['clusters_20'].isin([0,1,2,4,6,7,8,9,10,11,12,13,14,15,16,18,19])]
x = a[features].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.columns = features
df['clusters'] = everything['clusters']

fi = pd.read_csv('/home/emily/phd/2_interpretability/features/outputs/RF_shap_class_fi.csv')
fi = fi.drop(columns='0')
sign = pd.read_csv('/home/emily/phd/2_interpretability/features/outputs/RF_shap_class_sign.csv')
sign = sign.drop(columns='0')
categories = np.arange(0,20,1)
N = len(categories)
angles = np.linspace(0, 2 * pi, N, endpoint=False)
angles_mids = angles + (angles[1] / 2)

fi.columns = df.columns[:-1]
sign.columns = df.columns[:-1]

def polar_twin(ax):
    ax2 = ax.figure.add_axes(ax.get_position(), projection='polar', 
                             label='twin', frameon=False,
                             theta_direction=ax.get_theta_direction(),
                             theta_offset=ax.get_theta_offset())
    ax2.xaxis.set_visible(False)

    # There should be a method for this, but there isn't... Pull request?
    ax2._r_label_position._t = (22.5 + 180, 0.0)
    ax2._r_label_position.invalidate()

    # Bit of a hack to ensure that the original axes tick labels are on top of
    # whatever is plotted in the twinned axes. Tick labels will be drawn twice.
    for label in ax.get_yticklabels():
        ax.figure.texts.append(label)

    return ax2

for c, K in enumerate(list(df.columns[:-1])):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles_mids)
    ax.set_xticklabels(categories)
    ax.xaxis.set_minor_locator(FixedLocator(angles))

    # duplicate axes
    ax_fi = polar_twin(ax)
    #ax_fi.set_theta_offset(pi / 2)
    #ax_fi.set_theta_direction(-1)
    ax_fi.set_xticks(angles)
    ax_fi.set_xticklabels(categories)
    ax_fi.xaxis.set_minor_locator(FixedLocator(angles_mids))

    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.set_yticks([])
    # ax_fi.set_yticklabels([".2", ".4", ".6"], color="black", size=8)
    # ax_fi.set_ylim(0, 0.6)

    #Draw ylabels
    ax_fi.set_rlabel_position(0)
    ax_fi.set_yticks([])
    # ax_fi.set_yticklabels([ "", "", "0", "", ""], color="black", size=8)
    # ax_fi.set_ylim(-0.06, 0.06)

    # values0 = fi.T.loc[K]
    # ax_fi.text(angles_mids[1], values0.median(), np.round(values0.median(),3), size=14)
    # HANGLES = np.linspace(0, 2 * np.pi)
    # H1 = np.ones(len(HANGLES)) * values0.median()
    # ax.plot(HANGLES, H1, linewidth=3, color='black')

    grouped = df.groupby('clusters').mean()
    values0 = grouped.T.loc[K]
    for i, value in enumerate(values0):
        ax.bar(angles_mids[i], values0[i], width=angles[1] - angles[0],
        facecolor=colors[c], alpha=0.7, edgecolor='k', linewidth=1)
    ax.grid(True, axis='y', which='major')

    ax.text(angles_mids[1], values0.median(), np.round(values0.median(),3), size=14)
    HANGLES = np.linspace(0, 2 * np.pi)
    H1 = np.ones(len(HANGLES)) * values0.median()
    ax.plot(HANGLES, H1, linewidth=3, color='black')
    # ax.legend(loc='upper left', bbox_to_anchor=(0.9, 1))
    # plt.title('Cluster %s' % str(K))
    # plt.show()     

    # ax.grid(True, axis='x', which='minor')
    # ax.grid(False, axis='x', which='major')
    ax.grid(False, axis='y', which='major')
    # ax.legend(loc='upper left', bbox_to_anchor=(0.9, 1))
    # plt.title('Cluster %s' % str(K))
    plt.savefig('/home/emily/phd/2_interpretability/features/outputs/radar_feature_%s.png' % str(K))
    # plt.show()


