import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import glob
from shapely.geometry import Point
from PIL import Image

df = pd.read_csv('003_image_matching/keras_rmac-master/output/apr_all_rmac_feature_vector_clusters_20.csv')
df['0'] = df['0'].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))
#df = df.sample(10000)
X = np.array(df['0'].values.tolist())
n_clusters = df['clusters'].max() + 1


def get_cluster_centers(df):
    total = df[['0', 'clusters']].groupby('clusters').sum()   
    count = df[['0', 'clusters']].groupby('clusters').count()
    centroid = total/count
    return list(centroid['0'])

def k_mean_distance(X, centroids):
    # Calculate Euclidean distance for each data point assigned to centroid 
    distance = [(X[i]-centroids[i])**2 for i in range(512)]
    dist = np.sqrt(sum(distance))
    # return the mean value
    return dist

# def plot_one_centroid(i, collection):
#     #n = len(subspace.index)
#     centroid = Point( 50,50 )

#     # again, a workaround for indexing difference
#     candidates = collection[collection.clusters==i]
#     candidates.sort_values("distance")
#     it = np.min([1000, candidates.shape[0]])
#     for j in range(15):
#         try:
#             best = candidates.iloc[j]
#             im = Image.open(best.local_path)
#             im.thumbnail((thumb_side,thumb_side),Image.ANTIALIAS)
#             closest_open = min(open_grid,key=lambda x: centroid.distance(x))
#             x = int(closest_open.x) * thumb_side
#             y = int(closest_open.y) * thumb_side
#             canvas.paste(im,(x,y))
#             #idx = collection[collection.local_path==best.local_path].index
#             open_grid.remove(closest_open)
#             print (i)

#         except:
#             print ('error')

def plot_one_centroid(i, collection):
    #n = len(subspace.index)
    centroid = Point( 50,50 )

    # again, a workaround for indexing difference
    candidates = collection[collection.clusters==i]
    candidates.sort_values("distance")
    it = np.min([1000, candidates.shape[0]])
    for j in range(it):
        try:
            best = candidates.iloc[j]
            im = Image.open(best.local_path)
            im.thumbnail((thumb_side,thumb_side),Image.ANTIALIAS)
            closest_open = min(open_grid,key=lambda x: centroid.distance(x))
            x = int(closest_open.x) * thumb_side
            y = int(closest_open.y) * thumb_side
            canvas.paste(im,(x,y))
            #idx = collection[collection.local_path==best.local_path].index
            open_grid.remove(closest_open)
            print (i)
        except Exception as e:
            print (e)

# plot k-means centroid subspace
X_center = get_cluster_centers(df) 
centroids = X_center
pca = PCA(n_components=2)
pca.fit(X_center)
subspace = pd.DataFrame(pca.fit_transform(X_center), columns=['x','y'])
x = subspace.x
y = subspace.y
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(x, y)
plt.savefig('2_interpretability/growing_entourage/outputs/all_apr/centroid_subspace_1.png')

# calculate distance from centroid
total_distance = []
for i in range(df.shape[0]):
    cluster = df.iloc[i]['clusters']
    distance = k_mean_distance(X[i], centroids[cluster])
    total_distance.append(distance)
df['distance'] = total_distance

# image data
BASE = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/project/pathways/live/Transferability/emily_phd_images/all_apr_2021/'
local_path = []
for i in range(df.shape[0]):
    local_path.append( BASE + os.path.basename(str(df['Unnamed: 0.1.1'].iloc[i]) +  '.png' ))
df['local_path'] = local_path

# grid
num_bins = 100
print (subspace.x.min(), subspace.x.max(), subspace.y.min(), subspace.y.max())
x = [-5,5]
y = [-5,5]

tmp = pd.DataFrame(x,columns=["x"])
tmp["y"] = y
subspace = subspace.append(tmp)
subspace['x_bin'] = pd.cut(subspace['x'],num_bins,labels=False)
subspace['y_bin'] = pd.cut(subspace['y'],num_bins,labels=False)
x = subspace.x_bin
y = subspace.y_bin
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(x, y)
plt.savefig('2_interpretability/growing_entourage/outputs/all_apr/centroid_subspace_2.png')
# now we can remove the extreme points we used as grid expanders
subspace = subspace[:n_clusters]
# now to expand the grid by simple multiplication
factor = 1
subspace["x_grid"] = subspace.x_bin * factor
subspace["y_grid"] = subspace.y_bin * factor
centroid_point = []
n = len(subspace.index)
for i in range(n):
    centroid_point.append(Point(subspace.x_grid.loc[i],subspace.y_grid.loc[i]))
#centroid_point.append(Point(0,0))
subspace['centroid_point'] = centroid_point

# GRID LIST
grid_side = num_bins * factor
x,y = list(range(grid_side))*grid_side, np.repeat(range(grid_side),grid_side)
grid_list = pd.DataFrame(x,columns=['x'])
grid_list['y'] = y
point = []
n = len(grid_list.index)
for i in range(n):
    point.append(Point(grid_list.x.loc[i],grid_list.y.loc[i]))

grid_list['point'] = point
open_grid = list(grid_list.point)
centroids = list(subspace.centroid_point)

# REMOVAL OF CENTROIDS FROM OPEN_GRID LIST
open_grid = [item for item in open_grid if item not in centroids]

# PLOT
thumb_side=64
px_w = thumb_side * grid_side
px_h = thumb_side * grid_side
canvas = Image.new('RGB',(px_w,px_h),(50,50,50))

descriptor = 'growing_entourage'
for i in range(n_clusters):
    canvas = Image.new('RGB',(px_w,px_h),(50,50,50))
    open_grid = list(grid_list.point)
    open_grid = [item for item in open_grid if item not in centroids]
    plot_one_centroid(i, df)
    canvas.save('2_interpretability/growing_entourage/outputs/all_apr/%s_all_apr_%s_of_%s.png' % (descriptor, str(i), str(n_clusters)))

