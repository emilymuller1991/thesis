import pandas as pd 
import numpy as np 

# read in output from qgis spatial merge
df = pd.read_csv('/home/emily/phd/002_validation/buildings/ukbuildings_centroid_5nn_30m_panoid_join.csv')
df['distance_m'] = df['distance'].apply(lambda x: x*111139)

both = df.copy()
# get the angle at which to merge to be building facing
from numpy import arctan2,degrees 
# get angle between two lat,lon's in python. # nearest corresponds to building centroid

both['build_lat'] = pd.to_numeric(both['nearest_y'])
both['build_lon'] = pd.to_numeric(both['nearest_x'])
both['dY'] = both['build_lon']-both['lon']  
both['dX']= both['build_lat']-both['lat']  
both['angle'] = degrees(arctan2(both['dX'],both['dY']))
both['quad_b'] = ((both['azi'] + 90) % 360 ) // 90 
both['quad_d'] = ((both['azi'] + 270) % 360 ) // 90  
both['quad_angle'] = ((both['angle'] + 270) % 360 ) // 90 
both['half_b'] = np.where((both['quad_b'] > 1.0) & (both['quad_angle'] > 1.0), 'd', 0) 
both['half_d'] = np.where((both['quad_d'] > 1.0) & (both['quad_angle'] > 1.0), 'b', 0)  
both['half_dd'] = np.where((both['quad_d'] < 2.0) & (both['quad_angle'] < 2.0), 'b', 0) 
both['half_bb'] = np.where((both['quad_b'] < 2.0) & (both['quad_angle'] < 2.0), 'd', 0)  

# create new column with idx and angle
both['fov'] = both[['half_b', 'half_d', 'half_dd', 'half_bb']].max(1) 
both['image'] = both['idx'].astype(int).astype(str) + '_' + both['fov']

# create new column bui_id + prop_id 
both['bui_prop_id'] = both.apply(lambda x: str(x.bui_id) + '_' + str(x.prop_id), axis=1 )
both['unique_id'] = [str(i) for i in range(both.shape[0])]

# select unqiue bui_prop_id based on shortest distance from panoid
dfc = both.groupby('bui_prop_id')['distance_m']
unique = both.copy()
unique['minimum'] = unique.assign(min=dfc.transform(min))['min']
unique['keep'] = unique.apply(lambda x: 1 if np.float(x.distance_m) == np.float(x.minimum) else 0, axis = 1)

keep = unique[unique['keep'] == 1]
dfc = keep.groupby('image')['distance_m']
unique = keep.copy()
unique['minimum'] = unique.assign(min=dfc.transform(min))['min']
unique['keep'] = unique.apply(lambda x: 1 if np.float(x.distance_m) == np.float(x.minimum) else 0, axis = 1)
keep = unique[unique['keep'] == 1]

# merge cluster assignments 
clusters = pd.read_csv('/media/emily/south/rmac_feature_vectors/2018_rmac_feature_vector_clusters_20.csv')
merged = keep.merge(clusters[['Unnamed: 0.1', 'clusters']], left_on='image', right_on=['Unnamed: 0.1'], how='left', indicator=True)

# keep useful columns and save to .csv
useful = ['bui_prop_id', 'height', 'age', 'use', 'distance_m', 'image', 'clusters']
keep = merged[useful]

keep.to_csv('/home/emily/phd/002_validation/buildings/1to1_panoids_building.csv')