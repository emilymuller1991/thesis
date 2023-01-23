import pandas as pd 
import numpy as np 
from pyproj import Proj, transform
from numpy import arctan2,degrees 

# read in output from qgis spatial merge
df = pd.read_csv('/home/emily/phd/002_validation/green/clipped_panoids_merge_buffer_green.csv')

# convert 27700 into 4326 crs 
df['osx'] = df['Added geom info_xcoord']
df['osy'] = df['Added geom info_ycoord']
inProj = Proj(init='epsg:27700')
outProj = Proj(init='epsg:4326')
df['osgreen'] = df.apply(lambda x: transform(inProj, outProj, x.osx, x.osy), axis = 1)

df['osx'] = df['osgreen'].apply(lambda x: x[0])
df['osy'] = df['osgreen'].apply(lambda x: x[1])

df = df.dropna(axis=1)
df = df.drop(columns=['Added geom info_xcoord', 'Added geom info_ycoord', 'osgreen'])

both = df
####### get greenspace facing image
both['os_lat'] = pd.to_numeric(both['osy'])
both['os_lon'] = pd.to_numeric(both['osx'])
both['dY'] = both['os_lon']-both['lon']  
both['dX']= both['os_lat']-both['lat']  
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

both.to_csv('/home/emily/phd/002_validation/green/osgreen_image_angle.csv')