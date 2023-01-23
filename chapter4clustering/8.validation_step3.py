import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import numpy as np
from pySankey import sankey

# read in panoids merged to validation files (verisk and green)
verisk = pd.read_csv('/home/emily/phd/002_validation/buildings/1to1_panoids_building_buiid.csv')
os = pd.read_csv('/home/emily/phd/002_validation/green/osgreen_image_angle.csv')
clusters = pd.read_csv('/media/emily/south/rmac_feature_vectors/2018_rmac_feature_vector_clusters_20.csv')

# edit verisk labels
commercial = ['GENERAL COMMERCIAL - MIXED USE', 'GENERAL COMMERCIAL - MIXED USE - DERELICT']
retail = ['RETAIL WITH OFFICE/RESIDENTIAL ABOVE', 'RETAIL ONLY']
offices = [ 'OFFICE ONLY', 'OFFICE WITH RETAIL ON GROUND FLOOR']
industry = ['INDUSTRY - MANUFACTURING/PROCESSING']
all = commercial + retail + offices + industry

verisk['Terraced'] = verisk.apply(lambda x: 1 if x.use == 'RESIDENTIAL ONLY' and x.age == 'HISTORIC' else 0, axis=1)
verisk['Residential'] = verisk.apply(lambda x: 1 if x.use == 'RESIDENTIAL ONLY' and x.Terraced == 0 else 0, axis=1)
verisk['Commercial'] = verisk.apply(lambda x: 1 if x.use in all else 0, axis=1)

verisk["category"] = verisk[['Terraced', 'Residential', 'Commercial']].idxmax(axis=1)

# add greenspace image id's to dataframe
os['category'] = os.apply(lambda x: 'Green', axis=1)

# remove all images from verisk which should be in green
green_images = list(os[['image']]['image'])
building_images = list(verisk[['image']]['image'])
building_less_green = set(building_images) - set(green_images)

# sub-select buildings less green from verisk
v = verisk[verisk['image'].isin(list(building_less_green))][['image', 'clusters', 'category']]

# merge opengreenspace to clusters
osm = os[['image', 'category']].merge(clusters, left_on='image', right_on='Unnamed: 0')
osm = osm[['image', 'clusters', 'category']]
osm['category'] = osm.apply(lambda x: 'Green', axis=1)

# concatenate both
both = pd.concat([v,osm])

def cleanup(x):
    if x in [2,3,5,7,16,18,12,19,17]:
        return np.nan
    elif x in [4,8,9,14]:
        return 'Low-density'
    elif x in [1, 11]:
        return 'Green'
    elif x == 0:
        return 'Terraced'
    elif x == 15:
        return 'Commercial'
    else:
        return x
    
both['clusters_edited'] = both['clusters'].apply(lambda x: cleanup(x))
keep2 = both.dropna()
sankey.sankey(keep2['clusters_edited'], keep2['category'], aspect=20, fontsize=12)

# Get current figure
fig = plt.gcf()

# Set size in inches
fig.set_size_inches(6, 10)

# Set the color of the background to white
fig.set_facecolor("w")

# Save the figure
fig.savefig("outputs/step2val_building_less_green.png", bbox_inches="tight", dpi=150)

################################################################ STEP 3
both = pd.concat([v,osm])

def cleanup(x):
    if x in [2,3,5,7,10,17,6,16,18,12,19,13]:
        return np.nan
    elif x in [4,8,9,14]:
        return 'Low-density'
    elif x in [1, 11]:
        return 'Green'
    elif x == 0:
        return 'Terraced'
    elif x == 15:
        return 'Commercial'
    else:
        return x
    
both['clusters_edited'] = both['clusters'].apply(lambda x: cleanup(x))
keep2 = both.dropna()
sankey.sankey(keep2['clusters_edited'], keep2['category'], aspect=20, fontsize=12)

# Get current figure
fig = plt.gcf()

# Set size in inches
fig.set_size_inches(6, 10)

# Set the color of the background to white
fig.set_facecolor("w")

# Save the figure
fig.savefig("outputs/step3val_building_less_green.png", bbox_inches="tight", dpi=150)
