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
verisk['Low-density'] = verisk.apply(lambda x: 1 if x.use == 'RESIDENTIAL ONLY' and x.height < 5 and x.Terraced == 0 else 0, axis=1)
verisk['Medium-density'] = verisk.apply(lambda x: 1 if x.use == 'RESIDENTIAL ONLY' and x.height >= 5 and x.height < 10 and x.Terraced == 0 else 0, axis=1)
verisk['High-density'] = verisk.apply(lambda x: 1 if x.use == 'RESIDENTIAL ONLY' and x.height >= 10 and x.Terraced == 0 else 0, axis=1)
verisk['Commercial'] = verisk.apply(lambda x: 1 if x.use in all else 0, axis=1)
# verisk['Industrial'] = verisk.apply(lambda x: 1 if x.use == 'INDUSTRY - MANUFACTURING/PROCESSING' else 0, axis=1)

verisk["category"] = verisk[['Terraced', 'Low-density', 'Medium-density', 'High-density', 'Commercial']].idxmax(axis=1)

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
    if x in [3,5,17]:
        return np.nan
    elif x in [8,9]:
        return 'Low-density'
    elif x == 4:
        return 'Leafy green residential'
    elif x in [1, 11]:
        return 'Open green space'
    elif x == 0:
        return 'Terraced'
    elif x == 15:
        return 'Commercial'
    elif x in [2,6]:
        return 'Other green'
    elif x in [7,14,16,18]:
        return 'Vehicles'
    elif x == 10:
        return 'High-density'
    elif x == 12:
        return 'Sheds'
    elif x == 13:
        return 'Estates'
    elif x == 19:
        return 'Fences'
    else:
        return x

colorDict = {
    'Low-density':'#00fffffd',
    'Medium-density':'#00fffffd',
    'Leafy green residential':'#00fffffd',
    'Open green space':'#00ff00fd',
    'Green':'#00ff00fd',
    'Terraced':'#0000fffd',
    'Commercial':'#ffcc00fd',
    'Other green': '#008000fd',
    'Vehicles': '#ff0000fd',
    'High-density': '#ff6600fd' ,
    'Sheds': '#c87137fd', 
    'Estates': '#ff6600fd',
    'Fences': '#c87137fd'
}

both['clusters_grouped'] = both['clusters'].apply(lambda x: cleanup(x))
keep2 = both.dropna()
sankey.sankey(keep2['clusters_grouped'], keep2['category'], aspect=20, colorDict=colorDict, fontsize=12)

# Get current figure
fig = plt.gcf()

# Set size in inches
fig.set_size_inches(6, 10)

# Set the color of the background to white
fig.set_facecolor("w")

# Save the figure
fig.savefig("outputs/all_clusters_val_building_less_green.png", bbox_inches="tight", dpi=150)

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
fig.savefig("outputs/step1val_building_less_green.png", bbox_inches="tight", dpi=150)