# environmental exposure across urban typologies
import pandas as pd 

# read in urban classifications
df_8 =  pd.read_csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/both_years_proportions_all_nonna_lsoa_hierarchical8.csv')
df_13 = pd.read_csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/both_years_proportions_all_nonna_lsoa_hierarchical13.csv')
df_24 = pd.read_csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/both_years_proportions_all_nonna_lsoa_hierarchical24.csv')

df_8 = df_8[df_8['same'] == 1]
df_13 = df_13[df_13['same'] == 1]
df_24 = df_24[df_24['same'] == 1]

df_8.columns = ['Unnamed:0', 'lsoa', 'hierarchical8x', 'hierarchical8y', 'same']
df_13.columns = ['Unnamed:0', 'lsoa', 'hierarchical13x', 'hierarchical13y', 'same']
df_24.columns = ['Unnamed:0', 'lsoa', 'hierarchical24x', 'hierarchical24y', 'same']

# https://data.london.gov.uk/download/laei-2008-concentration-maps/dcb54931-9295-4d02-9650-dcefc645d58b/laei-2008-pm25e-20mgrid-shp.zip# 
# pre-processing of (lat,lon) for sql merge to oa
# from pyproj import Proj, transform
# exposures = pd.read_csv('/home/emily/phd/drives/phd/chapter6substantive/source/exposures/laei-2008-pm25e-20mgrid-shp/LAEI08_NOXa.csv') # exported to csv in QGIS
# inProj = Proj('epsg:27700')
# outProj = Proj('epsg:4326')
# x1, y1 = transform(inProj, outProj, exposures['Easting'], exposures['Northing'])
# exposures['lat'] = x1
# exposures['lon'] = y1
# exposures.to_csv('/home/emily/phd/drives/phd/chapter6substantive/source/exposures/laei-2008-pm25e-20mgrid-shp/LAEI08_NOXa_lat_lon.csv', index=False)

# read in environmental exposure by OA
#exposures = pd.read_csv('/home/emily/phd/drives/phd/chapter3data/outputs/exposure_pm25_merge_oa.csv')
#exposures = pd.read_csv('/home/emily/phd/drives/phd/chapter3data/outputs/exposure_pm10a_merge_oa.csv')
#exposures = pd.read_csv('/home/emily/phd/drives/phd/chapter3data/outputs/exposure_NO2a_merge_oa.csv')
exposures = pd.read_csv('/home/emily/phd/drives/phd/chapter3data/outputs/exposure_NOXa_merge_oa.csv')
# read in OA, LSOA file.
oa = pd.read_csv('/home/emily/phd/002_validation/source/oa/OA_2011_London_gen_MHW_4326_all_fields.csv')
exposures = exposures.merge(oa[['OA11CD', 'LSOA11CD']], left_on='oa_name', right_on='OA11CD' )
exp = exposures.groupby('LSOA11CD').mean()
exp['lsoa'] = exp.index

merged = exp.merge(df_8[['lsoa', 'hierarchical8x']], left_on='lsoa', right_on='lsoa', how='left')
merged = merged.merge(df_13[['lsoa', 'hierarchical13x']], left_on='lsoa', right_on='lsoa', how='left')
merged = merged.merge(df_24[['lsoa', 'hierarchical24x']], left_on='lsoa', right_on='lsoa', how='left')

merged.to_csv('/home/emily/phd/drives/phd/chapter3data/outputs/exposure_NOXa_merge_hierarchicalx.csv')

# mask uncertain LSOA's
mask = pd.read_csv('chapter3data/outputs/sampling_rate_lsoa_all_years_sql.csv')
masked = mask[mask['2021'] > 0.75]
merged_mased = merged[merged['lsoa'].isin(masked['lsoa11cd.1'])]
merged.to_csv('/home/emily/phd/drives/phd/chapter3data/outputs/exposure_pm10a_merge_hierarchicalx.csv')
