# 3 plots: elbow method, silhoette score, davies-bouldin score
import pandas as pd 

prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home/emily/phd/003_image_matching/clustering/output/2018/'
dataframes = [] 
for k in [4,6,8,10,12,14,16,18,20,25,30]:
    df = pd.read_csv(prefix + 'metrics_for_%s_clusters.csv' % str(k))
    dataframes.append(df) 
df = pd.concat(dataframes)

