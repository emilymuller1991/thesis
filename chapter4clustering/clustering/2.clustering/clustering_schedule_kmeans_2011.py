import clustering
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
#import models
from plots import plots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import time
import sys

# get number of clusters
n_clusters = int(sys.argv[1])
# prefix for hpc
prefix = '/rds/general/user/emuller/home'
year = 'census2011'

# Using MAC as the first input vector
df = pd.read_csv(prefix + '/emily/phd/003_image_matching/keras_rmac-master/census2021outputs/census2011_rmac_feature_vector.csv')
# df = pd.read_csv(prefix + '/emily/phd/003_image_matching/auto_encoders/python_outputs/inference/half_resnet_vae/half_resnet_latent=256_final_encodings_epoch_68_dictionary_save_in_one_batch_T.csv')
# '/rds/general/user/emuller/home003_image_matching/auto_encoders/python_outputs/inference/resnet_vae_encodings_model_checkpoint_epoch_20_half_resnet_latent=256_lr=1e-3_drop=0.3_8th_copy_dictionary_.csv'
# format dataframe
df['0'] = df['0'].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))
# df['encoding'] = df['encoding'].apply(lambda x: 
#                            np.fromstring(
#                                x.replace('\n','')
#                                 .replace('[','')
#                                 .replace(']','')
#                                 .replace('  ',' '), sep=' '))
# X = np.array(df['encoding'].values.tolist())  
X = np.array(df['0'].values.tolist()) 
X, indices = np.unique(X, axis = 0, return_index =True )

# create dataframe without corrupt data
df2 = df.iloc[indices]
if X.shape[0] == df2.shape[0]:
    print ('Objects have same dimensions')

# fit PCA
pca = PCA(n_components=2)
pca.fit(X)
pca_X = pca.fit_transform(X)
#plots.heatmap(X, fitted_ = None, k=None, data = year + '_years_rmac.png' )
plt.clf()
print (pca.explained_variance_ratio_.sum())
metric = 'kmeans'
res = {}
for k in [n_clusters]:
    start_time = time.time()
    init = clustering.accuracy(n_clusters=k)
    scs, chs, dbs, fitted_, X_dist = init.kmeans(X)
    end_time = time.time()
    d_time = (end_time - start_time)/3600
    print ('Completed clustering in %s hours' % str(d_time))
    df2['clusters'] = fitted_
    #df2.to_csv(prefix + '/emily/phd/003_image_matching/keras_rmac-master/output/apr_all_rmac_feature_vector_clusters_%s.csv' % str(k))
    df2.to_csv(prefix + '/emily/phd/003_image_matching/clustering/output/census2011_clusters_%s.csv' % str(k))
    # get distance from each cluster to center and plot
    center_dists = np.array([X_dist[i][x] for i,x in enumerate(fitted_)])
    total_cluster_dist = [center_dists[np.where(fitted_ == i)[0]].sum() for i in range(k) ]
    sns.barplot(np.arange(0, k, 1), total_cluster_dist)
    plt.xlabel('Cluster')
    plt.ylabel('Total distance to centroid')
    # plt.savefig(prefix + '/emily/phd/003_image_matching/auto_encoders/python_outputs/clustering/half_resnet_vae/autoencoder_half_resnet_256_%s_magnitude_%s.png' % (str(year), str(k)))
    plt.savefig(prefix + '/emily/phd/003_image_matching/clustering/output/%s_magnitude_%s.png' % (str(year), str(k)))  
    plt.clf()
    # get cardinality 
    total_cluster_count = [np.where(fitted_ == i)[0].shape[0] for i in range(k) ]
    sns.barplot(np.arange(0, k, 1), total_cluster_count)
    plt.xlabel('Cluster')
    plt.ylabel('Points in Cluster')
    # plt.savefig(prefix + '/emily/phd/003_image_matching/auto_encoders/python_outputs/clustering/half_resnet_vae/autoencoder_half_resnet_256_%s_cardinality_%s.png' % (str(year), str(k)))  
    plt.savefig(prefix + '/emily/phd/003_image_matching/clustering/output/%s_cardinality_%s.png' % (str(year), str(k)))  
    plt.clf()
    # get magnitude versus cardinality
    sns.regplot(total_cluster_count, total_cluster_dist) 
    plt.xlabel('Counts')
    plt.ylabel('Total distance to centroid')
    # plt.savefig(prefix + '/emily/phd/003_image_matching/auto_encoders/python_outputs/clustering/half_resnet_vae/autoencoder_half_resnet_256_%s_mag_vs_card_%s.png' % (str(year), str(k))) 
    plt.savefig(prefix + '/emily/phd/003_image_matching/clustering/output/%s_mag_vs_card_%s.png' % (str(year), str(k)))  
    plt.clf()

    res[k] = [k, scs, chs, dbs, sum(total_cluster_dist), d_time]
    #plots.pca_2d(pca_X, fitted_, fitted = metric + 'rmac_block4' + str(k))
    #plots.heatmap(X, fitted_, k, data = year + '_rmac')
    #plt.clf()
    
res_df = pd.DataFrame(res).T
res_df.columns = ['k', 'scs', 'chs', 'dbs', 'dist', 'time']
res_df.to_csv(prefix + '/emily/phd/003_image_matching/clustering/output/%s_metrics_for_%s_clusters.csv' % (str(year), str(k)))  
# res_df.to_csv(prefix + '/emily/phd/003_image_matching/auto_encoders/python_outputs/clustering/half_resnet_vae/autoencoder_half_resnet_256_%s_metrics_for_%s_clusters.csv' % (str(year), str(k)))  

print ('Completed programme for clustering')
# plt.figure(figsize = (16,16))
# fig, axs = plt.subplots(2,2)
# for ax in axs.flat:
#     #ax.grid(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
# axs[0,0].plot(res_df['k'], res_df['scs'])
# axs[0,0].set_title('Silhouette')
# # axs[0,1].plot(res_df['k'], res_df['chs'], 'tab:orange')
# # axs[0,1].set_title('Calinski Harabz')
# axs[1,0].plot(res_df['k'], res_df['dbs'], 'tab:green')
# axs[1,0].set_title('David Bouldin')
# axs[1,1].plot(res_df['k'], res_df['time'], 'tab:red')
# axs[1,1].set_title('Time (h)')
# fig.tight_layout(pad=3.0)
# plt.savefig(prefix + '/emily/phd/003_image_matching/clustering/output/%s_rmac_metrics.png' % year)
# plt.clf()

# plt.plot(res_df['k'], res_df['dist'], marker="o")
# #plt.plot(res_df['k'], res_df['dist'].apply(lambda x: sum(x)) , marker="o")
# plt.xlabel('Number of Clusters')
# plt.ylabel('Total Distance to Centroids')
# plt.savefig(prefix + '/emily/phd/003_image_matching/clustering/output/%s_years_rmac_optimum.png' % year)
# plt.clf()




