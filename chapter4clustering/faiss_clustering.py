import faiss 
import pandas as pd
import wandb
import time
import sys

wandb_name = sys.argv[1]
wandb.login(key='188ce11fc669c7ea709e957bece5360053daabe8')
wandb.init(id = id, project='faiss_clustering', entity='emilymuller1991')


n_clusters = 20
# prefix for hpc
# prefix = '/rds/general/user/emuller/home'
prefix = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home/'
year = 'both_years_zoom'

df = pd.read_csv(prefix + '/emily/phd/003_image_matching/keras_rmac-master/census2021outputs/census_2011_and_census_2021_zoom.csv')
df['0'] = df['0'].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))
df = df[1:1000]
X = np.array(df['0'].values.tolist()) 

def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        (list: ids of data in each cluster, float: loss value)
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    #clus.seed = np.random.randint(1234)
    clus.seed = 0

    clus.niter = 20000000
    clus.max_points_per_centroid = 1000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)

    # losses = faiss.vector_to_array(clus.obj)

    stats = clus.iteration_stats
    losses = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])
    if verbose:
        print('Iteration %s' % str(stats.size()))
        print('k-means loss evolution: {0}'.format(losses))
        wandb.log(
            {
                "loss_train": losses[-1]
            }
        )

    return [int(n[0]) for n in I], losses

# start clustering 
start = time.time()
c, l = run_kmeans(X, n_clusters, verbose=True)
finish = time.time()
print ('clustering algorithm finished in %s seconds' % str(finish-start))

df['clusters'] = c 
df.to_csv(prefix + '/emily/phd/003_image_matching/clustering/output/both_years_zoom_clusters_seed0.csv')
print ('losses')
