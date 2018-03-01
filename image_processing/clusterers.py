import hdbscan
import sklearn.cluster
import _pickle as pickle

# Density-Based Spatial Clustering of Applications with Noise
class DBSCAN:

    # initialize DBSCAN model
    def __init__(self, eps=0.3, min_samples=2, metric='braycurtis',
                 algorithm='auto', n_jobs=-1):
        self.clusterer = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                                                algorithm=algorithm, n_jobs=n_jobs)

    # cluster data
    def fit(self, dataset):
        self.model = self.clusterer.fit(dataset)

        self.labels = self.model.labels_
        n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise_ = len([i for i, val in enumerate(self.labels) if val == -1])

        print('[INFO] Estimated number of clusters: %d' % n_clusters_)
        print('[INFO] Estimated number of noise points: %d' % n_noise_)

    # persist model
    def save(self, ims, prefix):
        self.labels = dict(zip(ims, self.labels))
        self.core_samples = [list(ims)[idx] for idx in self.model.core_sample_indices_]

        with open('./models/' + prefix + '_dbscan_model.pickle', 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=4)

        print('[INFO] DBSCAN model saved to \'./models/' + prefix + '_dbscan_model.pickle\'')

# Hierarchical DBSCAN
class HDBSCAN:

    # initialize HDBSCAN model
    def __init__(self, min_cluster_size=2, metric='braycurtis', algorithm='best'):
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric,
                                         algorithm=algorithm, core_dist_n_jobs=-1)

    # cluster data
    def fit(self, dataset):
        self.model = self.clusterer.fit(dataset)

        self.labels = self.model.labels_
        n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise_ = len([i for i, val in enumerate(self.labels) if val == -1])

        print('[INFO] Estimated number of clusters: %d' % n_clusters_)
        print('[INFO] Estimated number of noise points: %d' % n_noise_)

    # persist model
    def save(self, ims, prefix):
        self.labels = dict(zip(ims, self.labels))

        with open('./models/' + prefix + '_hdbscan_model.pickle', 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=4)

        print('[INFO] HDBSCAN model saved to \'./models/' + prefix + '_hdbscan_model.pickle\'')

