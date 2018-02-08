import hdbscan
import sklearn.cluster
import _pickle as pickle

# Density-Based Spatial Clustering of Applications with Noise
class DBSCAN:

    # initialize DBSCAN model
    def __init__(self, eps=0.3, min_samples=2, metric='cosine',
                 algorithm='brute', n_jobs=-1):
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

        with open('./models/' + prefix + '_dbscan_model.pickle', 'wb') as handle:
            pickle.dump(self.model, handle, protocol=4)

        with open('./labels/' + prefix + '_dbscan_labels.pickle', 'wb') as handle:
            pickle.dump(self.labels, handle)

        print('[INFO] DBSCAN model saved to \'./models/' + prefix + '_dbscan_model.pickle\'')
        print('[INFO] Cluster labels saved to \'./labels/' + prefix + '_dbscan_labels.pickle\'')

# Hierarchical DBSCAN
class HDBSCAN:

    # initialize HDBSCAN model
    def __init__(self, min_cluster_size=2, metric='cosine', algorithm='best'):
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric,
                                         algorithm=algorithm)

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
            pickle.dump(self.model, handle, protocol=4)

        with open('./labels/' + prefix + '_hdbscan_labels.pickle', 'wb') as handle:
            pickle.dump(self.labels, handle)

        print('[INFO] HDBSCAN model saved to \'./models/' + prefix + '_hdbscan_model.pickle\'')
        print('[INFO] Cluster labels saved to \'./labels/' + prefix + '_hdbscan_labels.pickle\'')

