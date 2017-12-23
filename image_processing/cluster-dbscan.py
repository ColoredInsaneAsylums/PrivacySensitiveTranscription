import numpy as np
import _pickle as pickle

#from sklearn import metrics
from sklearn.cluster import DBSCAN

# train a DBSCAN model using the feature vectors
def main():
    print('[INFO] Working...')

    # load feature vectors
    with open('./features/HOG_features.pickle', 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        index = unpickler.load()

    # reshape 3d vectors to 2d
    dataset = list(index.values())
    dataset = np.asarray(dataset)

    # train model
    dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine', algorithm='brute', n_jobs=-1)
    model = dbscan.fit(dataset)

    # cluster analysis
    core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True
    labels = model.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = len([i for i, val in enumerate(labels) if val == -1])

    print('[INFO] Estimated number of clusters: %d' % n_clusters_)
    print('[INFO] Estimated number of noise points: %d' % n_noise_)
    #print('[INFO] Homogeneity: %0.3f' % metrics.homogeneity_score(labels_true, labels))
    #print('[INFO] Completeness: %0.3f' % metrics.completeness_score(labels_true, labels))
    #print('[INFO] V-measure: %0.3f' % metrics.v_measure_score(labels_true, labels))
    #print('[INFO] Adjusted Rand Index: %0.3f'
    #      % metrics.adjusted_rand_score(labels_true, labels))
    #print('[INFO] Adjusted Mutual Information: %0.3f'
    #      % metrics.adjusted_mutual_info_score(labels_true, labels))
    #print('[INFO] Silhouette Coefficient: %0.3f'
    #      % metrics.silhouette_score(X, labels))

    labels = dict(zip(index.keys(), labels))

    # save model and labels
    with open('dbscan_model.pickle', 'wb') as handle:
        pickle.dump(model, handle)

    with open('labels.pickle', 'wb') as handle:
        pickle.dump(labels, handle)

    print('[INFO] DBSCAN model trained and saved to dbscan_model.pickle')
    print('[INFO] Cluster labels saved to labels.pickle')

if __name__ == '__main__':
    main()
