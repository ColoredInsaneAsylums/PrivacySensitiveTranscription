import argparse
import numpy as np
import _pickle as pickle

from clusterers import HDBSCAN

# train a density-based model using the feature vectors
def main(feats_path, max_cluster_size):
    print('[INFO] Preparing to cluster features from \'' + feats_path + '\'')

    # load feature vectors
    with open(feats_path, 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        index = unpickler.load()

    # labels
    labels = {}

    # cluster iteratively
    min_cluster_size = max_cluster_size
    while len(index) > 0 and min_cluster_size >= 2:
        print('[INFO] Clustering with min_cluster_size = ' + str(min_cluster_size))
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)

        # reshape 3d vectors to 2d
        dataset = list(index.values())
        dataset = np.asarray(dataset)

        # fit clusterer
        clusterer.fit(dataset)

        # update labels
        new_labels = clusterer.model.labels_
        new_labels = dict(zip(index.keys(), new_labels))

        current_max = max(labels.values()) if len(labels) > 0 else 0
        new_labels = [label + current_max for label in new_labels]
        labels.update(new_labels)

        # get PHOCS of noise points to recluster
        index = {k: v for k, v in index if new_labels[k] < 1}
        min_cluster_size -= 1

    # save labels to disk
    output = './labels/iterative_hdbscan_' + str(max_cluster_size) + '_hdbscan_model.pickle'
    print('[INFO] Saving labels to ' + output)
    with open(output, 'wb') as handle:
        pickle.dump(self.__dict__, handle, protocol=4)

if __name__ == '__main__':
    # require filepath of features and name of clusterer to use
    parser = argparse.ArgumentParser(description='Extract image feature vectors using feature descriptors')
    parser.add_argument('-p', '--path', required=True,
                        nargs='?', action='store', const='./features/hog_features.pickle',
                        type=str, dest='feats_path',
                        help='The filepath of the feature vectors')
    parser.add_argument('-m', '--max_cluster_size', required=True,
                        nargs='?', action='store', const=2,
                        type=int, dest='max_cluster_size',
                        help='The max clustering size to begin with')

    args = vars(parser.parse_args())
    feats_path = args['feats_path']
    max_cluster_size = args['max_cluster_size']

    main(feats_path, max_cluster_size)

