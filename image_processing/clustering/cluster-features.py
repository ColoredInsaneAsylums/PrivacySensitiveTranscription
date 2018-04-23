import argparse
import numpy as np
import _pickle as pickle

from clusterers import DBSCAN, HDBSCAN

# train a density-based model using the feature vectors
def main(feats_path, clstr_algo):
    print('[INFO] Preparing to cluster features from ' + feats_path)

    # load feature vectors
    with open(feats_path, 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        index = unpickler.load()
        index = {name: vector for name, vector in index.items() if vector is not None}

    # reshape 3d vectors to 2d
    dataset = list(index.values())
    dataset = np.asarray(dataset)
    # clusterer model
    print('[INFO] Using the ' + clstr_algo.upper() + ' algorithm to cluster features')
    if clstr_algo == 'dbscan':
        clusterer = DBSCAN()
    elif clstr_algo == 'hdbscan':
        clusterer = HDBSCAN()

    # cluster and persist labels
    clusterer.fit(dataset)
    output = '../labels/hdbscan_' + feats_path.split('.pickle')[0].split('/')[-1] + '.pickle'
    with open(output, 'wb') as handle:
        pickle.dump(dict(zip(index.keys(), clusterer.labels)), handle, protocol=4)

if __name__ == '__main__':
    # require filepath of features and name of clusterer to use
    parser = argparse.ArgumentParser(description='Extract image feature vectors using feature descriptors')
    parser.add_argument('-f', '--feats', required=True,
                        nargs='?', action='store', const='./features/hog_features.pickle',
                        type=str, dest='feats_path',
                        help='The filepath of the feature vectors')
    parser.add_argument('-a', '--algorithm', required=True,
                        choices=['dbscan', 'hdbscan'],
                        nargs='?', action='store', const='hdbscan',
                        type=str, dest='clstr_algo',
                        help='The clustering algorithm to use')

    args = vars(parser.parse_args())
    feats_path = args['feats_path']
    clstr_algo = args['clstr_algo']

    main(feats_path, clstr_algo)

