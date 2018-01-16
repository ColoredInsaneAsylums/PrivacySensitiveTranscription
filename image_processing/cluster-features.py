import argparse
import numpy as np
import _pickle as pickle

from clusterers import DBSCAN, HDBSCAN

# train a DBSCAN model using the feature vectors
def main(feats_path, clstr_algo):
    print('[INFO] Preparing to cluster features from \'' + feats_path + '\'')

    # load feature vectors
    with open(feats_path, 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        index = unpickler.load()

    # reshape 3d vectors to 2d
    dataset = list(index.values())
    dataset = np.asarray(dataset)
    # clusterer model
    print('[INFO] Using the ' + clstr_algo.upper() + ' algorithm to cluster features')
    if clstr_algo == 'dbscan':
        clusterer = DBSCAN(metric='cosine', algorithm='brute')
    elif clstr_algo == 'hdbscan':
        clusterer = HDBSCAN(metric='cosine')

    # cluster and persist model
    clusterer.fit(dataset)
    clusterer.save(index.keys())

if __name__ == '__main__':
    # require filepath of features and name of clusterer to use
    parser = argparse.ArgumentParser(description='Extract image feature vectors using feature descriptors')
    parser.add_argument('-p', '--path', required=True,
                        nargs='?', action='store', const='./features/hog_features.pickle',
                        type=str, dest='feats_path',
                        help='The filepath of the feature vectors')
    parser.add_argument('-a', '--algorithm', required=True,
                        choices=['dbscan', 'hdbscan'],
                        nargs='?', action='store', const='dbscan',
                        type=str, dest='clstr_algo',
                        help='The clustering algorithm to use')

    args = vars(parser.parse_args())
    feats_path = args['feats_path']
    clstr_algo = args['clstr_algo']

    main(feats_path, clstr_algo)

