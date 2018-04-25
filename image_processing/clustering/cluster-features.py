import argparse
import hdbscan
import numpy as np
import os.path as path
import _pickle as pickle

# cluster feature vectors using HDBSCAN
def main(feats_path):
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
    print('[INFO] Using HDBSCAN to cluster features')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=None,
                                metric='braycurtis', algorithm='best',
                                core_dist_n_jobs=-1)

    # cluster
    clusterer.fit(dataset)

    # save labels
    base = path.basename(feats_path)
    name = path.splitext(base)[0]

    output = '../labels/hdbscan_' + name + '.pickle'
    with open(output, 'wb') as handle:
        pickle.dump(dict(zip(index.keys(), clusterer.labels)), handle, protocol=4)

if __name__ == '__main__':
    # require filepath of features
    parser = argparse.ArgumentParser(description='Extract image feature vectors using feature descriptors')
    parser.add_argument('-f', '--feats', required=True,
                        nargs=1, action='store',
                        type=str, dest='feats_path',
                        help='The filepath of the feature vectors')

    args = vars(parser.parse_args())
    feats_path = args['feats_path']

    main(feats_path)

