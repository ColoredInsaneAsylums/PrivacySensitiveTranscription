import argparse
import os.path as path
import _pickle as pickle

from sklearn.decomposition import PCA

# conduct Principle Component Analysis to reduce vector dimensionality
def main(feats_path, variance):
    with open(feats_path, 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        labels = unpickler.load()

    labels = {name: vector for name, vector in labels.items() if vector is not None}
    names = list(labels.keys())
    vectors = list(labels.values())

    print('[INFO] Conducting PCA on ' + feats_path + ' with variance ' + str(variance))
    pca = PCA(n_components=variance)
    vectors = pca.fit_transform(vectors)

    # save reduced vectors
    base = path.basename(feats_path)
    name = path.splitext(base)[0]

    output = name + '_var' + str(int(variance * 100)) + '.pickle'
    print('[INFO] Saving reduced vectors to ' + output)
    with open(output, 'wb') as handle:
        pickle.dump(dict(zip(names, vectors)), handle)

if __name__ == '__main__':
    # require features filepath and variance
    parser = argparse.ArgumentParser(description='Reduce vector dimensionality')
    parser.add_argument('-f', '--feats', required=True,
                        nargs=1, action='store',
                        type=str, dest='feats_path',
                        help='The filepath of the features')
    parser.add_argument('-v', '--variance', required=True,
                        nargs=1, action='store',
                        type=float, dest='variance',
                        help='The amount of variance to preserve')

    args = vars(parser.parse_args())
    feats_path = args['feats_path'][0]
    variance = args['variance'][0]

    main(feats_path, variance)

