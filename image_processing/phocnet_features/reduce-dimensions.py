import argparse
import _pickle as pickle

from sklearn.decomposition import PCA

# conduct Principle Component Analysis to reduce vector dimensionality
def main(feats_path, variance):
    with open(feats_path, 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        labels = unpickler.load()

    names = list(labels.keys())
    vectors = list(labels.values())

    print('[INFO] Conducting PCA on ' + feats_path + ' with variance ' + str(variance))
    pca = PCA(n_components=variance)
    vectors = pca.fit_transform(vectors)

    feats_path = feats_path.split('.')
    feats_path = feats_path[0] + '_pca_' + str(int(variance * 100)) + '.' + feats_path[1]

    print('[INFO] Saving reduced vectors to ' + feats_path)
    with open(feats_path, 'wb') as handle:
        pickle.dump(dict(zip(names, vectors)), handle)

if __name__ == '__main__':
    # require features filepath and variance
    parser = argparse.ArgumentParser(description='Reduce vector dimensionality')
    parser.add_argument('-p', '--path', required=True,
                        nargs=1, action='store',
                        type=str, dest='feats_path',
                        help='The filepath of the features')
    parser.add_argument('-v', '--variance', required=True,
                        nargs=1, action='store',
                        type=float, dest='variance')

    args = vars(parser.parse_args())
    feats_path = args['feats_path'][0]
    variance = args['variance'][0]

    main(feats_path, variance)

