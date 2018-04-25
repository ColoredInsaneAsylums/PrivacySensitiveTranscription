import argparse
import _pickle as pickle

from sklearn.decomposition import PCA

# conduct Principle Component Analysis to reduce vector dimensionality
def main(feats_path, n_comps):
    with open(feats_path, 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        labels = unpickler.load()

    labels = {name: vector for name, vector in labels.items() if vector is not None}
    names = list(labels.keys())
    vectors = list(labels.values())

    print('[INFO] Conducting PCA on ' + feats_path + ' with ' + str(n_comps) + ' components')
    pca = PCA(n_components=n_comps)
    vectors = pca.fit_transform(vectors)

    feats_path = feats_path.split('.')
    feats_path = feats_path[0] + '_ncomps' + str(int(n_comps)) + '.pickle'

    print('[INFO] Saving reduced vectors to ' + feats_path)
    with open(feats_path, 'wb') as handle:
        pickle.dump(dict(zip(names, vectors)), handle)

if __name__ == '__main__':
    # require features filepath and number of components
    parser = argparse.ArgumentParser(description='Reduce vector dimensionality')
    parser.add_argument('-f', '--feats', required=True,
                        nargs=1, action='store',
                        type=str, dest='feats_path',
                        help='The filepath of the features')
    parser.add_argument('-n', '--n_comps', required=True,
                        nargs=1, action='store',
                        type=int, dest='n_comps')

    args = vars(parser.parse_args())
    feats_path = args['feats_path'][0]
    n_comps = args['n_comps'][0]

    main(feats_path, n_comps)

