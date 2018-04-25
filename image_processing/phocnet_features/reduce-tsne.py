import argparse
import numpy as np
import os.path as path
import _pickle as pickle

#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

# conduct t-SNE to reduce vector dimensionality for visualization
def main(feats_path):
    with open(feats_path, 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        labels = unpickler.load()

    labels = {name: vector for name, vector in labels.items() if vector is not None}
    features = np.asarray(list(labels.values()))

    print('[INFO] Conducting t-SNE on ' + feats_path)
    tsne = TSNE(metric='braycurtis', verbose=1,
                n_iter=5000, random_state=42, n_jobs=-1)
    projection = tsne.fit_transform(features)

    # save reduced vectors
    base = path.basename(feats_path)
    name = path.splitext(base)[0]

    output = name + '_tsne.pickle'
    print('[INFO] Saving reduced vectors to ' + output)
    with open(output, 'wb') as handle:
        pickle.dump(projection, handle)

if __name__ == '__main__':
    # require features filepath
    parser = argparse.ArgumentParser(description='Reduce vector dimensionality')
    parser.add_argument('-f', '--feats', required=True,
                        nargs=1, action='store',
                        type=str, dest='feats_path',
                        help='The filepath of the features')

    args = vars(parser.parse_args())
    feats_path = args['feats_path'][0]

    main(feats_path)

