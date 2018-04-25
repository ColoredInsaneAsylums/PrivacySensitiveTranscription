import argparse
import os.path as path
import random
import _pickle as pickle

# sample a subset of the data 
def main(feats_path, n):
    with open(feats_path, 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        labels = unpickler.load()

    labels = {name: vector for name, vector in labels.items() if vector is not None}
    names = list(labels.keys())
    vectors = list(labels.values())

    print('[INFO] Sampling ' + str(n) + ' instances from ' + feats_path)
    random.seed(42)
    indices = random.sample(range(len(labels)), n)
    subset = {names[i]: vectors[i] for i in indices}

    # save subset
    base = path.basename(feats_path)
    name = path.splitext(base)[0]

    output = name + '_sub' + str(n) + '.pickle'
    print('[INFO] Saving subset to ' + output)
    with open(output, 'wb') as handle:
        pickle.dump(subset, handle)

if __name__ == '__main__':
    # require features filepath and number of components
    parser = argparse.ArgumentParser(description='Reduce vector dimensionality')
    parser.add_argument('-f', '--feats', required=True,
                        nargs=1, action='store',
                        type=str, dest='feats_path',
                        help='The filepath of the features')
    parser.add_argument('-n', '--n_samples', required=True,
                        nargs=1, action='store',
                        type=int, dest='n',
                        help='The number of samples')

    args = vars(parser.parse_args())
    feats_path = args['feats_path'][0]
    n = args['n'][0]

    main(feats_path, n)

