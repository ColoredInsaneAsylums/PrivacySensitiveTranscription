import argparse
import _pickle as pickle

from scipy.spatial import distance
from sklearn.neighbors import KDTree

# predict text in image given the PHOCs
def main(feat_path, dict_path):
    print('[INFO] Working...')

    # load image PHOCs
    print('[INFO] Loading image PHOCs from \'' + feat_path + '\'')
    with open(feat_path, 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        image_phocs = unpickler.load()

    # load dictionary PHOCs and create KDTree
    with open(dict_path, 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        dict_phocs = unpickler.load()
        words = list(dict_phocs.keys())
        word_phocs = list(dict_phocs.values())
        kdtree = KDTree(word_phocs, metric=distance.braycurtis)

    while True:
        results = {}

        # get user input
        image_q = input('[INPUT] Input an image to predict text for: ')

        if image_q not in index:
            print('[INFO] Query image not found, please try another.')
            continue

        phoc_q = index[image_q]

        # calculate braycurtis disimilaritiesi and find nearest neighbors
        print('[INFO] Finding top candidate predictions...')
        dist, idx = kdtree.query(phoc_n, k=5)

        # present results
        print('[INFO] Results:')
        for i in range(5):
            print('{0}\t{1}'.format(words[idx[i]], dist[i]))

if __name__ == '__main__':
    # require filepath of features
    parser = argparse.ArgumentParser(description='Return the top predictions for a word image')
    parser.add_argument('-p', '--path', required=True,
                        nargs='?', action='store', const='./features/phoc_features.pickle',
                        type=str, dest='feat_path',
                        help='The filepath of the image PHOCs')
    parser.add_argument('-d', '--dictionary', required=True,
                        nargs='?', action='store', const='./dictionary-phocs.pickle',
                        type=str, dest='dict_path',
                        help='The filepath of the dictionary (string) PHOCs')

    args = vars(parser.parse_args())
    feat_path = args['feat_path']
    dict_path = args['dict_path']

    main(feat_path, dict_path)

