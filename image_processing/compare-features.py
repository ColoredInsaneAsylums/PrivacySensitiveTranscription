import argparse
import cv2
import _pickle as pickle

from scipy import spatial

# find the most similar images given a query image
def main(desc_name):
    print('[INFO] Working...')

    # similarity threshold
    threshold = 0.6

    # load feature vectors
    print('[INFO] Loading saved ' + desc_name + ' features')
    with open('features/' + desc_name + '_features.pickle', 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        index = unpickler.load()

    while True:
        results = {}

        # get user input
        image_q = input('[INFO] Input an image to compare: ')

        if image_q not in index:
            print('[INFO] Query image not found, please try another.')
            continue

        feature_q = index[image_q]

        # calculate cosine similarities
        print('[INFO] Calculating cosine similarities...')
        for image_n, feature_n in index.items():
            score = 1 - spatial.distance.cosine(feature_n, feature_q)
            if score > threshold:
                results[image_n] = score

        del results[image_q]

        results = [(image_n, results[image_n]) for image_n in \
                  sorted(results, key=results.get, reverse=True)]

        # present results
        print('[INFO] Results:')
        for image_n, score in results:
            print('{0}\t{1}'.format(image_n, score))

            # im = cv2.imread(image_n, 0)
            # cv2.imshow(image_n, im)

if __name__ == '__main__':
    # require name of descriptor to use
    parser = argparse.ArgumentParser(description='Extract image feature vectors using feature descriptors (i.e., SIFT, SURF, HOG, ORB).')
    parser.add_argument('-d', '--descriptor', required=True,
                        choices=['SIFT', 'SURF', 'HOG', 'ORB'],
                        nargs=1, action='store', type=str, dest='desc_name',
                        help='The name of the descriptor to use (i.e., SIFT, SURF, HOG, ORB)')

    args = vars(parser.parse_args())
    desc_name = args['desc_name'][0]

    main(desc_name)
