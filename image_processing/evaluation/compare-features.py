import argparse
import cv2
import _pickle as pickle

from scipy.spatial import distance

# find the most similar images given a query image
def main(feat_path, images_path):
    print('[INFO] Working...')

    # similarity threshold
    threshold = 0.6

    # load feature vectors
    print('[INFO] Loading features from ' + feat_path)
    with open(feat_path, 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        index = unpickler.load()

    while True:
        results = {}

        # get user input
        image_q = input('[INPUT] Input an image to compare: ')

        if image_q not in index:
            print('[INFO] Query image not found, please try another.')
            continue

        feature_q = index[image_q]

        # calculate Bray Curtis dissimilarities
        print('[INFO] Calculating Bray Curtis dissimilarities...')
        for image_n, feature_n in index.items():
            score = 1 - distance.braycurtis(feature_n, feature_q)
            if score > threshold:
                results[image_n] = score

        del results[image_q]

        results = [(image_n, results[image_n]) for image_n in \
                  sorted(results, key=results.get, reverse=True)]

        # present results
        print('[INFO] Results:')
        for image_n, score in results:
            print('{0}\t{1}'.format(image_n, score))

            img = cv2.imread(images_path + '/'  + image_n, 0)
            cv2.imshow(image_n, img)

            k = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            # press ESC to exit
            if k == 27:
                break

if __name__ == '__main__':
    # require filepath of features
    parser = argparse.ArgumentParser(description='Compare image feature vectors via cosine similarity')
    parser.add_argument('-f', '--feats', required=True,
                        nargs=1, action='store',
                        type=str, dest='feat_path',
                        help='The filepath of the feature vectors to compare')
    parser.add_argument('-i', '--images', required=True,
                        nargs=1, action='store',
                        type=str, dest='images_path',
                        help='The directory path of the images to compare')


    args = vars(parser.parse_args())
    feat_path = args['feat_path'][0]
    images_path = args['images_path'][0]

    main(feat_path, images_path)

