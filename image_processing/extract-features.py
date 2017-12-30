import argparse
import cv2
import numpy as np
import os
import _pickle as pickle

from descriptors import HOG
from skimage.morphology import skeletonize

# run image filtering and HOG feature extraction
def main(im_path, desc_name):
    print('[INFO] Preparing to extract features for images in \'' + im_path + '\'')

    # track HOG feature vectors and corresponding images
    features = {}

    # image dimensions
    width = 128
    height = 64

    # feature descriptor
    print('[INFO] Using the ' + desc_name.upper() + ' feature descriptor')
    if desc_name == 'hog':
        descriptor = HOG()

   # evaluate image files
    print('[INFO] Processing images and computing features')
    for filename in os.listdir(im_path):
        if not filename.endswith('.jpg'):
            continue

        im = cv2.imread(im_path + filename, cv2.COLOR_BGR2GRAY)

        # resize image
        im = cv2.resize(im, (width,height))

        # binarize using Otsu's method
        im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        im[im == 255] = 1

        # thin using Zhang and Suen's method
        im = skeletonize(im)
        im = im.astype(np.uint8)

        # compute features
        v = descriptor.compute(im)
        features[filename] = v

    # save data
    print('[INFO] Saving features and corresponding image name to \'features/' + desc_name + '_features.pickle\'')
    with open('./features/' + desc_name + '_features.pickle', 'wb') as handle:
        pickle.dump(features, handle)

if __name__ == '__main__':
    # require image directory and name of descriptor to use
    parser = argparse.ArgumentParser(description='Extract image feature vectors using feature descriptors')
    parser.add_argument('-p', '--path', required=True,
                        nargs='?', action='store', const='./images/',
                        type=str, dest='im_path',
                        help='The filepath of the image directory')

    parser.add_argument('-d', '--descriptor', required=True,
                        choices=['hog'],
                        nargs='?', action='store', const='hog',
                        type=str, dest='desc_name',
                        help='The name of the descriptor to use')

    args = vars(parser.parse_args())
    im_path = args['im_path']
    desc_name = args['desc_name']

    main(im_path, desc_name)

