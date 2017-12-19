import cv2
import numpy
import os
import _pickle as pickle

# run image filtering and HOG feature extraction
def main():
    print('[INFO] Working...')

    # image directory
    path = './images/'

    # track HOG feature vectors and corresponding images
    features = {}
    h_features = []

    # image dimensions
    width = 128
    height = 64

    # HOG feature descriptor
    hog = cv2.HOGDescriptor(_winSize = (width,height),
                            _blockSize = (16,16),
                            _blockStride = (8,8),
                            _cellSize = (8,8),
                            _nbins = 9)

    # evaluate image files
    print('[INFO] Reshaping images and computing HOG features')
    for filename in os.listdir(path):
        im = cv2.imread(path + filename, 0)

        # resize image and compute features
        im = cv2.resize(im, (width,height))
        h = hog.compute(im)

        features[filename] = h
        h_features.append(h)

    # save data
    print('[INFO] Saving HOG features and corresponding image name to \'features.pickle\'')
    with open('features.pickle', 'wb') as handle:
        pickle.dump(features, handle)

    print('[INFO] Saving HOG feature vectors to \'hog_features.csv\'')
    numpy.savetxt('hog_features.csv',
                  numpy.array(h_features),
                  delimiter=',')

if __name__ == '__main__':
    main()

