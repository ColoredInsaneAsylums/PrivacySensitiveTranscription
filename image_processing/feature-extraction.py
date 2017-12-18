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

    # load images in black/white
    ims = {filename: cv2.imread(path + filename, 0) for filename in os.listdir(path)}

    # find largest image dimensions
    height = max([im.shape[0] for im in ims.values()])
    width = max([im.shape[1] for im in ims.values()])

    # align image dimensions to block size and stride
    height += 8 - (height % 8)
    width += 8 - (width % 8)

    # HOG feature descriptor
    hog = cv2.HOGDescriptor(_winSize = (width,height),
                            _blockSize = (16,16),
                            _blockStride = (8,8),
                            _cellSize = (8,8),
                            _nbins = 9)

    # evaluate image files
    for filename, im in ims.items():
        # resize image and compute features
        im = cv2.resize(im, (width, height))
        h = hog.compute(im)

        features[filename] = h
        h_features.append(h)

    # save data
    with open('features.pickle', 'wb') as handle:
        pickle.dump(features, handle)

    numpy.savetxt('hog_features.csv',
                  numpy.array(h_features),
                  delimiter=',')

    print('[INFO] HOG features and corresponding image name saved to \'features.pickle\'.')
    print('[INFO] HOG feature vectors saved to \'hog_features.csv\'.')

if __name__ == '__main__':
    main()

