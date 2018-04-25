import argparse
import numpy as np
import os
import _pickle as pickle

# associate numpy feature arrays to their images
def main(images_path, feats_path):
    # load image names and features
    image_names = sorted(os.listdir(images_path))
    features = np.load(feats_path)

    # pack image names and features into a dict and remove nan arrays
    feats = dict(zip(image_names, features))
    feats = {name: (feat if not np.isnan(feat[0]) else None) \
             for name, feat in feats.items()}

    # save data
    base = os.path.basename(feats_path)
    name = os.path.splitext(base)[0]

    output = name + '.pickle'
    print('[INFO] Saving features to ./' + output)
    with open('./' + output, 'wb') as handle:
        pickle.dump(feats, handle, protocol=4)

if __name__ == '__main__':
    # require filepath of the image directory and features
    parser = argparse.ArgumentParser(description='Associate numpy feature arrays to their images')
    parser.add_argument('-i', '--images', required=True,
                        nargs='?', action='store', const='../images/',
                        type=str, dest='images_path',
                        help='The filepath of the image directory')
    parser.add_argument('-f', '--features', required=True,
                        nargs=1, action='store',
                        type=str, dest='feats_path',
                        help='The filepath of the feature vectors')

    args = vars(parser.parse_args())
    images_path = args['images_path']
    feats_path = args['feats_path'][0]

    main(images_path, feats_path)

