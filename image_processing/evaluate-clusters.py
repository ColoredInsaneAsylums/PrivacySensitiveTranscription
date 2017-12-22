import cv2
import _pickle as pickle

# view items in clusters
def main():
    print('[INFO] Working...')

    with open('labels.pickle', 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        labels = unpickler.load()

    unique_labels = set(labels.values())
    print('[INFO] ' + str(len(unique_labels) - 1) + ' unique clusters found')

    print('[INFO] Press ESC to stop viewing a cluster, press any other key to continue')

    while True:
        # get user input
        label = int(input('[INFO] Input a number from ' + str(min(unique_labels)) + \
                          ' to ' + str(max(unique_labels)) + ' to view a cluster: '))

        if label not in unique_labels:
            print('[ERR] Input not in specified range')
            continue

        for im, l in labels.items():
            if l == label:
                img = cv2.imread('./images/' + im, 0)
                cv2.imshow(im, img)

                k = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()

                if k == 27:
                    break

if __name__ == '__main__':
    main()

