import argparse
import _pickle as pickle

from pymongo import MongoClient
import csh_db_cfg

def main(label_path):
    # connection to mongodb
    mongoConn = MongoClient(csh_db_cfg.DB_HOST + ":" + str(csh_db_cfg.DB_PORT))
    cshTransDB = mongoConn[csh_db_cfg.TRANSCRIPTION_DB_NAME]
    cshTransDB.authenticate(csh_db_cfg.TRANSCRIPTION_DB_USER,
                            csh_db_cfg.TRANSCRIPTION_DB_PASS)
    cshCollection = cshTransDB[csh_db_cfg.TRANS_DB_MeetingMinColl]

    print('[INFO] Loading saved cluster labels from ' + label_path)
    with open(label_path, 'rb') as handle:
        unpickler = pickle.Unpickler(handle)
        labels = unpickler.load()

    print('[INFO] Saving cluster labels to MongoDB')
    for name, label in labels.items():
        # set 0th cluster to noise
        if label == 0:
            label = -1

        searchQuery = {'anonymizedImageFile': name}
        updateQuery = {
            '$set': {
                'cluster': str(label),
            }
        }
        cshCollection.find_one_and_update(searchQuery, updateQuery)

if __name__ == '__main__':
    # require cluster labels filepath
    parser = argparse.ArgumentParser(description='Save cluster labels to MonogoDB')
    parser.add_argument('-p', '--path', required=True,
                        nargs=1, action='store',
                        type=str, dest='label_path',
                        help='The filepath of the cluster labels')

    args = vars(parser.parse_args())
    label_path = args['label_path'][0]

    main(label_path)

