import http.client, urllib.request, urllib.parse, urllib.error, base64, requests, time, json, os
import cv2

from azure_cfg import api_key
from pymongo import MongoClient
import csh_db_cfg


def main():
    # azure setup
    uri_base = 'https://westcentralus.api.cognitive.microsoft.com'
    im_path = './images/'
    requestHeaders = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': api_key,
    }

    # connection to mongodb
    mongoConn = MongoClient(csh_db_cfg.DB_HOST + ":" + str(csh_db_cfg.DB_PORT))
    cshTransDB = mongoConn[csh_db_cfg.TRANSCRIPTION_DB_NAME]
    cshTransDB.authenticate(csh_db_cfg.TRANSCRIPTION_DB_USER,
                            csh_db_cfg.TRANSCRIPTION_DB_PASS)
    cshCollection = cshTransDB[csh_db_cfg.TRANS_DB_MeetingMinColl]

    # get images that have not yet been processed
    searchQuery = {'meetsDimensionThreshold': {'$exists': False}}
    selectionQuery = {'anonymizedImageFile': 1, '_id': 0}
    images = list(cshCollection.find(searchQuery, selectionQuery))
    images = set(map(lambda x: x['anonymizedImageFile'], images))
    images = images.intersection(os.listdir(im_path))

    # loop through images
    failed_images = []
    for idx, filename in enumerate(images):
        print('[INFO] Predicting text for ' + filename + ' (' + str(idx + 1) + '/' + str(len(images)) + ')')

        im = cv2.imread(im_path + filename, cv2.COLOR_BGR2GRAY)
        height, width = im.shape[:2]
        max_dim = max(height, width)
        min_dim = min(height, width)

        # image must be larger than 40x40
        if min_dim < 40:
            searchQuery = {'anonymizedImageFile': filename}
            updateQuery = {
                '$set': {
                    'meetsDimensionThreshold': False
                }
            }
            cshCollection.find_one_and_update(searchQuery, updateQuery)

            continue

        # image must be smaller than 3600x3600
        if max_dim > 3200:
            dim = 3200
            r = dim / width
            dim = (dim, int(height * r))
            im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
            print(im.shape)
            cv2.imwrite(im_path + 'tmp.jpg', im)
            filename = 'tmp.jpg'

        body = open(im_path + filename, 'rb')
        params = {'handwriting' : 'true'}

        # ping Azure REST API
        try:
            response = requests.request('POST', uri_base + '/vision/v1.0/RecognizeText', json=None, data=body, headers=requestHeaders, params=params)

            if response.status_code != 202:
                parsed = json.loads(response.text)
                print ('[ERR] Error:')
                print (json.dumps(parsed, sort_keys=True, indent=2))
                failed_images.append(filename)
                continue

            operationLocation = response.headers['Operation-Location']

            time.sleep(8)

            response = requests.request('GET', operationLocation, json=None, data=None, headers=requestHeaders, params=None)

            # check if API detected text
            parsed = json.loads(response.text)
            lines = parsed['recognitionResult']['lines']
            if len(lines):
                text = []
                for line in lines:
                    text.append(line['text'])
                prediction = ('\n').join(text)
            else:
                prediction = None

            searchQuery = {'anonymizedImageFile': filename}
            updateQuery = {
                '$set': {
                    'azurePrediction': prediction,
                    'meetsDimensionThreshold': True
                }
            }
            cshCollection.find_one_and_update(searchQuery, updateQuery)

        except Exception as e:
            print('[ERR] Error:')
            print(e)
            failed_images.append(filename)

    print('\n[INFO] Done. Failed to get predictions for ' + str(len(failed_images)) + ' images:')
    print('\n\t'.join(failed_images))

if __name__ == '__main__':
    main()

