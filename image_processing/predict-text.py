import http.client, urllib.request, urllib.parse, urllib.error, base64, requests, time, json, os
import cv2

from azure_cfg import api_key
subscription_key = api_key

def main():
    uri_base = 'https://westcentralus.api.cognitive.microsoft.com'
    im_path = './images/'

    requestHeaders = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    images = os.listdir(im_path)
    for idx, filename in enumerate(images):
        if not filename.endswith('.jpg'):
            continue

        print('[INFO] Predicting text for ' + filename + ' (' + str(idx + 1) + '/' + str(len(images)) + ')')

        im = cv2.imread(im_path + filename, cv2.COLOR_BGR2GRAY)
        height, width = im.shape[:2]
        max_dim = max(height, width)
        min_dim = min(height, width)

        if min_dim < 40:
            print('Image height or width too small... skipping')
            continue

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

        try:
            response = requests.request('POST', uri_base + '/vision/v1.0/RecognizeText', json=None, data=body, headers=requestHeaders, params=params)

            if response.status_code != 202:
                parsed = json.loads(response.text)
                print ("Error:")
                print (json.dumps(parsed, sort_keys=True, indent=2))
                exit()

            operationLocation = response.headers['Operation-Location']

            time.sleep(8)

            response = requests.request('GET', operationLocation, json=None, data=None, headers=requestHeaders, params=None)

            parsed = json.loads(response.text)
            lines = parsed['recognitionResult']['lines']
            if len(lines):
                text = []
                for line in lines:
                    text.append(line['text'])
                print ("Response: " + ('\n').join(text))
            else:
                print ("No words detected")

        except Exception as e:
            print('Error:')
            print(e)

if __name__ == '__main__':
    main()

