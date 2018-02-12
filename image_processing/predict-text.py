import http.client, urllib.request, urllib.parse, urllib.error, base64, requests, time, json, os

from azure_cfg import api_key
subscription_key = api_key

def main():
    uri_base = 'https://westcentralus.api.cognitive.microsoft.com'
    im_path = './images/'

    requestHeaders = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    for filename in os.listdir(im_path):
        if not filename.endswith('.jpg'):
            continue

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

            print('\nHandwritten text submitted. Waiting 10 seconds to retrieve the recognized text.\n')
            time.sleep(10)

            response = requests.request('GET', operationLocation, json=None, data=None, headers=requestHeaders, params=None)

            parsed = json.loads(response.text)
            print ("Response:")
            print (json.dumps(parsed, sort_keys=True, indent=2))

        except Exception as e:
            print('Error:')
            print(e)

if __name__ == '__main__':
    main()

