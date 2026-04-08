import requests


STITCHER_URL = 'http://ecdysis01.local:8090/'


def get_stitcher_data(upload_dir_name=None):
    api_list_url = STITCHER_URL + 'list-upload-files/'
    all_data = []
    offset = 0
    limit = 100

    while True:

        params = {
            'offset': offset,
            'limit': limit,
            'approved': True}
        if upload_dir_name:
            params.update({
                'upload_dir_name': upload_dir_name
            })

        try:
            response = requests.get(api_list_url, params=params)
        except Exception as e:
            print(e)
            break

        if response.status_code == 200:
            data = response.json()
            if not data:
                break

            all_data.extend(data)

            offset += limit
        else:
            print(f"Error: {response.status_code}")
            break

    print(f"Retrieved {len(all_data)} items.")
    return all_data


def get_abridged_data_by_guid(guid):
    api_list_url = STITCHER_URL + 'list-upload-abridged/'
    try:
        response = requests.get(api_list_url, params={'guid': guid})
        if response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
        else:
            print('Response returned None')
    except Exception as e:
        print(e)


def post_sent_ls(post_params):
    api_post_url = STITCHER_URL + '/sent_label_studio/'
    try:
        response = requests.post(api_post_url, params=post_params)
        if response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
        else:
            print('Response returned None')
    except Exception as e:
        print(e)
