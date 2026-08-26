import requests
from datetime import datetime, timezone
from config_secrets import STITCHER_API_URL


ERROR_MSG_KEY = 'ERROR'


def get_root_message():
    api_url = STITCHER_API_URL
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return {ERROR_MSG_KEY: response.status_code}
    except Exception as e:
        print(e)
        return {ERROR_MSG_KEY: e}



def get_stitcher_data(upload_dir_name=None):
    api_list_url = STITCHER_API_URL + 'list-upload-files/'
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
    api_list_url = STITCHER_API_URL + 'list-upload-abridged/'
    try:
        response = requests.get(api_list_url, params={'guid': guid})
        if response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
        else:
            print('Response returned None')
    except Exception as e:
        print(e)


def simple_post_w_params(api_post_url, post_params):
    try:
        response = requests.post(api_post_url, params=post_params)
        if response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
        else:
            print('Response returned None')
    except Exception as e:
        print(f'error with post_params {post_params}')
        print(e)


def post_sent_ls(post_params):
    api_post_url = STITCHER_API_URL + 'sent_label_studio/'
    simple_post_w_params(api_post_url, post_params)


def updated_label_post(post_params):
    api_post_url = STITCHER_API_URL + 'updated_label/'
    simple_post_w_params(api_post_url, post_params)


def mark_rejected_post(post_params):
    api_post_url = STITCHER_API_URL + 'mark_labels_rejected/'
    simple_post_w_params(api_post_url, post_params)


def get_label_file_status(data):
    """
    Check if label file updated. Small difference in time can be api call artifact.
    """

    def format_time(d):
        try:
            d = datetime.fromisoformat(d)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d
        except Exception as e:
            print(f'Exception in get_label_file_status: {e}')
            return d

    d1 = format_time(data.get('label_file_updated_at'))
    d2 = format_time(data.get('label_studio_project_created_at'))

    if isinstance(d1, datetime) and isinstance(d2, datetime):

        # Check if they are on the same calendar day
        is_same_day = d1.date() == d2.date()

        if not is_same_day:
            return True

        # Calculate the difference in seconds
        difference_in_seconds = abs((d1 - d2).total_seconds())

        # Check if the gap is strictly larger than 1 minute (60 seconds)
        if difference_in_seconds > 60:
            return True

    return False


def get_filtered_panos_list(label_studio_project):
    """
    Using a strin label_studio_project filter those to get a list of upload_dir_name
    """
    api_list_url = STITCHER_API_URL + 'list-upload-files-abridged/'
    panos = []
    offset = 0
    limit = 100
    while True:

        params = {
            'offset': offset,
            'limit': limit,
            'approved': True}

        try:
            response = requests.get(api_list_url, params=params)
        except Exception as e:
            print(e)
            break

        if response.status_code == 200:
            data = response.json()
            if not data:
                break

            for d in data:
                if d['label_studio_project'] == label_studio_project:
                    if not get_label_file_status(d):
                        panos.append(d['upload_dir_name'])

            offset += limit
        else:
            print(f"Error: {response.status_code}")
            break

    print(f"Retrieved {len(panos)} items.")
    return panos
