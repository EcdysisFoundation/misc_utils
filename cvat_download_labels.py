import os
import zipfile
import shutil

from cvat_sdk import make_client

from secrets import CVAT_APIKEY
from stitcher_api import updated_label_post

##############################################################
### Using the cvat.ai project id and the known local TASK_DIR
### Update the label files in that directory from the download.
### TODO: make project name and TASK_DIR the same in future
###       in order to only need the id or name
##############################################################

ORGANIZATION_SLUG = 'Ecdysis'
PROJECT_ID = 389494
TASK_DIR = 'sdk_test'

BASE_DIR = '/pool1/srv/cvat-tasks/'
DATA_DIR = f'{BASE_DIR}{TASK_DIR}'


def download_labels():
# Connect to the server
    with make_client('https://app.cvat.ai/', access_token=CVAT_APIKEY) as client:
        client.organization_slug = ORGANIZATION_SLUG

        # Retrieve the project object
        project = client.projects.retrieve(PROJECT_ID)
        zip_file_path = f'{DATA_DIR}.zip'

        # Export the entire project as one dataset
        # By setting include_images=False, you get only the YOLO segmentation .txt files and data.yaml
        project.export_dataset(
            format_name="Ultralytics YOLO Segmentation 1.0",
            filename=zip_file_path,
            include_images=False
        )
        return zip_file_path


def extract_and_cleanup_labels(zip_path):
    # Ensure final destination exists
    os.makedirs(DATA_DIR, exist_ok=True)

    temp_extract_dir = "temp_cvat_labels"
    new_file_guids = []

    try:
        # Extract the zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        # Walk through extracted files and find .txt files
        for root, dirs, files in os.walk(temp_extract_dir):
            for file in files:
                if file.endswith(".txt") and file not in ["classes.txt", "train.txt"]:
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(DATA_DIR, file)
                    # Move, overwrite, and keep track
                    shutil.move(source_path, destination_path)
                    new_file_guids.append(file)
                    print(f'Moved/replaced file {file}')

        print(f"Labels successfully moved to: {DATA_DIR}")

    finally:
        # Clean up: remove temp folder and the original zip
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Deleted the zip file.")

    return new_file_guids


if __name__ == '__main__':
    zip_file_path = download_labels()
    new_file_guids = extract_and_cleanup_labels(zip_file_path)
    for guid in new_file_guids:
        post_params = {
            'guid': guid
        }
        updated_label_post(post_params)
        print(f'Updated label for guid {guid}')

    print('COMPLETED DOWNLOAD AND UPDATE PROCESS')
