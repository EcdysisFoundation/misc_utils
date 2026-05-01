import os
import time
import zipfile
import shutil

from cvat_sdk.api_client import Configuration, ApiClient
from cvat_sdk import make_client

from config_secrets import CVAT_APIKEY
from gen_utils import extract_guid
from stitcher_api import updated_label_post

##############################################################
### Using the cvat.ai project id and the known local TASK_DIR
### Update the label files in that directory from the download.
### TODO: make project name and TASK_DIR the same in future
###       in order to only need the id or name
##############################################################

ORGANIZATION_SLUG = 'Ecdysis'
EXPORT_FORMAT = 'Ultralytics YOLO Segmentation 1.0'

# This  is used for Project specific downloads
PROJECT_ID = 389494

# These are used when a task on cvat has a related task folder locally, normally produced at task or project creation
TASK_DIR = 'sdk_test'
BASE_DIR_TASKS = '/pool1/srv/cvat-tasks/'
DATA_DIR_TASK = f'{BASE_DIR_TASKS}{TASK_DIR}'

# These are for making a new dataset from cvat images, potentially across multiple projects and tasks
LOCAL_BASE_DIR = '/home/ecdysis/ultralytics/local_files'
LOCAL_DOWNLOAD_DIR = 'completed_may_1'
LOCAL_DOWNLOAD_PATH = f'{LOCAL_BASE_DIR}/{LOCAL_DOWNLOAD_DIR}'


CONFIGURATION = Configuration(
    host='https://app.cvat.ai/',
    access_token=CVAT_APIKEY
)


def download_labels_project():
# Connect to the server
    with make_client('https://app.cvat.ai/', access_token=CVAT_APIKEY) as client:
        client.organization_slug = ORGANIZATION_SLUG

        # Retrieve the project object
        project = client.projects.retrieve(PROJECT_ID)
        zip_file_path = f'{DATA_DIR_TASK}.zip'

        # Export the entire project as one dataset
        # By setting include_images=False, you get only the YOLO segmentation .txt files and data.yaml
        project.export_dataset(
            format_name=EXPORT_FORMAT,
            filename=zip_file_path,
            include_images=False
        )
        return zip_file_path


def extract_and_cleanup_labels(zip_path):
    # Ensure final destination exists
    os.makedirs(DATA_DIR_TASK, exist_ok=True)

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
                    destination_path = os.path.join(DATA_DIR_TASK, file)
                    # Move, overwrite, and keep track
                    shutil.move(source_path, destination_path)
                    new_file_guids.append(extract_guid(file))
                    print(f'Moved/replaced file {file}')

        print(f"Labels successfully moved to: {DATA_DIR_TASK}")

    finally:
        # Clean up: remove temp folder and the original zip
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Deleted the zip file.")

    return new_file_guids


def download_by_task():
    """
    Downloads all filtered tasks into a single directory.
    """
    # see client filters https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/tasks-api/#list
    # project_id
    # TBD

    with ApiClient(CONFIGURATION) as api_client:
        task_ids = []
        page = 1
        status = "completed"
        while True:
            # 1. Fetch only completed tasks for the project
            (data, response) = api_client.tasks_api.list(
                x_organization=ORGANIZATION_SLUG,
                page=page,
                status=status,
                page_size=10
            )
            for task in data['results']:
                task_ids.append(task.id)
            if data['next'] is None:
                break
            page += 1
        print(f'Found {len(task_ids)} task ids')

        # 2. Export data
        os.makedirs(LOCAL_DOWNLOAD_PATH, exist_ok=True)

        for task_id in task_ids:
            print(f"Processing Task {task_id}...")

            # 1. Download
            (data, _) = api_client.tasks_api.retrieve_dataset(
                task_id,
                format=EXPORT_FORMAT,
                _parse_response=False
            )

            zip_path = f"{LOCAL_DOWNLOAD_PATH}/task_{task_id}.zip"
            with open(zip_path, "wb") as f:
                f.write(data.data)

            # 2. Extract to a temp folder
            temp_extract_dir = f"{LOCAL_DOWNLOAD_PATH}/{task_id}"
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)

            # 3. Move and rename files into the master directory
            # We walk through the extracted files to find images and labels
            for root, dirs, files in os.walk(temp_extract_dir):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.txt')) and "data.yaml" not in file:
                        # Determine if it's an image or a label based on the folder it's in
                        subfolder = os.path.relpath(root, temp_extract_dir)
                        dest_path = os.path.join(LOCAL_DOWNLOAD_PATH, subfolder)
                        os.makedirs(dest_path, exist_ok=True)

                        # Prefix with task_id to avoid filename collisions
                        new_filename = f"task_{task_id}_{file}"
                        shutil.move(os.path.join(root, file), os.path.join(dest_path, new_filename))

                    elif file == "data.yaml":
                        # Just copy the yaml from the first task; assuming all tasks share the same classes
                        if not os.path.exists(os.path.join(LOCAL_DOWNLOAD_PATH, "data.yaml")):
                            shutil.copy(os.path.join(root, file), os.path.join(LOCAL_DOWNLOAD_PATH, "data.yaml"))

            # 4. Cleanup
            os.remove(zip_path)
            shutil.rmtree(temp_extract_dir)
            print(f"Task {task_id} merged.")

    print(f"Done! Your dataset is ready in: {os.path.abspath(LOCAL_DOWNLOAD_PATH)}")


if __name__ == '__main__':
    #zip_file_path = download_labels_project()
    #new_file_guids = extract_and_cleanup_labels(zip_file_path)
    #for guid in new_file_guids:
    #    post_params = {
    #        'guid': guid
    #    }
    #    updated_label_post(post_params)
    download_by_task()

    print('COMPLETED DOWNLOAD AND UPDATE PROCESS')
