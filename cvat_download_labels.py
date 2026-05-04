import argparse
import os
import zipfile
import shutil

from cvat_sdk import make_client
from cvat_sdk.api_client import Configuration, ApiClient
from cvat_sdk.api_client.exceptions import ApiException

from config_secrets import CVAT_APIKEY
from gen_utils import extract_guid
from stitcher_api import updated_label_post

##############################################################
### Download CVAT.ai labels for project, tasks, or jobs.
##############################################################

CVAT_URL = 'https://app.cvat.ai/'
ORGANIZATION_SLUG = 'Ecdysis'
EXPORT_FORMAT = 'Ultralytics YOLO Segmentation 1.0'
BASE_DIR_TASKS = '/pool1/srv/cvat-tasks/'

# These are for making a new dataset from cvat images, potentially across multiple projects and tasks
LOCAL_BASE_DIR = '/home/ecdysis/ultralytics/local_files'

CONFIGURATION = Configuration(
    host=CVAT_URL,
    access_token=CVAT_APIKEY
)

ARGS_PROJECT = 'project'
ARGS_TASKS = 'tasks'
ARGS_JOBS = 'jobs'


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download cvat.ai labels')
    parser.add_argument(
        '--source',
        choices=[ARGS_PROJECT, ARGS_TASKS, ARGS_JOBS],
        required=True,
        help="Select the source level (required)."
    )
    parser.add_argument(
        '--localsave',
        action='store_true',
        help="Enables saving to local_download_path"
    )
    args = parser.parse_args()
    print(f'--source set to {args.source}')
    return args


def download_labels_project(project_id, data_dir_task):
# Connect to the server
    with make_client(CVAT_URL, access_token=CVAT_APIKEY) as client:
        client.organization_slug = ORGANIZATION_SLUG

        # Retrieve the project object
        project = client.projects.retrieve(project_id)
        zip_file_path = f'{data_dir_task}.zip'

        # Export the entire project as one dataset
        # By setting include_images=False, you get only the YOLO segmentation .txt files and data.yaml
        project.export_dataset(
            format_name=EXPORT_FORMAT,
            filename=zip_file_path,
            include_images=False
        )
        return zip_file_path


def extract_and_cleanup_labels(zip_path, data_dir_task):

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
                    destination_path = os.path.join(data_dir_task, file)
                    # Move, overwrite, and keep track
                    shutil.move(source_path, destination_path)
                    new_file_guids.append(extract_guid(file))
                    print(f'Moved/replaced file {file}')

        print(f"Labels successfully moved to: {data_dir_task}")

    finally:
        # Clean up: remove temp folder and the original zip
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Deleted the zip file.")

    return new_file_guids


def get_tasks_by_status(status):
    """
    Returns task ids filtered by status.
    """
    # see client filters https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/tasks-api/#list

    with ApiClient(CONFIGURATION) as api_client:
        task_ids = []
        page = 1
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
    return task_ids


def download_by_task_id(task_ids, local_download_path):
    """
    Downloads task images by id to central directory.
    """
    retrieved_task_count = 0
    os.makedirs(local_download_path, exist_ok=True)
    with make_client(CVAT_URL, access_token=CVAT_APIKEY) as client:
        client.organization = ORGANIZATION_SLUG

        for task_id in task_ids:
            try:
                task = client.tasks.retrieve(task_id)
                retrieved_task_count += 1
                # This one method handles the POST, the polling, and the download
                task.export_dataset(EXPORT_FORMAT, f"{local_download_path}/task_{task_id}_labels.zip")
                retrieved_task_count += 1
            except ApiException as e:
                if e.status == 404:
                    print(f"Warning: Task {task_id} no longer exists. Skipping...")
                else:
                    print(f"Error retrieving Task. Skipping {task_id}: {e}")
    if retrieved_task_count:
        print(f'Downloaded {retrieved_task_count} records to {local_download_path}')
        return local_download_path
    return None


def get_zip_file_paths(local_download_path):
    result = []
    for root, _, files in os.walk(local_download_path):
         result += [f'{root}/{f}' for f in files if f.endswith('.zip')]
    return result


def extract_zipped_labels(local_download_path):
    for root, dirs, files in os.walk(local_download_path):
        print(root)
        print(dirs)
        print(files) # ['task_2160239_labels.zip', 'task_2160223_labels.zip', 'task_2172577_labels.zip', 'task_2160224_labels.zip', 'task_2172752_labels.zip', 'task_2160148_labels.zip', 'task_2160225_labels.zip', 'task_2160226_labels.zip', 'task_2160157_labels.zip', 'task_2160159_labels.zip', 'task_2160212_labels.zip', 'task_2172583_labels.zip', 'task_2160236_labels.zip', 'task_2163329_labels.zip', 'task_2172578_labels.zip', 'task_2160229_labels.zip', 'task_2160221_labels.zip', 'task_2172755_labels.zip']

        return [f'{root}/{f}' for f in files]
        temp_extract_dir = f"{LOCAL_DOWNLOAD_PATH}/{task_id}"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
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

    print(f"Done! Your dataset is ready in: {os.path.abspath(local_download_path)}")


if __name__ == '__main__':
    args = get_args()

    if args.source == ARGS_PROJECT:

        project_id = 389494  # move this variable to args
        task_dir = 'sdk_test'  # move this variable to args

        data_dir_task = f'{BASE_DIR_TASKS}{task_dir}'
        zip_file_path = download_labels_project(project_id, data_dir_task)
        new_file_guids = extract_and_cleanup_labels(zip_file_path, data_dir_task)
        for guid in new_file_guids:
            post_params = {
                'guid': guid
            }
            # BASE_DIR_TASKS is tied to Stitcher app, must notify
            updated_label_post(post_params)

    elif args.source == ARGS_TASKS:

        local_download_dir = 'completed_may_4'  # move this variable to args
        status = "completed"  # move this variable to args

        local_download_path = f'{LOCAL_BASE_DIR}/{local_download_dir}'
        local_extracted_path = f'{LOCAL_BASE_DIR}/{local_download_dir}_extracted'
        if args.localsave:
            # Ensure final destination exists
            os.makedirs(local_download_path, exist_ok=True)
            os.makedirs(local_extracted_path, exits_ok=True)
            task_ids = get_tasks_by_status(status)
            populated_download_dir = download_by_task_id(task_ids, local_download_path)
            if not populated_download_dir:
                exit()
            zip_paths = get_zip_file_paths(populated_download_dir)
            for task_zip_path in zip_paths:
                extracted_files = extract_and_cleanup_labels(task_zip_path, local_extracted_path)
                print(f'extracted_files: {extracted_files}')

        else:
            print(f'localsave is the only current save setting for source {ARGS_TASKS}')

    elif args.source == ARGS_JOBS:
        print(f'Source == {ARGS_JOBS} is not implemented')

    print('COMPLETED DOWNLOAD AND UPDATE PROCESS')
