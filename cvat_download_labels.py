import argparse
import os
import zipfile
import shutil
from pathlib import Path

from cvat_sdk import make_client
from cvat_sdk.api_client import Configuration, ApiClient, exceptions
from cvat_sdk.api_client.exceptions import ApiException

from config_secrets import CVAT_APIKEY
from gen_utils import extract_guid
from stitcher_api import updated_label_post

##############################################################
# Download CVAT.ai labels for project, tasks, or jobs.
#
# Standard use is to use options...
# if by jobs
# python -m cvat_download_labels --source jobs --project-name 'my_project' --task-name 'my_task_name'
# --stage and --state used and set with defaults
# --task-dir optionally overrides --task-name as the cvat-tasks directory name
# --skip-stitcher-notify optionally skips notifying stitcher api
#
###############################################################
# Non Standard uses.
#
# if only_get_rejected, bypasses download labels, but is also ran if using cvat_download_labels with --source jobs
# python -m cvat_download_labels --only-get-rejected --project-name 'my_project' --task-name 'my_task_name'
#
# These may have been one off use cases, possibly unsupported ...
#
# if by project_jobs
# this option finds jobs within a project that are marked as completed and downloads those to their --task-dir
# python -m cvat_download_labels --source project_jobs --task-dir my_task_dir --project-name my_project
#
# if by task, we are using this for evaluation testing
# --source task
# --task-name 'mytask'
# --localsave # saves the files locally, not related to stitcher
#
# if by project, we download all files in the project irregardless if marked as completed.
# --source project
# --project-name 'myproject'
# --project-id 0 # the id from cvat.ai project
# --task-dir 'mydir' # the dir at BASE_DIR_TASKS where the download will replace files
#
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
ARGS_PROJECT_JOBS = 'project_jobs'
ARGS_TASKS = 'tasks'
ARGS_JOBS = 'jobs'


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download cvat.ai labels')
    parser.add_argument(
        '--source',
        choices=[ARGS_PROJECT, ARGS_PROJECT_JOBS, ARGS_TASKS, ARGS_JOBS],
        help="Select the source level (required)."
    )
    parser.add_argument(
        '--localsave',
        action='store_true',
        help="Enables saving to local_download_path"
    )
    parser.add_argument(
        '--skip-stitcher-notify',
        action='store_true',
        help="Skips notifing stitcher of new download"
    )
    parser.add_argument(
        '--status',
        default='completed',
        help='Status filter, defaults to completed'
    )
    parser.add_argument(
        '--local-download-dir',
        help='For use with localsave, required in that case'
    )
    parser.add_argument(
        '--project-id',
        type=int,
        help="The project ID, required with source=project"
    )
    parser.add_argument(
        '--task-dir',
        help=f'Directory in {BASE_DIR_TASKS} where tasks exist'
    )
    parser.add_argument(
        '--project-name',
        help='The project name search string'
    )
    parser.add_argument(
        '--task-name',
        help='The cvat.ai task name filter'
    )
    parser.add_argument(
        '--stage',
        default='acceptance',
        help='The cvat.ai stage field filter'
    )
    parser.add_argument(
        '--state',
        default='completed',
        help='The cvat.ai stage field filter'
    )
    parser.add_argument(
        '--only-get-rejected',
        action='store_true',
        help='bypasses download labels, but is also ran if using cvat_download_labels with --source jobs'
    )
    args = parser.parse_args()
    print(f'--source set to {args.source}')
    return args


def download_labels_project(project_id, data_dir_task):
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
        moved_files = set()
        for root, dirs, files in os.walk(temp_extract_dir):
            for file in files:
                if file.endswith(".txt") and file.lower() not in ["classes.txt", "train.txt"]:
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(data_dir_task, file)
                    # Move, overwrite, and keep track
                    shutil.move(source_path, destination_path)
                    moved_files.add(file.lower())
                    new_file_guids.append(extract_guid(file))
            # check for train.txt images with zero annotations. These dont get a file created.
            if 'train.txt' in files:
                traintxt_path = os.path.join(root, 'train.txt')
                # check that each image entry has a releated .txt file that exists in 'files'.
                # If not, create an empty .txt file for it.
                with open(traintxt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # CVAT paths in train.txt usually look like: data/images/img1.jpg or ./images/img1.jpg
                        image_filename = os.path.basename(line)
                        image_stem = os.path.splitext(image_filename)[0]
                        expected_label_file = f"{image_stem}.txt"

                        # If this image did not have a corresponding moved label file, create an empty one
                        if expected_label_file.lower() not in moved_files:
                            empty_label_path = os.path.join(data_dir_task, expected_label_file)
                            with open(empty_label_path, 'w', encoding='utf-8') as empty_file:
                                pass  # Creates a completely empty text file

                            new_file_guids.append(extract_guid(expected_label_file))

        print(f"Labels successfully moved to: {data_dir_task}")

    finally:
        # Clean up: remove temp folder and the original zip
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)

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
            (data, _) = api_client.tasks_api.list(
                x_organization=ORGANIZATION_SLUG,
                page=page,
                status=status,
                page_size=10
            )
            task_ids += [task.id for task in data['results']]
            if data['next'] is None:
                break
            page += 1
        print(f'Found {len(task_ids)} task ids')
    return task_ids


def download_by_task_id(task_ids, download_path):
    """
    Downloads task images by id to central directory.
    """
    retrieved_task_count = 0
    with make_client(CVAT_URL, access_token=CVAT_APIKEY) as client:
        client.organization = ORGANIZATION_SLUG

        for task_id in task_ids:
            print(f'Downloading task_id {task_id}, this may take some time...')
            try:
                task = client.tasks.retrieve(task_id)
                task.export_dataset(EXPORT_FORMAT, f"{download_path}/task_{task_id}_labels.zip")
                retrieved_task_count += 1
            except ApiException as e:
                if e.status == 404:
                    print(f"Warning: Task {task_id} no longer exists. Skipping...")
                else:
                    print(f"Error retrieving Task. Skipping {task_id}: {e}")
    if retrieved_task_count:
        print(f'Downloaded {retrieved_task_count} records to {download_path}')
        return download_path
    return None


def download_by_job_ids(job_ids, download_path):
    """
    Downloads job annotations by id to central directory.
    """
    retrieved_job_count = 0
    with make_client(CVAT_URL, access_token=CVAT_APIKEY) as client:
        print(f'Downloading {len(job_ids)} jobs, this may take some time...')
        for job_id in job_ids:
            try:
                job = client.jobs.retrieve(job_id)
                local_zip_path = f"{download_path}/job_{job_id}_labels.zip"
                job.export_dataset(
                    format_name=EXPORT_FORMAT,
                    filename=local_zip_path
                )
                retrieved_job_count += 1
                print(f"Successfully downloaded Job {job_id} to {local_zip_path}")
            except Exception as e:
                print(f"Error retrieving Job {job_id}: {e}")
    if retrieved_job_count:
        print(f'Downloaded {retrieved_job_count} records to {download_path}')
        return download_path
    return None


def get_zip_file_paths(local_download_path):
    result = []
    for root, _, files in os.walk(local_download_path):
        result += [f'{root}/{f}' for f in files if f.endswith('.zip')]
    return result


def get_filtered_job_ids(args):
    """
    https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/jobs-api/#example-5
    """
    if not args.project_name:
        print('get_filtered_job_ids requires option --project-name, returning None')
        return
    if args.task_name:
        with ApiClient(CONFIGURATION) as api_client:
            job_ids = []
            page = 1
            while True:
                try:
                    (data, response) = api_client.jobs_api.list(
                        x_organization=ORGANIZATION_SLUG,
                        page=page,
                        page_size=10,
                        project_name=args.project_name,
                        stage=args.stage,
                        state=args.state,
                        task_name=args.task_name,
                    )
                    job_ids += [job.id for job in data['results']]
                    if data['next'] is None:
                        break
                    page += 1
                except exceptions.ApiException as e:
                    print("Exception when calling JobsApi.list(): %s\n" % e)
                    break
            print(f'Found {len(job_ids)} job ids')
            return job_ids
    elif args.source == ARGS_PROJECT_JOBS:
        with ApiClient(CONFIGURATION) as api_client:
            job_ids = []
            page = 1
            while True:
                try:
                    (data, response) = api_client.jobs_api.list(
                        x_organization=ORGANIZATION_SLUG,
                        page=page,
                        page_size=10,
                        project_name=args.project_name,
                        stage=args.stage,
                        state=args.state,
                    )
                    job_ids += [job.id for job in data['results']]
                    if data['next'] is None:
                        break
                    page += 1
                except exceptions.ApiException as e:
                    print("Exception when calling JobsApi.list(): %s\n" % e)
                    break
            print(f'Found {len(job_ids)} job ids')
            return job_ids
    else:
        print(f'valid args not given to get_filtered_job_ids, args {args}..')
        return None


def get_rejected_job_ids(args):
    with ApiClient(CONFIGURATION) as api_client:
        job_ids = []
        page = 1
        while True:
            try:
                (data, response) = api_client.jobs_api.list(
                    x_organization=ORGANIZATION_SLUG,
                    page=page,
                    page_size=10,
                    project_name=args.project_name,
                    state='rejected',
                    task_name=args.task_name,
                )
                job_ids += [job.id for job in data['results']]
                if data['next'] is None:
                    break
                page += 1
            except exceptions.ApiException as e:
                print("Exception when calling JobsApi.list(): %s\n" % e)
                break
        print(f'Found {len(job_ids)} job ids marked as rejected')
        return job_ids


def report_rejected(args):

    jobids = get_rejected_job_ids(args)
    print(jobids)
    with ApiClient(CONFIGURATION) as api_client:
        for job_id in jobids:
            try:
                (data, response) = api_client.jobs_api.retrieve_data_meta(job_id)
                print(data)
            except exceptions.ApiException as e:
                print("Exception when calling JobsApi.retrieve_data_meta(): %s\n" % e)



def main(args):

    if args.source == ARGS_PROJECT:
        """
        This only supports the case where
        CVAT.ai project == data_dir_task == f'{BASE_DIR_TASKS}{args.task_dir}'
        """
        if not args.project_id and args.task_dir:
            print('args --project-id and --task-dir are required when source==project')
            return

        data_dir_task = f'{BASE_DIR_TASKS}{args.task_dir}'
        zip_file_path = download_labels_project(args.project_id, data_dir_task)
        new_file_guids = extract_and_cleanup_labels(zip_file_path, data_dir_task)
        for guid in new_file_guids:
            post_params = {
                'guid': guid
            }
            # BASE_DIR_TASKS is tied to Stitcher app, must notify
            updated_label_post(post_params)

    elif args.source == ARGS_PROJECT_JOBS:
        if not args.task_dir and not args.project_name:
            print(f'args --task-dir and --project-name are required when --source is {ARGS_PROJECT_JOBS}')
            return
        data_dir_task = f'{BASE_DIR_TASKS}{args.task_dir}'
        print(f'the data_dir_task is {data_dir_task}. Files will be saved here.')
        job_ids = get_filtered_job_ids(args)
        guids = []
        if not job_ids:
            print('Found no jobs, exiting..')
            return
        print(f'Found {len(job_ids)} job ids')
        populated_download_dir = download_by_job_ids(job_ids, data_dir_task)
        if not populated_download_dir:
            print('There was no populated_download_dir, exiting..')
            return
        print(f'Finished downloading jobs, into {populated_download_dir}')
        print(f'Getting zip files from {populated_download_dir}')
        zip_paths = get_zip_file_paths(populated_download_dir)
        print(f'Extracting {len(zip_paths)} zip_paths')
        for task_zip_path in zip_paths:
            extracted_files = extract_and_cleanup_labels(task_zip_path, populated_download_dir)
            guids += extracted_files
        print(f'Extracted labels for {len(guids)} guids')
        if not args.skip_stitcher_notify:
            print('Notifying stitcher of these guids')
            for guid in guids:
                post_params = {
                    'guid': guid
                }
                # BASE_DIR_TASKS is tied to Stitcher app, must notify
                updated_label_post(post_params)

    elif args.source == ARGS_TASKS:

        guids = []

        if args.localsave:
            if not args.local_download_dir:
                print('Option --local-download-dir when using --localsave, exiting..')
                return
            if args.task_dir:
                print(f'Option --task-dir not supported when localsave is True, ignoring {args.task_dir}')
            local_download_path = f'{LOCAL_BASE_DIR}/{args.local_download_dir}'
            os.makedirs(local_download_path, exist_ok=True)
            task_ids = get_tasks_by_status(args.status)
            populated_download_dir = download_by_task_id(task_ids, local_download_path)
            if not populated_download_dir:
                return
            zip_paths = get_zip_file_paths(populated_download_dir)
            for task_zip_path in zip_paths:
                extracted_files = extract_and_cleanup_labels(task_zip_path, populated_download_dir)
                guids += extracted_files

            print(f'Extracted labels for {len(guids)} guids')
            print(f"Dataset is ready in: {os.path.abspath(local_download_path)}")

        else:
            if not args.task_dir:
                print('Option --task-dir is required when source=tasks and localsave is False')
                return
            print(f'localsave is the only current save setting for source {ARGS_TASKS}')

    elif args.source == ARGS_JOBS:
        if not args.task_name:
            print(f'--task-name is required when --source is {ARGS_JOBS}')
            return
        if not args.project_name:
            print(f'--project-name is required when --source is {ARGS_JOBS}')
            return
        guids = []
        if args.task_dir:
            data_dir_task = f'{BASE_DIR_TASKS}{args.task_dir}'
        else:
            data_dir_task = f'{BASE_DIR_TASKS}{args.task_name}'
        print(f'the data_dir_task is {data_dir_task}. Files will be saved here.')
        job_ids = get_filtered_job_ids(args)
        if not job_ids:
            print('Found no jobs, exiting..')
            return
        print(f'Found {len(job_ids)} job ids')
        print(f'Ensuring {data_dir_task} exists, will create if not.')
        Path(data_dir_task).mkdir(parents=True, exist_ok=True)
        populated_download_dir = download_by_job_ids(job_ids, data_dir_task)
        if not populated_download_dir:
            print('There was no populated_download_dir, exiting..')
            return
        print(f'Finished downloading jobs, into {populated_download_dir}')
        print(f'Getting zip files from {populated_download_dir}')
        zip_paths = get_zip_file_paths(populated_download_dir)
        print(f'Extracting {len(zip_paths)} zip_paths')
        for task_zip_path in zip_paths:
            extracted_files = extract_and_cleanup_labels(task_zip_path, populated_download_dir)
            guids += extracted_files
        print(f'Extracted labels for {len(guids)} guids')
        if not args.skip_stitcher_notify:
            print('Notifying stitcher of these guids')
            for guid in guids:
                post_params = {
                    'guid': guid
                }
                # BASE_DIR_TASKS is tied to Stitcher app, must notify
                updated_label_post(post_params)

    print('COMPLETED DOWNLOAD AND UPDATE PROCESS')


if __name__ == '__main__':
    args = get_args()
    if args.only_get_rejected:
        report_rejected(args)
    elif args.source:
        main(args)
    else:
        print('No actions taken, check args for conformity: {args}')
