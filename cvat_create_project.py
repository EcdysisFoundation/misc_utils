
import argparse
import zipfile
import tempfile
import os
import time
from PIL import Image
from cvat_sdk import make_client, models
from cvat_sdk.api_client import Configuration, ApiClient

from gen_utils import extract_guid
from config_secrets import CVAT_APIKEY
from stitcher_api import post_sent_ls

##############################################################
# Create a cvat.ai project and populate from outputs of our
# ultralytics repo inference module
# where inference creates a TASK_DIR with .txt,
# YOLO predictions on high res images
# and resized images get created and added to TASK_DIR
##############################################################


BASE_DIR = '/pool1/srv/cvat-tasks/'
LABEL_MAP = {0: 'Arthropod'}
ORGANIZATION_SLUG = 'Ecdysis'
CVAT_CLIENT_URL = 'https://app.cvat.ai/'
CONFIGURATION = Configuration(
    host=CVAT_CLIENT_URL,
    access_token=CVAT_APIKEY
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download cvat.ai labels')
    parser.add_argument(
        '--task-dir',
        required=True,
        help="The cvat.ai task name and the directory at BASE_DIR"
    )
    parser.add_argument(
        '--project-name',
        required=True,
        help="The project-name and cvat.ai"
    )
    args = parser.parse_args()
    return args


def get_image_files(data_dir):
    return sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])


def get_data_dir(args):
    return f'{BASE_DIR}{args.task_dir}'


def create_task_from_directory(args):
    # Create a Client instance bound to a local server and authenticate using basic auth
    data_dir = get_data_dir(args)
    print(f'Data directory is {data_dir}')
    with make_client(CVAT_CLIENT_URL, access_token=CVAT_APIKEY) as client:
        client.organization_slug = ORGANIZATION_SLUG

        # Get or set project
        project = None
        existing_projects = client.projects.list()
        if existing_projects:
            projects = [v.name for v in existing_projects]
            if args.project_name in projects:
                project = existing_projects[projects.index(args.project_name)]
                print(f"Using existing project: {project.name} (ID: {project.id})")
        if not project:
            # Define labels at the Project level
            labels = [{"name": name} for name in LABEL_MAP.values()]
            project_spec = models.ProjectWriteRequest(name=args.project_name, labels=labels)
            project = client.projects.create(project_spec)
            print(f"Created new project: {project.name}")

        # get the images
        image_files = get_image_files(data_dir)
        image_paths = [os.path.join(data_dir, f) for f in image_files]

        # Create one Task for all images
        # segment_size=1 creates one job per image
        task_spec = models.TaskWriteRequest(
            name=args.task_dir,
            project_id=project.id,
            segment_size=1
        )
        task = client.tasks.create(task_spec)
        print(f'Created Task ID: {task.id}. Uploading {len(image_paths)} images...')

        # Create a temporary ZIP file
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
            tmp_zip_path = tmp_zip.name
            with zipfile.ZipFile(tmp_zip_path, 'w') as zf:
                for filename in image_files:
                    full_path = os.path.join(data_dir, filename)
                    # We add the file to the zip. Frame order follows 'image_files' order.
                    zf.write(full_path, arcname=filename)

            print(f"Created temporary zip ({os.path.getsize(tmp_zip_path) // 1024**2} MB). Uploading...")

            # Upload the ZIP
            task.upload_data([tmp_zip_path])

        # Clean up the temp file
        if os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)

        print("Waiting for server to process images and create jobs...")
        time.sleep(2)
        max_retries = 300
        for i in range(max_retries):
            # Refresh the task object from the server
            task.fetch()

            # Check if the jobs are generated
            jobs = task.get_jobs()
            if len(jobs) == len(image_files):
                print(f"Server ready! All {len(jobs)} jobs created.")
                break

            print(f"Attempt {i+1}: Jobs not ready yet. Retrying in 10s...")
            time.sleep(10)
        else:
            raise TimeoutError("CVAT took too long to create jobs. Check server logs.")

    return True


def search_tasks_by_project(project_id, task_name):
    """
    Returns one task id filtered by criteria, or None.
    """
    # see client filters https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/tasks-api/#list
    task_ids = []
    with ApiClient(CONFIGURATION) as api_client:
        page = 1
        while True:
            (data, _) = api_client.tasks_api.list(
                x_organization=ORGANIZATION_SLUG,
                page=page,
                status=project_id,
                search=task_name,
                page_size=100
            )
            task_ids += [task.id for task in data['results'] if task.name == task_name]
            if data['next'] is None:
                break
            page += 1
    if len(task_ids) != 1:
        print(f'Found {len(task_ids)} task ids which is != 1. Returning None')
        return None
    return task_ids[0]


def patch_annotations(args):
    data_dir = get_data_dir(args)
    with make_client(CVAT_CLIENT_URL, access_token=CVAT_APIKEY) as client:

        print(f"Searching for project: '{args.project_name}'...")
        project = None
        existing_projects = client.projects.list()
        if existing_projects:
            projects = [v.name for v in existing_projects]
            if args.project_name in projects:
                project = existing_projects[projects.index(args.project_name)]
                print(f"Using existing project: {project.name} (ID: {project.id})")
        if not project:
            raise ValueError(f"Project '{args.project_name}' not found.")

        # Find the task inside that specific project using the task name, we use the TASK_DIR
        print(f"Searching for task: '{args.task_dir}' within project...")
        task_id = search_tasks_by_project(project.id, args.task_dir)
        if not task_id:
            raise ValueError(f"Task '{args.task_dir}' not found in project '{args.project_name}'.")
        print(f"Found Task! ID: {task_id}")
        retrieved_task = client.tasks.retrieve(task_id)

        image_files = get_image_files(data_dir)
        cvat_labels = project.get_labels()
        label_name_to_id = {v.name: v.id for v in cvat_labels}

        for idx, filename in enumerate(image_files):
            label_file = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(data_dir, label_file)

            if not os.path.exists(label_path):
                print(f'WARNING: Not Found: {label_path} continuing...')
                continue
            with Image.open(os.path.join(data_dir, filename)) as img:
                w, h = img.size

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    # De-normalize coordinates
                    pixel_coords = []
                    for i in range(0, len(coords), 2):
                        pixel_coords.append(coords[i] * w)     # x
                        pixel_coords.append(coords[i+1] * h)   # y

                    shape = (models.LabeledShapeRequest(
                        frame=idx,  # Assign to the correct frame index
                        label_id=label_name_to_id[LABEL_MAP[class_id]],
                        type="polygon",
                        points=pixel_coords,
                        occluded=False,
                        attributes=[],
                    ))
                    retrieved_task.update_annotations(models.PatchedLabeledDataRequest(shapes=[shape]))
                    print(f"Uploaded label for {idx}.")
        return task.id


def notify_stitcher(args, task_id):
    img_files = get_image_files()
    with make_client(CVAT_CLIENT_URL, access_token=CVAT_APIKEY) as client:
        task = client.tasks.retrieve(task_id)
        meta = task.get_meta()
        jobs = client.jobs.list(task_id=task.id)

        for img in img_files:
            frame_id = None
            for idx, frame_meta in enumerate(meta.frames):
                if frame_meta.name == img:
                    frame_id = idx
                    break

            if frame_id is None:
                print(f"WARNING: Image '{img}' not found in task {task_id}. Continuing...")
                continue

            print(f"Found image! Frame ID maps to: {frame_id}")

            target_job = None
            for job in jobs:
                if job.start_frame <= frame_id <= job.stop_frame:
                    target_job = job
                    break

            if not target_job:
                print(f"WARNING: Could not find an active job enclosing frame {frame_id}. Continuing..")
                continue

            print(f"The image belongs to Job ID: {target_job.id}")

            print(f"Notifying Stitcher of Job{target_job.id}.")
            label_file = os.path.splitext(img)[0] + ".txt"
            guid = extract_guid(img)
            post_params = {
                'guid': guid,
                'project': args.project_name,
                'label_project_dir': args.task_name,
                'label_file': label_file,
                'label_job_id': target_job.id,
                'label_task_id': task_id
            }
            post_sent_ls(post_params)


if __name__ == '__main__':
    args = get_args()
    task_created = create_task_from_directory(args)

    if task_created:
        task_id = patch_annotations(args)
        notify_stitcher(args, task_id)
