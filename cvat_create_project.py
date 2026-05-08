
import os
from PIL import Image
from cvat_sdk import make_client, models

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

TASK_DIR = 'sdk_test'

PROJECT_NAME = "SDK Test Project"

BASE_DIR = '/pool1/srv/cvat-tasks/'
DATA_DIR = f'{BASE_DIR}{TASK_DIR}'
LABEL_MAP = {0: 'Arthropod'}
ORGANIZATION_SLUG = 'Ecdysis'

def create_task_from_directory():
    # Create a Client instance bound to a local server and authenticate using basic auth
    with make_client('https://app.cvat.ai/', access_token=CVAT_APIKEY) as client:
        client.organization_slug = ORGANIZATION_SLUG

        # Get or set project
        project = None
        existing_projects = client.projects.list()
        if existing_projects:
            projects = [v.name for v in existing_projects]
            if PROJECT_NAME in projects:
                project = existing_projects[projects.index(PROJECT_NAME)]
                print(f"Using existing project: {project.name} (ID: {project.id})")
        if not project:
            # Define labels at the Project level
            labels = [{"name": name} for name in LABEL_MAP.values()]
            project_spec = models.ProjectWriteRequest(name=PROJECT_NAME, labels=labels)
            project = client.projects.create(project_spec)
            print(f"Created new project: {project.name}")

        # Map the labels, get the images
        cvat_labels = project.get_labels()
        label_name_to_id = {l.name: l.id for l in cvat_labels}
        image_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        image_paths = [os.path.join(DATA_DIR, f) for f in image_files]

        # Create one Task for all images
        # segment_size=1 creates one job per image
        task_spec = models.TaskWriteRequest(
            name=TASK_DIR,
            project_id=project.id,
            segment_size=1
        )
        task = client.tasks.create(task_spec)
        print(f'Created Task ID: {task.id}. Uploading {len(image_paths)} images...')
        task.upload_data(image_paths)

        # Prepare and upload annotations for all frames
        all_shapes = []
        for idx, filename in enumerate(image_files):
            label_file = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(DATA_DIR, label_file)

            if os.path.exists(label_path):
                with Image.open(os.path.join(DATA_DIR, filename)) as img:
                    w, h = img.size

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue

                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]

                    # De-normalize coordinates
                    pixel_coords = []
                    for i in range(0, len(coords), 2):
                        pixel_coords.append(coords[i] * w)     # x
                        pixel_coords.append(coords[i+1] * h)   # y

                    all_shapes.append(models.LabeledShapeRequest(
                        frame=idx, # Assign to the correct frame index
                        label_id=label_name_to_id[LABEL_MAP[class_id]],
                        type="polygon",
                        points=pixel_coords,
                        occluded=False,
                        attributes=[],
                    ))

        if all_shapes:
            task.update_annotations(models.PatchedLabeledDataRequest(shapes=all_shapes))
            print(f"Uploaded {len(all_shapes)} total shapes across {len(image_files)} jobs.")


    print(f"\nSuccessfully uploaded {len(image_files)} to task: {TASK_DIR}.")
    return [(extract_guid(i), os.path.splitext(i)[0] + ".txt") for i in image_files]


if __name__ == '__main__':
    sent_guids = create_task_from_directory()
    for v in sent_guids:
        post_params = {
            'guid': v[0],
            'project': PROJECT_NAME,
            'label_project_dir': TASK_DIR,
            'label_file': v[1]
        }
        post_sent_ls(post_params)
