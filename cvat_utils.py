
import os
from tqdm import tqdm
from PIL import Image
from cvat_sdk import make_client, models
from cvat_sdk.core.helpers import TqdmProgressReporter

from secrets import CVAT_APIKEY

TASK_NAME = 'sdk_test'

PROJECT_NAME = "SDK Test Project"

BASE_DIR = '/pool1/srv/cvat-tasks/'
DATA_DIR = f'{BASE_DIR}{TASK_NAME}'
LABEL_MAP = {0: 'Arthropod'}
ORGANIZATION_SLUG = 'Ecdysis'

def create_task_from_directory():
    # Create a Client instance bound to a local server and authenticate using basic auth
    with make_client('https://app.cvat.ai/', access_token=CVAT_APIKEY) as client:
        client.organization_slug = ORGANIZATION_SLUG

        # 1. Get or set project
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

        # 2. Map the labels, get the images
        cvat_labels = task.get_labels()
        label_name_to_id = {l.name: l.id for l in cvat_labels}
        image_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])

        for filename in tqdm(image_files, desc="Creating Tasks"):
            image_path = os.path.join(DATA_DIR, filename)
            label_file = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(DATA_DIR, label_file)

            # 2. Create the Task
            task_spec = models.TaskWriteRequest(
                name=f'Task_{filename}',
                project_id=project.id,
            )
            task = client.tasks.create(task_spec)
            print(f"Created task ID: {task.id}")

            task.upload_data(image_path, pbar=TqdmProgressReporter(tqdm(disable=True)))

            if os.path.exists(label_path):
                # Get image dimensions for de-normalization
                with Image.open(os.path.join(DATA_DIR, filename)) as img:
                    w, h = img.sizes

            shapes = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue

                    class_id = int(parts[0])
                    pixel_coords = []
                    coords = [float(x) for x in parts[1:]]

                    for i in range(0, len(coords), 2):
                        pixel_coords.append(coords[i] * w)     # x
                        pixel_coords.append(coords[i+1] * h)   # y

                    # Create the CVAT Shape object
                    shapes.append(models.LabeledShapeRequest(
                        frame=0,
                        label_id=label_name_to_id[LABEL_MAP[class_id]],
                        type="polygon",
                        points=pixel_coords,
                        occluded=False,
                        attributes=[],
                    ))

            if shapes:
                task.update_annotations(models.LabeledDataRequest(shapes=shapes))

    print(f"\nSuccessfully created {len(image_files)} tasks in project '{PROJECT_NAME}'.")


if __name__ == '__main__':
    create_task_from_directory()
