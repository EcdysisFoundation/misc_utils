
import os
import time
from PIL import Image
from cvat_sdk import make_client, models

from config_secrets import CVAT_APIKEY



import time
from cvat_sdk.api_client import exceptions


TASK_DIR = '71_KS_NE_2025'

PROJECT_NAME = "2025 Clusters"

BASE_DIR = '/pool1/srv/cvat-tasks/'
DATA_DIR = f'{BASE_DIR}{TASK_DIR}'
LABEL_MAP = {0: 'Arthropod'}
ORGANIZATION_SLUG = 'Ecdysis'
CVAT_TASK_ID = 2234658



def upload_annotations_with_retry(task, all_shapes, max_retries=10):
    print(f"Checking task status before uploading {len(all_shapes)} shapes...")

    for i in range(max_retries):
        task.fetch() # Refresh task data from server

        # Check if CVAT has finished processing the images
        # Status can be: "queued", "started", or "completed"
        if task.status.value == 'completed':
            print("Server is ready. Uploading annotations...")
            try:
                task.update_annotations(models.PatchedLabeledDataRequest(shapes=all_shapes))
                print("Annotations uploaded successfully!")
                return True
            except exceptions.ApiException as e:
                print(f"Annotation upload failed with error: {e}. Retrying...")
        else:
            print(f"Task status is '{task.status.value}'. Images are likely still tiling. Waiting 15s...")

        time.sleep(15) # Give the server time to breathe

    print("Failed to upload annotations after multiple attempts.")
    return False



def main():

    with make_client('https://app.cvat.ai/', access_token=CVAT_APIKEY) as client:
        # Replace 1234567 with your actual Task ID
        task = client.tasks.retrieve(CVAT_TASK_ID)

        existing_projects = client.projects.list()

        projects = [v.name for v in existing_projects]
        if PROJECT_NAME in projects:
            project = existing_projects[projects.index(PROJECT_NAME)]
            print(f"Using existing project: {project.name} (ID: {project.id})")

        image_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        image_paths = [os.path.join(DATA_DIR, f) for f in image_files]

        # Map the labels, get the images
        cvat_labels = project.get_labels()
        label_name_to_id = {l.name: l.id for l in cvat_labels}
        image_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        image_paths = [os.path.join(DATA_DIR, f) for f in image_files]

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

        # Now you can run the annotation logic
        if all_shapes:
            upload_annotations_with_retry(task, all_shapes)


if __name__ == '__main__':
    main()
