
import os
from PIL import Image
from cvat_sdk import make_client, models
from cvat_sdk.core.proxies.tasks import ResourceType, Task
from cvat_sdk.core.helpers import TqdmProgressReporter

from secrets import CVAT_APIKEY

TASK_NAME = 'mytask'

BASE_DIR = '/pool1/srv/cvat-tasks/'
DATA_DIR = f'{BASE_DIR}{TASK_NAME}'
LABEL_MAP = {0: 'Arthropod'}

# Create a Client instance bound to a local server and authenticate using basic auth
with make_client('https://app.cvat.ai/', access_token=CVAT_APIKEY) as client:
    client.organization_slug = 'Ecdysis'
    data_dir = f'{BASE_DIR}{TASK_NAME}'

    # 1. Define the Task Labels
    labels = [
        {"name": name, "attributes": []} for name in LABEL_MAP.values()
    ]

    # 2. Create the Task
    task_spec = models.TaskWriteRequest(
        name="YOLO Segmentation Import",
        labels=labels,
    )
    task = client.tasks.create(task_spec)
    print(f"Created task ID: {task.id}")

    # 3. Upload Images
    # Note: Sorting ensures file-to-frame consistency
    image_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
    image_paths = [os.path.join(DATA_DIR, f) for f in image_files]

    task.upload_data(image_paths, pbar=TqdmProgressReporter())
    print("Images uploaded. Processing...")

    # 4. Prepare Annotations
    shapes = []

    for frame_id, filename in enumerate(image_files):
        label_file = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(DATA_DIR, label_file)

        if not os.path.exists(label_path):
            continue

        # Get image dimensions for de-normalization
        with Image.open(os.path.join(DATA_DIR, filename)) as img:
            w, h = img.size

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                # Convert normalized YOLO [x1, y1, x2, y2...] to absolute pixels
                # CVAT expects a flat list: [x1, y1, x2, y2, ...]
                pixel_coords = []
                for i in range(0, len(coords), 2):
                    pixel_coords.append(coords[i] * w)     # x
                    pixel_coords.append(coords[i+1] * h)   # y

                # Create the CVAT Shape object
                shape = models.LabeledShapeRequest(
                    frame=frame_id,
                    label_id=next(l.id for l in task.labels if l.name == LABEL_MAP[class_id]),
                    type="polygon",
                    points=pixel_coords,
                    occluded=False,
                    attributes=[],
                )
                shapes.append(shape)

    # 5. Push Annotations to CVAT
    annotation_data = models.LabeledDataRequest(shapes=shapes)
    task.update_annotations(annotation_data)
    print(f"Successfully uploaded {len(shapes)} polygons as pre-annotations.")
