import json
import os
import re
import shutil
import yaml

from pathlib import Path
from collections import Counter
from PIL import Image

from gen_utils import extract_guid

############################################
#
# Creates a YOLO Evaluation dataset for use with our Ultralytics SAHI repo
#
############################################


# These vars vary per test set
LABEL_DIR = '/pool1/srv/cvat-tasks/sdk_test'
DATASET = 'evaluation_dataset_1'
# These typically will not
IMG_SOURCE_DIR = '/pool1/srv/label-studio/mydata/stitchermedia'
DATASET_DIR_BASE = '/home/ecdysis/ultralytics/local_files'
DATASET_DIR = f'{DATASET_DIR_BASE}/{DATASET}'
TEST_DIR = 'test'
Image.MAX_IMAGE_PIXELS = None

CLASS_NAMES = {
    0: "Arthropod",
}  # as they appear in YOLO
# shift by one. YOLO starts at 0 but some COCO formats ignore 0
CLASS_NAMES_COCO = {i + 1: name for i, name in CLASS_NAMES.items()}


def get_image_info(image_path, image_id):
    # PIL.Image.open is "lazy" - it reads metadata without loading all pixels
    with Image.open(image_path) as img:
        width, height = img.size
    return {
        "id": image_id,
        "file_name": os.path.basename(image_path),
        "width": width,
        "height": height
    }


def yolo_to_coco_poly(yolo_poly, w, h):
    """Converts normalized YOLO [x, y, x, y...] to pixel-space [x, y, x, y...]"""
    return [coord * w if i % 2 == 0 else coord * h for i, coord in enumerate(yolo_poly)]


def convert_to_coco_lite(images_dir, labels_dir, output_json, class_names):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in class_names.items()]
    }

    ann_id = 1
    image_files = sorted(list(Path(images_dir).glob("*.jpg")))

    for i, img_path in enumerate(image_files):
        print(f"Processing {img_path.name}...")

        # 1. Get Image Info (Lightweight)
        img_info = get_image_info(img_path, i)
        coco["images"].append(img_info)

        # 2. Match with Label
        label_path = Path(labels_dir) / img_path.with_suffix('.txt').name
        if not label_path.exists():
            continue

        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0]) + 1
                poly_normalized = parts[1:]

                # Convert to pixel coordinates
                poly_pixels = yolo_to_coco_poly(poly_normalized, img_info["width"], img_info["height"])

                # Calculate simple Bbox from polygon (min/max x, min/max y)
                xs = poly_pixels[0::2]
                ys = poly_pixels[1::2]
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
                width, height = x_max - x_min, y_max - y_min

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": class_id,
                    "segmentation": [poly_pixels],
                    "area": width * height, # Simplified area
                    "bbox": [x_min, y_min, width, height],
                    "iscrowd": 0
                })
                ann_id += 1

    with open(output_json, 'w') as f:
        json.dump(coco, f)
    print(f"Done! Created {output_json}")


def create_yaml():
    """
    Generates the data.yaml file required for YOLO training.
    """
    dataset_root = Path(DATASET_DIR)
    yaml_content = {
        'path': str(dataset_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': f'images/{TEST_DIR}',
        'names': CLASS_NAMES_COCO
    }

    yaml_path = dataset_root / 'data.yaml'

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully created metadata: {yaml_path}")


def get_label_stats(label_dir):
    stats = Counter()
    for label_file in Path(label_dir).glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                class_id = line.split()[0]
                class_name = CLASS_NAMES_COCO[int(class_id) + 1]
                stats[class_name] += 1
    return stats


def extract_pano_part(filename):
    # Search for '__panorama' followed by anything that isn't a dot
    match = re.search(r'__panorama[^.]*', filename)
    if match:
        v = match.group(0)[2:]
        return v
    return None


def create_clear_dirs(dir_path, subfolders=(TEST_DIR,)):
    parent_images = Path(dir_path) / 'images'
    parent_labels = Path(dir_path) / 'labels'

    # Clear previous runs, make fresh directories
    if os.path.exists(parent_images):
        shutil.rmtree(parent_images)
    if os.path.exists(parent_labels):
        shutil.rmtree(parent_labels)

    for name in subfolders:
        i = parent_images / name
        i.mkdir(parents=True)
        ld = parent_labels / name
        ld.mkdir(parents=True)

    return {
        'images': parent_images,
        'labels': parent_labels
    }


def generate_dataset():
    """
    From a directory of YOLO Segmentation 1.0 labels,
    get their guid and structure YOLO dataset with the high res images.
    Requires guid within the label file name.
    TODO: modify to use a .csv export from bugbox when selected labels are
          identified on bugbox and distributed among multiple cvat.ai projects
          Download pano to archive dir, like in metaformer_ecdysis to ensure exact matching to db.
    """
    dir_path = Path(DATASET_DIR)
    dir_path.mkdir(exist_ok=True)
    dirs = create_clear_dirs(dir_path)

    label_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith(('.txt'))])
    for file in label_files:
        guid = extract_guid(file)
        pano_name = extract_pano_part(file)
        if not guid or not pano_name:
            print(f'found non-conforming txt file: {file}')
            continue
        dirs['images']

        source_label_path = Path(f'{LABEL_DIR}/{file}')
        pano_img = Path(f'{IMG_SOURCE_DIR}/{guid}/{pano_name}.jpg')
        file_img = file.replace('.txt', '.jpg')
        dst_img_path = dirs['images'] / TEST_DIR / file_img
        dst_label_path = dirs['labels'] / TEST_DIR / file
        if pano_img.is_file():
            dst_img_path.symlink_to(pano_img)
            shutil.copy(source_label_path, dst_label_path)
        else:
            print(f'Warning: pano_img is not a file, skipped: {pano_img}')

    stats = get_label_stats(dirs['labels'] / TEST_DIR)
    print(f"Class distribution in {TEST_DIR}: {stats}")


if __name__ == '__main__':
    generate_dataset()
    create_yaml()
    convert_to_coco_lite(
        f'{DATASET_DIR}/images/{TEST_DIR}',
        f'{DATASET_DIR}/labels/{TEST_DIR}',
        f'{DATASET_DIR}/dataset_{TEST_DIR}.json',
        CLASS_NAMES_COCO)
