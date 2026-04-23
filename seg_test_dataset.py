import os
import re
import shutil
import supervision as sv

from pathlib import Path

from gen_utils import extract_guid


LABEL_DIR = '/pool1/srv/cvat-tasks/sdk_test'
IMG_SOURCE_DIR = '/pool1/srv/label-studio/mydata/stitchermedia'
DATASET_DIR_BASE = '/home/ecdysis/ultralytics/local_files'
DATASET = 'evaluation_dataset_1'
DATASET_DIR = f'{DATASET_DIR_BASE}/{DATASET}'
TEST_DIR = 'test'


def convert_dataset_to_coco():
    # Load your YOLO segmentation dataset
    ds = sv.DetectionDataset.from_yolo(
        images_directory_path="path/to/test/images",
        annotations_directory_path="path/to/test/labels",
        data_yaml_path="data.yaml"
    )

    # Export to COCO format (This creates your dataset.json)
    ds.as_coco(
        images_directory_path="path/to/test/images",
        annotations_path="dataset.json"
    )


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
        # THIS needs to return the final dest for image and labels in test dir.
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


if __name__ == '__main__':
    generate_dataset()
