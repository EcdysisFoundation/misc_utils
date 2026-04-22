import os
import supervision as sv

from pathlib import Path

from gen_utils import extract_guid


LABEL_DIR = '/pool1/srv/cvat-tasks/sdk_test'
DATASET_DIR_BASE = '/home/ecdysis/ultralytics/local_files'
DATASET = 'evaluation_dataset_1'
DATASET_DIR = f'{DATASET_DIR_BASE}/{DATASET}'


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


def generate_dataset():
    """
    From a directory of YOLO Segmentation 1.0 labels,
    get their guid and structure YOLO dataset with the high res images.
    Requires guid within the label file name.
    TODO: modify to use a .csv export from bugbox when selected labels are
          identified on bugbox and distributed among multiple cvat.ai projects
    """
    dir_path = Path(DATASET_DIR)
    dir_path.mkdir(exist_ok=True)
    label_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith(('.txt'))])
    for file in label_files:
        guid = extract_guid(file)
        if not guid:
            print(f'found non-conforming txt file: {file}')
            continue
        print(guid)
        print(file)



if __name__ == '__main__':
    generate_dataset()


