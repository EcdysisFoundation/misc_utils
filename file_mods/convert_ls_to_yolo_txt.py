import requests
import numpy as np
from PIL import Image
from pathlib import Path
from config_secrets import STITCHER_API_URL
from stitcher_api import get_root_message, ERROR_MSG_KEY


# convert label studio to coco, then convert that to yolo
# do this per FastAPI retrieved record, saving the label in a cvat-tasks directory, without an image.
# name the label consitent with other yolo/cvat labels


CVAT_LABEL_DIR_LS_CONVERTIONS  = '/pool1/srv/cvat-tasks/label_studio_conversions'
IMG_FILE_MOUNT = '/pool1/srv/label-studio/mydata/stitchermedia'


def get_ls_polygon_bounding_box(x, y):
    """
    From https://github.com/HumanSignal/label-studio-sdk/blob/master/src/label_studio_sdk/converter/utils.py
    """

    assert len(x) == len(y)
    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
    return [x1, y1, x2 - x1, y2 - y1]


def get_polygon_area(x, y):
    """
    From https://github.com/HumanSignal/label-studio-sdk/blob/master/src/label_studio_sdk/converter/utils.py
    https://en.wikipedia.org/wiki/Shoelace_formula

    """

    assert len(x) == len(y)
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def convert_ls_polygonlabels_to_coco(
        annotation_id, image_id,
        points, width, height):
    """
    From https://github.com/HumanSignal/label-studio-sdk/blob/master/src/label_studio_sdk/converter/converter.py#L836
    """
    points_abs = [
        (x / 100 * width, y / 100 * height) for x, y in points
    ]
    x, y = zip(*points_abs)

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 0,  # single category
        "segmentation":
            [
                [coord for point in points_abs for coord in point]
            ],
        "bbox": get_ls_polygon_bounding_box(x, y),
        "ignore": 0,
        "iscrowd": 0,
        "area": get_polygon_area(x, y),
    }

def min_index(arr1: np.ndarray, arr2: np.ndarray):
    """Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        (tuple[int, int]): A tuple (idx1, idx2) where idx1 is the index in arr1 and idx2 is the index in arr2 of the
            pair with the shortest distance.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments: list[list]):
    """Merge multiple segments into one list by connecting the coordinates with the minimum distance between each
    segment.

    This function connects these coordinates with a thin line to merge all segments into one.

    Args:
        segments (list[list]): Original segmentations in COCO's JSON file. Each element is a list of coordinates, like
            [segmentation1, segmentation2,...].

    Returns:
        (list[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def convert_coco_to_yolo(coco_result, image_width, image_height, use_keypoints=False):
    """
    Modified from convert_coco at
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/converter.py
    and requirements of
    https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/
    """
    boxes = []
    segments = []
    keypoints = []
    classificaions = []
    for anno in coco_result:
        if anno.get("iscrowd", False):
            continue
        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(anno["bbox"], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= image_width  # normalize x
        box[[1, 3]] /= image_height  # normalize y
        if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
            continue
        boxes.append(box)
        classificaions.append(anno['category_id'])

        if not anno.get("segmentation"):
            segments.append([])
        elif len(anno["segmentation"]) > 1:
            # sometimes multiple polygons are predicted for a single object
            s = merge_multi_segment(anno["segmentation"])
            s = (np.concatenate(s, axis=0) / np.array([image_width, image_height])).reshape(-1).tolist()
        else:
            s = [j for i in anno["segmentation"] for j in i]  # all segments concatenated
            s = (np.array(s).reshape(-1, 2) / np.array([image_width, image_height])).reshape(-1).tolist()
        segments.append(s)

        if use_keypoints:
            if anno.get("keypoints") is None:
                keypoints.append([])
            keypoints.append(
                box + (np.array(anno["keypoints"]).reshape(-1, 3) / np.array([image_width, image_height, 1])).reshape(-1).tolist()
            )
    assert len(boxes) == len(classificaions)
    if segments:
        assert len(boxes) == len(classificaions) == len(segments)
    if keypoints:
        assert len(boxes) == len(classificaions) == len(segments) == len(keypoints)

    return {
        'boxes': boxes,
        'classificaions': classificaions,
        'segments': segments,
        'keypoints': keypoints
    }


def ls_segmentation_yolo_conversions():
    """
    Use the api and get .json file of training set.
    anno_size_gte, if not None, filters annotations to have
    at least a width and height of the bounding box in
    # of anno_size_gte pixels.
    """
    api_ping = get_root_message()
    print(api_ping)
    if ERROR_MSG_KEY in api_ping.keys():
        return
    api_list_url = STITCHER_API_URL + '/list-upload-files/'
    offset = 0
    limit = 10

    anno_size_gte=50

    image_id = 0

    while True:
        if True:
            # stop early for testing
            if image_id > 1:
                break
        params = {
            'offset': offset,
            'limit': limit,
            'approved': True
        }
        print('-list-upload-files-' * 6)
        print(params)

        try:
            response = requests.get(api_list_url, params=params)
        except Exception as e:
            print(e)
            break

        if response.status_code == 200:
            data = response.json()
            if not data:
                break

            print(f'data returned from api for next {limit} records')

            for row in data:
                if not row['annotations_segment']:
                    # this record does not have label studio annotations
                    continue
                if row['label_file'] or row['label_project_dir']:
                    # this record has cvat labels already
                    continue
                # clean incomplete annotations
                row['annotations_segment'] = [
                    v for v in row['annotations_segment'] if v['closed']
                ]
                if not row['annotations_segment']:
                    # There are no annotations
                    continue

                print(f"{row['upload_dir_name']} passed filtering, checking additional criteria in process")

                # get image dims, make sure its there
                img_dir = f"{IMG_FILE_MOUNT}{row['panorama_path'].replace('/media', '')}"
                img_filename = str(Path(img_dir).stem)
                label_name = f"{row['upload_dir_name']}__{row['guid']}__{img_filename}"
                label_path = Path(f"{CVAT_LABEL_DIR_LS_CONVERTIONS}/{label_name}.txt")

                if not Path(img_dir).is_file():
                    print(f"IMAGE NOT FOUND{img_dir}")
                    continue

                with Image.open(img_dir) as img:
                    width, height = img.size

                # convert and format the annotations and other info
                coco_annotations = [convert_ls_polygonlabels_to_coco(i, image_id,
                    v['points'], width, height) for i, v in enumerate(row['annotations_segment'])]
                # filter to anno_size_gte
                coco_annotations = [
                    v for v in coco_annotations if v['bbox'][2] >= anno_size_gte or v['bbox'][3] >= anno_size_gte
                ]
                if not coco_annotations:
                    print(f"{row['upload_dir_name']} filtered out to no objects larger than anno_size_gte {anno_size_gte}")
                    continue
                yolo_annotations = convert_coco_to_yolo(coco_annotations, width, height)

                # write the yolo .txt file
                full_label_save_path = str(label_path)
                with open(full_label_save_path, mode="w", encoding="utf-8") as file:
                    for i, cat in enumerate(yolo_annotations['classificaions']):
                        polygon = yolo_annotations['segments'][i]
                        if polygon:
                            file.write(f"{cat} {' '.join(str(v) for v in polygon)}\n")
                print(f"wrote {row['upload_dir_name']} label to {full_label_save_path}")
                image_id += 1

            offset += limit
        else:
            print(f"Error: {response.status_code}")
            break


if __name__ == '__main__':

    # avoid DecompressionBombError
    max_image_pixels = Image.MAX_IMAGE_PIXELS
    print(f'MAX_IMAGE_PIXES is {Image.MAX_IMAGE_PIXELS}')
    if max_image_pixels < 180000000:
        Image.MAX_IMAGE_PIXELS = max_image_pixels * 4
        print(f'raised MAX_IMAGE_PIXES to {Image.MAX_IMAGE_PIXELS}')

    ls_segmentation_yolo_conversions()
