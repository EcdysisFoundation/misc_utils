import boto3
import os
import json
import pyvips
from pathlib import Path

import stitcher_api
import secrets
from coco_json_tool import StreamingCOCOWriter


#######
# Using Stitcher API, get images and predictions,
# create tiles with pyvips to S3 for specific cvat.ai tasks
# predictions saved to Stitcher in coco format as follows,
# where coco_result is result = sahi.predict.get_sliced_prediction()
# coco_result = result.to_coco_predictions(image_id=os.path.basename(img_path))
# json.dumps([{
#     'predictions': coco_result,
#     'original_width': original_width,
#     'original_height': original_height
# }])
#######

# production file mount
# FILE_MOUNT = '/pool1/srv/label-studio/mydata/stitchermedia'
# dev file mount
FILE_MOUNT = '/Users/michaelawilson/repos/label-studio/mydata/stitchermedia'
TASK_DIR_BASE_LOCAL = 'local_files'
TASK_DIR_BASE_CLOUD = 'dzi_projects'
AWS_STORAGE_BUCKET_NAME = 'ecdysis-label-studio'


S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id=secrets.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=secrets.AWS_SECRET_ACCESS_KEY,
    region_name=secrets.AWS_REGION,
)


def create_dzi_local(input_image_path, local_task_dir, tile_size=256):
    """
    Converts a large image to Deep Zoom Image (DZI) format.

    Args:
        input_image_path (str): Path to the high-resolution image.
        output_base_name (str): The base name for the output (e.g., 'image_1').
                                This will create 'image_1.dzi' and 'image_1_files/'.
        tile_size (int): Size of the tiles in the pyramid. Default is 256.
    """
    img_name = Path(input_image_path).stem
    output_base_name = str(Path(local_task_dir) / img_name)

    print(f"Creating DZI from {input_image_path}...")
    try:
        image = pyvips.Image.new_from_file(input_image_path, access="sequential")
        # Quality can be adjusted e.g., suffix='.jpg[Q=90]'
        image.dzsave(output_base_name, tile_size=tile_size, suffix='.jpg[Q=90]')
        print(f"Successfully created {img_name}.dzi locally.")
        return f"{img_name}.dzi"
    except Exception as e:
        print(f"Error creating DZI: {e}")
        return None


def create_cvat_manifest(dzi_files, local_task_dir):
    """Creates a CVAT manifest file listing the DZI files."""
    manifest_path = Path(local_task_dir) / "manifest.jsonl"
    print(f"Creating manifest file: {manifest_path}...")
    with open(manifest_path, 'w') as f:
        # Simple manifest for DZI. No extra details needed by CVAT,
        # just the relative filename.
        # Format: {"name": "filename.dzi"}
        for dzi_file in dzi_files:
            f.write(json.dumps({"name": dzi_file}) + '\n')
    print("Manifest creation complete.")


def upload_directory_to_s3(local_directory, s3_bucket, s3_prefix):
    """Recursively uploads a local directory to S3."""
    print(f"Uploading directory {local_directory} to s3://{s3_bucket}/{s3_prefix}...")
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            # Create relative path from the start of local_directory
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            # Optionally: Set Content-Type for jpg tiles for browser efficiency
            extra_args = {}
            if file.lower().endswith(('.jpg', '.jpeg')):
                extra_args['ContentType'] = 'image/jpeg'
            elif file.lower().endswith('.dzi'):
                extra_args['ContentType'] = 'application/xml'
            elif file.lower().endswith('.json') or file.lower().endswith('.jsonl'):
                extra_args['ContentType'] = 'application/json'

            try:
                S3_CLIENT.upload_file(local_path, s3_bucket, s3_path, ExtraArgs=extra_args)
            except Exception as e:
                print(f"Failed to upload {local_path} to {s3_path}: {e}")

    print("Upload complete.")


def run_main(task_name, send_these_sites=[], send_these_panos=[], require_predictions=True):

    if not task_name:
        print('WARNING: no task_name provided, exiting...')

    print(f"=== Starting labeling resource preparation for {task_name} ===")

    local_task_dir = Path(TASK_DIR_BASE_LOCAL) / task_name
    local_task_dir.mkdir(parents=True, exist_ok=True)
    dzi_filenames = []
    image_id_counter = 1

    if require_predictions:
        final_coco_path = local_task_dir / "pre_annotations.json"
        my_categories = [{"id": 0, "name": "Arthropod"}]
        writer = StreamingCOCOWriter(f'./{final_coco_path}', my_categories)

    all_filters = send_these_sites + send_these_panos
    for site_or_dir in all_filters:
        filtered_data = stitcher_api.get_stitcher_data(site_or_dir)

        for d in filtered_data:
            # we use a name convention in first for characters, filter those
            if d['upload_dir_name'][:4] not in send_these_sites \
                    and d['upload_dir_name'] not in send_these_panos:
                continue
            if d['panorama_path']:
                p = FILE_MOUNT + d['panorama_path']
                p = p.replace('/media', '')
                if os.path.exists(p):
                    print(f'Image found: {p}')
                else:
                    print(f'WARNING: Image not found, continuing: {p}')
                    continue
            if not d['predictions_coco'] and require_predictions:
                print(f"require_predictions enabled, skipping {d['upload_dir_name']} does not have predictions")
                continue

            dzi_file = create_dzi_local(p, f'./{TASK_DIR_BASE_LOCAL}/{task_name}/{d['upload_dir_name']}')
            if dzi_file:
                dzi_filenames.append(dzi_file)

            if require_predictions and dzi_file:
                # Stream it to disk!
                writer.add_image_and_predictions(
                    image_id=image_id_counter,
                    file_name=dzi_file,
                    width=d['panorama_width'],
                    height=d['panorama_height'],
                    sahi_predictions=d['predictions_coco']
                )

                image_id_counter += 1

    if require_predictions:
        writer.close()
    print(f'Completed creating {len(dzi_filenames)} dzi files')

    create_cvat_manifest(dzi_filenames, local_task_dir)

    s3_base_path = Path(TASK_DIR_BASE_CLOUD) / task_name
    upload_directory_to_s3(str(local_task_dir), AWS_STORAGE_BUCKET_NAME, s3_base_path)

    print(f"=== Resources ready in s3://{AWS_STORAGE_BUCKET_NAME}/{s3_base_path} ===")


if __name__ == '__main__':

    task_name = 'test2'  # is used as directory name, and expected to be task_name on cvat.ai
    send_these_sites = []   # send based on sitecode example [str(i) for i in range(4111, 4131)]
    send_these_panos = []  # use the upload_dir, example [4308_sw_T2, ...]

    run_main(
        task_name=task_name,
        send_these_sites=send_these_sites,
        send_these_panos=send_these_panos)
