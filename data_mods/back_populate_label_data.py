import argparse
import os
import time

from ..gen_utils import extract_guid
from ..stitcher_api import post_sent_ls, updated_label_post

####################################################
### Backpopluate fields used in cvat_utils.py,
### task_dir is the cvat-tasks directory to search,
### example use case, for task not actually on cvat
### python -m misc_utils.data_mods.back_populate_label_data --task-dir label_studio_conversions
####################################################


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Dataset generation')
    parser.add_argument(
        '--task-dir', type=str, required=True,
        help='The task dir in DATA_DIR')
    parser.add_argument(
        '--data-dir', type=str, default='/pool1/srv/cvat-tasks/',
        help='The task dir in DATA_DIR')
    return parser.parse_args()


def update_recs_from_directory(args):

    label_files = sorted([f for f in os.listdir(f"{args.data_dir}/{args.task_dir}") if f.endswith(('.txt'))])
    guid_map = [(extract_guid(i), i) for i in label_files]
    for v in guid_map:
        post_params = {
            'guid': v[0],
            'project': args.task_dir,
            'label_project_dir': args.task_dir,
            'label_file': v[1],
            'label_job_id': 0,  # no job, required field
            'label_task_id': 0  # no id, required field
        }
        post_sent_ls(post_params)
    return guid_map


if __name__ == '__main__':
    args = get_args()
    seconds = 60
    print(f"update_recs_from_directory...{args.data_dir}")
    guid_map = update_recs_from_directory(args)
    print(f"Completed, waiting {seconds} seconds before updated_label_post")
    time.sleep(seconds)
    print(f"next will updated_label_post for {len(guid_map)} records")
    for guid in guid_map:
        post_params = {
            'guid': guid[0]
        }
        updated_label_post(post_params)
