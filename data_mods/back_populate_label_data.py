import os

from ..gen_utils import extract_guid
from ..stitcher_api import post_sent_ls

####################################################
### Backpopluate fields used in cvat_utils.py
### These are fields that did not formerly exist
### TASK_NAME is the cvat-tasks directory to search
### python -m misc_utils.data_mods.back_populate_label_data
####################################################

TASK_NAME = 'sdk_test'
PROJECT_NAME = "SDK Test Project"

DATA_DIR = f'/pool1/srv/cvat-tasks/{TASK_NAME}'


def update_recs_from_directory():

    label_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(('.txt'))])
    print(label_files)
    guid_map = [(extract_guid(i), i) for i in label_files]
    print(guid_map)
    for v in guid_map:
        post_params = {
            'guid': v[0],
            'project': PROJECT_NAME,
            'label_project_dir': TASK_NAME,
            'label_file': v[1]
        }
        post_sent_ls(post_params)


if __name__ == '__main__':
    update_recs_from_directory()
