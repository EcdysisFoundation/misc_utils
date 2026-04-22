import boto3
import csv
import os
import re
from datetime import datetime
from pathlib import Path
from botocore.exceptions import ClientError

import config_secrets

S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id=config_secrets.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config_secrets.AWS_SECRET_ACCESS_KEY,
    region_name=config_secrets.AWS_REGION,
)

CURRENT_DIRECTORY = os.getcwd()
BASE_DIRECTORY = CURRENT_DIRECTORY.replace('/misc_utils', '')
BASE_DIRECTORY = ''  # use blank on Ecdysis02
ROOT_DIR = Path(BASE_DIRECTORY + "/pool1/smb/0007 - Audio Files")
S3_SOURCE_DIR = 'source'
SITECODE_PATTERN = r'__\d{4}_'
AWS_STORAGE_BUCKET_NAME = 'ecdysis-eco-acoustics'


def get_all_files():
    # get all files (not directories), as their full path
    return [p for p in ROOT_DIR.rglob("*") if p.is_file()]


def get_filename(file_base_path):
    c = file_base_path.replace(' ', '_')
    c = c.replace("'", '')
    c = c.replace('"', '')
    c = c.replace('/', '__')  # directories as __
    return c


def check_if_file(object_key):

    try:
        S3_CLIENT.head_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=object_key)
        print(f"Object '{object_key}' exists.")
        return True

    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"Object '{object_key}' does not exist.")
        else:
            # Handle other errors, such as permission issues (403)
            print(f"An error occurred: {e.response['Error']['Message']}")

    return False


def get_sitecode(filename_clean):
    match = re.search(SITECODE_PATTERN, filename_clean)
    if match:
        extracted = match.group()
        extracted = extracted.replace('_', '')
        return extracted
    return ''


inventory = [('file_base_path', 'filename_clean', 'filesize', 'file_extension', 's3_object_key', 'on_s3', 'sitecode')]

if __name__ == '__main__':
    print(BASE_DIRECTORY)
    if ROOT_DIR.is_dir():
        print(f"The directory '{ROOT_DIR}' exists.")
    else:
        print(f'{ROOT_DIR} does not exist')
        exit()

    all_files = get_all_files()

    file_count = 0
    for f in all_files:

        file_base_path = str(f).replace(f'{str(ROOT_DIR)}/', '')
        filebasename = os.path.basename(file_base_path)
        if filebasename[0] == '.':
            continue

        file_extension = os.path.splitext(filebasename)[1]
        if file_extension != '.wav':
            continue

        filesize = os.path.getsize(f)

        filename_clean = get_filename(file_base_path)
        sitecode = get_sitecode(filename_clean)

        s3_object_key = f'{S3_SOURCE_DIR}/{filename_clean}'
        on_s3 = check_if_file(s3_object_key)

        if not on_s3:
            S3_CLIENT.upload_file(
                Filename=f,
                Bucket=AWS_STORAGE_BUCKET_NAME,
                Key=s3_object_key
            )
            on_s3 = check_if_file(s3_object_key)

        inventory.append((file_base_path, filename_clean, filesize, file_extension, s3_object_key, on_s3, sitecode))

        file_count += 1

    inventory_filename = 'file_inventory.csv'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20251230_135801
    inventory_filename = f"{inventory_filename.rsplit('.', 1)[0]}_{timestamp}.csv"

    with open('/local_files/' + inventory_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write all rows at once
        writer.writerows(inventory)
    print(f'Completed {file_count} records')
