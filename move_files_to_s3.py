import boto3
import logging

import config_secrets

logging.basicConfig(level=logging.DEBUG)

S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id=config_secrets.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config_secrets.AWS_SECRET_ACCESS_KEY,
    region_name=config_secrets.AWS_REGION,
)


BUCKET_NAME = 'ecdysis-public'
BUCKET_SUBDIR = 'qiime2'

# if nameing individual files within a single directory
BASE_LOCAL_PATH = '/home/ecdysis/microbiome-omics/output'
FILES_TO_SEND = []


def upload_progress(bytes_transferred):
    print(f"{bytes_transferred} bytes uploaded...")


def send_file_list():
    for file in FILES_TO_SEND:
        print(f'sending {file}')
        file_path = f'{BASE_LOCAL_PATH}/{file}'
        s3_key = f'{BUCKET_SUBDIR}/{file}'
        S3_CLIENT.upload_file(file_path, BUCKET_NAME, s3_key, Callback=upload_progress)


if __name__ == '__main__':
    send_file_list()
