import boto3

import secrets


S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id=secrets.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=secrets.AWS_SECRET_ACCESS_KEY,
    region_name=secrets.AWS_REGION,
)


BUCKET_NAME = 'ecdysis-public'
BUCKET_SUBDIR = 'qiime2'

# if nameing individual files within a single directory
BASE_LOCAL_PATH = '/home/ecdysis/microbiome-omics/output'
FILES_TO_SEND = [
    '2022_16s_demux-paired.qzv', '2023_16s_demux-paired.qzv', '2024_16s_demux-paired.qzv',
    '2022_its_demux-paired.qzv', '2023_its_demux-paired.qzv', '2024_its_demux-paired.qzv'
]


def send_file_list():
    for file in FILES_TO_SEND:
        file_path = f'{BASE_LOCAL_PATH}/{file}'
        s3_key = f'{BUCKET_SUBDIR}/{file_path}'
        S3_CLIENT.upload_file(file_path, BUCKET_NAME, s3_key)


if __name__ == '__main__':
    send_file_list()
