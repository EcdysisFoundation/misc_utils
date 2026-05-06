import argparse
import os
import logging

import boto3
from botocore.exceptions import ClientError

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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download cvat.ai labels')
    parser.add_argument(
        '--file',
        help="specifiy file instead of syncing all"
    )
    parser.add_argument(
        '--base-local-path',
        default='/home/ecdysis/microbiome-omics/output',
        help='Change the default base local path'
    )
    return parser.parse_args()


def upload_progress(bytes_transferred):
    print(f"{bytes_transferred} bytes uploaded...")


def send_file(args):
    print(f'sending {args.file}')
    file_path = f'{args.base_local_path}/{args.file}'
    s3_key = f'{BUCKET_SUBDIR}/{args.file}'
    S3_CLIENT.upload_file(file_path, BUCKET_NAME, s3_key, Callback=upload_progress)


def file_exists_in_s3(s3_key):
    """Checks if a file exists in the S3 bucket."""
    try:
        S3_CLIENT.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        return True
    except ClientError as e:
        # 404 means the object does not exist
        if e.response['Error']['Code'] == "404":
            return False
        # If it's another error (like 403), re-raise it
        raise


def sync_local_to_s3(args):
    """Scans local directory and uploads missing files to S3."""
    # os.walk yields a 3-tuple (dirpath, dirnames, filenames)
    for root, dirs, files in os.walk(args.base_local_path):
        for filename in files:
            # Construct the full local path
            local_path = os.path.join(root, filename)

            # Construct the S3 key relative to the BASE_LOCAL_PATH
            # This maintains your directory structure in S3
            relative_path = os.path.relpath(local_path, args.base_local_path)
            s3_key = os.path.join(BUCKET_SUBDIR, relative_path).replace("\\", "/")

            if not file_exists_in_s3(s3_key):
                print(f"Uploading: {relative_path} -> s3://{BUCKET_NAME}/{s3_key}")
                try:
                    S3_CLIENT.upload_file(
                        local_path,
                        BUCKET_NAME,
                        s3_key,
                        Callback=upload_progress
                    )
                except Exception as e:
                    print(f"Failed to upload {filename}: {e}")
            else:
                print(f"Skipping (already exists): {s3_key}")


if __name__ == '__main__':
    args = get_args()
    if args.file:
        print(f'Sending file {args.file}...')
        send_file(args)
        print('..Completed')
    else:
        print(f'Sending files in {args.base_local_path} that dont already exist, by filename')
        sync_local_to_s3(args)
        print('..Completed')
