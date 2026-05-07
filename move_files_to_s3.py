import argparse
import os
import logging
import boto3
from botocore.exceptions import ClientError
import config_secrets


logging.basicConfig(level=logging.INFO)
# Specifically silence the chatty libraries
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id=config_secrets.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config_secrets.AWS_SECRET_ACCESS_KEY,
    region_name=config_secrets.AWS_REGION,
)

BUCKET_NAME = 'ecdysis-public'
# Setting threshold to 1GB. Adjust this as needed.
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 * 1024


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
    parser.add_argument(
        '--bucket-subdir',
        default='qiime2',
        help='Change the deault bucket subdir'
    )
    return parser.parse_args()


def upload_progress(bytes_transferred):
    print(f"{bytes_transferred} bytes uploaded...")


def format_bytes(size):
    """Converts bytes to a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024


def send_file(args):
    print(f'sending {args.file}')
    file_path = f'{args.base_local_path}/{args.file}'
    s3_key = f'{args.bucket_subdir}/{args.file}'
    S3_CLIENT.upload_file(file_path, BUCKET_NAME, s3_key, Callback=upload_progress)


def get_all_s3_keys(bucket, prefix):
    """
    Lists all keys in S3.
    """
    keys = set()
    paginator = S3_CLIENT.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                keys.add(obj['Key'])
    return keys


def sync_local_to_s3(args):
    """
    Scans local directory and uploads missing files to S3.
    """
    print("Fetching existing file list from S3 (please wait)...")
    existing_keys = get_all_s3_keys(BUCKET_NAME, args.bucket_subdir)
    print(f"Found {len(existing_keys)} files already in S3.")

    for root, dirs, files in os.walk(args.base_local_path):
        for filename in files:
            if filename.endswith('.py'):
                continue
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, args.base_local_path)
            flattened_name = relative_path.replace(os.sep, "_")
            s3_key = f"{args.bucket_subdir}/{flattened_name}".replace("\\", "/")

            # Check File Size First
            file_size = os.path.getsize(local_path)
            if file_size > MAX_FILE_SIZE_BYTES:
                print(f"⚠️  SKIPPING: {relative_path} is too large ({format_bytes(file_size)})")
                continue

            if s3_key not in existing_keys:
                print(f"Uploading: {relative_path} -> s3://{BUCKET_NAME}/{s3_key}")
                try:
                    S3_CLIENT.upload_file(
                        local_path,
                        BUCKET_NAME,
                        s3_key,
                        Callback=upload_progress
                    )
                except Exception as e:
                    print(f"❌ Failed to upload {filename}: {e}")
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
        print(f'Max files size is limited to {format_bytes(MAX_FILE_SIZE_BYTES)}')
        print('Use option --file to specifiy large files')
        sync_local_to_s3(args)
        print('..Completed')
