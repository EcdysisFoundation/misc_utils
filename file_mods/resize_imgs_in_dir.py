import argparse
import os
from pathlib import Path

from ..gen_utils import load_resize_and_save_thumbnail


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download cvat.ai labels')
    parser.add_argument(
        '--file',
        help="specifiy file instead of resizing all"
    )
    parser.add_argument(
        '--base-local-path',
        default='/home/ecdysis/ultralytics/local_files/output',
        help='Change the default base local path'
    )
    parser.add_argument(
        '--delete-original',
        action='store_true',
        help='Deletes the original files that get resized if this option provided'
    )
    parser.add_argument(
        '--resize-width',
        default=600,
        type=int,
        help='Change the default resize width'
    )
    return parser.parse_args()


def get_file_path(root, file):
    return f'{root}/{file}'


def get_thumbnail_path(args, file_path):
    name_only = Path(file_path).stem
    suffix = Path(file_path).suffix
    file_path_thumbnail = f'{args.base_local_path}/{name_only}_thumbnail{suffix}'
    return file_path_thumbnail


def resize_all_imgs_in_dir(args):
    resized_files = 0
    deleted_files = 0
    for root, dirs, files in os.walk(args.base_local_path):
        for filename in files:
            suffix = Path(args.file).suffix
            if suffix.lower() not in ['.jpg', '.png']:
                continue
            file_path = get_file_path(root, filename)
            file_path_thumbnail = get_thumbnail_path(args, file_path)
            print(f'Resizing {file_path}')
            load_resize_and_save_thumbnail(
                file_path, file_path_thumbnail, args.resize_width)
            resized_files += 1
            if args.delete_original:
                Path(file_path).unlink()
                deleted_files += 1
    print(f'Resized file count: {resized_files}')
    print(f'Deleted file count: {deleted_files}')


if __name__ == '__main__':
    args = get_args()
    if args.delete_original:
        print(f'Warning: --delete-original is set to True')
    if args.file:
        file_path = get_file_path(args.base_local_path, args.file)
        file_path_thumbnail = get_thumbnail_path(args, file_path)
        load_resize_and_save_thumbnail(
            file_path, file_path_thumbnail, args.resize_width)
        print(f'Completed resize of {file_path}')
    else:
        resize_all_imgs_in_dir(args)
