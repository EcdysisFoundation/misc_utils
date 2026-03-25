############
## clean and copy fastq files to a new directory for processing
## fastq files need removed from .zip directories
## omits unneccessary files and retains original directory structure
############
import os
import zipfile
import re
import shutil


STARTING_LOCATION = '/pool1/smb/fastq/ALL years fastQ'
ENDING_LOCATION = '/pool1/srv/fastq'
YEAR_DIRS = ['2022', '2023', '2024']

# Match a space, an open parenthesis, one or more digits, and a closing parenthesis
duplicate_pattern = r' \(\d+\)'

def make_dir(dir_to_make):
    # make_dir within ENDING_LOCATION only
    new_dir = ENDING_LOCATION + f'/{dir_to_make}'
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)
    return new_dir


def handle_zipfile(zip_path, save_to_dir):

    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.infolist():
            # Skip directories
            if member.is_dir():
                continue
            if not member.filename.endswith('.fastq.gz'):
                print(f'Skipping: {member.filename}')
                continue

            filename = os.path.basename(member.filename)
            target_path = os.path.join(save_to_dir, filename)
            with z.open(member) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)


def do_main():

    zips_processed = 0

    for year in YEAR_DIRS:
        # stop early
        if zips_processed > 0:
            break
        print(f'processing year {year}')

        make_dir(year)

        zip_dir = STARTING_LOCATION + f'/{year}'
        zip_files = [f for f in os.listdir(zip_dir) if f.endswith('.zip')]

        print(f"Found {len(zip_files)} ZIP archives. Starting extraction...")
        for zip_name in zip_files:
            print(f'working on {zip_name}')
            if re.search(duplicate_pattern, zip_name):
                print(f'Skipping duplicate: {zip_name}')
                continue
            unzip_dir = f'{year}/{zip_name.replace('.zip', '')}'
            save_to_dir = make_dir(unzip_dir)
            zip_path = os.path.join(zip_dir, zip_name)
            handle_zipfile(zip_path, save_to_dir)
            # stop early
            if zips_processed > 0:
                break

if __name__ == '__main__':
    do_main()
