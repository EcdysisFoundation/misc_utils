import csv

BASE_LOCAL_PATH = '/home/ecdysis/microbiome-omics/output'

def write_metadata(header, file):
    filename_prefix = file[:9]
    in_file_path = f'{BASE_LOCAL_PATH}/{file}'
    out_file_path = f'{BASE_LOCAL_PATH}/{filename_prefix}metadata.tsv'

    with open(in_file_path, mode='r') as file:
        reader = csv.reader(file)

        with open(out_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(header)  # Write the new header

            for row in reader:
                # sampleid in first postion
                writer.writerow([row[0]])


def main():
    files = ['2022_16s_manifest.tsv', '2023_16s_manifest.tsv', '2024_16s_manifest.tsv']
    header = ['sampleid']
    for file in files:
        write_metadata(header, file)


if __name__ == '__main__':
    main()
