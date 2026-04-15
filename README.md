# Misc_utils

This repo is for Ecdysis Foundation miscellaneous utilities and etc.

## CVAT.AI labeling projects and the Stitcher API

The files, `cvat_create_project.py` and `cvat_download_labels.py` are used to create cvat.ai projects and update our Stitcher API with cvat.ai exports, respectively. To create a new project, identify site codes or panorama upload directories from the Stitcher (https://github.com/EcdysisFoundation/stitcher) data to use. Create predictions for these using the https://github.com/EcdysisFoundation/ultralytics inference module. This will create a directory of prediction labels. Next, use `cvat_create_project.py`, directing to this new directory, to create a cvat.ai project. Once the project is completed on cvat.ai, use `cvat_download_labels.py` to replace our local copy of labels and notify the Stitcher API they are updated.

## cleancopy_fastq_dirs.py

This unzips and cleans directories of fastq files in preperation for qiime2

## DZI image processing

Deep Zoom Image processing, see panorama_tile_to_s3

## Bioacoustics

Files in a directory on our local server (0007 - Audio Files) are structured in the same way audio files are stored on an external drive where new files are added from devices as site visits complete. These are generally structured in directories of a 'cluster' with sub directoires labeled with four digit sitecodes. However, there is some variation. Files outside of these sitecode folders may be mistakes of some sort and generally files outside of site-code folders can be ignored.

A process was developed to traverse all the sub folders for .wav files and save them to AWS S3 in a single directory. The name of each file is structured to capture the original file structure where a space (' ') in a file path is replaced with a single underscore ('_') and a slash in the path denoting a directory ('/') is replaced with a double underscore ('__'). Therefore, site codes can be parsed from the filename for a pattern of '__xxxx_' where x represents integers 0 - 9 (regex r'__\d{4}_'). However, with the variations, there may be some filenames where a sitecode may not be captured by this pattern. During the process, an inventory file was generated with the original path, filename, filesize, and the parsed site code if available titled 'file_inventory.csv'. The inventory also includes if the file exists on S3. As new audio files are added to the local server, the inventory can be ran again and new files uploaded to S3 at that time.

In the S3 bucket, the .wav files in directory 'source'. In directory 'meta_info' are files such as the 'file_inventory.csv', 'birds_merged.csv', and 'gps_by_sites.csv' intended to be used as meta information about the audio files or sites.
