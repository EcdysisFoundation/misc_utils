import cv2
import re
from uuid import UUID
from pathlib import Path


def extract_guid(text):
    # Pattern for a standard UUID (hexadecimal chars in 8-4-4-4-12 format)
    guid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'

    match = re.search(guid_pattern, text, re.IGNORECASE)

    if match:
        found_str = match.group(0)
        try:
            # Final validation: check if it actually loads as a UUID object
            return str(UUID(found_str))
        except ValueError:
            return None
    return None


def load_resize_and_save_thumbnail(path, thumbnail_path, new_width=600):

    try:
        img = cv2.imread(path)
    except cv2.error as e:
        print(f"Error loading image: {e}")
        img = None
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # Resize, keeping aspect ratio, or skip
    h, w = img.shape[:2]
    if float(w) <= new_width:
        return
    scale = new_width / float(w)
    new_height = int(h * scale)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    success = cv2.imwrite(thumbnail_path, resized)
    if not success:
        raise IOError(f"Could not write image: {thumbnail_path}")
    return


def analyze_txt_files_for_empties(directory_path: str):
    """
    Opens all .txt files in a directory and counts how many are empty
    versus how many have actual content.
    """
    target_dir = Path(directory_path)

    # Safety check: Ensure the directory exists
    if not target_dir.is_dir():
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    empty_count = 0
    content_count = 0
    total_files = 0

    print(f"Analyzing .txt files in '{directory_path}'...\n")

    # Look for all files ending with .txt
    for file_path in target_dir.glob("*.txt"):
        if file_path.is_file():
            total_files += 1
            try:
                # Open and read the file
                content = file_path.read_text(encoding="utf-8")

                # .strip() removes whitespace, ensuring files with just spaces/newlines count as empty
                if not content.strip():
                    print(f"  [Empty]  {file_path.name}")
                    empty_count += 1
                else:
                    print(f"  [{len(content)} chars] {file_path.name}")
                    content_count += 1
            except Exception as e:
                print(f"  [Error]   Could not read {file_path.name}. Error: {e}")

    # Display the final summary
    print("\n" + "-" * 40)
    print("ANALYSIS SUMMARY")
    print("-" * 40)
    print(f"Total .txt files scanned: {total_files}")
    print(f"Files with content:       {content_count}")
    print(f"Empty files:              {empty_count}")
    print("-" * 40)
    return {
        'total_files': total_files,
        'content_count': content_count,
        'empty_count': empty_count
    }
