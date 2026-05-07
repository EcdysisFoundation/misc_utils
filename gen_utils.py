import re
from uuid import UUID
import cv2


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
