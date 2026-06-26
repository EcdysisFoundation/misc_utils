import json

# Replace with the path to the JSON file you are passing to SAHI
coco_json_path = "local_files/cvat_dataset.json"

with open(coco_json_path, "r") as f:
    data = json.load(f)

corrupted_count = 0
for i, anno in enumerate(data.get("annotations", [])):
    bbox = anno.get("bbox")

    if bbox is None:
        print(f"❌ Entry {i}: Annotation ID {anno.get('id')} is missing the 'bbox' key entirely!")
        corrupted_count += 1
    elif len(bbox) != 4:
        print(f"❌ Entry {i}: Annotation ID {anno.get('id')} (Image ID: {anno.get('image_id')}) has a bbox of length {len(bbox)}: {bbox}")
        corrupted_count += 1

if corrupted_count == 0:
    print("✅ All bboxes in the JSON have exactly 4 entries. The issue might be a mismatch in data types.")
else:
    print(f"🔍 Found {corrupted_count} corrupted annotation(s).")
