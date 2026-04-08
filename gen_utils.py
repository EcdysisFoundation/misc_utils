import re
from uuid import UUID


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
