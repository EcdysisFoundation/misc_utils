#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from statistics import mean


# python -m inspect_json --path path/to/file.json --sublist-key annotations

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download cvat.ai labels')
    parser.add_argument(
        '--path',
        required=True,
        help="Path to file required."
    )
    parser.add_argument(
        '--limit',
        type=int,
        help="Limit the number of top level records to investigate"
    )
    parser.add_argument(
        '--sublist-key',
        help="Investigate a sub-list dictionary entry using the dict key to the list"
    )
    return parser.parse_args()


def type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__

def summarize_json(args):
    with open(args.path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Top-level type: {type_name(data)}")

    if not isinstance(data, list):
        if isinstance(data, dict):
            print('Top-level JSON is a dict. Converting to a list.')
            data = [data]
        else:
            print("Expected top-level JSON to be a list.")
            return

    print(f"Top-level list length: {len(data)}")

    key_presence = Counter()
    key_type_counts = defaultdict(Counter)
    list_length_stats = defaultdict(list)
    nested_dict_keys = defaultdict(int)
    bad_records = []

    records = data if args.limit is None else data[:args.limit]
    print(f"Records inspected: {len(records)}")

    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            bad_records.append((idx, type_name(item)))
            continue

        if args.sublist_key:
            sub_records = item[args.sublist_key]
            for idx, item in enumerate(sub_records):
                if not isinstance(item, dict):
                    bad_records.append((idx, type_name(item)))
                    continue
                for key, value in item.items():
                    key_presence[key] += 1
                    t = type_name(value)
                    key_type_counts[key][t] += 1

                    if isinstance(value, list):
                        list_length_stats[key].append(len(value))
                    elif isinstance(value, dict):
                        nested_dict_keys[key] += 1
        else:
            for key, value in item.items():
                key_presence[key] += 1
                t = type_name(value)
                key_type_counts[key][t] += 1

                if isinstance(value, list):
                    list_length_stats[key].append(len(value))
                elif isinstance(value, dict):
                    nested_dict_keys[key] += 1

    print("\n=== Record shape issues ===")
    if bad_records:
        print(f"Non-dict items found: {len(bad_records)}")
        for idx, t in bad_records[:20]:
            print(f"  Index {idx}: {t}")
        if len(bad_records) > 20:
            print("  ...")
    else:
        print("All inspected items are dictionaries.")

    print("\n=== Key summary ===")
    all_keys = sorted(set(key_presence) | set(key_type_counts) | set(list_length_stats) | set(nested_dict_keys))

    for key in all_keys:
        presence = key_presence[key]
        types = dict(key_type_counts[key])

        print(f"\nKey: {key}")
        print(f"  Present in records: {presence}/{len(records)}")
        print(f"  Types seen: {types}")

        if key in list_length_stats and list_length_stats[key]:
            lengths = list_length_stats[key]
            print(
                "  List lengths: "
                f"count={len(lengths)}, "
                f"min={min(lengths)}, "
                f"max={max(lengths)}, "
                f"avg={mean(lengths):.2f}"
            )

        if nested_dict_keys[key]:
            print(f"  Nested dict values seen: {nested_dict_keys[key]}")


if __name__ == "__main__":
    args = get_args()
    summarize_json(args)
