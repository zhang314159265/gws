"""
Dedupe rows in input CSV using the unique key specified by user.

User can also specify the list of fields that should be concat together for
merged rows. All other fields much be the same for rows to be merged
"""
import argparse
import csv
from collections import namedtuple

def merge(acc, line_obj):
    fields = acc._fields
    out_dict = {}
    for f in fields:
        acc_val = getattr(acc, f)
        new_val = getattr(line_obj, f)

        if f in args.concat_fields:
            out_dict[f] = f"{acc_val}|{new_val}"
        else:
            assert acc_val == new_val, f"{acc_val} v.s. {new_val}"
            out_dict[f] = acc_val
    return type(acc)(**out_dict)

args = None
def main():
    parser = argparse.ArgumentParser(description="A general tool to dedupe rows in a CSV file according to user provided keys.")
    parser.add_argument(
        "-i", "--input-csv", help="The input CSV file to process", required=True
    )
    parser.add_argument(
        "-k", "--unique-key-name", help="A single column name used for deduping", required=True
    )
    parser.add_argument(
        "-c", "--concat-fields", help="The comma separated list for fields that should be concat when merging rows", default=None,
    )
    parser.add_argument(
        "-o", "--output-csv", help="The output CSV file.", default=None,
    )
    global args
    args = parser.parse_args()
    if not args.concat_fields:
        args.concat_fields = set()
    else:
        args.concat_fields = set(args.concat_fields.split(','))

    seen = dict()
    num_input_lines = 0
    with open(args.input_csv) as f:
        csv_reader = csv.reader(f)

        header = next(csv_reader)
        LogLine = namedtuple("LogLine", list(header))
        for line in csv_reader:
            line_obj = LogLine(*line)
            num_input_lines += 1

            key = getattr(line_obj, args.unique_key_name)
            if key in seen:
                seen[key] = merge(seen[key], line_obj)
            else:
                seen[key] = line_obj

    print(f"{num_input_lines} input lines -> {len(seen)} output lines")

    if args.output_csv:
        assert len(seen) > 0
        fields = next(iter(seen.values()))._fields

        with open(args.output_csv, "w") as f:
            writer = csv.writer(f)
            writer.writerow(fields)

            for line in seen.values():
                values = [getattr(line, name) for name in fields]
                writer.writerow(values)
        print(f"Output is written to {args.output_csv}")

if __name__ == "__main__":
    main()
    print("bye")
