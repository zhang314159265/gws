"""
Augument the CSV with gist created based on kernel path in the file system.
"""
import argparse
import csv
import subprocess
import re

def create_gist(kernel_path):
    """
    Output from 'gh gist create' looks like:

        - Creating gist with-gist.csv
        ✓ Created public gist with-gist.csv
        https://gist.github.com/shunting314/64cf67a2a72ae3e81c16aa0a5f189d4b
    """
    full_command = f"gh gist create -p {kernel_path}"
    output = subprocess.check_output(full_command.split()).decode()

    m = re.search(r"(https://gist.github.com/[^\s$]+)", output)
    assert m, "gist url not found"
    return m.group(1)

args = None
def main():
    parser = argparse.ArgumentParser(description="A general tool to augment a CSV with gists for kernels.")
    parser.add_argument(
        "-i", "--input-csv", help="The input CSV file to process", required=True
    )
    parser.add_argument(
        "-o", "--output-csv", help="The output CSV file.", default=None,
    )
    global args
    args = parser.parse_args()

    with open(args.input_csv) as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)

        line_obj_list = []
        for line in csv_reader:
            line_obj = {k: v for k, v in zip(header, line)}
            kernel_path = line_obj["kernel_path"]
            gist_url = create_gist(kernel_path)
            line_obj["gist_url"] = gist_url
            line_obj_list.append(line_obj)

    if args.output_csv:
        assert len(line_obj_list) > 0
        fields = list(line_obj_list[0].keys())
        with open(args.output_csv, "w") as f:
            writer = csv.writer(f)
            writer.writerow(fields)

            for line_obj in line_obj_list:
                writer.writerow(list(line_obj.values()))
        print(f"Output is written to {args.output_csv}")

if __name__ == "__main__":
    main()
    print("bye")
