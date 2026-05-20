import os

import argparse
import requests
import json

class Downloader:
    def __init__(self, args):
        self.model = args.model
        self.token = args.token
        self.output_dir = os.path.join(args.output_dir, self.model)

    def ls_url(self, dirname=""):
        return f"https://huggingface.co/api/models/{self.model}/tree/main/{dirname}"

    def file_url(self, filename):
        return f"https://huggingface.co/{self.model}/resolve/main/{filename}"

    def download_url(self, url):
        r = requests.get(url)
        return r.text

    def download_file(self, url_path, expected_size):
        local_path = os.path.join(self.output_dir, url_path)
        headers = dict(
            Authorization=f"Bearer {self.token}",
        )
        with open(local_path, "wb") as f:
            full_url = self.file_url(url_path)
            print(f"Download {url_path}")
            # print(f"{full_url=}")
            r = requests.get(full_url, headers=headers)
            assert len(r.content) == expected_size, f"{len(r.content)} v.s. {expected_size}: {r.content[:1024]}"
            f.write(r.content)

    def recursive_download(self, url_dir):
        os.makedirs(os.path.join(self.output_dir, url_dir), exist_ok=True)
        ls_str = requests.get(self.ls_url(url_dir)).text
        ls_json = json.loads(ls_str)

        for entry in ls_json:
            if entry["type"] == "directory":
                self.recursive_download(entry["path"])
            else:
                assert entry["type"] == "file"
                self.download_file(entry["path"], entry["size"])

    def download(self):
        self.recursive_download("")

def main():
    parser = argparse.ArgumentParser("Checkpoint downloader.")
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    print(args)
    Downloader(args).download()

if __name__ == "__main__":
    main()
