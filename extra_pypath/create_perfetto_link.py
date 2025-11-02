# Note: if pass in gist url directly, it should be a url for the raw content.

import argparse
import sys
import subprocess
import re

def upload_tracefile(tracefile: str) -> str:
    """
    Upload tracefile and return the gist link
    """
    cmd_upload = f"gh gist create -p {tracefile}"
    cmd_state = subprocess.run(cmd_upload.split(), capture_output=True)
    if cmd_state.returncode != 0:
        raise RuntimeError(f"Command fail {cmd_upload}")
    raw_out = cmd_state.stdout
    return raw_out.decode().strip()

def is_gist_link(link):
    """
    A gist link (not a raw link.).
    """
    return link.startswith("https://gist.github.com/")

def is_raw_gist_link(link):
    return link.startswith("https://gist.githubusercontent.com/")

def parse_gist_link(link) -> tuple[str, str]:
    """
    Given a gist link return a tuple of uid and gistid
    """
    m = re.match(r"^https://gist.github.com/([^/]+)/([\w]+)$", link)
    if not m:
        raise RuntimeError(f"Invalid gist link {link}")
    return m.group(1), m.group(2)

def get_raw_gist_link(link):
    uid, gistid = parse_gist_link(link)
    cmd_list_file = f"gh gist view --files {link}"
    cmd_state = subprocess.run(cmd_list_file.split(), capture_output=True)
    if cmd_state.returncode != 0:
        raise RuntimeError(f"Command fail {cmd_list_file}")

    raw_out = cmd_state.stdout
    filelist = raw_out.decode().strip().split("\n")
    if len(filelist) != 1:
        raise RuntimeError(f"Only support gist with a single file: {raw_out}")
    filename = filelist[0]
    raw_link = f"https://gist.githubusercontent.com/{uid}/{gistid}/raw/{filename}"
    # print(f"Raw gist link {raw_link}")
    return raw_link

def create_perflink(traceurl: str) -> str:
    if not traceurl.startswith("http"):
        # localfile
        traceurl = upload_tracefile(traceurl)

    if is_gist_link(traceurl):
        traceurl = get_raw_gist_link(traceurl)
        assert is_raw_gist_link(traceurl)

    deep_linking_tool = "https://shunting314.github.io/open-perfetto.html"
    perflink = f"{deep_linking_tool}?url={traceurl}"

    print(f"perflink {perflink}")
    return perflink

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Missing trace file/url!")
        sys.exit(1)
    tracefile = sys.argv[1]
    create_perflink(tracefile)
