import tempfile
import re
import subprocess
import os

def ttir_to_ptx(ttir_code, ptx_version="8.0"):
    """
    The output ptx path is hardcoded as /tmp/tritoncc.ptx for now.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ttir_path = os.path.join(tmpdir, "input.ttir")
        with open(ttir_path, "w") as f:
            f.write(ttir_code)

        cmd = f"tritoncc {ttir_path}"
        subprocess.check_call(cmd, shell=True)

        # read the ptx code
        with open("/tmp/tritoncc.ptx") as f:
            ptx_code = f.read()

    # replace the ptx version in the ptx code
    ptx_code = re.sub(r"\.version \d+\.\d+", f".version {ptx_version}", ptx_code)
    return ptx_code
