# The script assumes it's run with conda environment activated already.

set -ex

function install_torch() {
  git clone https://github.com/pytorch/pytorch.git
  cd pytorch
  git submodule update --init --recursive

  USE_CUSPARSELT=0 python setup.py develop

  cat > /tmp/a.py <<EOF
import torch
import triton

def f(x):
    return x + 1

N = 2 ** 30
x = torch.randn(N, device="cuda")
ref = f(x)
opt_f = torch.compile(f)

for _ in range(5):
    opt_f(x)

ms = triton.testing.do_bench(lambda: opt_f(x))
nbytes = x.numel() * x.itemsize * 2
tbgs = nbytes * 1e-12 / (ms * 1e-3)
print(f"ms {ms:.3f}, tbgs {tbgs:.3f}")

act = opt_f(x)
torch.testing.assert_close(ref, act)
print("PASS")
EOF
  python /tmp/a.py > /tmp/log
  if ! grep -q "PASS" /tmp/log; then
    echo "torch.compile does not pass the test"
    exit 1
  fi
}

conda install -y python=3.12 pyyaml
# conda install -y gcc_linux-64 gxx_linux-64
# The following one installs glibc. Maybe quite slow
# conda install -c conda-forge sysroot_linux-64 
if ! ( python --version | grep -q '3.12'); then
  echo "incorrect python version"
  exit 1
fi

# install numpy beforing installing pytorch so pytorch will have numpy support
pip install typing_extensions packaging six triton cmake numpy

mkdir -p ~/ws

cd ~/ws

install_torch
