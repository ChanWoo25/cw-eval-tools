# CPP Utils

## Prerequisites

```bash
apt install -y ninja-build

# For Ubuntu 20.04
wget https://apt.llvm.org/llvm.sh &&
chmod +x llvm.sh &&
./llvm.sh 10 &&
apt-get install -y clang clangd
rm llvm.sh

# For running compdb
pip3 install -U pip
pip3 install -U wheel setuptools
pip3 install compdb
```
