#! /bin/bash

cd ~
git clone --single-branch -b v3.0 \
  https://github.com/p-ranav/argparse
cd argparse

# Build the tests
mkdir build
cd build
cmake -DARGPARSE_BUILD_SAMPLES=on -DARGPARSE_BUILD_TESTS=on ..
make

# Run tests
./test/tests

# Install the library
make install
cd ~ && rm -rf argparse
