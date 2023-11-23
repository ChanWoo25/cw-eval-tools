import os
from glob import glob
from pathlib import Path
from module.zip import Zip

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--root", type=str)
args = parser.parse_args()

def main():
    ROOT = Path(args.root)
    euroc_zip_paths = glob(str(ROOT / '*.zip'))
    for zip_fn in euroc_zip_paths:
        # Only Extract Indoor datasets
        if 'V1' in zip_fn or 'V2' in zip_fn:
            zip_fn = Path(zip_fn)
            dst_dir = zip_fn.parent / zip_fn.stem
            fzip = Zip(zip_fn)
            if fzip.exists():
                fzip.unzip(dst_dir=dst_dir)

if __name__ == '__main__':
    main()
