import os
from pathlib import Path
import zipfile
import gzip
from enum import Enum
import bz2
import pickle
import _pickle as cPickle

# # Pickle a file and then compress it into a file with extension
# def compressed_pickle(title, data):
#     with bz2.BZ2File(title + '.pbz2', 'w') as f:
#         cPickle.dump(data, f)

# # Load any compressed pickle file
# def decompress_pickle(file):
#     data = bz2.BZ2File(file, 'rb')
#     data = cPickle.load(data)
#     return data

class ZipType(Enum):
    ZIP=1
    TAR=2
    TAR_GZ=3
    PBZ2=4

# TODO_LCW : Add .tar .tar.gz support
class Zip:
    def __init__(self, zip_fn: Path) -> None:
        self.ok = False
        self.zip_fn: Path = zip_fn

        if not zip_fn.exists():
            print(f"[Error] {zip_fn} doesn't exist.")
            exit(1)

        if len(zip_fn.suffixes) == 1:
            if zip_fn.suffix == '.zip':
                self.ok = True
                self.zip_type = ZipType.ZIP
            elif zip_fn.suffix == '.tar':
                self.ok = True
                self.zip_type = ZipType.TAR
            elif zip_fn.suffix == '.pbz2':
                self.ok = True
                self.zip_type = ZipType.PBZ2
        elif len(zip_fn.suffixes) == 2:
            if (zip_fn.suffixes[0] == '.tar' and \
                zip_fn.suffixes[1] == '.gz'):
                self.ok = True
                self.zip_type = ZipType.TAR_GZ

        if not self.ok:
            print(f"[Error] {zip_fn} : Unknown zip type.")
            exit(1)

    def exists(self):
        return self.ok

    def unzip(self, dst_dir: Path=None):
        """Unzip into dst_dir or working directory"""
        if dst_dir is not None:
            if dst_dir.exists():
                print(f"[Error] {dst_dir} already exists.",
                       "Please check path one more time.")
                return
            else:
                os.makedirs(dst_dir)
        else:
            dst_dir = self.zip_fn.parent

        if self.zip_type == ZipType.ZIP:
            print(f"Extract {self.zip_fn.name} into {dst_dir} ... ")
            with zipfile.ZipFile(self.zip_fn, 'r') as f_zip:
                # Extract all contents into the directory
                f_zip.extractall(dst_dir)
            print("[Ok]")
        elif self.zip_type == ZipType.PBZ2:
            with bz2.BZ2File(self.zip_fn, 'rb') as f:
                data = cPickle.load(f)
            return data
        else:
            print("[Error] Not implemented.")
            exit(1)
