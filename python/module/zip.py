import os
from pathlib import Path
import zipfile
import gzip
from enum import Enum

class ZipType(Enum):
    ZIP=1
    TAR=2
    TAR_GZ=3

# TODO_LCW : Add .tar .tar.gz support
class Zip:
    def __init__(self, zip_fn: Path) -> None:
        self.ok = False
        self.zip_fn: Path = zip_fn

        if not zip_fn.exists():
            print(f"[Error] {zip_fn} doesn't exist.")
            return

        if len(zip_fn.suffixes) == 1:
            if zip_fn.suffix == '.zip':
                self.ok = True
                self.zip_type = ZipType.ZIP
                return
            elif zip_fn.suffix == '.tar':
                self.ok = True
                self.zip_type = ZipType.TAR
                return
        elif len(zip_fn.suffixes) == 2:
            if (zip_fn.suffixes[0] == '.tar' and \
                zip_fn.suffixes[1] == '.gz'):
                self.ok = True
                self.zip_type = ZipType.TAR_GZ
                return

        print(f"[Error] {zip_fn} : Unknown zip type.")
        return

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
        else:
            print("[Error] Not implemented.")

        return
