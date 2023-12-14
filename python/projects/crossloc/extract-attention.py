#%%
import numpy as np
import os
import sys
from pathlib import Path
import copy
import datetime
from glob import glob

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle as pkl
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange, reduce, repeat

sys.path.append('/root/shared/cw-eval-tools/python')
from module.zip import Zip

def get_current_ts():
    ct = datetime.datetime.now()
    return f"{ct.year}-{ct.month:02d}-{ct.day:02d}_{ct.hour:02d}-{ct.minute:02d}-{ct.second:02d}"

DATASETS_DIR = Path('/data/datasets')
cs_campus_dir = DATASETS_DIR / 'dataset_cs_campus'
results_dir = cs_campus_dir / 'results'
results_dir = results_dir / 'campus_ours' / '2023-12-14_17-28-35'
print(results_dir)
#%%
def process_n_save(unzip_dir:Path,
                   dbg_dict:dict):
    for i in range(3):
        cid = f"raw_{i}_coord"
        id = f"raw_{i}"
        fn = unzip_dir / f"{id}.txt"
        with open(fn, 'w') as f:
            data: torch.Tensor = dbg_dict[cid]
            print(id)
            print(data.shape)
            print(f"Write to {fn} | shape:={data.shape}")
            assert(data.ndim == 2)
            for j in range(data.shape[0]):
                line = f"{data[j,0]} {data[j,1]} {data[j,2]}"
                if j != 0:
                    line = "\n" + line
                f.write(line)

    valid_n_pts = []
    for i in range(3):
        cid = f"conv_{i}_coord"
        fid = f"conv_{i}_feats"
        id  = f"conv_{i}"
        avg_fn = unzip_dir / f"{id}_avg.txt"
        max_fn = unzip_dir / f"{id}_max.txt"

        coord_data: torch.Tensor = dbg_dict[cid]
        feats_data: torch.Tensor = dbg_dict[fid]
        valid_n_pts.append(coord_data.shape[0])

        print(id)
        print(coord_data.shape)
        print(feats_data.shape)

        avg_data = reduce(feats_data, 'n_pts n_feat -> n_pts 1', 'mean')
        max_data = reduce(feats_data, 'n_pts n_feat -> n_pts 1', 'max')
        coord_avg = torch.concat([coord_data, avg_data], dim=1)
        coord_max = torch.concat([coord_data, max_data], dim=1)

        assert(coord_avg.ndim == 2 and coord_avg.shape[1] == 4)
        assert(coord_max.ndim == 2 and coord_max.shape[1] == 4)
        n_pts = coord_avg.shape[0]

        f1 = open(avg_fn, 'w')
        f2 = open(max_fn, 'w')
        print(f"Write to {avg_fn} | shape:={coord_avg.shape}")
        print(f"Write to {max_fn} | shape:={coord_max.shape}")
        for j in range(n_pts):
            l1 = f"{coord_avg[j,0]} {coord_avg[j,1]} {coord_avg[j,2]} {coord_avg[j,3]}"
            l2 = f"{coord_max[j,0]} {coord_max[j,1]} {coord_max[j,2]} {coord_max[j,3]}"
            if j != 0:
                l1 = "\n" + l1
                l2 = "\n" + l2
            f1.write(l1)
            f2.write(l2)
        f1.close()
        f2.close()



    for i in range(3):
        for prefix in ['after_diff', 'after_fuse']:
            cid = f"{prefix}_{i}_coord"
            fid = f"{prefix}_{i}_feats"
            id = f"{prefix}_{2-i}"
            avg_fn = unzip_dir / f"{id}_avg.txt"
            max_fn = unzip_dir / f"{id}_max.txt"

            n_pts = valid_n_pts[2-i]

            coord_data: torch.Tensor = dbg_dict[cid]
            feats_data: torch.Tensor = dbg_dict[fid]
            if prefix != 'conv':
                coord_data = coord_data[:n_pts]
                feats_data = feats_data[:n_pts]
            print(id)
            print(coord_data.shape)
            print(feats_data.shape)

            avg_data = reduce(feats_data, 'n_pts n_feat -> n_pts 1', 'mean')
            max_data = reduce(feats_data, 'n_pts n_feat -> n_pts 1', 'max')
            coord_avg = torch.concat([coord_data, avg_data], dim=1)
            coord_max = torch.concat([coord_data, max_data], dim=1)

            assert(coord_avg.ndim == 2 and coord_avg.shape[1] == 4)
            assert(coord_max.ndim == 2 and coord_max.shape[1] == 4)

            f1 = open(avg_fn, 'w')
            f2 = open(max_fn, 'w')
            print(f"Write to {avg_fn} | shape:={coord_avg.shape}")
            print(f"Write to {max_fn} | shape:={coord_max.shape}")
            for j in range(n_pts):
                l1 = f"{coord_avg[j,0]} {coord_avg[j,1]} {coord_avg[j,2]} {coord_avg[j,3]}"
                l2 = f"{coord_max[j,0]} {coord_max[j,1]} {coord_max[j,2]} {coord_max[j,3]}"
                if j != 0:
                    l1 = "\n" + l1
                    l2 = "\n" + l2
                f1.write(l1)
                f2.write(l2)
            f1.close()
            f2.close()

    print('End')
#%%
zip_fns = glob(str(results_dir/'*.pbz2'))
for zip_fn in zip_fns:
    zip_name = zip_fn.split('/')[-1]
    zip_stem = zip_name.split('.')[0]
    print(f"\n[{zip_stem}]")

    unzip_dir = results_dir / zip_stem
    os.makedirs(unzip_dir, exist_ok=True)

    attn_dict = Zip(Path(zip_fn)).unzip()

    # if attn_dict['query_idx'] != 2767:
    #     continue

    qsize = [0.05, 0.12, 0.4]
    with open(unzip_dir/'meta.txt', 'w') as f:
        words = []
        words.append(attn_dict['filename'])
        words.append(f"{attn_dict['northing']:.6f}")
        words.append(f"{attn_dict['easting']:.6f}")
        f.write(' '.join(words))

    process_n_save(unzip_dir, attn_dict)
    # def extract_other(attn_dict: dict,
    #                   key: str,
    #                   type: str,
    #                   qsize: float):
    #     if type == 'coord':
    #         with open(unzip_dir/f'{key}.txt', 'w') as f:
    #             ts = attn_dict[key]
    #             N = ts.shape[0]
    #             for i in range(N):

    #             words = []
    #             words.append(attn_dict['filename'])
    #             words.append(f"{attn_dict['northing']:.6f}")
    #             words.append(f"{attn_dict['easting']:.6f}")
    #             f.write(' '.join(words))

    # print(f"raw_0_coord.shape: {attn_dict['raw_0_coord'].shape}")
    # print(f"raw_1_coord.shape: {attn_dict['raw_1_coord'].shape}")
    # print(f"raw_2_coord.shape: {attn_dict['raw_2_coord'].shape}")

    # print(f"conv_0_coord.shape: {attn_dict['conv_0_coord'].shape}")
    # attn_dict['conv_0_feats'] = reduce(attn_dict['conv_0_feats'], 'n_pts n_feat -> n_pts 1', 'max')
    # print(f"conv_0_feats.shape: {attn_dict['conv_0_feats'].shape}")
    # conv_0 = torch.concat([attn_dict['conv_0_coord'], attn_dict['conv_0_feats']], dim=1)
    # print(f"conv_0.shape: {conv_0.shape}")

    # print(f"conv_1_coord.shape: {attn_dict['conv_1_coord'].shape}")
    # attn_dict['conv_1_feats'] = reduce(attn_dict['conv_1_feats'], 'n_pts n_feat -> n_pts 1', 'max')
    # print(f"conv_1_feats.shape: {attn_dict['conv_1_feats'].shape}")
    # conv_1 = torch.concat([attn_dict['conv_1_coord'], attn_dict['conv_1_feats']], dim=1)
    # print(f"conv_1.shape: {conv_1.shape}")

    # print(f"conv_2_coord.shape: {attn_dict['conv_2_coord'].shape}")
    # attn_dict['conv_2_feats'] = reduce(attn_dict['conv_2_feats'], 'n_pts n_feat -> n_pts 1', 'max')
    # print(f"conv_2_feats.shape: {attn_dict['conv_2_feats'].shape}")
    # conv_2 = torch.concat([attn_dict['conv_2_coord'], attn_dict['conv_2_feats']], dim=1)
    # print(f"conv_2.shape: {conv_2.shape}")

    # attn_dict['after_diff_2_feats']  = reduce(attn_dict['after_diff_2_feats'] , 'n_pts n_feat -> n_pts 1', 'max')
    # after_diff_0 = torch.concat([attn_dict['after_diff_2_coord'], attn_dict['after_diff_2_feats']], dim=1)
    # after_diff_0 = after_diff_0[:conv_0.shape[0]]
    # print(f"after_diff_0.shape: {after_diff_0.shape}")

    # attn_dict['after_diff_1_feats']  = reduce(attn_dict['after_diff_1_feats'] , 'n_pts n_feat -> n_pts 1', 'max')
    # after_diff_1 = torch.concat([attn_dict['after_diff_1_coord'], attn_dict['after_diff_1_feats']], dim=1)
    # after_diff_1 = after_diff_1[:conv_1.shape[0]]
    # print(f"after_diff_1.shape: {after_diff_1.shape}")

    # attn_dict['after_diff_0_feats']  = reduce(attn_dict['after_diff_0_feats'] , 'n_pts n_feat -> n_pts 1', 'max')
    # after_diff_2 = torch.concat([attn_dict['after_diff_0_coord'], attn_dict['after_diff_0_feats']], dim=1)
    # after_diff_2 = after_diff_2[:conv_2.shape[0]]
    # print(f"after_diff_2.shape: {after_diff_2.shape}")

    # attn_dict['after_fuse_2_feats']  = reduce(attn_dict['after_fuse_2_feats'] , 'n_pts n_feat -> n_pts 1', 'max')
    # after_fuse_0 = torch.concat([attn_dict['after_fuse_2_coord'], attn_dict['after_fuse_2_feats']], dim=1)
    # after_fuse_0 = after_fuse_0[:conv_0.shape[0]]
    # print(f"after_fuse_0.shape: {after_fuse_0.shape}")

    # attn_dict['after_fuse_1_feats']  = reduce(attn_dict['after_fuse_1_feats'] , 'n_pts n_feat -> n_pts 1', 'max')
    # after_fuse_1 = torch.concat([attn_dict['after_fuse_1_coord'], attn_dict['after_fuse_1_feats']], dim=1)
    # after_fuse_1 = after_fuse_1[:conv_1.shape[0]]
    # print(f"after_fuse_1.shape: {after_fuse_1.shape}")

    # attn_dict['after_fuse_0_feats']  = reduce(attn_dict['after_fuse_0_feats'] , 'n_pts n_feat -> n_pts 1', 'max')
    # after_fuse_2 = torch.concat([attn_dict['after_fuse_0_coord'], attn_dict['after_fuse_0_feats']], dim=1)
    # print(f"after_fuse_2.shape: {after_fuse_2.shape}")
    # after_fuse_2 = after_fuse_2[:conv_2.shape[0]]

    # print(f"after_diff_2_coord.shape: {attn_dict['after_diff_2_coord'].shape}")
    # print(f"after_diff_2_feats.shape: {attn_dict['after_diff_2_feats'].shape}")
    # print(f"after_diff_1_coord.shape: {attn_dict['after_diff_1_coord'].shape}")
    # print(f"after_diff_1_feats.shape: {attn_dict['after_diff_1_feats'].shape}")
    # print(f"after_diff_0_coord.shape: {attn_dict['after_diff_0_coord'].shape}")
    # print(f"after_diff_0_feats.shape: {attn_dict['after_diff_0_feats'].shape}")
    # print(f"after_fuse_2_coord.shape: {attn_dict['after_fuse_2_coord'].shape}")
    # print(f"after_fuse_2_feats.shape: {attn_dict['after_fuse_2_feats'].shape}")
    # print(f"after_fuse_1_coord.shape: {attn_dict['after_fuse_1_coord'].shape}")
    # print(f"after_fuse_1_feats.shape: {attn_dict['after_fuse_1_feats'].shape}")
    # print(f"after_fuse_0_coord.shape: {attn_dict['after_fuse_0_coord'].shape}")
    # print(f"after_fuse_0_feats.shape: {attn_dict['after_fuse_0_feats'].shape}")

    # print(after_diff_0)
    # print(after_diff_1)
    # print(after_diff_2)
    # a = 10
        # ordering: fine => coarse
        # 'raw_coords_0',
        # 'raw_coords_1',
        # 'raw_coords_2',
        # 'conv_coords_0',
        # 'conv_feats_0',
        # 'conv_coords_1',
        # 'conv_feats_1',
        # 'conv_coords_2',
        # 'conv_feats_2',
        # 'after_diff_coords_2',
        # 'after_diff_feats_2',
        # 'after_diff_coords_1',
        # 'after_diff_feats_1',
        # 'after_diff_coords_0',
        # 'after_diff_feats_0',
        # 'after_fuse_coords_2',
        # 'after_fuse_feats_2'
        # 'after_fuse_coords_1',
        # 'after_fuse_feats_1',
        # 'after_fuse_coords_0',
        # 'after_fuse_feats_0',

# infer_zip_fn = cs_campus_dir / 'infer_dict.pbz2'
# infer_dict = Zip(infer_zip_fn).unzip()

# %%
