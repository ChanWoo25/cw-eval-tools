# Pre-processor for Oxford RobotCar, Inhouse and CS Campus Dataset[@CrossLoc3D]
# Extended from PointNetVLAD Repo: https://github.com/mikacuy/pointnetvlad
# Author: Chanwoo Lee (leechanwoo25@gmail.com)
#%%
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

import pickle
from tqdm import tqdm

DATASETS_DIR = Path('/data/datasets')
CS_CAMPUS_DIR = DATASETS_DIR / 'dataset_cs_campus'
BENCHMARK_DIR = CS_CAMPUS_DIR / 'benchmark_datasets'
UMD_DIR = BENCHMARK_DIR / 'umd'
assert (DATASETS_DIR.exists())
assert (CS_CAMPUS_DIR.exists())
assert (BENCHMARK_DIR.exists())

#%% Functions
def readMetas(seq_dir: Path, skip_header=1):
    list_fn = seq_dir / 'umd_aerial_cloud_20m_100coverage_4096_new.csv'
    paths = []
    xys = []
    cnt = 0
    with open(list_fn, 'r') as f:
        lines = f.readlines()
        lines = lines[skip_header:]
        for line in lines:
            if line.strip() == "":
                break
            cnt += 1
            words = line.strip().split(',')
            paths.append(words[0])
            xys.append([float(words[1]), float(words[2])])
    xys = np.array(xys, dtype=float)
    # print(f"[DIR] {seq_dir.name}: {xys.shape[0]}s")
    return paths, xys

def construct_training_tuples_new(aerial_training_paths: list,
                                  aerial_training_xys: np.ndarray,
                                  ground_training_paths: list,
                                  ground_training_xys: np.ndarray,
                                  save_pkl_fn: Path,
                                  save_txt_fn: Path,
                                  aerial_positive_range,
                                  aerial_negative_range,
                                  ground_positive_range,
                                  ground_negative_range):
    """ Construct Training Tuples

    Args:
        training_paths (list): N-dim paths
        training_xys (np.ndarray): [N x 2] shape ndarray [northing, easting]
        save_fn (Path): save filename
        ind_nn_r (_type_): _description_
        ind_r_r (int, optional): _description_. Defaults to 50.
        Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
        Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    """
    aerial_tree = KDTree(aerial_training_xys)
    ground_tree = KDTree(ground_training_xys)

    nn_idxs_aer_from_aer = aerial_tree.query_radius(aerial_training_xys, r=aerial_positive_range)
    nn_idxs_aer_from_gro = ground_tree.query_radius(aerial_training_xys, r=aerial_positive_range)
    nn_idxs_gro_from_aer = aerial_tree.query_radius(ground_training_xys, r=aerial_positive_range)
    nn_idxs_gro_from_gro = ground_tree.query_radius(ground_training_xys, r=ground_positive_range)

    rr_idxs_aer_from_aer = aerial_tree.query_radius(aerial_training_xys, r=aerial_negative_range)
    rr_idxs_aer_from_gro = ground_tree.query_radius(aerial_training_xys, r=aerial_negative_range)
    rr_idxs_gro_from_aer = aerial_tree.query_radius(ground_training_xys, r=aerial_negative_range)
    rr_idxs_gro_from_gro = ground_tree.query_radius(ground_training_xys, r=ground_negative_range)

    # print(type(nn_idxs_aer_from_aer), nn_idxs_aer_from_aer.shape)
    # print(type(rr_idxs_aer_from_aer), rr_idxs_aer_from_aer.shape)
    # print(type(nn_idxs_aer_from_aer[0]), nn_idxs_aer_from_aer[0].shape)
    # print(type(nn_idxs_aer_from_aer[10]), nn_idxs_aer_from_aer[10].shape)
    # # <class 'numpy.ndarray'> (6268,)
    # # <class 'numpy.ndarray'> (6268,)
    # # <class 'numpy.ndarray'> (30,)
    # # <class 'numpy.ndarray'> (36,)

    catalog = {}
    N_AERIAL = aerial_training_xys.shape[0]
    N_GROUND = ground_training_xys.shape[0]

    print(f"Save test file | '{save_txt_fn}'")
    with open(save_txt_fn, 'w') as f:
        for i in range(N_AERIAL + N_GROUND):
            if i < N_AERIAL:
                line = f"{aerial_training_paths[i]} {aerial_training_xys[i,0]:.9f} {aerial_training_xys[i,1]:.9f}"
            else:
                idx = i - N_AERIAL
                line = f"{ground_training_paths[idx]} {ground_training_xys[idx,0]:.9f} {ground_training_xys[idx,1]:.9f}"
            if i != 0:
                line = "\n" + line
            f.write(line)

    print("Create training catalog with aerial ...")
    for i in tqdm(range(N_AERIAL)):
        idx_dict = {}
        idx_dict['query']    = aerial_training_paths[i]
        idx_dict['northing'] = aerial_training_xys[i,0]
        idx_dict['easting']  = aerial_training_xys[i,1]

        positives = np.concatenate([nn_idxs_aer_from_aer[i],
                                    (nn_idxs_aer_from_gro[i]+N_AERIAL)],
                                   axis=0)
        print(f"positives: {positives.shape[0]}")
        print(nn_idxs_aer_from_gro[i]+N_AERIAL)
        positives = positives[positives != i].tolist()
        positives = sorted(positives)

        all_indexs = np.linspace(0,
                                 N_AERIAL+N_GROUND,
                                 N_AERIAL+N_GROUND,
                                 dtype=int, endpoint=False).tolist()
        non_negatives = np.concatenate([rr_idxs_aer_from_aer[i],
                                        (rr_idxs_aer_from_gro[i]+N_AERIAL)],
                                       axis=0).tolist()
        negatives = sorted(list(set(all_indexs) - set(non_negatives)))

        idx_dict['positives'] = positives
        idx_dict['negatives'] = negatives
        catalog[i] = idx_dict

    print("Create training catalog with ground ...")
    for i in tqdm(range(N_GROUND)):
        idx = i + N_AERIAL
        idx_dict = {}
        idx_dict['query']    = ground_training_paths[i]
        idx_dict['northing'] = ground_training_xys[i,0]
        idx_dict['easting']  = ground_training_xys[i,1]

        positives = np.concatenate([nn_idxs_gro_from_aer[i],
                                    (nn_idxs_gro_from_gro[i]+N_AERIAL)],
                                   axis=0)
        print(f"positives: {positives.shape[0]}")
        print(nn_idxs_gro_from_gro[i]+N_AERIAL)
        positives = positives[positives != idx].tolist()
        positives = sorted(positives)

        all_indexs = np.linspace(0,
                                 N_AERIAL+N_GROUND,
                                 N_AERIAL+N_GROUND,
                                 dtype=int, endpoint=False).tolist()
        non_negatives = np.concatenate([rr_idxs_gro_from_aer[i],
                                        (rr_idxs_gro_from_gro[i]+N_AERIAL)],
                                       axis=0).tolist()
        negatives = sorted(list(set(all_indexs) - set(non_negatives)))

        idx_dict['positives'] = positives
        idx_dict['negatives'] = negatives
        catalog[idx] = idx_dict

    print(f"Save catalog into pickle file | '{save_pkl_fn}'")
    with open(save_pkl_fn, 'wb') as handle:
        pickle.dump(catalog, handle, protocol=pickle.HIGHEST_PROTOCOL)


def construct_evaluation_dbase_n_query(db_aerial_set: list,
                                       db_ground_set: list,
                                       test_set: list,
                                       save_eval_db_prefix: Path,
                                       save_eval_query_pkl_fn: Path,
                                       save_eval_query_txt_fn: Path,
                                       aerial_positive_range,
                                       ground_positive_range):
    ############
    # Database #
    ############
    database_catalog = []
    di = 0
    for seq_name in db_aerial_set:
        dbase_paths, dbase_xys = readMetas(UMD_DIR / seq_name)
        dbase_dict = {}
        for i in range(len(dbase_paths)):
            dbase_dict[i] = {
                'query'   : dbase_paths[i],
                'northing': dbase_xys[i,0],
                'easting' : dbase_xys[i,1]
            }
        dbase_txt_fn = str(save_eval_db_prefix) + f"_{di:02d}.txt"
        print(f"[DBase{di:02d}] ({len(dbase_dict):5d}s) Save txt in '{dbase_txt_fn}'")
        with open(dbase_txt_fn, 'w') as f:
            for i in range(len(dbase_paths)):
                line = f"{dbase_paths[i]} {dbase_xys[i,0]:.9f} {dbase_xys[i,1]:.9f}"
                if i != 0:
                    line = "\n" + line
                f.write(line)
        database_catalog.append(dbase_dict)
        di += 1

    for seq_name in db_ground_set:
        dbase_paths, dbase_xys = readMetas(UMD_DIR / seq_name)
        dbase_dict = {}
        for i in range(len(dbase_paths)):
            dbase_dict[i] = {
                'query'   : dbase_paths[i],
                'northing': dbase_xys[i,0],
                'easting' : dbase_xys[i,1]
            }
        dbase_txt_fn = str(save_eval_db_prefix) + f"_{di:02d}.txt"
        print(f"[DBase{di:02d}] ({len(dbase_dict):5d}s) Save txt in '{dbase_txt_fn}'")
        with open(dbase_txt_fn, 'w') as f:
            for i in range(len(dbase_paths)):
                line = f"{dbase_paths[i]} {dbase_xys[i,0]:.9f} {dbase_xys[i,1]:.9f}"
                if i != 0:
                    line = "\n" + line
                f.write(line)
        database_catalog.append(dbase_dict)
        di += 1

    dbase_pkl_fn = str(save_eval_db_prefix) + f".pickle"
    print(f"[DBase] Save catalog in '{dbase_pkl_fn}'")
    with open(dbase_pkl_fn, 'wb') as handle:
        pickle.dump(database_catalog, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###########
    # Queries #
    ###########
    queries_paths = []
    queries_xys = []
    for seq_name in test_set:
        tmp_ground_paths, tmp_ground_xys= readMetas(UMD_DIR / seq_name)
        for j in range(len(tmp_ground_paths)):
            queries_paths.append(tmp_ground_paths[j])
        queries_xys.append(tmp_ground_xys)
    queries_xys = np.concatenate(queries_xys, axis=0)

    print(f"[Query] ({len(queries_paths):4d}s) Save txt in '{save_eval_query_txt_fn}'")
    with open(save_eval_query_txt_fn, 'w') as f:
        for i in range(len(queries_paths)):
            line = f"{queries_paths[i]} {queries_xys[i,0]:.9f} {queries_xys[i,1]:.9f}"
            if i != 0:
                line = "\n" + line
            f.write(line)

    dbase_trees = []
    for seq_name in db_aerial_set:
        dbase_paths, dbase_xys = readMetas(UMD_DIR / seq_name)
        tree = KDTree(dbase_xys)
        dbase_trees.append(tree)
    for seq_name in db_ground_set:
        dbase_paths, dbase_xys = readMetas(UMD_DIR / seq_name)
        tree = KDTree(dbase_xys)
        dbase_trees.append(tree)
    n_aerial = len(db_aerial_set)
    n_ground = len(db_ground_set)
    n_dbase = n_aerial + n_ground
    assert n_dbase == len(dbase_trees)

    queries_catalog = []
    queries_list = []
    n_matches = 0
    for qi in tqdm(range(len(queries_paths))):
        query_path = queries_paths[qi]
        query_xy   = queries_xys[qi].reshape((1,2))
        query_dict = {}
        query_dict['query'] = query_path
        query_dict['northing'] = queries_xys[0,0]
        query_dict['easting']  = queries_xys[0,1]

        # Aerial
        for i in range(n_aerial):
            nn_idxs = dbase_trees[i].query_radius(query_xy, r=aerial_positive_range)
            query_dict[i] = nn_idxs[0].tolist()
            n_matches += len(query_dict[i])
        # Ground
        for i in range(n_aerial, n_dbase):
            nn_idxs = dbase_trees[i].query_radius(query_xy, r=ground_positive_range)
            query_dict[i] = nn_idxs[0].tolist()
            n_matches += len(query_dict[i])

        queries_list.append(query_dict)
    queries_catalog.append(queries_list)

    print(f"[Query] Save catalog pickle in '{save_eval_query_pkl_fn}' | size: {len(queries_list)} | matches: ({n_matches})")
    with open(save_eval_query_pkl_fn, 'wb') as handle:
        pickle.dump(queries_catalog, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%% Main - training

def main(train_both:bool, scale:float=1.0):
    aerial_00 = f"umcp_aerial_resized{int(scale*100.0):03d}_00"
    aerial_01 = f"umcp_aerial_resized{int(scale*100.0):03d}_01"
    # print(f"aerial_00: {aerial_00}")
    # print(f"aerial_01: {aerial_01}")

    TRAIN_SET = []
    if train_both:
        TRAIN_SET.append(aerial_00)
        TRAIN_SET.append(aerial_01)
    else:
        TRAIN_SET.append(aerial_00)
    TRAIN_SET.append('umcp_lidar5_ground_umd_gps_1')
    TRAIN_SET.append('umcp_lidar5_ground_umd_gps_3')
    TRAIN_SET.append('umcp_lidar5_ground_umd_gps_4')
    TRAIN_SET.append('umcp_lidar5_ground_umd_gps_5')
    TRAIN_SET.append('umcp_lidar5_ground_umd_gps_6')
    TRAIN_SET.append('umcp_lidar5_ground_umd_gps_8')
    TRAIN_SET.append('umcp_lidar5_ground_umd_gps_9')
    TRAIN_SET.append('umcp_lidar5_ground_umd_gps_10')
    TRAIN_SET.append('umcp_lidar5_ground_umd_gps_12')

    TEST_SET = []
    TEST_SET.append('umcp_lidar5_ground_umd_gps_2')
    TEST_SET.append('umcp_lidar5_ground_umd_gps_7')
    TEST_SET.append('umcp_lidar5_ground_umd_gps_11')
    TEST_SET.append('umcp_lidar5_ground_umd_gps_15')
    TEST_SET.append('umcp_lidar5_ground_umd_gps_16')
    TEST_SET.append('umcp_lidar5_ground_umd_gps_17')
    TEST_SET.append('umcp_lidar5_ground_umd_gps_18')
    TEST_SET.append('umcp_lidar5_ground_umd_gps_19')

    DB_AERIAL_SET = [aerial_00, aerial_01]
    DB_GROUND_SET = []
    DB_GROUND_SET.append('umcp_lidar5_ground_umd_gps_1')
    DB_GROUND_SET.append('umcp_lidar5_ground_umd_gps_3')
    DB_GROUND_SET.append('umcp_lidar5_ground_umd_gps_4')
    DB_GROUND_SET.append('umcp_lidar5_ground_umd_gps_5')
    DB_GROUND_SET.append('umcp_lidar5_ground_umd_gps_6')
    DB_GROUND_SET.append('umcp_lidar5_ground_umd_gps_8')
    DB_GROUND_SET.append('umcp_lidar5_ground_umd_gps_9')
    DB_GROUND_SET.append('umcp_lidar5_ground_umd_gps_10')
    DB_GROUND_SET.append('umcp_lidar5_ground_umd_gps_12')
    # # For Debugging
    # print("[Training]")
    # for e in TRAIN_SET:
    #     print(e)
    # print("\n[Test]")
    # for e in TEST_SET:
    #     print(e)

    aerial_paths = None
    aerial_xys = None
    if train_both:
        aerial_paths = []
        aerial_xys = []
        aerial_paths_00, aerial_xys_00 = readMetas(UMD_DIR / TRAIN_SET[0])
        aerial_paths_01, aerial_xys_01 = readMetas(UMD_DIR / TRAIN_SET[1])
        for path in aerial_paths_00:
            aerial_paths.append(path)
        for path in aerial_paths_01:
            aerial_paths.append(path)
        aerial_xys.append(aerial_xys_00)
        aerial_xys.append(aerial_xys_01)
        aerial_xys = np.concatenate(aerial_xys, axis=0)
    else:
        aerial_paths, aerial_xys = readMetas(UMD_DIR / TRAIN_SET[0])
        aerial_paths_01, aerial_xys_01 = readMetas(UMD_DIR / TRAIN_SET[1])

    # # For Debugging
    # print(f"aerial_paths length: {len(aerial_paths)}")
    # print(f"aerial_xys.shape: {aerial_xys.shape}")

    ground_paths = []
    ground_xys = []
    ground_set = TRAIN_SET[2:] if train_both else TRAIN_SET[1:]
    for seq_name in ground_set:
        tmp_ground_paths, tmp_ground_xys= readMetas(UMD_DIR / seq_name)
        for j in range(len(tmp_ground_paths)):
            ground_paths.append(tmp_ground_paths[j])
        ground_xys.append(tmp_ground_xys)
    ground_xys = np.concatenate(ground_xys, axis=0)

    # # For Debugging
    # print(f"ground_paths length: {len(ground_paths)}")
    # print(f"ground_xys.shape: ground_xys.shape}")

    save_pkl_fn = None
    save_txt_fn = None
    save_eval_db_prefix = ""
    save_eval_query_pkl_fn = ""
    save_eval_query_txt_fn = ""
    if train_both:
        save_pkl_fn = BENCHMARK_DIR / f"training_both_scale{int(scale*100.0):03d}.pickle"
        save_txt_fn = BENCHMARK_DIR / f"training_both_scale{int(scale*100.0):03d}.txt"
        save_eval_db_prefix    = BENCHMARK_DIR / f"evaluation_dbase_both_scale{int(scale*100.0):03d}"
        save_eval_query_pkl_fn = BENCHMARK_DIR / f"evaluation_query_both_scale{int(scale*100.0):03d}.pickle"
        save_eval_query_txt_fn = BENCHMARK_DIR / f"evaluation_query_both_scale{int(scale*100.0):03d}.txt"
    else:
        save_pkl_fn = BENCHMARK_DIR / f"training_solo_scale{int(scale*100.0):03d}.pickle"
        save_txt_fn = BENCHMARK_DIR / f"training_solo_scale{int(scale*100.0):03d}.txt"
        save_eval_db_prefix    = BENCHMARK_DIR / f"evaluation_dbase_solo_scale{int(scale*100.0):03d}"
        save_eval_query_pkl_fn = BENCHMARK_DIR / f"evaluation_query_solo_scale{int(scale*100.0):03d}.pickle"
        save_eval_query_txt_fn = BENCHMARK_DIR / f"evaluation_query_solo_scale{int(scale*100.0):03d}.txt"

    iscale = int(scale*100.0)
    if iscale == 100:
        aprange =  80.0
        anrange = 120.0
        gprange =  20.0
        gnrange =  50.0
        construct_training_tuples_new(aerial_paths, aerial_xys,
                                      ground_paths, ground_xys,
                                      save_pkl_fn,  save_txt_fn,
                                      aerial_positive_range=aprange,
                                      aerial_negative_range=anrange,
                                      ground_positive_range=gprange,
                                      ground_negative_range=gnrange)
        construct_evaluation_dbase_n_query(DB_AERIAL_SET, DB_GROUND_SET, TEST_SET,
                                           save_eval_db_prefix,
                                           save_eval_query_pkl_fn,
                                           save_eval_query_txt_fn,
                                           aerial_positive_range=aprange,
                                           ground_positive_range=gprange)
    elif iscale == 80:
        aprange =  70.0
        anrange = 100.0
        gprange =  20.0
        gnrange =  50.0
        construct_training_tuples_new(aerial_paths, aerial_xys,
                                      ground_paths, ground_xys,
                                      save_pkl_fn,  save_txt_fn,
                                      aerial_positive_range=aprange,
                                      aerial_negative_range=anrange,
                                      ground_positive_range=gprange,
                                      ground_negative_range=gnrange)
        construct_evaluation_dbase_n_query(DB_AERIAL_SET, DB_GROUND_SET, TEST_SET,
                                           save_eval_db_prefix,
                                           save_eval_query_pkl_fn,
                                           save_eval_query_txt_fn,
                                           aerial_positive_range=aprange,
                                           ground_positive_range=gprange)
    elif iscale == 70:
        aprange = 50.0
        anrange = 80.0
        gprange = 20.0
        gnrange = 50.0
        construct_training_tuples_new(aerial_paths, aerial_xys,
                                      ground_paths, ground_xys,
                                      save_pkl_fn,  save_txt_fn,
                                      aerial_positive_range=aprange,
                                      aerial_negative_range=anrange,
                                      ground_positive_range=gprange,
                                      ground_negative_range=gnrange)
        construct_evaluation_dbase_n_query(DB_AERIAL_SET, DB_GROUND_SET, TEST_SET,
                                           save_eval_db_prefix,
                                           save_eval_query_pkl_fn,
                                           save_eval_query_txt_fn,
                                           aerial_positive_range=aprange,
                                           ground_positive_range=gprange)
    elif iscale == 60:
        aprange = 40.0
        anrange = 70.0
        gprange = 20.0
        gnrange = 50.0
        construct_training_tuples_new(aerial_paths, aerial_xys,
                                      ground_paths, ground_xys,
                                      save_pkl_fn,  save_txt_fn,
                                      aerial_positive_range=aprange,
                                      aerial_negative_range=anrange,
                                      ground_positive_range=gprange,
                                      ground_negative_range=gnrange)
        construct_evaluation_dbase_n_query(DB_AERIAL_SET, DB_GROUND_SET, TEST_SET,
                                           save_eval_db_prefix,
                                           save_eval_query_pkl_fn,
                                           save_eval_query_txt_fn,
                                           aerial_positive_range=aprange,
                                           ground_positive_range=gprange)
    else:
        print(f"Unvalid iscale: {iscale}")


#%%
# main(train_both=True,
#      scale=0.6)
main(train_both=False,
     scale=0.7)
# main(train_both=True,
#      scale=0.7)
# main(train_both=True,
#      scale=0.8)
# main(train_both=True,
#      scale=1.0)

#%%

# exit(0)

training_catalog_fn = CATALOG_V2_DIR / 'training_catalog_ver2.txt'
training_paths, training_xys = load_meta(training_catalog_fn)
aerial_training_paths = training_paths[:6268]
aerial_training_xys   = training_xys[:6268]
ground_training_paths = training_paths[6268:]
ground_training_xys   = training_xys[6268:]
print(f"aerial data: {len(aerial_training_paths)}")
print(f"ground data: {len(ground_training_paths)}")





# from datasets.base_datasets import TrainingTuple
# # Import test set boundaries
# from datasets.pointnetvlad.generate_test_sets import P1, P2, P3, P4, check_in_test_set
def check_in_test_set(northing, easting, points):
    in_test_set = False
    for point in points:
        if point[0] - X_WIDTH < northing < point[0] + X_WIDTH and point[1] - Y_WIDTH < easting < point[1] + Y_WIDTH:
            in_test_set = True
            break
    return in_test_set

#%%
DATASET_DIRS = {
    'oxford': '/data/datasets/dataset_oxford_inhouse/benchmark_datasets/oxford',
    'inhouse': '/data/datasets/dataset_oxford_inhouse/benchmark_datasets/inhouse_datasets',
    'cs_campus': '/data/datasets/dataset_cs_campus/benchmark_datasets/umd'
}

DATASET_META_INFOS = {
    'oxford': {
        'test_sectors': [
            [5735712.768124, 620084.402381],
            [5735611.299219, 620540.270327],
            [5735237.358209, 620543.094379],
            [5734749.303802, 619932.693364]],
        'width': 150.0,
        'height': 150.0
    },
    'inhouse': {
        'test_sectors': [
            # For University Sector
            [363621.292362, 142864.19756],
            [364788.795462, 143125.746609],
            [363597.507711, 144011.414174],
            # For Residential Area
            [360895.486453, 144999.915143],
            [362357.024536, 144894.825301],
            [361368.907155, 145209.663042]],
        'width': 150.0,
        'height': 150.0
    },
    'cs_campus_aerial': {
        'sequences': [
            'umcp_lidar5_cloud_6',
            'umcp_lidar5_cloud_7'
        ],
        'valid_area': {
            'xlim': [4316500.0, 4318500.0],
            'ylim': [331400.0 , 332600.0 ]
        },
        'test_sectors': [
            [4317250.0, 331950.0],
            [4317290.0, 332130.0],
            [4317390.0, 332260.0],
            [4317470.0, 331930.0],
            [4317480.0, 332100.0],
            [4317520.0, 332210.0]],
        'width': 50.0,
        'height': 50.0
    },
    'cs_campus_ground': {
        'sequences': [
            'umcp_lidar5_ground_umd_gps_1',
            'umcp_lidar5_ground_umd_gps_2',
            'umcp_lidar5_ground_umd_gps_3',
            'umcp_lidar5_ground_umd_gps_4',
            'umcp_lidar5_ground_umd_gps_5',
            'umcp_lidar5_ground_umd_gps_6',
            'umcp_lidar5_ground_umd_gps_7',
            'umcp_lidar5_ground_umd_gps_8',
            'umcp_lidar5_ground_umd_gps_9',
            'umcp_lidar5_ground_umd_gps_10',
            'umcp_lidar5_ground_umd_gps_11',
            'umcp_lidar5_ground_umd_gps_12',
            'umcp_lidar5_ground_umd_gps_13',
            'umcp_lidar5_ground_umd_gps_14',
            'umcp_lidar5_ground_umd_gps_15',
            'umcp_lidar5_ground_umd_gps_16',
            'umcp_lidar5_ground_umd_gps_17',
            'umcp_lidar5_ground_umd_gps_18',
            'umcp_lidar5_ground_umd_gps_19'
        ],
        'test_sectors': [
            [4317250.0, 331950.0],
            [4317290.0, 332130.0],
            [4317390.0, 332260.0],
            [4317470.0, 331930.0],
            [4317480.0, 332100.0],
            [4317520.0, 332210.0]],
        'width': 40.0,
        'height': 40.0
    }
}

# For training and test data splits
X_WIDTH = 150
Y_WIDTH = 150

# For Oxford
P1 = [5735712.768124, 620084.402381]
P2 = [5735611.299219, 620540.270327]
P3 = [5735237.358209, 620543.094379]
P4 = [5734749.303802, 619932.693364]

# For University Sector
P5 = [363621.292362, 142864.19756]
P6 = [364788.795462, 143125.746609]
P7 = [363597.507711, 144011.414174]

# For Residential Area
P8 = [360895.486453, 144999.915143]
P9 = [362357.024536, 144894.825301]
P10 = [361368.907155, 145209.663042]

P_DICT = {"oxford": [P1, P2, P3, P4], "university": [P5, P6, P7], "residential": [P8, P9, P10], "business": []}



# Test set boundaries
P = [P1, P2, P3, P4]

# RUNS_FOLDER = "oxford/"
# FILENAME = "pointcloud_locations_20m_10overlap.csv"
# POINTCLOUD_FOLS = "/pointcloud_20m_10overlap/"
#%%
class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position

def construct_query_dict(df_centroids, base_path, filename, ind_nn_r, ind_r_r=50):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    tree = KDTree(df_centroids[['northing', 'easting']])
    idxs_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)
    queries = {}

    for idx_nn in range(len(idxs_nn)):
        query = df_centroids.iloc[idx_nn]["file"]

        anchor_pos = np.array(df_centroids.iloc[idx_nn][['northing', 'easting']])
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.bin', f"Expected .bin file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])

        positives = idxs_nn[idx_nn]
        non_negatives = ind_r[idx_nn]

        positives = positives[positives != idx_nn]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[idx_nn] \
          = TrainingTuple(id=idx_nn,
                          timestamp=timestamp,
                          rel_scan_filepath=query,
                          positives=positives,
                          non_negatives=non_negatives,
                          position=anchor_pos)

    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

#%% Main
#%%
def construct_query_dict_ver2(aerial_training_paths: list,
                              aerial_training_xys: np.ndarray,
                              ground_training_paths: list,
                              ground_training_xys: np.ndarray,
                              aerial_positive_range=60.0,
                              aerial_negative_range=120.0,
                              ground_positive_range=20.0,
                              ground_negative_range=50.0):
    """ Construct Training Tuples

    Args:
        training_paths (list): N-dim paths
        training_xys (np.ndarray): [N x 2] shape ndarray [northing, easting]
        save_fn (Path): save filename
        ind_nn_r (_type_): _description_
        ind_r_r (int, optional): _description_. Defaults to 50.
        Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
        Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    """
    aerial_tree = KDTree(aerial_training_xys)
    ground_tree = KDTree(ground_training_xys)

    nn_idxs_aer_from_aer = aerial_tree.query_radius(aerial_training_xys, r=aerial_positive_range)
    nn_idxs_aer_from_gro = ground_tree.query_radius(aerial_training_xys, r=aerial_positive_range)
    nn_idxs_gro_from_aer = aerial_tree.query_radius(ground_training_xys, r=aerial_positive_range)
    nn_idxs_gro_from_gro = ground_tree.query_radius(ground_training_xys, r=ground_positive_range)

    rr_idxs_aer_from_aer = aerial_tree.query_radius(aerial_training_xys, r=aerial_negative_range)
    rr_idxs_aer_from_gro = ground_tree.query_radius(aerial_training_xys, r=aerial_negative_range)
    rr_idxs_gro_from_aer = aerial_tree.query_radius(ground_training_xys, r=aerial_negative_range)
    rr_idxs_gro_from_gro = ground_tree.query_radius(ground_training_xys, r=ground_negative_range)

    # print(type(nn_idxs_aer_from_aer), nn_idxs_aer_from_aer.shape)
    # print(type(rr_idxs_aer_from_aer), rr_idxs_aer_from_aer.shape)
    # print(type(nn_idxs_aer_from_aer[0]), nn_idxs_aer_from_aer[0].shape)
    # print(type(nn_idxs_aer_from_aer[10]), nn_idxs_aer_from_aer[10].shape)
    # # <class 'numpy.ndarray'> (6268,)
    # # <class 'numpy.ndarray'> (6268,)
    # # <class 'numpy.ndarray'> (30,)
    # # <class 'numpy.ndarray'> (36,)

    catalog = {}
    N_AERIAL = aerial_training_xys.shape[0]
    N_GROUND = ground_training_xys.shape[0]

    for i in range(N_AERIAL):
        idx = i
        idx_dict = {}
        idx_dict['query']    = aerial_training_paths[i]
        idx_dict['northing'] = aerial_training_xys[i,0]
        idx_dict['easting']  = aerial_training_xys[i,1]

        positives = np.concatenate([nn_idxs_aer_from_aer[i],
                                    (nn_idxs_aer_from_gro[i]+N_AERIAL)],
                                   axis=0)
        positives = positives[positives != idx].tolist()
        positives = sorted(positives)

        all_indexs = np.linspace(0,
                                 N_AERIAL+N_GROUND,
                                 N_AERIAL+N_GROUND,
                                 dtype=int, endpoint=False).tolist()
        non_negatives = np.concatenate([rr_idxs_aer_from_aer[i],
                                        (rr_idxs_aer_from_gro[i]+N_AERIAL)],
                                       axis=0).tolist()
        negatives = sorted(list(set(all_indexs) - set(non_negatives)))

        idx_dict['positives'] = positives
        idx_dict['negatives'] = negatives
        catalog[idx] = idx_dict

    for i in range(N_GROUND):
        idx = N_AERIAL + i
        idx_dict = {}
        idx_dict['query']    = ground_training_paths[i]
        idx_dict['northing'] = ground_training_xys[i,0]
        idx_dict['easting']  = ground_training_xys[i,1]

        positives = np.concatenate([nn_idxs_gro_from_aer[i],
                                    (nn_idxs_gro_from_gro[i]+N_AERIAL)],
                                   axis=0)
        positives = positives[positives != idx].tolist()
        positives = sorted(positives)

        all_indexs = np.linspace(0,
                                 N_AERIAL+N_GROUND,
                                 N_AERIAL+N_GROUND,
                                 dtype=int, endpoint=False).tolist()
        non_negatives = np.concatenate([rr_idxs_gro_from_aer[i],
                                        (rr_idxs_gro_from_gro[i]+N_AERIAL)],
                                       axis=0).tolist()
        negatives = sorted(list(set(all_indexs) - set(non_negatives)))

        idx_dict['positives'] = positives
        idx_dict['negatives'] = negatives
        catalog[idx] = idx_dict

    print("catalog length: %d" % len(catalog))
    return catalog

#%%
def construct_query_dict_new(aerial_training_paths: list,
                             aerial_training_xys: np.ndarray,
                             ground_training_paths: list,
                             ground_training_xys: np.ndarray,
                             aerial_positive_range=50.0,
                             aerial_negative_range=80.0,
                             ground_positive_range=15.0,
                             ground_negative_range=50.0):
    """ Construct Training Tuples

    Args:
        training_paths (list): N-dim paths
        training_xys (np.ndarray): [N x 2] shape ndarray [northing, easting]
        save_fn (Path): save filename
        ind_nn_r (_type_): _description_
        ind_r_r (int, optional): _description_. Defaults to 50.
        Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
        Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    """
    aerial_tree = KDTree(aerial_training_xys)
    ground_tree = KDTree(ground_training_xys)

    nn_idxs_aer_from_aer = aerial_tree.query_radius(aerial_training_xys, r=aerial_positive_range)
    nn_idxs_aer_from_gro = ground_tree.query_radius(aerial_training_xys, r=aerial_positive_range)
    nn_idxs_gro_from_aer = aerial_tree.query_radius(ground_training_xys, r=aerial_positive_range)
    nn_idxs_gro_from_gro = ground_tree.query_radius(ground_training_xys, r=ground_positive_range)

    rr_idxs_aer_from_aer = aerial_tree.query_radius(aerial_training_xys, r=aerial_negative_range)
    rr_idxs_aer_from_gro = ground_tree.query_radius(aerial_training_xys, r=aerial_negative_range)
    rr_idxs_gro_from_aer = aerial_tree.query_radius(ground_training_xys, r=aerial_negative_range)
    rr_idxs_gro_from_gro = ground_tree.query_radius(ground_training_xys, r=ground_negative_range)

    # print(type(nn_idxs_aer_from_aer), nn_idxs_aer_from_aer.shape)
    # print(type(rr_idxs_aer_from_aer), rr_idxs_aer_from_aer.shape)
    # print(type(nn_idxs_aer_from_aer[0]), nn_idxs_aer_from_aer[0].shape)
    # print(type(nn_idxs_aer_from_aer[10]), nn_idxs_aer_from_aer[10].shape)
    # # <class 'numpy.ndarray'> (6268,)
    # # <class 'numpy.ndarray'> (6268,)
    # # <class 'numpy.ndarray'> (30,)
    # # <class 'numpy.ndarray'> (36,)

    catalog = {}
    N_AERIAL = aerial_training_xys.shape[0]
    N_GROUND = ground_training_xys.shape[0]

    for i in range(N_AERIAL):
        idx = i
        idx_dict = {}
        idx_dict['query']    = aerial_training_paths[i]
        idx_dict['northing'] = aerial_training_xys[i,0]
        idx_dict['easting']  = aerial_training_xys[i,1]

        positives = np.concatenate([nn_idxs_aer_from_aer[i],
                                    (nn_idxs_aer_from_gro[i]+N_AERIAL)],
                                   axis=0)
        positives = positives[positives != idx].tolist()
        positives = sorted(positives)

        all_indexs = np.linspace(0,
                                 N_AERIAL+N_GROUND,
                                 N_AERIAL+N_GROUND,
                                 dtype=int, endpoint=False).tolist()
        non_negatives = np.concatenate([rr_idxs_aer_from_aer[i],
                                        (rr_idxs_aer_from_gro[i]+N_AERIAL)],
                                       axis=0).tolist()
        negatives = sorted(list(set(all_indexs) - set(non_negatives)))

        idx_dict['positives'] = positives
        idx_dict['negatives'] = negatives
        catalog[idx] = idx_dict

    for i in range(N_GROUND):
        idx = N_AERIAL + i
        idx_dict = {}
        idx_dict['query']    = ground_training_paths[i]
        idx_dict['northing'] = ground_training_xys[i,0]
        idx_dict['easting']  = ground_training_xys[i,1]

        positives = np.concatenate([nn_idxs_gro_from_aer[i],
                                    (nn_idxs_gro_from_gro[i]+N_AERIAL)],
                                   axis=0)
        positives = positives[positives != idx].tolist()
        positives = sorted(positives)

        all_indexs = np.linspace(0,
                                 N_AERIAL+N_GROUND,
                                 N_AERIAL+N_GROUND,
                                 dtype=int, endpoint=False).tolist()
        non_negatives = np.concatenate([rr_idxs_gro_from_aer[i],
                                        (rr_idxs_gro_from_gro[i]+N_AERIAL)],
                                       axis=0).tolist()
        negatives = sorted(list(set(all_indexs) - set(non_negatives)))

        idx_dict['positives'] = positives
        idx_dict['negatives'] = negatives
        catalog[idx] = idx_dict

    print("catalog length: %d" % len(catalog))
    return catalog


catalog = construct_query_dict_ver2(aerial_training_paths,
                                    aerial_training_xys,
                                    ground_training_paths,
                                    ground_training_xys)

save_fn = BENCHMARK_DIR / 'training_queries_umd_4096_new.pickle'
with open(save_fn, 'wb') as handle:
    pickle.dump(catalog, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
queries_catalog_fn  = CATALOG_V2_DIR / 'queries_catalog_ver2.txt'
database_catalog_fns = [
    CATALOG_V2_DIR / 'database_catalog_ver2_aerial_0.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_aerial_1.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_0.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_1.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_2.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_3.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_4.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_5.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_6.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_7.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_8.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_9.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_10.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_11.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_12.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_13.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_14.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_15.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_16.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_17.txt',
    CATALOG_V2_DIR / 'database_catalog_ver2_ground_18.txt'
]


# #%%
# l1 = np.linspace(0, 10, 10, dtype=int, endpoint=False)
# l2 = np.linspace(5, 15, 10, dtype=int, endpoint=False)
# l3 = list(set(l1) - set(l2))
# print(l3)
# #%%

#     tree = KDTree(df_centroids[['northing', 'easting']])
#     idxs_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
#     ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)
#     queries = {}

#     for idx_nn in range(len(idxs_nn)):
#         query = df_centroids.iloc[idx_nn]["file"]

#         anchor_pos = np.array(df_centroids.iloc[idx_nn][['northing', 'easting']])
#         # Extract timestamp from the filename
#         scan_filename = os.path.split(query)[1]
#         assert os.path.splitext(scan_filename)[1] == '.bin', f"Expected .bin file: {scan_filename}"
#         timestamp = int(os.path.splitext(scan_filename)[0])

#         positives = idxs_nn[idx_nn]
#         non_negatives = ind_r[idx_nn]

#         positives = positives[positives != idx_nn]
#         # Sort ascending order
#         positives = np.sort(positives)
#         non_negatives = np.sort(non_negatives)

#         # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
#         queries[idx_nn] \
#           = TrainingTuple(id=idx_nn,
#                           timestamp=timestamp,
#                           rel_scan_filepath=query,
#                           positives=positives,
#                           non_negatives=non_negatives,
#                           position=anchor_pos)

#     file_path = os.path.join(base_path, filename)
#     with open(file_path, 'wb') as handle:
#         pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     print("Done ", filename)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
#     # TODO_LCW dataset.required should be True later.
#     parser.add_argument('--dataset', type=str, required=False, help='Dataset root folder',
#                         choices=['oxford', 'inhouse', 'cs_campus'], default='cs_campus')
#     args = parser.parse_args()
#     DATASET_DIR = Path(DATASET_DIRS[args.dataset])
#     print(f"Dataset root: {DATASET_DIR}")
#     assert (DATASET_DIR.exists())

#     included = []
#     if args.dataset == 'oxford':
#         pass
#     elif args.dataset == 'inhouse':
#         pass
#     elif args.dataset == 'cs_campus':
#         included.append('cs_campus_aerial')
#         included.append('cs_campus_ground')

#     sequences = sorted(os.listdir(DATASET_DIR))
#     for seq in sequences:
#         print(seq)

#     exit(0)

#     base_path = args.dataset_root

#     all_folders = sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER)))
#     folders = []

#     # All runs are used for training (both full and partial)
#     index_list = range(len(all_folders) - 1)
#     print("Number of runs: " + str(len(index_list)))
#     for index in index_list:
#         folders.append(all_folders[index])
#     print(folders)

#     df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
#     df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

#     for folder in tqdm.tqdm(folders):
#         df_locations = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, folder, FILENAME), sep=',')
#         df_locations['timestamp'] = RUNS_FOLDER + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(str) + '.bin'
#         df_locations = df_locations.rename(columns={'timestamp': 'file'})

#         for index, row in df_locations.iterrows():
#             if check_in_test_set(row['northing'], row['easting'], P):
#                 df_test = df_test.append(row, ignore_index=True)
#             else:
#                 df_train = df_train.append(row, ignore_index=True)

#     print("Number of training submaps: " + str(len(df_train['file'])))
#     print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
#     # ind_nn_r is a threshold for positive elements - 10 is in original PointNetVLAD code for refined dataset
#     construct_query_dict(df_train, base_path, "training_queries_baseline2.pickle", ind_nn_r=10)
#     construct_query_dict(df_test, base_path, "test_queries_baseline2.pickle", ind_nn_r=10)

# # %%
