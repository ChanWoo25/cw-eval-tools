#%%
import numpy as np
import os
import sys
from pathlib import Path
import copy
import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle as pkl
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/root/shared/cw-eval-tools/python')
from module.zip import Zip

def get_current_ts():
    ct = datetime.datetime.now()
    return f"{ct.year}-{ct.month:02d}-{ct.day:02d}_{ct.hour:02d}-{ct.minute:02d}-{ct.second:02d}"


#%% Load Inference Results
DATASETS_DIR = Path('/data/datasets')
cs_campus_dir = DATASETS_DIR / 'dataset_cs_campus'
results_dir = cs_campus_dir / 'results'
infer_zip_fn = cs_campus_dir / 'infer_dict.pbz2'
infer_dict = Zip(infer_zip_fn).unzip()

all_db_embs = infer_dict['all_db_embs']
all_db_meta = infer_dict['all_db_meta']
all_q_embs  = infer_dict['all_q_embs']
all_q_meta  = infer_dict['all_q_meta']

#%% Create eval sets
iters = []
for idx_db in range(len(all_db_embs)):
    # Remove redundant aerial evaluation
    if idx_db == 0:
        continue

    if all_db_embs[idx_db].shape[0] == 0:
        print(f"Database[{idx_db}] length == 0")
        continue

    assert (all_q_embs.ndim == 2)
    assert (all_q_embs.shape[0] == len(all_q_meta['idx']))
    iters.append((idx_db, 0))

#%%
class Metrics(object):
    ITEMS = [{
        'name': 'Recall@1%',
        'enabled': True,
        'eval_func': 'cls._get_recall_one_percent',
        'is_greater_better': True,
        'init_value': 0.0
    }]

    @classmethod
    def get(cls, *args, **kwargs):
        _items = cls.items()
        _values = {item['name']: 0 for item in _items}
        for i, item in enumerate(_items):
            eval_func = eval(item['eval_func'])
            _values[item['name']] = eval_func(*args, **kwargs)

        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]

    @classmethod
    def _get_recall_one_percent(cls, *args, **kwargs):
        return args[0]

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value

class AverageValue(object):
    """Computes and stores the average and current value"""

    def __init__(self, value_names=None):
        self.value_names = value_names if value_names is not None else [
            'value']
        self.num_value = 1 if value_names is None else len(value_names)
        self.reset()

    def reset(self):
        self._val = {vn: 0 for vn in self.value_names}
        self._sum = {vn: 0 for vn in self.value_names}
        self._count = {vn: 0 for vn in self.value_names}

    def update(self, values):
        if type(values) == dict:
            for vn, v in values.items():
                self._val[vn] = v
                self._sum[vn] += v
                self._count[vn] += 1
        else:
            self._val['value'] = values
            self._sum['value'] += values
            self._count['value'] += 1

    def val(self, vn=None):
        if vn is None:
            return self._val['value'] if self.value_names == ['value'] else copy.deepcopy(self._val)
        else:
            return self._val[vn]

    def count(self, vn=None):
        if vn is None:
            return self._count['value'] if self.value_names == ['value'] else copy.deepcopy(self._count)
        else:
            return self._count[vn]

    def avg(self, vn=None):
        if vn is None:
            return self._sum['value'] / self._count['value'] if self.value_names == ['value'] else {
                vn: self._sum[vn] / self._count[vn] for vn in self.value_names
            }
        else:
            return self._sum[vn] / self._count[vn]

    def avg_str(self, vn=None, format_str='%.4f'):
        avg_vs = self.avg(vn)
        if type(avg_vs) == dict:
            result = ''
            for k, v in avg_vs.items():
                result += ("%s= " + format_str + " ") % (k, v)
            return result
        else:
            return format_str % avg_vs

#%%
def get_dist(dbase_north, dbase_east,
             query_north, query_east):
    return np.linalg.norm(np.array(
                [dbase_north - query_north,
                 dbase_east - query_east]))

def get_diff(dbase_embedding,
             query_embedding):
    tmp = dbase_embedding - query_embedding
    diff = np.linalg.norm(tmp)
    return diff

def get_recall(log_dir:Path,
               dbase_idx: int,
               dbase_embs: np.ndarray,
               dbase_meta: dict,
               query_embs: np.ndarray,
               query_meta: dict,
               query_catalog: list,
               num_neighbors=25):
    # Squeeze if needed
    if dbase_embs.ndim > 2:
        dbase_embs = dbase_embs.squeeze()
    if query_embs.ndim > 2:
        query_embs = query_embs.squeeze()

    ## Debugging ##
    # print(f"dbase_embs.shape: {dbase_embs.shape}")
    # print(f"query_embs.shape: {query_embs.shape}")
    # print(f"dbase_meta.keys: {dbase_meta.keys()}")
    # print(f"query_meta.keys: {query_meta.keys()}")
    # # dbase_embs.shape: (13760, 256)
    # # query_embs.shape: (1059, 256)
    # # dbase_meta.keys: dict_keys(['idx', 'filename', 'northing', 'easting'])
    # # query_meta.keys: dict_keys(['idx', 'filename', 'northing', 'easting'])

    ## Save results in a text file
    matches_fn = log_dir / f"query-to-dbase-{dbase_idx:02d}-matches.txt"
    f_match = open(matches_fn, 'w')

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    dbase_tree = KDTree(dbase_embs)
    n_dbase = dbase_embs.shape[0]
    feature_len = query_embs.shape[1]
    assert(query_embs.shape[1] == dbase_embs.shape[1])

    # num_neighbors = 25
    recall = [0] * num_neighbors
    top1_similarity_score = []
    one_percent_retrieved = 0
    top1p_idx = max(int(round(n_dbase/100.0)), 1)

    num_evaluated = 0
    top1_TP_xys = []
    top1_FP_xys = []
    top1p_TP_xys = []
    top1p_FP_xys = []
    for qi in range(query_embs.shape[0]):
        seq_idx = int(query_meta['filename'][qi].split('/')[1].split('_')[5]) - 1
        # print(f"[Q-{qi:04d}] filename: {query_meta['filename'][qi]}")
        # print(f"[Q-{qi:04d}] seq_idx: {seq_idx:02d} | idx: {query_meta['idx'][qi]:04d}")
        query_details = query_catalog[qi]
        true_indices: list = query_details[dbase_idx]

        # Only evaluates queries with answers.
        if len(true_indices) == 0:
            continue
        else:
            num_evaluated += 1

        distances, indices = dbase_tree.query(
            query_embs[qi].reshape((1,feature_len)), k=min(n_dbase, num_neighbors))

        top1_dists = []
        top1_diffs = []
        is_top1 = False
        is_top1p = False
        for ni in range(len(indices[0])):
            di = indices[0][ni]
            if di in true_indices:
                # similarity = np.dot(query_embs[qi], dbase_embs[di])
                recall[ni] += 1

                # For Top@1 Analysis
                dist = get_dist(dbase_meta['northing'][di], dbase_meta['easting'][di],
                                query_meta['northing'][qi], query_meta['easting'][qi])
                diff = get_diff(dbase_embs[di], query_embs[qi])
                top1_dists.append(dist)
                top1_diffs.append(diff)

                if ni == 0:
                    is_top1 = True

                if ni < top1p_idx:
                    is_top1p = True
                    one_percent_retrieved += 1

                break

        if is_top1:
            top1_TP_xys.append([query_meta['northing'][qi], query_meta['easting'][qi]])
        else:
            top1_FP_xys.append([query_meta['northing'][qi], query_meta['easting'][qi]])
        if is_top1p:
            top1p_TP_xys.append([query_meta['northing'][qi], query_meta['easting'][qi]])
        else:
            top1p_FP_xys.append([query_meta['northing'][qi], query_meta['easting'][qi]])



        log_mesg = f"{seq_idx:02d} " \
                 + f"{query_meta['idx'][qi]} " \
                 + f"{query_meta['filename'][qi]} " \
                 + f"{query_meta['northing'][qi]:.6f} " \
                 + f"{query_meta['easting'][qi]:.6f}"

        for i in range(10):
            di = indices[0][i]

            if di in true_indices:
                stat = "T"
            else:
                stat = "F"

            dist = get_dist(dbase_meta['northing'][di], dbase_meta['easting'][di],
                            query_meta['northing'][qi], query_meta['easting'][qi])
            top1_dists.append(dist)

            log_mesg = log_mesg \
                     + f" {stat} {di} {dist:.6f}"
        log_mesg = log_mesg + "\n"
        f_match.write(log_mesg)
    f_match.close()

    plot_top1_fn = log_dir / f"query-to-dbase-{dbase_idx:02d}-top1.png"
    top1_TP_xys = np.array(top1_TP_xys, dtype=float)
    top1_FP_xys = np.array(top1_FP_xys, dtype=float)
    top1p_TP_xys = np.array(top1p_TP_xys, dtype=float)
    top1p_FP_xys = np.array(top1p_FP_xys, dtype=float)
    all_xs = np.array(query_meta['northing'], dtype=float)
    all_ys = np.array(query_meta['easting'], dtype=float)

    ## Plot True & False
    plt.figure(figsize=(8, 6), dpi=400)
    sns.set_theme(context='paper', style='whitegrid')
    sns.scatterplot(x=all_xs, y=all_ys, marker='v', s=12, alpha=0.4)
    sns.scatterplot(x=top1_TP_xys[:,0], y=top1_TP_xys[:,1], marker='o', c='g', s=20, alpha=0.9)
    sns.scatterplot(x=top1_FP_xys[:,0], y=top1_FP_xys[:,1], marker='^', c='r', s=40, alpha=1.0)
    plt.title('Top@1 Queries')
    plt.xlabel('northing')
    plt.ylabel('easting')
    plt.savefig(plot_top1_fn)
    plt.clf()

    ## Plot True & False
    plot_top1p_fn = log_dir / f"query-to-dbase-{dbase_idx:02d}-top1p.png"
    plt.figure(figsize=(8, 6), dpi=400)
    sns.set_theme(context='paper', style='whitegrid')
    sns.scatterplot(x=all_xs, y=all_ys, marker='v', s=15, alpha=0.6)
    sns.scatterplot(x=top1p_TP_xys[:,0], y=top1p_TP_xys[:,1], marker='o', c='g', s=20, alpha=0.9)
    sns.scatterplot(x=top1p_FP_xys[:,0], y=top1p_FP_xys[:,1], marker='^', c='r', s=40, alpha=1.0)
    plt.title('Top@1% Queries')
    plt.xlabel('northing')
    plt.ylabel('easting')
    plt.savefig(plot_top1p_fn)
    plt.clf()

    top1_dists = np.array(top1_dists)
    top1_diffs = np.array(top1_diffs)

    avg_top1_dists = top1_dists.mean()
    avg_top1_diffs = top1_diffs.mean()
    min_top1_dists = top1_dists.min()
    min_top1_diffs = top1_diffs.min()
    max_top1_dists = top1_dists.max()
    max_top1_diffs = top1_diffs.max()
    std_top1_dists = top1_dists.std()
    std_top1_diffs = top1_diffs.std()
    top1_dist_dict = {
        'avg': avg_top1_dists,
        'min': min_top1_dists,
        'max': max_top1_dists,
        'std': std_top1_dists
    }
    top1_diff_dict = {
        'avg': avg_top1_diffs,
        'min': min_top1_diffs,
        'max': max_top1_diffs,
        'std': std_top1_diffs
    }

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=250)
    ax.boxplot([top1_dists], autorange=True)
    ax.set_ylabel('Distance[m]')
    # ax.set_ylim([avg_top1_dists - 3.0 * std_top1_dists,
    #              avg_top1_dists + 3.0 * std_top1_dists])
    plot_dists_fn = log_dir / f"query-to-dbase-{dbase_idx:02d}-dists.png"
    fig.savefig(plot_dists_fn)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=250)
    # ax.set_ylim([avg_top1_diffs - 3.0 * std_top1_diffs,
    #              avg_top1_diffs + 3.0 * std_top1_diffs])
    ax.boxplot([top1_diffs], autorange=True)
    ax.set_ylabel('global desc diff')
    plot_diffs_fn = log_dir / f"query-to-dbase-{dbase_idx:02d}-diffs.png"
    fig.savefig(plot_diffs_fn)

    if num_evaluated == 0:
        return None, None, None, None

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recalls = (np.cumsum(recall)/float(num_evaluated))*100
    # print(f"db[{idx_db}] - qr[{idx_qr}] | Query Num: {queries_output.shape[0]}, DBase Num: {database_output.shape[0]}")
    # print(f"db[{idx_db}] - qr[{idx_qr}] | Recall@1: {recall[0]:.3f}, Recall@1%:{one_percent_recall:.3f}")
    # from IPython import embed;embed()
    # print(recall)
    # print(np.mean(top1_similarity_score))
    # print(one_percent_recall)
    # f_match.write(f"{num_evaluated} {one_percent_recall} {recall}")
    # f_match.close()

    return recalls, one_percent_recall, top1_dist_dict, top1_diff_dict


#%%
## Debug
# for iter in iters:
#     print(f"dbase({iter[0]:02d}) - query({iter[1]:02d})")

with open('/data/datasets/dataset_cs_campus/benchmark_datasets/umd_evaluation_query.pickle', 'rb') as f:
    query_catalog = pkl.load(f)
    query_catalog = query_catalog[0]

# Query Count
# {12: 78, 13: 43, 14: 65, 15: 226, 16: 345, 17: 264, 18: 38}

log_dir = results_dir / get_current_ts()
os.makedirs(log_dir, exist_ok=True)
print(f"[+] Eval results is saved in '{log_dir}'")
neighbor=25
similarity = []
dist = []
all_recalls = np.zeros(neighbor, dtype=float)
metrics = AverageValue(Metrics.names())
with tqdm(total=len(iters)) as pbar:
    for i, j in iters:
        recalls, one_percent_recall, top1_dist_dict, top1_diff_dict \
            = get_recall(log_dir,
                         i, all_db_embs[i], all_db_meta[i],
                         all_q_embs, all_q_meta, query_catalog)

        # if recalls == None:
        #     continue

        _metrics = Metrics.get(one_percent_recall)
        metrics.update(_metrics)
        pbar.update(1)
        all_recalls += np.array(recalls)

        print(f"[{i:02d}-th dbase result]")
        print(f"Top@1% Recall: {one_percent_recall:.2f}")
        top5_print = [f"{i:.2f}" for i in recalls[:5].tolist()]
        print(f"Top@5 Recalls: {top5_print}")

all_recalls = all_recalls / len(iters)
# avg_similarity = np.mean(similarity)
# avg_dist = np.mean(dist)
print('====================== EVALUATE RESULTS ======================')
format_str = '{sample_num:<10} ' + \
    ' '.join(['{%s:<10}' % vn for vn in metrics.value_names])
title_dict = dict(
    sample_num='Sample'
)
title_dict.update({vn: vn for vn in metrics.value_names})

print(format_str.format(**title_dict))

overall_dict = dict(
    sample_num=len(iters)
)
# from IPython import embed;embed()
overall_dict.update(
    {vn: "%.4f" % metrics.avg(vn) for vn in metrics.value_names})

print(format_str.format(**overall_dict))

# t = 'Avg. similarity: {:.4f} Avg. dist: {:.4f} Avg. recall @N:\n'+str(all_recalls)
# print(t.format(avg_similarity, avg_dist))

print(f"Recall@1%:")
print(metrics.avg())
# return Metrics('Recall@1%', metrics.avg())


        # for x in pair_similarity:
        #     similarity.append(x)
        # if pair_dist != None:
        #     for x in pair_dist:
        #         dist.append(x)
        # break
        # pair_recall, pair_similarity, pair_opr, pair_dist = get_recall(
        #     i, j,
        #     database_vectors=all_db_embs, database_sets=all_db_set,
        #     query_vectors=all_q_embs, query_sets=q_data_loader.dataset.catalog,
        #     num_neighbors=neighbor)
        # if pair_recall is None:
        #     continue
        # _metrics = Metrics.get(pair_opr)

        # metrics.update(_metrics)
        # pbar.update(1)
        # recall += np.array(pair_recall)
        # for x in pair_similarity:
        #     similarity.append(x)
        # if pair_dist != None:
        #     for x in pair_dist:
        #         dist.append(x)

# avg_recall = recall / len(iters)
# avg_similarity = np.mean(similarity)
# avg_dist = np.mean(dist)
# log.info(
#     '====================== EVALUATE RESULTS ======================')
# format_str = '{sample_num:<10} ' + \
#     ' '.join(['{%s:<10}' % vn for vn in metrics.value_names])

# title_dict = dict(
#     sample_num='Sample'
# )
# title_dict.update({vn: vn for vn in metrics.value_names})

# log.info(format_str.format(**title_dict))

# overall_dict = dict(
#     sample_num=len(iters)
# )
# # from IPython import embed;embed()
# overall_dict.update(
#     {vn: "%.4f" % metrics.avg(vn) for vn in metrics.value_names})

# log.info(format_str.format(**overall_dict))

# t = 'Avg. similarity: {:.4f} Avg. dist: {:.4f} Avg. recall @N:\n'+str(avg_recall)
# log.info(t.format(avg_similarity, avg_dist))

# return Metrics('Recall@1%', metrics.avg())

#%% Query Catalog Analysis
with open('/data/datasets/dataset_cs_campus/benchmark_datasets/umd_evaluation_query.pickle', 'rb') as f:
    catalog = pkl.load(f)
print(catalog)

#%%
import datetime
ct = datetime.datetime.now()
print("current time:-", ct)
print(f"{ct.year}-{ct.month:02d}-{ct.day:02d}_{ct.hour:02d}-{ct.minute:02d}-{ct.second:02d}")
#%%
import logging
logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

logging.info('an info messge')
# 2017-05-25 00:58:28 INFO     an info messge
# >>> logging.debug('a debug messag is not shown')
# >>>
