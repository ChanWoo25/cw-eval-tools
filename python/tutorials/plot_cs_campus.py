# %%
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = Path('/data/datasets')
CS_CAMPUS_DIR = DATASET_DIR / 'dataset_cs_campus'

def readCatalog(mode: str, num: int):
    assert(0 <= num and num < 14)
    if mode == 'db':
        fn = CS_CAMPUS_DIR / 'catalog' / f"db_catalog_{num}.txt"
    elif mode == 'qr':
        fn = CS_CAMPUS_DIR / 'catalog' / 'qr_catalog.txt'
    with open(fn, 'r') as f:
        lines = f.readlines()
        paths = []
        xs = []
        ys = []
        for line in lines:
            words = line.strip().split(' ')
            paths.append(words[1])
            xs.append(float(words[2]))
            ys.append(float(words[3]))
    return paths, xs, ys

def readGPS(num: int):
    assert(13 <= num and num < 20)
    fn = CS_CAMPUS_DIR / f"benchmark_datasets/umd/umcp_lidar5_ground_umd_gps_{num}" / 'umd_aerial_cloud_20m_100coverage_4096.csv'
    with open(fn, 'r') as f:
        lines = f.readlines()
        paths = []
        xs = []
        ys = []
        for line in lines[1:]:
            words = line.strip().split(',')
            paths.append(words[0])
            xs.append(float(words[1]))
            ys.append(float(words[2]))
    return paths, xs, ys

# %%

paths, xs, ys = readCatalog('db', 1)
xs = np.array(xs, dtype=float)
ys = np.array(ys, dtype=float)

print(xs.max() - xs.min())
print(ys.max() - ys.min())

plt.figure(figsize=(12, 10), dpi=400)
plt.xlim(4316800, 4318000)
plt.ylim( 331500,  332500)

sns.set_theme(context='paper', style='whitegrid')
sns.scatterplot(x=xs, y=ys, marker='v', s=20, alpha=0.6)

for idx in range(2, 14):
    paths, xs, ys = readCatalog('db', idx)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    # sns.lineplot(x=xs, y=ys, alpha=0.6, linewidth=1)
    plt.plot(xs, ys, alpha=0.9, linewidth=3)

# for idx in range(13, 20):
#     paths, xs, ys = readGPS(idx)
#     xs = np.array(xs, dtype=float)
#     ys = np.array(ys, dtype=float)
#     # sns.lineplot(x=xs, y=ys, alpha=0.6, linewidth=1)
#     plt.plot(xs, ys, alpha=0.9, linewidth=3)

# Customize the plot
plt.title('CS Campus Dataset [w/ Eval Database]', {'fontsize':18})
plt.xlabel('northing')
plt.ylabel('easting')

# Display the plot
plt.show()

#%%
paths, xs, ys = readCatalog('db', 1)
xs = np.array(xs, dtype=float)
ys = np.array(ys, dtype=float)

print(xs.max() - xs.min())
print(ys.max() - ys.min())

plt.figure(figsize=(12, 10), dpi=400)
plt.xlim(4316800, 4318000)
plt.ylim( 331500,  332500)

sns.set_theme(context='paper', style='whitegrid')
sns.scatterplot(x=xs, y=ys, marker='v', s=20, alpha=0.6)

# for idx in range(2, 14):
#     paths, xs, ys = readCatalog('db', idx)
#     xs = np.array(xs, dtype=float)
#     ys = np.array(ys, dtype=float)
#     # sns.lineplot(x=xs, y=ys, alpha=0.6, linewidth=1)
#     plt.plot(xs, ys, alpha=0.9, linewidth=3)

for idx in range(13, 20):
    paths, xs, ys = readGPS(idx)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    # sns.lineplot(x=xs, y=ys, alpha=0.6, linewidth=1)
    plt.plot(xs, ys, alpha=0.9, linewidth=3)

# Customize the plot
plt.title('CS Campus Dataset [w/ Eval Queries]', {'fontsize':18})
plt.xlabel('northing')
plt.ylabel('easting')

# Display the plot
plt.show()

# %%
paths, xs, ys = readCatalog('db', 1)
xs = np.array(xs, dtype=float)
ys = np.array(ys, dtype=float)

print(xs.max() - xs.min())
print(ys.max() - ys.min())

plt.figure(figsize=(10, 6), dpi=400)
plt.xlim(4316500, 4318500)
plt.ylim( 331400,  332600)

sns.set_theme(context='paper', style='whitegrid')
sns.scatterplot(x=xs, y=ys, marker='v', s=20, alpha=0.6)

for idx in range(2, 14):
    paths, xs, ys = readCatalog('db', idx)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    # sns.lineplot(x=xs, y=ys, alpha=0.6, linewidth=1)
    plt.plot(xs, ys, alpha=0.9, linewidth=3)

# for idx in range(13, 20):
#     paths, xs, ys = readGPS(idx)
#     xs = np.array(xs, dtype=float)
#     ys = np.array(ys, dtype=float)
#     # sns.lineplot(x=xs, y=ys, alpha=0.6, linewidth=1)
#     plt.plot(xs, ys, alpha=0.9, linewidth=3)

# Customize the plot
plt.title('CS Campus Dataset [w/ Eval Database]', {'fontsize':15})
plt.xlabel('northing')
plt.ylabel('easting')

# Display the plot
plt.show()

# %%
