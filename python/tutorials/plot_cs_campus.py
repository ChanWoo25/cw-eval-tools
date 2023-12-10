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

#%% Plot Trajectories %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
paths, xs, ys = readCatalog('db', 1)
xs = np.array(xs, dtype=float)
ys = np.array(ys, dtype=float)

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

for idx in range(13, 20):
    paths, xs, ys = readGPS(idx)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    # sns.lineplot(x=xs, y=ys, alpha=0.6, linewidth=1)
    plt.plot(xs, ys, alpha=0.9, linewidth=3)

rects = [(4317250-40, 331950-40, 80, 80),
         (4317290-40, 332130-40, 80, 80),
         (4317390-40, 332260-40, 80, 80),
         (4317470-40, 331930-40, 80, 80),
         (4317480-40, 332100-40, 80, 80),
         (4317520-40, 332210-40, 80, 80)]

import matplotlib.patches as patches
ax = plt.gca()
for x, y, w, h in rects:
    ax.add_patch(
        patches.Rectangle(
            (x, y), w, h,
            edgecolor='blue',
            facecolor='palegreen',
            linewidth=0.5,
            fill=True,
            alpha=0.6))

# Customize the plot
plt.title('CS Campus Dataset [w/ All Trajectories]', {'fontsize':18})
plt.xlabel('northing')
plt.ylabel('easting')

# Display the plot
plt.savefig('/data/datasets/dataset_cs_campus/plots/all-trajectories.png')
plt.show()

#%% Save Each Trajectories %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for idx in range(13, 14):
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

    paths, xs, ys = readCatalog('db', idx)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    plt.plot(xs, ys, alpha=0.9, linewidth=3)

    plt.title('CS Campus Dataset [w/ Ground-%02d]' % (idx-1), {'fontsize':18})
    plt.xlabel('northing')
    plt.ylabel('easting')
    plt.savefig('/data/datasets/dataset_cs_campus/plots/ground-%02d' % (idx-1))

for idx in range(13, 20):
    plt.figure(figsize=(12, 10), dpi=400)
    plt.xlim(4316800, 4318000)
    plt.ylim( 331500,  332500)
    sns.set_theme(context='paper', style='whitegrid')

    paths, xs, ys = readCatalog('db', 1)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    sns.scatterplot(x=xs, y=ys, marker='v', s=20, alpha=0.6)

    paths, xs, ys = readGPS(idx)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    plt.plot(xs, ys, alpha=0.9, linewidth=3)

    plt.title('CS Campus Dataset [w/ Ground-%02d]' % (idx), {'fontsize':18})
    plt.xlabel('northing')
    plt.ylabel('easting')
    plt.savefig('/data/datasets/dataset_cs_campus/plots/ground-%02d' % (idx))

# Display the plot
# plt.show()

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
