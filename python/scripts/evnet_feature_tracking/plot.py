import os
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# The package for colored Command Line Interface (CLI)
from colorama import init as colorama_init
from colorama import Fore, Back, Style
colorama_init(autoreset=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
parent_dir = Path(__file__).parent

# # Dataset
# x=np.array([1, 2, 3, 4, 5, 6, 7, 8])
# y=np.array([20, 30, 5, 12, 39, 48, 50, 3])

# cubic_interpolation_model = interp1d(x, y, kind = "cubic")

# # Plotting the Graph
# X_=np.linspace(x.min(), x.max(), 500)
# Y_=cubic_interpolation_model(X_)

# plt.plot(X_, Y_)
# plt.title("Plot Smooth Curve Using the scipy.interpolate.interp1d Class")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

def plot_scatter(dataset: str, method: str):
    error_fn = Path("/home/leecw/results/event_feature_tracking") / dataset / (method+"_errors.csv")
    save_fn  = parent_dir / (dataset + "_" + method + "_scatter.png")
    sim_save_fn  = parent_dir / (dataset + "_" + method + "_sim_scatter.png")
    sm_save_fn  = parent_dir / (dataset + "_" + method + "_smoothed.png")
    error_np = np.genfromtxt(error_fn, dtype=float, skip_header=1, delimiter=',')

    indexs     = error_np[:, 0]
    y_error    = error_np[:, 1]
    x_lifetime = error_np[:, 2]
    validity   = error_np[:, 3]

    list_x = []
    list_y = []
    idx = indexs[0]
    for i in range(1, error_np.shape[0]):
        if indexs[i] != idx:
            list_x.append(x_lifetime[i-1])
            list_y.append(y_error[i-1])
            idx = indexs[i]

    N = len(list_x)
    list_x = np.array(list_x).reshape(N, 1)
    list_y = np.array(list_y).reshape(N, 1)
    new_np = np.hstack([list_x, list_y])
    idxs = np.argsort(new_np[:, 0])
    new_np = new_np[idxs]
    print(new_np.shape)
    print(new_np[:10])

    new_list = []
    ts = new_np[:, 0]
    old_t = ts[0]
    cnt_redun = 0
    for i in range(1, N):
        if (ts[i] - old_t) < 1e-6:
            cnt_redun += 1
        else:
            new_list.append(new_np[i, :])
            old_t = ts[i]

    print("Removed: %d" % cnt_redun)

    final_np = np.vstack(new_list)
    print(final_np.shape)


    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=300)
    fig.suptitle('scatter %s | %s' % (dataset, method), fontsize=12)
    axs.scatter(x_lifetime, y_error, s=5)
    fig.tight_layout()
    fig.savefig(save_fn)
    plt.close(fig)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=300)
    fig.suptitle('simple_scatter %s | %s' % (dataset, method), fontsize=12)
    axs.scatter(final_np[:,0], final_np[:,1], s=8)
    fig.tight_layout()
    fig.savefig(sim_save_fn)
    plt.close(fig)

    # cubic_interpolation_model = interp1d(new_list_x_np, new_list_y_np, kind = "cubic")
    # cubic_interpolation_model = make_interp_spline(final_np[:, 0], final_np[:, 1])

    # # Plotting the Graph
    # X_=np.linspace(list_x.min(), list_x.max(), 500)
    # Y_=cubic_interpolation_model(X_)
    # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=300)
    # fig.suptitle('smoothed %s | %s' % (dataset, method), fontsize=12)
    # axs.plot(X_, Y_, 'b', linewidth=1)
    # fig.tight_layout()
    # fig.savefig(sm_save_fn)
    # plt.close(fig)

if __name__ == '__main__':

    # error_fn = Path("/home/leecw/results/event_feature_tracking/poster_translation/haste_correlation_errors.csv")
    # error_np = np.genfromtxt(error_fn, dtype=float, skip_header=1, delimiter=',')

    plot_scatter("boxes_6dof", "haste_correlation")
    plot_scatter("boxes_rotation", "haste_correlation")
    plot_scatter("boxes_translation", "haste_correlation")
    plot_scatter("poster_6dof", "haste_correlation")
    plot_scatter("poster_rotation", "haste_correlation")
    plot_scatter("poster_translation", "haste_correlation")
    plot_scatter("shapes_6dof", "haste_correlation")
    plot_scatter("shapes_rotation", "haste_correlation")
    plot_scatter("shapes_translation", "haste_correlation")



    plot_scatter("boxes_6dof", "ours_v1")
    plot_scatter("boxes_rotation", "ours_v1")
    plot_scatter("boxes_translation", "ours_v1")
    plot_scatter("poster_6dof", "ours_v1")
    plot_scatter("poster_rotation", "ours_v1")
    plot_scatter("poster_translation", "ours_v1")
    plot_scatter("shapes_6dof", "ours_v1")
    plot_scatter("shapes_rotation", "ours_v1")
    plot_scatter("shapes_translation", "ours_v1")
