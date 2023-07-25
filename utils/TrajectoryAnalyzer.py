import numpy as np
import torch
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt

import os
import sys
# print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Trajectory import Trajectory
from TrajectoryParser import TrajectoryParser
from Rotation import Rotation
from torch_utils import bmtm


# The package for colored Command Line Interface (CLI)
from colorama import init as colorama_init
from colorama import Fore, Back, Style
colorama_init(autoreset=True)

class TrajectoryAnalyzer:
    """
        Analyze a single or multiple trajectories. \n
        One TrajectoryAnalyzer Instance per one dataset-algorithm pair
        ### Notation
        - time: normally means timestamp [sec]
        - T: normally means tranformation (t & R)
        - t: normally means translation
        - R: normally means rotation
        - c: normally means scale factor
    """

    def __init__(self, single_dir:str=None, root_dir:str=None, gt_abs_path:str=None,
                 start:float=None, end:float=None, estimate_dir_prefix:str='event_imu_',
                 use_cache=False, do_align=True):
        assert (gt_abs_path is not None and os.path.exists(gt_abs_path)), Fore.RED+'You need to specify an absolute path for proper GT file'

        self.gt_abs_path = Path(gt_abs_path)
        self.use_cache = use_cache
        self.cudable = torch.cuda.is_available()

        # Extract GT first
        self.gt = Trajectory(use_file=True, trajectory_file_abs_path=self.gt_abs_path)
        print(Fore.RED+Style.DIM+'GT::path::%s'%gt_abs_path)

        if single_dir is not None:
            single_dir = Path(single_dir)
            print(Fore.GREEN+'EVAL::single -- %s' % single_dir.name)
            estimate_path          = single_dir / 'estimate.txt'
            cache_path             = single_dir / 'cache_dict.pt'
            plot_dir               = single_dir / 'plots'
            if not estimate_path.exists():
                raise AssertionError(Fore.RED+'%s doesn\'t exists!'%estimate_path.name)

            if not cache_path.exists() or not use_cache:
                print(Fore.GREEN+Style.DIM+'[info] -- not load from cache')
                estimate = Trajectory(use_file=True, trajectory_file_abs_path=estimate_path, start=start, end=end)
                TrajectoryParser.parse_trajectory(
                    estimate, self.gt,
                    cache_path, do_align=do_align, save_dir=estimate_path.parent)
            else:
                print(Fore.GREEN+Style.DIM+'[info] -- load from %s file exists in %s' % (cache_path.name, single_dir.name))

            cache:dict = torch.load(cache_path)

            if not plot_dir.exists():
                os.mkdir(plot_dir)

            # self.plot_all(cache, plot_dir)
            # self.dump(cache, plot_dir)
            exit(0)

        self.root_dir = Path(root_dir)
        assert (root_dir is not None and os.path.exists(root_dir)), Fore.RED+'You need to specify root directory containing estimates and ground-truth files'
        self.dirs = list(self.root_dir.glob(estimate_dir_prefix+'*'))
        self.dirs.sort()

        print(Fore.GREEN+'Registered subdirs:')
        for dir in self.dirs:
            print('- %s'%dir)


        for dir in self.dirs:
            print(Fore.GREEN+'%s Processing ... ' % dir.name)
            estimate_path          = dir / 'estimate.txt'
            cache_path             = dir / 'cache_dict.pt'
            plot_dir               = dir / 'plots'

            if not estimate_path.exists():
                raise AssertionError(Fore.RED+'%s doesn\'t exists!'%estimate_path.name)

            if not cache_path.exists() or not use_cache:
                estimate = Trajectory(use_file=True, trajectory_file_abs_path=estimate_path)
                TrajectoryParser.parse_trajectory(
                    estimate, self.gt,
                    cache_fn=cache_path, save_dir=estimate_path.parent)
            else:
                print(Fore.GREEN+Style.DIM+'[info] -- %s file exists in %s' % (cache_path.name, dir.name))

            cache:dict = torch.load(cache_path)
            if not plot_dir.exists():
                os.mkdir(plot_dir)

            self.plot_all(cache, plot_dir)
            self.dump(cache, plot_dir)

            print()

        self.abs_error_dict = {}
        self.rel_error_dict = {}

    def dump(self, cache:dict, plot_dir:Path):

        result_dir:dict = {}

        abs_t_traveled_gt = cache['abs_t_traveled_gt']
        total_distance = abs_t_traveled_gt[-1].item()
        result_dir['total_distance'] = total_distance

        gt = cache['abs_t_gt']
        est = cache['abs_t_est']
        error = gt - est
        error_norm = torch.norm(error, dim=1)
        result_dir['abs_t_mean_error'] = error_norm.mean().item()

        abs_t_error_norm_percent = error_norm.mean() / total_distance
        result_dir['mean_position_error'] = abs_t_error_norm_percent.item()

        abs_r_error = torch.rad2deg(cache['abs_r_error'])
        abs_r_error_norm = abs_r_error.norm(dim=1)
        result_dir['abs_r_rmse_error'] = abs_r_error_norm.mean().item()

        abs_ypr_error = cache['abs_ypr_error']
        abs_yaw_error = torch.abs(abs_ypr_error[:, 0])
        abs_yaw_mean_error = abs_yaw_error.mean() / total_distance
        result_dir['mean_yaw_error'] = abs_yaw_mean_error.item()


        result_dir['last_position_error_ratio'] = error_norm[-1].item() / total_distance
        result_dir['last_yaw_error_ratio'] = abs_yaw_error[-1].item() / total_distance

        yaml_fn = plot_dir / 'other_numeric_results.yaml'
        import yaml
        with open(yaml_fn, 'w', encoding = 'utf-8') as f:
            yaml.dump(result_dir, f, default_flow_style=False)

        # perform file operations

    def plot_all(self, cache:dict, plot_dir:Path):
        ######################### Dict Keys ###############
        # abs_t_est             # abs_q_xyzw_est          #
        # abs_t_gt              # abs_q_xyzw_gt           #
        # abs_r_est             # rel_t_est               #
        # abs_r_gt              # rel_t_gt                #
        # abs_ypr_est           # rel_t_error             #
        # abs_ypr_gt            # rel_yaw_gt              #
        # abs_t_error           # yaw_traveled_gt         #
        # abs_r_error           # yaw_error_per_meter     #
        # abs_q_xyzw_error      # yaw_error_per_meter_t   #
        # abs_ypr_error         # timestamp_sec           #
        ###################################################
        self.plot_3d_traj(cache, plot_dir)
        self.plot_abs_t(cache, plot_dir)
        self.plot_abs_t_error(cache, plot_dir)
        self.plot_abs_r(cache, plot_dir)
        self.plot_abs_r_error(cache, plot_dir)
        self.plot_abs_q(cache, plot_dir)
        self.plot_abs_yaw(cache, plot_dir)

        # self.plot_absolute_ypr_error(cache['timestamp_sec'], cache['abs_ypr_error'], plot_dir=plot_dir)
        # self.plot_yaw_error_per_meter(cache, plot_dir=plot_dir)
        # self.plot_relative_errors(cache, plot_dir=plot_dir)
        # self.plot_abs_r_error(cache['timestamp_sec'], cache['abs_r_error'], plot_dir=plot_dir)

    def plot_3d_traj(self, cache, plot_dir:Path):
        abs_t_est = cache['abs_t_est']
        abs_t_gt = cache['abs_t_gt']
        abs_t_est = abs_t_est + (abs_t_gt[0] - abs_t_est[0]).reshape((1, 3))
        xs_gt = abs_t_gt[:, 0].cpu()
        ys_gt = abs_t_gt[:, 1].cpu()
        zs_gt = abs_t_gt[:, 2].cpu()
        xs_est = abs_t_est[:, 0].cpu()
        ys_est = abs_t_est[:, 1].cpu()
        zs_est = abs_t_est[:, 2].cpu()

        fig = plt.figure(figsize=(8, 8), dpi=200)
        ax0 = fig.add_subplot(221, projection="3d")
        ax1 = fig.add_subplot(222)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(224)

        fontlabel = {"fontsize":"large", "color":"gray", "fontweight":"bold"}

        ax0.set_title('3D trajectory plot', fontdict=fontlabel)
        ax0.set_xlabel("X [m]", fontdict=fontlabel, labelpad=10)
        ax0.set_ylabel("Y [m]", fontdict=fontlabel, labelpad=10)
        ax0.set_zlabel("Z [m]", fontdict=fontlabel, labelpad=10)
        ax0.set_xlim(xs_est.min()-0.1, xs_est.max()+0.1)
        ax0.set_ylim(ys_est.min()-0.1, ys_est.max()+0.1)
        ax0.set_zlim(zs_est.min()-0.1, zs_est.max()+0.1)
        ax0.view_init(elev=30., azim=120)    # 각도 지정

        # Ref: https://stackoverflow.com/questions/40489378/matplotlib-how-to-efficiently-plot-a-large-number-of-line-segments-in-3d
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        lines_gt = torch.hstack([abs_t_gt[:-1], abs_t_gt[1:]]).cpu()
        lines_gt = lines_gt.reshape((-1, 2, 3))
        lines_collection_gt = Line3DCollection(lines_gt, linewidths=0.8, colors='k')
        ax0.add_collection(lines_collection_gt)

        lines_est = torch.hstack([abs_t_est[:-1], abs_t_est[1:]]).cpu()
        lines_est = lines_est.reshape((-1, 2, 3))
        lines_collection_est = Line3DCollection(lines_est, linewidths=0.8, colors='b')
        ax0.add_collection(lines_collection_est)

        from matplotlib.collections import LineCollection

        # Plot XY Plane :: Ref :: https://matplotlib.org/stable/gallery/shapes_and_collections/line_collection.html
        ax1.set_title("XY plane", fontdict=fontlabel)
        ax1.set_xlabel("X", fontdict=fontlabel, labelpad=10)
        ax1.set_ylabel("Y", fontdict=fontlabel, labelpad=10)
        ax1.set_xlim(xs_est.min()-0.1, xs_est.max()+0.1)
        ax1.set_ylim(ys_est.min()-0.1, ys_est.max()+0.1)
        xys_gt = torch.stack([xs_gt, ys_gt], dim=1)
        xys_est = torch.stack([xs_est, ys_est], dim=1)
        xy_lines_gt = torch.hstack([xys_gt[:-1], xys_gt[1:]]).reshape((-1, 2, 2)).numpy()
        xy_line_collection_gt = LineCollection(xy_lines_gt, linewidths=0.8, colors='k', linestyle='solid')
        xy_lines_est = torch.hstack([xys_est[:-1], xys_est[1:]]).reshape((-1, 2, 2)).numpy()
        xy_line_collection_est = LineCollection(xy_lines_est, linewidths=0.8, colors='b', linestyle='solid')
        ax1.add_collection(xy_line_collection_gt)
        ax1.add_collection(xy_line_collection_est)
        # Plot YZ Plane
        ax2.set_title("YZ plane", fontdict=fontlabel)
        ax2.set_xlabel("Y", fontdict=fontlabel, labelpad=10)
        ax2.set_ylabel("Z", fontdict=fontlabel, labelpad=10)
        ax2.set_xlim(ys_est.min()-0.1, ys_est.max()+0.1)
        ax2.set_ylim(zs_est.min()-0.1, zs_est.max()+0.1)
        yzs_gt = torch.stack([ys_gt, zs_gt], dim=1)
        yzs_est = torch.stack([ys_est, zs_est], dim=1)
        yz_lines_gt = torch.hstack([yzs_gt[:-1], yzs_gt[1:]]).reshape(-1, 2, 2).numpy()
        yz_line_collection_gt = LineCollection(yz_lines_gt, linewidths=0.8, colors='k', linestyle='solid')
        yz_lines_est = torch.hstack([yzs_est[:-1], yzs_est[1:]]).reshape(-1, 2, 2).numpy()
        yz_line_collection_est = LineCollection(yz_lines_est, linewidths=0.8, colors='b', linestyle='solid')
        ax2.add_collection(yz_line_collection_gt)
        ax2.add_collection(yz_line_collection_est)
        # Plot XZ Plane
        ax3.set_title("XZ plane", fontdict=fontlabel)
        ax3.set_xlabel("X", fontdict=fontlabel, labelpad=10)
        ax3.set_ylabel("Z", fontdict=fontlabel, labelpad=10)
        ax3.set_xlim(xs_est.min()-0.1, xs_est.max()+0.1)
        ax3.set_ylim(zs_est.min()-0.1, zs_est.max()+0.1)
        xzs_gt = torch.stack([xs_gt, zs_gt], dim=1)
        xzs_est = torch.stack([xs_est, zs_est], dim=1)
        xz_lines_gt = torch.hstack([xzs_gt[:-1], xzs_gt[1:]]).reshape(-1, 2, 2).numpy()
        xz_line_collection_gt = LineCollection(xz_lines_gt, linewidths=0.8, colors='k', linestyle='solid')
        xz_lines_est = torch.hstack([xzs_est[:-1], xzs_est[1:]]).reshape(-1, 2, 2).numpy()
        xz_line_collection_est = LineCollection(xz_lines_est, linewidths=0.8, colors='b', linestyle='solid')
        ax3.add_collection(xz_line_collection_gt)
        ax3.add_collection(xz_line_collection_est)

        fn = plot_dir / '3d_plot.png'
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)
        print(Fore.GREEN+'PLOT::%s'%fn)

        # Animate
        from matplotlib import animation

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6), dpi=150, subplot_kw={"projection":"3d"})
        # fig = plt.figure(figsize=(6, 6), dpi=150)
        # ax = fig.add_subplot(221, projection="3d")

        def init():
            ax.set_title('3D trajectory plot', fontdict=fontlabel)
            ax.set_xlabel("X [m]", fontdict=fontlabel, labelpad=10)
            ax.set_ylabel("Y [m]", fontdict=fontlabel, labelpad=10)
            ax.set_zlabel("Z [m]", fontdict=fontlabel, labelpad=10)
            ax.set_xlim(xs_est.min()-0.1, xs_est.max()+0.1)
            ax.set_ylim(ys_est.min()-0.1, ys_est.max()+0.1)
            ax.set_zlim(zs_est.min()-0.1, zs_est.max()+0.1)
            # ax.view_init(elev=30., azim=120)    # 각도 지정

            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            lines_gt = torch.hstack([abs_t_gt[:-1], abs_t_gt[1:]]).cpu()
            lines_gt = lines_gt.reshape((-1, 2, 3))
            lines_collection_gt = Line3DCollection(lines_gt, linewidths=0.5, colors='k')
            ax.add_collection(lines_collection_gt)
            lines_est = torch.hstack([abs_t_est[:-1], abs_t_est[1:]]).cpu()
            lines_est = lines_est.reshape((-1, 2, 3))
            lines_collection_est = Line3DCollection(lines_est, linewidths=0.5, colors='b')
            ax.add_collection(lines_collection_est)
            return fig

        def animate(i):
            ax.view_init(elev=30., azim=i)
            return fig

        animate_3d = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=360, interval=20)
        gif_fn = plot_dir / 'animate_3d.gif'
        animate_3d.save(gif_fn, fps=30)
        plt.close(fig)
        print(Fore.GREEN+'PLOT::%s'%gif_fn)

    def plot_abs_t(self, cache:dict, plot_dir:Path):
        time = cache['timestamp_sec']
        gt = cache['abs_t_gt']
        est = cache['abs_t_est']

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9), dpi=200)
        fig.suptitle('Absolute Translation (SE3 Alignment)', fontsize=18)

        axs[0].set_title('x-axis [m]', {'fontsize':16,'fontweight':2})
        axs[0].plot(time, gt.cpu().numpy()[:, 0],  'k', linewidth=1, label="GT")
        axs[0].plot(time, est.cpu().numpy()[:, 0],  'b', linewidth=1, label="est.")
        axs[0].legend(loc='best')

        axs[1].set_title('y-axis [m]', {'fontsize':16,'fontweight':2})
        axs[1].plot(time, gt.cpu().numpy()[:, 1],  'k', linewidth=1, label="GT")
        axs[1].plot(time, est.cpu().numpy()[:, 1],  'b', linewidth=1, label="est.")
        axs[1].legend(loc='best')

        axs[2].set_title('z-axis [m]', {'fontsize':16,'fontweight':2})
        axs[2].plot(time, gt.cpu().numpy()[:, 2],  'k', linewidth=1, label="GT")
        axs[2].plot(time, est.cpu().numpy()[:, 2],  'b', linewidth=1, label="est.")
        axs[2].legend(loc='best')

        fn = plot_dir / 'abs_t.png'
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

        est = est + (gt[0] - est[0]).reshape((1, 3))
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9), dpi=200)
        fig.suptitle('Absolute Translation (match initial point)', fontsize=18)

        axs[0].set_title('x-axis [m]', {'fontsize':16,'fontweight':2})
        axs[0].plot(time, gt.cpu().numpy()[:, 0],  'k', linewidth=1, label="GT")
        axs[0].plot(time, est.cpu().numpy()[:, 0],  'b', linewidth=1, label="est.ba")
        axs[0].legend(loc='best')

        axs[1].set_title('y-axis [m]', {'fontsize':16,'fontweight':2})
        axs[1].plot(time, gt.cpu().numpy()[:, 1],  'k', linewidth=1, label="GT")
        axs[1].plot(time, est.cpu().numpy()[:, 1],  'b', linewidth=1, label="est.ba")
        axs[1].legend(loc='best')

        axs[2].set_title('z-axis [m]', {'fontsize':16,'fontweight':2})
        axs[2].plot(time, gt.cpu().numpy()[:, 2],  'k', linewidth=1, label="GT")
        axs[2].plot(time, est.cpu().numpy()[:, 2],  'b', linewidth=1, label="est.ba")
        axs[2].legend(loc='best')

        fn = plot_dir / 'abs_t_init.png'
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

    def plot_abs_t_error(self, cache:dict, plot_dir:Path):
        time = cache['timestamp_sec']
        gt = cache['abs_t_gt']
        est = cache['abs_t_est']
        error = gt - est
        error_norm = torch.norm(error, dim=1)

        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(9, 10), dpi=200)
        fig.suptitle('Absolute Translation', fontsize=20)

        fontlabel = {'fontsize':14, "fontweight":"bold"}
        axs[0].set_title('x-axis [m]', fontlabel)
        axs[0].plot(time, error.cpu().numpy()[:, 0],  'k', linewidth=1.5, label="error")
        axs[0].legend(loc='best')

        axs[1].set_title('y-axis [m]', fontlabel)
        axs[1].plot(time, error.cpu().numpy()[:, 1],  'k', linewidth=1.5, label="error")
        axs[1].legend(loc='best')

        axs[2].set_title('z-axis [m]', fontlabel)
        axs[2].plot(time, error.cpu().numpy()[:, 2],  'k', linewidth=1.5, label="error")
        axs[2].legend(loc='best')

        axs[3].set_title('t error norm [m]', fontlabel)
        axs[3].plot(time, error_norm.cpu().numpy(),  'k', linewidth=1.5, label="error")
        axs[3].legend(loc='best')

        fn = plot_dir / 'abs_t_error.png'
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

    def plot_abs_r(self, cache:dict, plot_dir:Path):
        time = cache['timestamp_sec']
        est = torch.rad2deg(cache['abs_r_est'])
        gt = torch.rad2deg(cache['abs_r_gt'])

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9), dpi=200)
        fig.suptitle('Absolute Rotation Vector', fontsize=16)

        axs[0].set_ylabel('wx [deg]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[0].plot(time, est.cpu().numpy()[:, 0],  'b', linewidth=1, label="est[deg]")
        axs[0].plot(time, gt.cpu().numpy()[:, 0],  'k', linewidth=1, label="gt[deg]")
        axs[0].legend(loc='best')

        axs[1].set_ylabel('wy [deg]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[1].plot(time, est.cpu().numpy()[:, 1],  'b', linewidth=1, label="est[deg]")
        axs[1].plot(time, gt.cpu().numpy()[:, 1],  'k', linewidth=1, label="gt[deg]")
        axs[1].legend(loc='best')

        axs[2].set_ylabel('wz error [deg]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[2].plot(time, est.cpu().numpy()[:, 2],  'b', linewidth=1, label="est[deg]")
        axs[2].plot(time, gt.cpu().numpy()[:, 2],  'k', linewidth=1, label="gt[deg]")
        axs[2].legend(loc='best')

        fn = plot_dir / 'abs_r.png'
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

    def plot_abs_r_error(self, cache:dict, plot_dir:Path):
        time = cache['timestamp_sec']
        error = torch.rad2deg(cache['abs_r_error'])

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9), dpi=200)
        fig.suptitle('Absolute Rotation Vector', fontsize=16)

        axs[0].set_ylabel('wx [deg]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[0].plot(time, error.cpu().numpy()[:, 0],  'b', linewidth=1, label="err[deg]")
        axs[0].legend(loc='best')

        axs[1].set_ylabel('wy [deg]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[1].plot(time, error.cpu().numpy()[:, 1],  'b', linewidth=1, label="err[deg]")
        axs[1].legend(loc='best')

        axs[2].set_ylabel('wz error [deg]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[2].plot(time, error.cpu().numpy()[:, 2],  'b', linewidth=1, label="err[deg]")
        axs[2].legend(loc='best')

        fn = plot_dir / 'abs_r_error.png'
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

    def plot_abs_q(self, cache:dict, plot_dir:Path):
        time = cache['timestamp_sec']
        est = cache['abs_q_xyzw_est']
        gt = cache['abs_q_xyzw_gt']

        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(9, 12), dpi=200)
        fig.suptitle('Absolute quaternion', fontsize=16)

        est[est[:,3] < 0] = -est[est[:,3] < 0]
        gt[gt[:,3] < 0] = -gt[gt[:,3] < 0]

        # mask = (est[:, 3] * gt[:, 3]) < 0
        # est[mask] = -est[mask]
        # mask = torch.abs(est[:, 0] - gt[:, 0]) > 1.0
        # est[mask] = -est[mask]
        # mask = torch.abs(est[:, 1] - gt[:, 1]) > 1.0
        # est[mask] = -est[mask]
        # mask = torch.abs(est[:, 2] - gt[:, 2]) > 1.0
        # est[mask] = -est[mask]

        axs[0].set_ylabel('qx', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[0].plot(time, est.cpu().numpy()[:, 0],  'b', linewidth=1, label="est")
        axs[0].plot(time, gt.cpu().numpy()[:, 0],  'k', linewidth=1, label="gt")
        axs[0].legend(loc='best')

        axs[1].set_ylabel('qy', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[1].plot(time, est.cpu().numpy()[:, 1],  'b', linewidth=1, label="est")
        axs[1].plot(time, gt.cpu().numpy()[:, 1],  'k', linewidth=1, label="gt")
        axs[1].legend(loc='best')

        axs[2].set_ylabel('qz', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[2].plot(time, est.cpu().numpy()[:, 2],  'b', linewidth=1, label="est")
        axs[2].plot(time, gt.cpu().numpy()[:, 2],  'k', linewidth=1, label="gt")
        axs[2].legend(loc='best')

        axs[3].set_ylabel('qw', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[3].plot(time, est.cpu().numpy()[:, 3],  'b', linewidth=1, label="est")
        axs[3].plot(time, gt.cpu().numpy()[:, 3],  'k', linewidth=1, label="gt")
        axs[3].legend(loc='best')

        fn = plot_dir / 'abs_q_xyzw.png'
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

    def plot_abs_yaw(self, cache:dict, plot_dir:Path):
        time = cache['timestamp_sec']
        est = torch.rad2deg(cache['abs_ypr_est'][:, 0])
        gt = torch.rad2deg(cache['abs_ypr_gt'][:, 0])
        est = est + (gt[0] - est[0])

        fig, ax = plt.subplots(figsize=(9, 3), dpi=200)
        fig.suptitle('absolute yaw', fontsize=16)

        ax.set_ylabel('yaw [deg]', fontdict={'fontsize':10,'fontweight':'bold'})
        ax.plot(time, est.cpu().numpy(),  'b', linewidth=1, label="est.")
        ax.plot(time, gt.cpu().numpy(),  'k', linewidth=1, label="gt")
        ax.legend(loc='best')

        fn = plot_dir / 'abs_yaw.png'
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)


    def plot_absolute_ypr_error(self, time:torch.Tensor, error:torch.Tensor, plot_dir:Path):
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9), dpi=200)
        fig.suptitle('Absolute ypr error', fontsize=20)

        axs[0].set_ylabel('yaw error [deg]', fontdict={'fontsize':12,'fontweight':'bold'})
        axs[0].plot(time, error.cpu().numpy()[:, 0],  'k', linewidth=1, label="yaw_error[deg]")
        axs[0].legend(loc='best')

        axs[1].set_ylabel('pitch error [deg]', fontdict={'fontsize':12,'fontweight':'bold'})
        axs[1].plot(time, error.cpu().numpy()[:, 1],  'k', linewidth=1, label="pitch_error[deg]")
        axs[1].legend(loc='best')

        axs[2].set_ylabel('roll error [deg]', fontdict={'fontsize':12,'fontweight':'bold'})
        axs[2].plot(time, error.cpu().numpy()[:, 2],  'k', linewidth=1, label="roll_error[deg]")
        axs[2].legend(loc='best')
        axs[2].set_xlabel('Time [sec]', fontdict={'fontsize':12,'fontweight':'bold'})

        fn = plot_dir / 'absolute_ypr_error.png'
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)



    # def plot_abs_t_error_norm_percent(self, traveled:torch.Tensor, error:torch.Tensor, plot_dir:Path):
    #     fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 3), dpi=200)
    #     fig.suptitle('Absolute Translation Errors / Traveled distance', fontsize=14)

    #     axs.plot(traveled, error,  'k', linewidth=1, label="ate/dist")
    #     axs.legend(loc='best')
    #     axs.set_xlabel('Distance traveled [m]', fontdict={'fontsize':10,'fontweight':'bold'})
    #     axs.set_ylabel('Mean position error [%]', fontdict={'fontsize':10,'fontweight':'bold'})

    #     fn = plot_dir / 'abs_t_error_norm_percent.png'
    #     fig.tight_layout()
    #     fig.savefig(fn)
    #     plt.close(fig)

    # def plot_abs_yaw_error_norm_percent(self, traveled:torch.Tensor, error:torch.Tensor, plot_dir:Path):
    #     fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 3), dpi=200)
    #     fig.suptitle('Absolute Yaw Errors / Traveled', fontsize=14)

    #     axs.plot(traveled, error,  'k', linewidth=1)
    #     axs.legend(loc='best')
    #     axs.set_ylim(0.0, 1.5)
    #     axs.set_xlabel('Yaw traveled [deg]', fontdict={'fontsize':10,'fontweight':'bold'})
    #     axs.set_ylabel('Mean yaw error [%]', fontdict={'fontsize':10,'fontweight':'bold'})

    #     fn = plot_dir / 'abs_yaw_error_percent.png'
    #     fig.tight_layout()
    #     fig.savefig(fn)
    #     plt.close(fig)

    def plot_yaw_error_per_meter(self, cache:dict, plot_dir:Path):
        yaw_error_per_meter = cache['yaw_error_per_meter']
        yaw_error_per_meter_t = cache['yaw_error_per_meter_t']

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 3), dpi=200)
        fig.suptitle('Abs yaw error per meter[deg/m]', fontsize=16)

        axs.plot(yaw_error_per_meter_t, yaw_error_per_meter,  'k', linewidth=1, label="yaw error per meter [deg/m]")
        axs.legend(loc='best')
        axs.set_xlabel('Distance traveled [m]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs.set_ylabel('Mean yaw error [deg/m]', fontdict={'fontsize':10,'fontweight':'bold'})

        fn = plot_dir / 'yaw_error_per_meter_graph.png'
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

    def plot_relative_errors(self, time:torch.Tensor, t_error:torch.Tensor, r_error:torch.Tensor, plot_dir:Path):
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), dpi=200)
        fig.suptitle('Relative Translation Error', fontsize=16)

        axs[0].set_ylabel('tx error [m]', fontdict={'fontsize':12,'fontweight':'bold'})
        axs[0].plot(time, t_error.cpu().numpy()[:, 0],  'k', linewidth=1, label="GT")
        axs[0].legend(loc='best')

        axs[1].set_ylabel('ty error [m]', fontdict={'fontsize':12,'fontweight':'bold'})
        axs[1].plot(time, t_error.cpu().numpy()[:, 1],  'k', linewidth=1, label="GT")
        axs[1].legend(loc='best')

        axs[2].set_ylabel('tz error [m]', fontdict={'fontsize':12,'fontweight':'bold'})
        axs[2].plot(time, t_error.cpu().numpy()[:, 2],  'k', linewidth=1, label="GT")
        axs[2].legend(loc='best')
        axs[2].set_xlabel('Time [sec]', fontdict={'fontweight':'bold'})

        fn = plot_dir / 'rel_t_error.png'
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

if __name__ == '__main__':

    # eval = TrajectoryAnalyzer(root_dir='/data/RESULT/event_se3/desktop/uslam_event_imu/desktop_uslam_event_imu_boxes_6dof',
    #                           gt_abs_path='/data/RESULT/event_se3/desktop/uslam_event_imu/desktop_uslam_event_imu_boxes_6dof/stamped_groundtruth.txt')

    # eval = TrajectoryAnalyzer(single_dir='/data/RESULT/dynamic_6dof',root_dir=None,
    #                           gt_abs_path='/data/RESULT/dynamic_6dof/stamped_groundtruth.txt',
    #                           start=None, end=None, use_cache=False)

    # eval = TrajectoryAnalyzer(single_dir='/data/RESULT/mvsec_test',root_dir=None,
    #                           gt_abs_path='/data/RESULT/mvsec_test/stamped_groundtruth.txt',
    #                           start=0.0, end=25.0, use_cache=False)


    eval = TrajectoryAnalyzer( single_dir='/data/results/lvi_slam/indoor10',root_dir=None,
                              gt_abs_path='/data/results/lvi_slam/indoor10/groundtruth.txt',
                              start=0.0, end=999999999999999999.0, use_cache=False, do_align=True)
    eval = TrajectoryAnalyzer( single_dir='/data/results/lvi_slam/indoor10',root_dir=None,
                              gt_abs_path='/data/results/lvi_slam/indoor10/groundtruth.txt',
                              start=0.0, end=999999999999999999.0, use_cache=False, do_align=False)


    # def save_interpolated_gt_n_estimate(self, estimate:Trajectory, gt:Trajectory, save_dir:Path):
    #     est_ts = estimate.times.data
    #     est_t = estimate.poses.t_data.data
    #     est_R = estimate.poses.R_data.quat().data
    #     gt_ts = gt.times.data
    #     gt_t = gt.poses.t_data.data
    #     gt_R = gt.poses.R_data.quat().data

    #     t0 = torch.max(est_ts[0], gt_ts[0]).item()
    #     gt_i0 = torch.searchsorted(gt_ts, t0, right=True) - 1
    #     est_i0 = torch.searchsorted(est_ts, t0, right=True)
    #     assert (gt_ts[gt_i0].item() <= est_ts[est_i0].item())

    #     tn = torch.min(est_ts[-1], gt_ts[-1]).item()
    #     gt_in = torch.searchsorted(gt_ts, tn, right=False)
    #     est_in = torch.searchsorted(est_ts, tn, right=True) - 1
    #     assert (est_ts[est_in].item() <= gt_ts[gt_in].item())

    #     assert (gt_ts[gt_i0].item() <= gt_ts[gt_in].item())
    #     assert (est_ts[est_i0].item() <= est_ts[est_in].item())

    #     est_ts = est_ts[est_i0:est_in]
    #     est_t = est_t[est_i0:est_in, :]
    #     est_R = est_R[est_i0:est_in, :]
    #     gt_ts = gt_ts[gt_i0:gt_in]
    #     gt_t = gt_t[gt_i0:gt_in, :]
    #     gt_R = gt_R[gt_i0:gt_in, :]

    #     t0 = torch.max(est_ts[0], gt_ts[0]).item()
    #     tn = torch.min(est_ts[-1], gt_ts[-1]).item()

    #     idxs_left  = torch.searchsorted(gt_ts, est_ts, right=True) - 1
    #     idxs_right = torch.searchsorted(gt_ts, est_ts, right=True)

    #     ql = gt_R[idxs_left]
    #     qr = gt_R[idxs_right]
    #     tl = gt_t[idxs_left]
    #     tr = gt_t[idxs_right]

    #     dt = gt_ts[idxs_right] - gt_ts[idxs_left]
    #     tau = (est_ts - gt_ts[idxs_left]) / dt

    #     interpolated_q = Rotation.slerp(ql, qr, tau) # ok
    #     interpolated_t = torch.lerp(tl, tr, tau.unsqueeze(1)) # ok
    #     interpolated_time = est_ts.unsqueeze(1)

    #     est_t = est_t - est_t[0, :]
    #     est_t = est_t + interpolated_t[0, :]

    #     est_R = Rotation(est_R, param_type='quat', qtype='xyzw')
    #     gt_R = Rotation(interpolated_q, param_type='quat', qtype='xyzw')
    #     est_R = est_R.SO3()
    #     gt_R = gt_R.SO3()

    #     est_R = est_R - est_R.data[0]
    #     est_R = est_R + gt_R.data[0]
    #     est_q = est_R.quat().data

    #     header = '8 columns :: time[sec] tx ty tz qx qy qz qw'

    #     interpolated_gt = torch.cat((interpolated_time,
    #                                  interpolated_t,
    #                                  interpolated_q), dim=1).cpu().numpy()
    #     interpolated_gt_fn = save_dir / 'interpolated_gt.txt'
    #     np.savetxt(interpolated_gt_fn, interpolated_gt, fmt='%1.8f', header=header)

    #     interpolated_est = torch.cat((est_ts.unsqueeze(1),
    #                                   est_t,
    #                                   est_q), dim=1).cpu().numpy()
    #     interpolated_est_fn = save_dir / 'interpolated_estimate.txt'
    #     np.savetxt(interpolated_est_fn, interpolated_est, fmt='%1.8f', header=header)

    # def save_relative_estimate(self, save_dir:Path):
    #     interpolated_est_fn = save_dir / 'interpolated_estimate.txt'
    #     if interpolated_est_fn.exists():
    #         interpolated_est = np.genfromtxt(interpolated_est_fn)
    #         interpolated_est = torch.from_numpy(interpolated_est)
    #         if self.cudable:
    #             interpolated_est = interpolated_est.cuda()

    #         time = interpolated_est[:, 0]
    #         t = interpolated_est[:, 1:4]
    #         R = interpolated_est[:, 4:8]
    #         time = time[:-1].unsqueeze(1) # torch.lerp(time[:-1], time[1:], 0.5)
    #         dt = t[1:] - t[:-1]
    #         R = Rotation(R, 'quat', qtype='xyzw')
    #         R = R.SO3().data
    #         R = bmtm(R[:-1], R[1:])
    #         R = Rotation(R, 'SO3')
    #         dR = R.quat().data
    #         dr = R.so3().data

    #         dt_norm = dt.norm(dim=1).unsqueeze(1)
    #         dr_norm = dr.norm(dim=1).unsqueeze(1)

    #         header = '13 columns :: time[sec] dt[1x3] dt_norm dq[1x4] dr[1x3] dr_norm'
    #         relative_estimate = torch.cat((time, dt, dt_norm, dR, dr, dr_norm), dim=1).cpu().numpy()
    #         relative_estimate_fn = save_dir / 'relative_estimate.txt'
    #         np.savetxt(relative_estimate_fn, relative_estimate, fmt='%1.8f', header=header)

    #     else:
    #         raise AssertionError('interpolated data does\'t exists!')

    # def save_relative_gt(self, save_dir:Path):
    #     interpolated_gt_fn = save_dir / 'interpolated_gt.txt'
    #     if interpolated_gt_fn.exists():
    #         interpolated_gt = np.genfromtxt(interpolated_gt_fn)
    #         interpolated_gt = torch.from_numpy(interpolated_gt)
    #         if self.cudable:
    #             interpolated_gt = interpolated_gt.cuda()

    #         time = interpolated_gt[:, 0]
    #         t = interpolated_gt[:, 1:4]
    #         R = interpolated_gt[:, 4:8]
    #         time = time[:-1].unsqueeze(1) # torch.lerp(time[:-1], time[1:], 0.5)
    #         dt = t[1:] - t[:-1]
    #         R = Rotation(R, 'quat', qtype='xyzw')
    #         R = R.SO3().data
    #         R = bmtm(R[:-1], R[1:])
    #         R = Rotation(R, 'SO3')
    #         dR = R.quat().data
    #         dr = R.so3().data

    #         dt_norm = dt.norm(dim=1).unsqueeze(1)
    #         dr_norm = dr.norm(dim=1).unsqueeze(1)

    #         header = '13 columns :: time[sec] dt[1x3] dt_norm dq[1x4] dr[1x3] dr_norm'
    #         relative_gt = torch.cat((time, dt, dt_norm, dR, dr, dr_norm), dim=1).cpu().numpy()
    #         relative_gt_fn = save_dir / 'relative_gt.txt'
    #         np.savetxt(relative_gt_fn, relative_gt, fmt='%1.8f', header=header)
    #     else:
    #         raise AssertionError('interpolated data does\'t exists!')
