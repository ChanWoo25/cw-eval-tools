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

    def __init__(self, root_dir:str=None, gt_abs_path:str=None, estimate_dir_prefix:str='event_imu_', use_cache=False):
        assert (root_dir is not None and os.path.exists(root_dir)), Fore.RED+'You need to specify root directory containing estimates and ground-truth files'
        assert (gt_abs_path is not None and os.path.exists(gt_abs_path)), Fore.RED+'You need to specify an absolute path for proper GT file'

        self.root_dir = Path(root_dir)
        self.gt_abs_path = Path(gt_abs_path)
        self.use_cache = use_cache
        self.cudable = torch.cuda.is_available()

        self.dirs = list(self.root_dir.glob(estimate_dir_prefix+'*'))
        self.dirs.sort()

        print(Fore.GREEN+'Registered subdirs:')
        for dir in self.dirs:
            print('- %s'%dir)
        print('+ %s -- [GT]\n'%gt_abs_path)

        self.gt = Trajectory(use_file=True, trajectory_file_abs_path=self.gt_abs_path)
        print()

        for dir in self.dirs:
            print(Fore.GREEN+'%s Processing ... ' % dir.name)
            estimate_path          = dir / 'estimate.txt'
            cache_path             = dir / 'cache_dict.pt'
            plot_dir               = dir / 'plots'

            if not estimate_path.exists():
                raise AssertionError(Fore.RED+'%s doesn\'t exists!'%estimate_path.name)

            if not cache_path.exists() or not use_cache:
                estimate = Trajectory(use_file=True, trajectory_file_abs_path=estimate_path)
                TrajectoryParser.parse_trajectory(estimate, self.gt, cache_path)
            else:
                print(Fore.GREEN+Style.DIM+'[info] -- %s file exists in %s' % (cache_path.name, dir.name))

            cache:dict = torch.load(cache_path)
            print(cache.keys())

            ######################### Dict Keys #########################################
            # timestamp_sec             # abs_ypr_error            # rel_timestamp_sec  #
            # abs_t_est                 # abs_yaw_error            # rel_t_est          #
            # abs_t_gt                  # yaw_traveled             # rel_q_xyzw_est     #
            # abs_t_error               # abs_yaw_error_percent    # rel_t_gt           #
            # abs_t_error_norm          # yaw_error_per_meter      # rel_q_xyzw_gt      #
            # abs_t_error_norm_percent  # abs_yaw_error_per_meter  # rel_t_error        #
            # t_traveled_gt             # abs_q_xyzw_est           # rel_t_error_norm   #
            # abs_r_est                 # abs_q_xyzw_gt            # rel_r_error        #
            # abs_r_gt                                             # rel_r_error_norm   #
            #############################################################################
            if not plot_dir.exists():
                os.mkdir(plot_dir)

            self.plot_absolute_t_error(cache['timestamp_sec'], cache['abs_t_est'], cache['abs_t_gt'], plot_dir=plot_dir)
            self.plot_absolute_r(cache['timestamp_sec'], cache['abs_r_est'], cache['abs_r_gt'], plot_dir=plot_dir)
            self.plot_absolute_q(cache['timestamp_sec'], cache['abs_q_xyzw_est'], cache['abs_q_xyzw_gt'], plot_dir=plot_dir)
            self.plot_abs_t_error_norm_percent(cache['t_traveled_gt'], cache['abs_t_error_norm_percent'], plot_dir=plot_dir)
            self.plot_absolute_ypr_error(cache['timestamp_sec'], cache['abs_ypr_error'], plot_dir=plot_dir)
            self.plot_abs_yaw_error_norm_percent(cache['yaw_traveled'], cache['abs_yaw_error_percent'], plot_dir=plot_dir)
            self.plot_abs_yaw_error_per_meter(cache['t_traveled_gt'], cache['abs_yaw_error'], plot_dir=plot_dir)
            self.plot_relative_errors(cache['rel_timestamp_sec'], cache['rel_t_error'], cache['rel_r_error'], plot_dir=plot_dir)
            print()

        self.abs_error_dict = {}
        self.rel_error_dict = {}

    def plot_absolute_q(self, time:torch.Tensor, est:torch.Tensor, gt:torch.Tensor, plot_dir:Path):
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 10), dpi=200)
        fig.suptitle('Absolute quaternion', fontsize=16)

        mask = (est[:, 3] * gt[:, 3]) < 0
        est[mask] = -est[mask]
        mask = torch.abs(est[:, 0] - gt[:, 0]) > 1.0
        est[mask] = -est[mask]
        mask = torch.abs(est[:, 1] - gt[:, 1]) > 1.0
        est[mask] = -est[mask]
        mask = torch.abs(est[:, 2] - gt[:, 2]) > 1.0
        est[mask] = -est[mask]

        axs[0].set_ylim(-1, 1.0)
        axs[0].set_ylabel('qx', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[0].plot(time, est.cpu().numpy()[:, 0],  'b', linewidth=1, label="est[deg]")
        axs[0].plot(time, gt.cpu().numpy()[:, 0],  'k', linewidth=1, label="gt[deg]")
        axs[0].legend(loc='best')

        axs[1].set_ylim(-1, 1.0)
        axs[1].set_ylabel('qy', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[1].plot(time, est.cpu().numpy()[:, 1],  'b', linewidth=1, label="est[deg]")
        axs[1].plot(time, gt.cpu().numpy()[:, 1],  'k', linewidth=1, label="gt[deg]")
        axs[1].legend(loc='best')

        axs[2].set_ylim(-1, 1.0)
        axs[2].set_ylabel('qz', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[2].plot(time, est.cpu().numpy()[:, 2],  'b', linewidth=1, label="est[deg]")
        axs[2].plot(time, gt.cpu().numpy()[:, 2],  'k', linewidth=1, label="gt[deg]")
        axs[2].legend(loc='best')

        axs[3].set_ylim(-1, 1.0)
        axs[3].set_ylabel('qw', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[3].plot(time, est.cpu().numpy()[:, 3],  'b', linewidth=1, label="est[deg]")
        axs[3].plot(time, gt.cpu().numpy()[:, 3],  'k', linewidth=1, label="gt[deg]")
        axs[3].legend(loc='best')

        fn = plot_dir / 'absolute_q.png'
        plt.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

    def plot_absolute_r(self, time:torch.Tensor, est:torch.Tensor, gt:torch.Tensor, plot_dir:Path):
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9), dpi=200)
        fig.suptitle('Absolute Rotation Vector', fontsize=16)

        axs[0].set_ylim(-180.0, 180.0)
        axs[0].set_ylabel('wx [deg]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[0].plot(time, est.cpu().numpy()[:, 0],  'b', linewidth=1, label="est[deg]")
        axs[0].plot(time, gt.cpu().numpy()[:, 0],  'k', linewidth=1, label="gt[deg]")
        axs[0].legend(loc='best')

        axs[1].set_ylim(-180.0, 180.0)
        axs[1].set_ylabel('wy [deg]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[1].plot(time, est.cpu().numpy()[:, 1],  'b', linewidth=1, label="est[deg]")
        axs[1].plot(time, gt.cpu().numpy()[:, 1],  'k', linewidth=1, label="gt[deg]")
        axs[1].legend(loc='best')

        axs[2].set_ylim(-180.0, 180.0)
        axs[2].set_ylabel('wz error [deg]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs[2].plot(time, est.cpu().numpy()[:, 2],  'b', linewidth=1, label="est[deg]")
        axs[2].plot(time, gt.cpu().numpy()[:, 2],  'k', linewidth=1, label="gt[deg]")
        axs[2].legend(loc='best')

        fn = plot_dir / 'absolute_r.png'
        plt.tight_layout()
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
        plt.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

    def plot_absolute_t_error(self, time:torch.Tensor, est:torch.Tensor, gt:torch.Tensor, plot_dir:Path):
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9), dpi=200)
        fig.suptitle('Absolute Translation', fontsize=20)

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

        fn = plot_dir / 'absolute_t_error.png'
        plt.tight_layout()
        fig.savefig(fn)
        plt.close(fig)


    def plot_abs_t_error_norm_percent(self, traveled:torch.Tensor, error:torch.Tensor, plot_dir:Path):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 3), dpi=200)
        fig.suptitle('Absolute Translation Errors / Traveled distance', fontsize=14)

        axs.plot(traveled, error,  'k', linewidth=1, label="ate/dist")
        axs.legend(loc='best')
        axs.set_xlabel('Distance traveled [m]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs.set_ylabel('Mean position error [%]', fontdict={'fontsize':10,'fontweight':'bold'})

        fn = plot_dir / 'abs_t_error_norm_percent.png'
        plt.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

    def plot_abs_yaw_error_norm_percent(self, traveled:torch.Tensor, error:torch.Tensor, plot_dir:Path):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 3), dpi=200)
        fig.suptitle('Absolute Yaw Errors / Traveled', fontsize=14)

        axs.plot(traveled, error,  'k', linewidth=1)
        axs.legend(loc='best')
        axs.set_ylim(0.0, 1.5)
        axs.set_xlabel('Yaw traveled [deg]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs.set_ylabel('Mean yaw error [%]', fontdict={'fontsize':10,'fontweight':'bold'})

        fn = plot_dir / 'abs_yaw_error_percent.png'
        plt.tight_layout()
        fig.savefig(fn)
        plt.close(fig)


    def plot_abs_yaw_error_per_meter(self, traveled:torch.Tensor, error:torch.Tensor, plot_dir:Path):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 3), dpi=200)
        fig.suptitle('Absolute Yaw Errors / Traveled distance', fontsize=16)

        axs.plot(traveled, error,  'k', linewidth=1, label="yaw error [deg]")
        axs.legend(loc='best')
        axs.set_xlabel('Distance traveled [m]', fontdict={'fontsize':10,'fontweight':'bold'})
        axs.set_ylabel('Mean yaw error [deg/m]', fontdict={'fontsize':10,'fontweight':'bold'})

        fn = plot_dir / 'abs_yaw_error_per_meter.png'
        plt.tight_layout()
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
        plt.tight_layout()
        fig.savefig(fn)
        plt.close(fig)

if __name__ == '__main__':

    eval = TrajectoryAnalyzer(root_dir='/data/RESULT/event_se3/desktop/uslam_event_imu/desktop_uslam_event_imu_boxes_6dof',
                              gt_abs_path='/data/RESULT/event_se3/desktop/uslam_event_imu/desktop_uslam_event_imu_boxes_6dof/stamped_groundtruth.txt')

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
