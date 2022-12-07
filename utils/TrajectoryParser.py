import numpy as np
import torch
from typing import Tuple
from pathlib import Path

import os
import sys
# print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Trajectory import Trajectory
from Rotation import Rotation
from torch_utils import bmtm


# The package for colored Command Line Interface (CLI)
from colorama import init as colorama_init
from colorama import Fore, Back, Style
colorama_init(autoreset=True)


class TrajectoryParser:
    cudable = torch.cuda.is_available()
    available_keys = [
        'timestamp_sec',
        'abs_t_est',
        'abs_t_gt',
        't_traveled_gt',
        'abs_q_xyzw_est',
        'abs_q_xyzw_gt',
        'abs_ypr_error',
        'yaw_traveled',
        'rel_timestamp_sec'
        'rel_t_est',
        'rel_t_gt',
        'rel_q_xyzw_est',
        'rel_q_xyzw_gt'
    ]

    @classmethod
    def parse_trajectory(cls, estimate:Trajectory, gt:Trajectory, cache_fn:Path) -> None:
        interpolated_est, interpolated_gt = cls.interpolate_data(estimate, gt)

        if cls.cudable:
            interpolated_est = interpolated_est.cuda()
            interpolated_gt = interpolated_gt.cuda()

        timestamp_sec = interpolated_est[:, 0]
        abs_t_est = interpolated_est[:, 1:4]
        abs_q_xyzw_est = interpolated_est[:, 4:8]
        abs_t_gt = interpolated_gt[:, 1:4]
        abs_q_xyzw_gt = interpolated_gt[:, 4:8]
        print("timestamp_sec:", timestamp_sec.shape)
        print("abs_t_est:", abs_t_est.shape)
        print("abs_q_xyzw_est:", abs_q_xyzw_est.shape)
        print("abs_t_gt:", abs_t_gt.shape)
        print("abs_q_xyzw_gt:", abs_q_xyzw_gt.shape)

        rel_timestamp_sec = timestamp_sec[:-1]
        rel_t_est, rel_q_xyzw_est = cls.relative_estimate(interpolated_est)
        rel_t_gt, rel_q_xyzw_gt = cls.relative_gt(interpolated_gt)
        print("rel_timestamp_sec:", rel_timestamp_sec.shape)
        print("rel_t_est:", rel_t_est.shape)
        print("rel_q_xyzw_est:", rel_q_xyzw_est.shape)
        print("rel_t_gt:", rel_t_gt.shape)
        print("rel_q_xyzw_gt:", rel_q_xyzw_gt.shape)

        rel_t_norm_gt = rel_t_gt.norm(dim=1)
        t_traveled_gt = torch.zeros_like(timestamp_sec)
        traveled = 0.0
        for i in range(t_traveled_gt.size(0)):
            t_traveled_gt[i] = traveled
            assert (t_traveled_gt[i-1] <= t_traveled_gt[i])

            if i == (t_traveled_gt.size(0)-1):
                break
            else:
                traveled += rel_t_norm_gt[i].item()
        print("t_traveled_gt:", t_traveled_gt.shape)

        R_est = Rotation(abs_q_xyzw_est, param_type='quat', qtype='xyzw').SO3()
        R_gt = Rotation(abs_q_xyzw_gt, param_type='quat', qtype='xyzw').SO3()
        abs_R_error = R_gt - R_est
        abs_ypr_error = abs_R_error.SO3_to_euler_rzyx()
        abs_ypr_error = torch.rad2deg(abs_ypr_error)
        print(abs_ypr_error.shape)

        dR_gt = Rotation(rel_q_xyzw_gt, param_type='quat', qtype='xyzw').SO3()
        dyaw = dR_gt.SO3_to_euler_rzyx()[:, 0]
        print(dyaw.shape)

        yaw_traveled = torch.zeros_like(timestamp_sec)
        traveled = 0.0
        for i in range(yaw_traveled.size(0)):
            yaw_traveled[i] = traveled
            assert (yaw_traveled[i-1] <= yaw_traveled[i])

            if i == (yaw_traveled.size(0)-1):
                break
            else:
                traveled += abs(dyaw[i].item())
        print("yaw_traveled:", yaw_traveled.shape)


        # self.result_dir:Path = result_dir
        # self.gt_abs_path:Path = gt_abs_path
        # self.use_cache = use_cache
        # self.cudable = torch.cuda.is_available()



    def __init__(self, result_dir:Path, gt_abs_path:Path, use_cache=False):
        assert (result_dir.exists())
        assert (gt_abs_path.exists())

        self.cache_fn:Path = result_dir / 'cache_dict.pt'

        if self.cache_fn.exists():
            return torch.load(self.cache_fn)

        self.result_dir:Path = result_dir
        self.gt_abs_path:Path = gt_abs_path
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
            inter_est_path         = dir / 'interpolated_estimate.txt'
            inter_gt_path          = dir / 'interpolated_gt.txt'
            relative_estimate_path = dir / 'relative_estimate.txt'
            relative_gt_path       = dir / 'relative_gt.txt'

            if not estimate_path.exists():
                raise AssertionError(Fore.RED+'%s doesn\'t exists!'%estimate_path.name)

            if not inter_est_path.exists() or not inter_gt_path.exists() or not use_cache:
                estimate = Trajectory(use_file=True, trajectory_file_abs_path=estimate_path)
                self.save_interpolated_gt_n_estimate(estimate, self.gt, dir)
            else:
                print('Cached %s,%s is checked!' % (inter_est_path.name, inter_gt_path.name))

            if not relative_estimate_path.exists() or not use_cache:
                self.save_relative_estimate(dir)
            else:
                print('Cached %s is checked!' % (relative_estimate_path.name))

            if not relative_gt_path.exists() or not use_cache:
                self.save_relative_gt(dir)
            else:
                print('Cached %s is checked!' % (relative_gt_path.name))

            print()

        self.abs_error_dict = {}
        self.rel_error_dict = {}

    @classmethod
    def interpolate_data(cls, estimate:Trajectory, gt:Trajectory):
        est_ts = estimate.times.data
        est_t = estimate.poses.t_data.data
        est_R = estimate.poses.R_data.quat().data
        gt_ts = gt.times.data
        gt_t = gt.poses.t_data.data
        gt_R = gt.poses.R_data.quat().data

        t0 = torch.max(est_ts[0], gt_ts[0]).item()
        gt_i0 = torch.searchsorted(gt_ts, t0, right=True) - 1
        est_i0 = torch.searchsorted(est_ts, t0, right=True)
        assert (gt_ts[gt_i0].item() <= est_ts[est_i0].item())

        tn = torch.min(est_ts[-1], gt_ts[-1]).item()
        gt_in = torch.searchsorted(gt_ts, tn, right=False)
        est_in = torch.searchsorted(est_ts, tn, right=True) - 1
        assert (est_ts[est_in].item() <= gt_ts[gt_in].item())

        assert (gt_ts[gt_i0].item() <= gt_ts[gt_in].item())
        assert (est_ts[est_i0].item() <= est_ts[est_in].item())

        est_ts = est_ts[est_i0:est_in]
        est_t = est_t[est_i0:est_in, :]
        est_R = est_R[est_i0:est_in, :]
        gt_ts = gt_ts[gt_i0:gt_in]
        gt_t = gt_t[gt_i0:gt_in, :]
        gt_R = gt_R[gt_i0:gt_in, :]

        t0 = torch.max(est_ts[0], gt_ts[0]).item()
        tn = torch.min(est_ts[-1], gt_ts[-1]).item()

        idxs_left  = torch.searchsorted(gt_ts, est_ts, right=True) - 1
        idxs_right = torch.searchsorted(gt_ts, est_ts, right=True)

        ql = gt_R[idxs_left]
        qr = gt_R[idxs_right]
        tl = gt_t[idxs_left]
        tr = gt_t[idxs_right]

        dt = gt_ts[idxs_right] - gt_ts[idxs_left]
        tau = (est_ts - gt_ts[idxs_left]) / dt

        interpolated_q = Rotation.slerp(ql, qr, tau) # ok
        interpolated_t = torch.lerp(tl, tr, tau.unsqueeze(1)) # ok
        interpolated_time = est_ts.unsqueeze(1)

        est_t = est_t - est_t[0, :]
        est_t = est_t + interpolated_t[0, :]

        est_R = Rotation(est_R, param_type='quat', qtype='xyzw')
        gt_R = Rotation(interpolated_q, param_type='quat', qtype='xyzw')
        est_R = est_R.SO3()
        gt_R = gt_R.SO3()

        est_R = est_R - est_R.data[0]
        est_R = est_R + gt_R.data[0]
        est_q = est_R.quat().data

        interpolated_gt = torch.cat((interpolated_time,
                                     interpolated_t,
                                     interpolated_q), dim=1)
        interpolated_est = torch.cat((est_ts.unsqueeze(1),
                                      est_t,
                                      est_q), dim=1)
        return interpolated_est, interpolated_gt



    def save_interpolated_gt_n_estimate(self, estimate:Trajectory, gt:Trajectory, save_dir:Path):
        est_ts = estimate.times.data
        est_t = estimate.poses.t_data.data
        est_R = estimate.poses.R_data.quat().data
        gt_ts = gt.times.data
        gt_t = gt.poses.t_data.data
        gt_R = gt.poses.R_data.quat().data

        t0 = torch.max(est_ts[0], gt_ts[0]).item()
        gt_i0 = torch.searchsorted(gt_ts, t0, right=True) - 1
        est_i0 = torch.searchsorted(est_ts, t0, right=True)
        assert (gt_ts[gt_i0].item() <= est_ts[est_i0].item())

        tn = torch.min(est_ts[-1], gt_ts[-1]).item()
        gt_in = torch.searchsorted(gt_ts, tn, right=False)
        est_in = torch.searchsorted(est_ts, tn, right=True) - 1
        assert (est_ts[est_in].item() <= gt_ts[gt_in].item())

        assert (gt_ts[gt_i0].item() <= gt_ts[gt_in].item())
        assert (est_ts[est_i0].item() <= est_ts[est_in].item())

        est_ts = est_ts[est_i0:est_in]
        est_t = est_t[est_i0:est_in, :]
        est_R = est_R[est_i0:est_in, :]
        gt_ts = gt_ts[gt_i0:gt_in]
        gt_t = gt_t[gt_i0:gt_in, :]
        gt_R = gt_R[gt_i0:gt_in, :]

        t0 = torch.max(est_ts[0], gt_ts[0]).item()
        tn = torch.min(est_ts[-1], gt_ts[-1]).item()

        idxs_left  = torch.searchsorted(gt_ts, est_ts, right=True) - 1
        idxs_right = torch.searchsorted(gt_ts, est_ts, right=True)

        ql = gt_R[idxs_left]
        qr = gt_R[idxs_right]
        tl = gt_t[idxs_left]
        tr = gt_t[idxs_right]

        dt = gt_ts[idxs_right] - gt_ts[idxs_left]
        tau = (est_ts - gt_ts[idxs_left]) / dt

        interpolated_q = Rotation.slerp(ql, qr, tau) # ok
        interpolated_t = torch.lerp(tl, tr, tau.unsqueeze(1)) # ok
        interpolated_time = est_ts.unsqueeze(1)

        est_t = est_t - est_t[0, :]
        est_t = est_t + interpolated_t[0, :]

        est_R = Rotation(est_R, param_type='quat', qtype='xyzw')
        gt_R = Rotation(interpolated_q, param_type='quat', qtype='xyzw')
        est_R = est_R.SO3()
        gt_R = gt_R.SO3()

        est_R = est_R - est_R.data[0]
        est_R = est_R + gt_R.data[0]
        est_q = est_R.quat().data

        header = '8 columns :: time[sec] tx ty tz qx qy qz qw'

        interpolated_gt = torch.cat((interpolated_time,
                                     interpolated_t,
                                     interpolated_q), dim=1).cpu().numpy()
        interpolated_gt_fn = save_dir / 'interpolated_gt.txt'
        np.savetxt(interpolated_gt_fn, interpolated_gt, fmt='%1.8f', header=header)

        interpolated_est = torch.cat((est_ts.unsqueeze(1),
                                      est_t,
                                      est_q), dim=1).cpu().numpy()
        interpolated_est_fn = save_dir / 'interpolated_estimate.txt'
        np.savetxt(interpolated_est_fn, interpolated_est, fmt='%1.8f', header=header)

    @classmethod
    def relative_estimate(cls, interpolated_est:torch.Tensor):
        if cls.cudable:
            interpolated_est = interpolated_est.cuda()

        t = interpolated_est[:, 1:4]
        R = interpolated_est[:, 4:8]
        rel_t_est = t[1:] - t[:-1]
        R = Rotation(R, 'quat', qtype='xyzw')
        R = R.SO3().data
        R = bmtm(R[:-1], R[1:])
        R = Rotation(R, 'SO3')
        rel_q_xyzw_est = R.quat().data
        return rel_t_est, rel_q_xyzw_est

    @classmethod
    def relative_gt(cls, interpolated_gt:torch.Tensor):
        if cls.cudable:
            interpolated_gt = interpolated_gt.cuda()

        t = interpolated_gt[:, 1:4]
        R = interpolated_gt[:, 4:8]
        rel_t_gt = t[1:] - t[:-1]
        R = Rotation(R, 'quat', qtype='xyzw')
        R = R.SO3().data
        R = bmtm(R[:-1], R[1:])
        R = Rotation(R, 'SO3')
        rel_q_xyzw_gt = R.quat().data
        return rel_t_gt, rel_q_xyzw_gt


if __name__ == '__main__':
    TrajectoryParser.parse_trajectory("", "")
