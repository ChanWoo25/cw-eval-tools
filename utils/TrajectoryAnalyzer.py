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

    def __init__(self, root_dir:str=None, gt_abs_path:str=None, estimate_dir_prefix:str='result_', use_cache=True):
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

    def save_relative_estimate(self, save_dir:Path):
        interpolated_est_fn = save_dir / 'interpolated_estimate.txt'
        if interpolated_est_fn.exists():
            interpolated_est = np.genfromtxt(interpolated_est_fn)
            interpolated_est = torch.from_numpy(interpolated_est)
            if self.cudable:
                interpolated_est = interpolated_est.cuda()

            time = interpolated_est[:, 0]
            t = interpolated_est[:, 1:4]
            R = interpolated_est[:, 4:8]
            time = time[:-1].unsqueeze(1) # torch.lerp(time[:-1], time[1:], 0.5)
            dt = t[1:] - t[:-1]
            R = Rotation(R, 'quat', qtype='xyzw')
            R = R.SO3().data
            R = bmtm(R[:-1], R[1:])
            R = Rotation(R, 'SO3')
            dR = R.quat().data
            dr = R.so3().data

            dt_norm = dt.norm(dim=1).unsqueeze(1)
            dr_norm = dr.norm(dim=1).unsqueeze(1)

            header = '13 columns :: time[sec] dt[1x3] dt_norm dq[1x4] dr[1x3] dr_norm'
            relative_estimate = torch.cat((time, dt, dt_norm, dR, dr, dr_norm), dim=1).cpu().numpy()
            relative_estimate_fn = save_dir / 'relative_estimate.txt'
            np.savetxt(relative_estimate_fn, relative_estimate, fmt='%1.8f', header=header)

        else:
            raise AssertionError('interpolated data does\'t exists!')

    def save_relative_gt(self, save_dir:Path):
        interpolated_gt_fn = save_dir / 'interpolated_gt.txt'
        if interpolated_gt_fn.exists():
            interpolated_gt = np.genfromtxt(interpolated_gt_fn)
            interpolated_gt = torch.from_numpy(interpolated_gt)
            if self.cudable:
                interpolated_gt = interpolated_gt.cuda()

            time = interpolated_gt[:, 0]
            t = interpolated_gt[:, 1:4]
            R = interpolated_gt[:, 4:8]
            time = time[:-1].unsqueeze(1) # torch.lerp(time[:-1], time[1:], 0.5)
            dt = t[1:] - t[:-1]
            R = Rotation(R, 'quat', qtype='xyzw')
            R = R.SO3().data
            R = bmtm(R[:-1], R[1:])
            R = Rotation(R, 'SO3')
            dR = R.quat().data
            dr = R.so3().data

            dt_norm = dt.norm(dim=1).unsqueeze(1)
            dr_norm = dr.norm(dim=1).unsqueeze(1)

            header = '13 columns :: time[sec] dt[1x3] dt_norm dq[1x4] dr[1x3] dr_norm'
            relative_gt = torch.cat((time, dt, dt_norm, dR, dr, dr_norm), dim=1).cpu().numpy()
            relative_gt_fn = save_dir / 'relative_gt.txt'
            np.savetxt(relative_gt_fn, relative_gt, fmt='%1.8f', header=header)
        else:
            raise AssertionError('interpolated data does\'t exists!')


    def SE3(self):
        raise NotImplementedError

    def se3(self):
        raise NotImplementedError

if __name__ == '__main__':

    eval = TrajectoryAnalyzer(root_dir='/root/eslam/namu_trajectory_evaluation/test',
                              gt_abs_path='/root/eslam/namu_trajectory_evaluation/test/stamped_groundtruth.txt')
