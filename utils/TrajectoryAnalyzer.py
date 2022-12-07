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

    def __init__(self, root_dir:str=None, gt_abs_path:str=None, estimate_dir_prefix:str='result_', use_cache=False):
        assert (root_dir is not None and os.path.exists(root_dir)), Fore.RED+'You need to specify root directory containing estimates and ground-truth files'
        assert (gt_abs_path is not None and os.path.exists(gt_abs_path)), Fore.RED+'You need to specify an absolute path for proper GT file'

        self.root_dir = Path(root_dir)
        self.gt_abs_path = Path(gt_abs_path)
        self.use_cache = use_cache

        self.dirs = list(self.root_dir.glob(estimate_dir_prefix+'*'))
        self.dirs.sort()

        print(Fore.GREEN+'Registered subdirs:')
        for dir in self.dirs:
            print('- %s'%dir)
        print('+ %s -- [GT]\n'%gt_abs_path)

        self.gt = Trajectory(use_file=True, trajectory_file_abs_path=self.gt_abs_path)

        for dir in self.dirs:
            print(Fore.GREEN+'%s Processing ... ' % dir.name)
            estimate_path          = dir / 'estimate.txt'
            inter_est_path         = dir / 'interpolated_estimate.txt'
            inter_gt_path          = dir / 'interpolated_gt.txt'
            relative_estimate_path = dir / 'relative_estimate.txt'
            relative_gt_path       = dir / 'relative_gt.txt'

            if not estimate_path.exists():
                raise AssertionError(Fore.RED+'%s doesn\'t exists!'%estimate_path.name)

            estimate = Trajectory(use_file=True, trajectory_file_abs_path=estimate_path)

            if not inter_est_path.exists() or not inter_gt_path.exists() or not use_cache:
                self.save_interpolated_gt_n_estimate(estimate, self.gt, dir)
            if not relative_estimate_path.exists() or not use_cache:
                pass
            if not relative_gt_path.exists() or not use_cache:
                pass

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





    def save_relative_estimate(self, estimate:Trajectory):
        raise NotImplementedError()

    def save_relative_gt(self, interpolated_gt:Trajectory):
        raise NotImplementedError()

    def interpolate_by(self, t_int:'Time') -> 'Trajectory':
        """
            Interpolation is performed by 'in-place' way,
            according to the time vector given by t_int.
            You need to sure self.t and t_int have same unit base.
        """
        # assert (self.)
        raise NotImplementedError()

    def __str__(self) -> str:
        message = 'Print only first 10 lines.'

    @property
    def length(self):
        assert (self.times.length == self.poses.length)
        return self.times.length

    def SE3(self):
        raise NotImplementedError

    def se3(self):
        raise NotImplementedError

if __name__ == '__main__':

    eval = TrajectoryAnalyzer(root_dir='/root/eslam/namu_trajectory_evaluation/test',
                              gt_abs_path='/root/eslam/namu_trajectory_evaluation/test/stamped_groundtruth.txt')
