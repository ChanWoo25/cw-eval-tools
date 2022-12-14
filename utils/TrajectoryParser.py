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
from TrajectoryAlign import TrajectoryAlign

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

        # Compute on GPU if possible.
        if cls.cudable:
            interpolated_est = interpolated_est.cuda()
            interpolated_gt = interpolated_gt.cuda()

        # Parse basic
        timestamp_sec = interpolated_est[:, 0]
        timestamp_sec = timestamp_sec - timestamp_sec[0] #cache#
        abs_t_est = interpolated_est[:, 1:4]             #cache#
        abs_q_xyzw_est = interpolated_est[:, 4:8]        #cache#
        abs_t_gt = interpolated_gt[:, 1:4]               #cache#
        abs_q_xyzw_gt = interpolated_gt[:, 4:8]          #cache#

        # Do SE3 Alignment
        align_R, align_t = TrajectoryAlign.alignSE3(abs_t_est, abs_t_gt)
        if cls.cudable:
            align_R = align_R.cuda()
            align_t = align_t.cuda()
        abs_t_est = torch.matmul(abs_t_est, align_R.transpose(0, 1)) + align_t.expand_as(abs_t_est)

        # Compute absolute translation & rotation error
        abs_q_xyzw_est = Rotation(abs_q_xyzw_est, param_type='quat', qtype='xyzw')
        abs_q_xyzw_gt = Rotation(abs_q_xyzw_gt, param_type='quat', qtype='xyzw')
        abs_R_est = abs_q_xyzw_est.clone().SO3()
        abs_R_gt = abs_q_xyzw_gt.clone().SO3()
        abs_r_est = abs_R_est.clone().so3()            #cache#
        abs_r_gt = abs_R_gt.clone().so3()              #cache#
        abs_ypr_est = abs_R_est.SO3_to_euler_rzyx()    #cache#
        abs_ypr_gt = abs_R_gt.SO3_to_euler_rzyx()      #cache#

        abs_t_error = abs_t_gt - abs_t_est             #cache#
        abs_R_error = abs_R_gt - abs_R_est
        abs_r_error = abs_R_error.clone().so3()        #cache#
        abs_q_xyzw_error = abs_R_error.clone().quat()  #cache#
        abs_ypr_error = abs_ypr_gt - abs_ypr_est
        abs_ypr_error = torch.rad2deg(abs_ypr_error)
        abs_ypr_error[abs_ypr_error>=180.0] -= 360.0
        abs_ypr_error[abs_ypr_error<180.0] += 360.0  #cache#

        # Compute relative errors
        rel_t_est = abs_t_est[1:] - abs_t_est[:-1]   #cache#
        rel_t_gt  = abs_t_gt[1:] - abs_t_gt[:-1]     #cache#
        rel_t_error = rel_t_gt - rel_t_est           #cache#
        rel_R_est = bmtm(abs_R_est.data[:-1], abs_R_est.data[1:])
        rel_R_est = Rotation(rel_R_est, param_type='SO3')
        rel_R_gt = bmtm(abs_R_gt.data[:-1], abs_R_gt.data[1:])
        rel_R_gt = Rotation(rel_R_gt, param_type='SO3')
        rel_R_error = rel_R_gt - rel_R_est
        rel_yaw_est = abs_ypr_est[1:,0] - abs_ypr_est[:-1,0]
        rel_yaw_gt  = abs_ypr_gt[1:,0]  - abs_ypr_gt[:-1,0]

        # Compute total translation distance
        t_traveled_gt = torch.zeros_like(timestamp_sec, dtype=torch.float64)
        for i in range(rel_t_gt.size(0)):
            t_traveled_gt[i+1] = t_traveled_gt[i] + torch.norm(rel_t_gt[i])

        # Compute total yaw distance
        rel_yaw_gt = torch.abs(abs_ypr_gt[1:, 0] - abs_ypr_gt[:-1, 0])
        rel_yaw_gt = torch.rad2deg(rel_yaw_gt)                                    #cache#
        rel_yaw_gt[rel_yaw_gt>180.0] = rel_yaw_gt[rel_yaw_gt>180.0] - 360.0
        yaw_traveled_gt = torch.zeros_like(timestamp_sec, dtype=torch.float64) #cache#
        for i in range(rel_yaw_gt.size(0)):
            yaw_traveled_gt[i+1] = yaw_traveled_gt[i] + rel_yaw_gt[i]
        print("yaw_traveled_gt:", yaw_traveled_gt.shape)

        # Compute yaw error per meter
        yaw_error_per_meter = []
        yaw_error_per_meter_t = []
        prev_i = 0
        for i in range(t_traveled_gt.size(0)):
            dt = t_traveled_gt[i] - t_traveled_gt[prev_i]
            if dt.item() > 0.1:
                dyaw_gt = abs_ypr_gt[i, 0] - abs_ypr_gt[prev_i, 0]
                dyaw_est = abs_ypr_est[i, 0] - abs_ypr_est[prev_i, 0]
                dyaw_error = torch.rad2deg(torch.abs(dyaw_gt - dyaw_est))
                dyaw_error[dyaw_error>180.0] = dyaw_error[dyaw_error>180.0] - 360.0
                dyaw_error[dyaw_error<=-180.0] = dyaw_error[dyaw_error<=-180.0] + 360.0
                dyaw_error_per_meter = dyaw_error / dt
                yaw_error_per_meter.append(dyaw_error_per_meter.item())
                yaw_error_per_meter_t.append(t_traveled_gt[i].item())
                prev_i = i
        yaw_error_per_meter = torch.from_numpy(np.array(yaw_error_per_meter, dtype=float))     #cache#
        yaw_error_per_meter_t = torch.from_numpy(np.array(yaw_error_per_meter_t, dtype=float)) #cache#

        # Create dictionary for caching in pickle format
        cache_dict = {}
        cache_dict['timestamp_sec'] = timestamp_sec.cpu()
        cache_dict['abs_t_est'] = abs_t_est.cpu()
        cache_dict['abs_q_xyzw_est'] = abs_q_xyzw_est.data.cpu()
        cache_dict['abs_t_gt'] = abs_t_gt.cpu()
        cache_dict['abs_q_xyzw_gt'] = abs_q_xyzw_gt.data.cpu()
        cache_dict['abs_r_est'] = abs_r_est.data.cpu()
        cache_dict['abs_r_gt'] = abs_r_gt.data.cpu()
        cache_dict['abs_ypr_est'] = abs_ypr_est.data.cpu()
        cache_dict['abs_ypr_gt'] = abs_ypr_gt.data.cpu()
        cache_dict['abs_t_error'] = abs_t_error.cpu()
        cache_dict['abs_r_error'] = abs_r_error.data.cpu()
        cache_dict['abs_q_xyzw_error'] = abs_q_xyzw_error.data.cpu()
        cache_dict['abs_ypr_error'] = abs_ypr_error.cpu()
        cache_dict['rel_t_est'] = rel_t_est.cpu()
        cache_dict['rel_t_gt'] = rel_t_gt.cpu()
        cache_dict['rel_t_error'] = rel_t_error.cpu()
        cache_dict['rel_yaw_gt'] = rel_yaw_gt.cpu()
        cache_dict['yaw_traveled_gt'] = yaw_traveled_gt.cpu()
        cache_dict['yaw_error_per_meter'] = yaw_error_per_meter.cpu()
        cache_dict['yaw_error_per_meter_t'] = yaw_error_per_meter_t.cpu()
        torch.save(cache_dict, cache_fn)

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
