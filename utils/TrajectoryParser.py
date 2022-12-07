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
        # print("timestamp_sec:", timestamp_sec.shape)
        # print("abs_t_est:", abs_t_est.shape)
        # print("abs_q_xyzw_est:", abs_q_xyzw_est.shape)
        # print("abs_t_gt:", abs_t_gt.shape)
        # print("abs_q_xyzw_gt:", abs_q_xyzw_gt.shape)

        abs_t_error = torch.abs(abs_t_gt - abs_t_est)
        abs_t_error_norm:torch.Tensor = torch.norm(abs_t_error, dim=1)
        # print("abs_t_error:", abs_t_error.shape)
        # print("abs_t_error_norm:", abs_t_error_norm.shape)


        rel_timestamp_sec = timestamp_sec[:-1]
        rel_t_est, rel_q_xyzw_est = cls.relative_estimate(interpolated_est)
        rel_t_gt, rel_q_xyzw_gt = cls.relative_gt(interpolated_gt)
        # print("rel_timestamp_sec:", rel_timestamp_sec.shape)
        # print("rel_t_est:", rel_t_est.shape)
        # print("rel_q_xyzw_est:", rel_q_xyzw_est.shape)
        # print("rel_t_gt:", rel_t_gt.shape)
        # print("rel_q_xyzw_gt:", rel_q_xyzw_gt.shape)

        rel_t_error = rel_t_gt - rel_t_est
        rel_t_error_norm:torch.Tensor = rel_t_error.norm(dim=1)
        # print("rel_t_error:", rel_t_error.shape)
        # print("rel_t_error_norm:", rel_t_error_norm.shape)

        rel_R_est = Rotation(rel_q_xyzw_est, param_type='quat', qtype='xyzw').SO3()
        rel_R_gt = Rotation(rel_q_xyzw_gt, param_type='quat', qtype='xyzw').SO3()
        rel_R = rel_R_gt - rel_R_est
        rel_R = rel_R.so3()

        rel_r_error = rel_R.data
        rel_r_error_norm:torch.Tensor = rel_r_error.norm(dim=1)
        # print("rel_r_error:", rel_r_error.shape)
        # print("rel_r_error_norm:", rel_r_error_norm.shape)



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
        # print("t_traveled_gt:", t_traveled_gt.shape)

        abs_t_error_norm_percent = abs_t_error_norm / t_traveled_gt
        abs_t_error_norm_percent[0] = 0.0
        # print("abs_t_error_norm_percent:", abs_t_error_norm_percent.shape)
        # print('abs_t_error_norm_percent:', abs_t_error_norm_percent[:10])

        R_est = Rotation(abs_q_xyzw_est, param_type='quat', qtype='xyzw').SO3()
        R_gt = Rotation(abs_q_xyzw_gt, param_type='quat', qtype='xyzw').SO3()
        abs_ypr_est = R_est.SO3_to_euler_rzyx()
        abs_ypr_gt = R_gt.SO3_to_euler_rzyx()
        abs_R_error = R_gt - R_est

        R_est = R_est.so3()
        R_gt = R_gt.so3()
        abs_r_est = torch.rad2deg(R_est.data)
        abs_r_gt = torch.rad2deg(R_gt.data)


        abs_ypr_error = abs_ypr_gt - abs_ypr_est
        abs_ypr_error = torch.rad2deg(abs_ypr_error)
        # print("abs_ypr_error", abs_ypr_error.shape)

        abs_yaw_error = abs_ypr_error[:, 0]
        # print("abs_yaw_error", abs_yaw_error.shape)

        yaw_error_per_meter = []
        prev_i = 0
        for i in range(t_traveled_gt.size(0)):
            dt = t_traveled_gt[i] - t_traveled_gt[prev_i]
            if dt.item() > 0.1:
                dyaw_gt = abs_ypr_gt[i, 0] - abs_ypr_gt[prev_i, 0]
                dyaw_est = abs_ypr_est[i, 0] - abs_ypr_est[prev_i, 0]
                # print('dyaw_gt: %1.8f'%dyaw_gt)
                # print('dyaw_est: %1.8f'%dyaw_est)
                dyaw_error = torch.abs(dyaw_gt - dyaw_est)
                dyaw_error = torch.rad2deg(dyaw_error)
                dyaw_error_per_meter = dyaw_error / dt
                yaw_error_per_meter.append(dyaw_error_per_meter.item())
                prev_i = i
            else:
                continue

        yaw_error_per_meter = torch.from_numpy(np.array(yaw_error_per_meter))
        # print('yaw_error_per_meter:', yaw_error_per_meter.shape)
        # print('mean yaw_error_per_meter:', yaw_error_per_meter.mean())

        abs_yaw_error_per_meter = abs_yaw_error[-1] / t_traveled_gt[-1]
        # print('abs_yaw_error_per_meter:', abs_yaw_error_per_meter.shape)
        # print('paper mean yaw_error_per_meter: %1.8f'%abs_yaw_error_per_meter.item())


        abs_R_error = abs_R_error.so3()
        abs_r_error = abs_R_error.data
        abs_r_error_norm = abs_r_error.norm(dim=1)
        # print("abs_r_error:", abs_r_error.shape)
        # print("abs_r_error_norm:", abs_r_error_norm.shape)

        dR_gt = Rotation(rel_q_xyzw_gt, param_type='quat', qtype='xyzw').SO3()
        dyaw = dR_gt.SO3_to_euler_rzyx()[:, 0]
        # print("dyaw", dyaw.shape)

        yaw_traveled = torch.zeros_like(timestamp_sec)
        traveled = 0.0
        for i in range(yaw_traveled.size(0)):
            yaw_traveled[i] = traveled
            assert (yaw_traveled[i-1] <= yaw_traveled[i])

            if i == (yaw_traveled.size(0)-1):
                break
            else:
                traveled += abs(dyaw[i].item())
        yaw_traveled = torch.rad2deg(yaw_traveled)
        # print("yaw_traveled:", yaw_traveled.shape)

        abs_yaw_error_percent = abs(abs_ypr_error[:, 0]) / yaw_traveled
        abs_yaw_error_percent[0] = 0.0
        # print('abs_yaw_error_percent:', abs_yaw_error_percent.shape)
        # print('abs_yaw_error_percent:', abs_yaw_error_percent[:10])

        cache_dict = {}
        cache_dict['timestamp_sec'] = timestamp_sec.cpu()
        cache_dict['abs_t_est'] = abs_t_est.cpu()
        cache_dict['abs_t_gt'] = abs_t_gt.cpu()
        cache_dict['abs_t_error'] = abs_t_error.cpu()
        cache_dict['abs_t_error_norm'] = abs_t_error_norm.cpu()
        cache_dict['abs_t_error_norm_percent'] = abs_t_error_norm_percent.cpu()
        cache_dict['t_traveled_gt'] = t_traveled_gt.cpu()
        cache_dict['abs_r_est'] = abs_r_est.cpu()
        cache_dict['abs_r_gt'] = abs_r_gt.cpu()
        cache_dict['abs_ypr_error'] = abs_ypr_error.cpu()
        cache_dict['abs_yaw_error'] = abs_yaw_error.cpu()
        cache_dict['yaw_traveled'] = yaw_traveled.cpu()
        cache_dict['abs_yaw_error_percent'] = abs_yaw_error_percent.cpu()
        cache_dict['yaw_error_per_meter'] = yaw_error_per_meter.cpu()
        cache_dict['abs_yaw_error_per_meter'] = abs_yaw_error_per_meter.cpu()
        cache_dict['rel_timestamp_sec'] = rel_timestamp_sec.cpu()
        cache_dict['rel_t_est'] = rel_t_est.cpu()
        cache_dict['rel_q_xyzw_est'] = rel_q_xyzw_est.cpu()
        cache_dict['rel_t_gt'] = rel_t_gt.cpu()
        cache_dict['rel_q_xyzw_gt'] = rel_q_xyzw_gt.cpu()
        cache_dict['rel_t_error'] = rel_t_error.cpu()
        cache_dict['rel_t_error_norm'] = rel_t_error_norm.cpu()
        cache_dict['rel_r_error'] = rel_r_error.cpu()
        cache_dict['rel_r_error_norm'] = rel_r_error_norm.cpu()

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
