import numpy as np
import torch
from typing import Tuple

import os
import sys
# print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Rotation import Rotation
from Translation import Translation
from Pose import Pose
from Time import Time

# The package for colored Command Line Interface (CLI)
from colorama import init as colorama_init
from colorama import Fore, Back, Style
colorama_init(autoreset=True)

class Trajectory:
    """
        Class for a single Trajectory
        ### Notation
        - time: normally means timestamp [sec]
        - T: normally means tranformation (t & R)
        - t: normally means translation
        - R: normally means rotation
        - c: normally means scale factor
    """
    print_format_note_only_once = True

    def __init__(self, use_file:bool=False, trajectory_file_abs_path:str=None,
                       use_tensor:bool=False, time_sec:torch.Tensor=None, t_data:torch.Tensor=None, R_data:torch.Tensor=None,
                       start:float=None, end:float=None):


        assert (int(use_file) + int(use_tensor) == 1), 'Should be choose only one flag to initialize Trajectory class, from file or from tensors'

        if use_file:
            assert (os.path.isabs(trajectory_file_abs_path)), 'For simplicity, I recommend to use absolute path always...'
            assert (os.path.exists(trajectory_file_abs_path)), 'Estiamte file doesn\'t exist!'

            # internally initialize member vaiables
            # self.times and self.poses
            self.load(trajectory_file_abs_path, start, end)
        else:
            raise NotImplementedError()

    def load(self, est_filename:str, start:float, end:float) -> np.ndarray:
        """
            **This method deals with loading trajectory data.** \n
            Fix or Override thid method according to your format
            In this file, assume that [time(sec) tx ty tz qx qy qz qw]
            with 1-line header
        """
        message = "Loaded file: %s" % est_filename
        print(Fore.GREEN + message)
        if Trajectory.print_format_note_only_once:
            Trajectory.print_format_note_only_once = False
            message  = "- Note that the text file format should be"
            message += " '# time(sec) tx(m) ty(m) tz(m) qx qy qz qw'.\n"
            message += "- You can follow the format or change this load() method"
            print(message)

        data = torch.from_numpy(np.loadtxt(est_filename))
        assert (data.shape[0] > 0 and data.ndim == 2), 'Trajectory instance must have multiple poses'
        data = data[:, 0:8] # Clipping
        if start is not None and end is not None:
            beforeN = data.shape[0]
            est_times = data[:, 0].clone() - data[0, 0]
            mask = (start <= est_times) & (est_times <= end)
            data = data[mask]
            afterN = data.shape[0]
            print('Clip trajectory data from %.4f to %.4f, size %d to %d'%(start, end, beforeN, afterN))

        self.N = data.shape[0]
        self.times:Time = Time(data=data[:, 0])
        self.poses:Pose = Pose(t_data=data[:, 1:4],
                                R_data=data[:, 4:8],
                                R_type='quat',
                                R_qtype='xyzw')
        self.data = data

    def numpy(self, out_Rtype:str) -> Tuple[np.ndarray, np.ndarray, str]:
        t = self.poses.t_data
        R = self.poses.R_data

        if out_Rtype == 'SO3':
            R = R.SO3()
        elif out_Rtype == 'so3':
            R = R.so3()
        if out_Rtype == 'quat':
            R = R.quat()
        else:
            raise AssertionError('Unvalid rotation type!')

        t = t.numpy()
        R, R_type = R.numpy()
        return t, R, R_type

    @property
    def length(self):
        assert (self.times.length == self.poses.length)
        return self.times.length

if __name__ == '__main__':

    traj = Trajectory(use_file=True, trajectory_file_abs_path='/root/eslam/namu_trajectory_evaluation/test/result_00/est.txt')
    print('Trajectory length: ', traj.length)

    gt = Trajectory(use_file=True, trajectory_file_abs_path='/root/eslam/namu_trajectory_evaluation/test/stamped_groundtruth.txt')
    print('GT length: ', gt.length)
