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

class TrajectoryAlign:
    cudable = torch.cuda.is_available()

    @classmethod
    def get_best_yaw(cls, C:torch.Tensor) -> torch.Tensor:
        '''
        maximize trace(Rz(theta) * C)
        '''
        assert C.shape == (3, 3)

        A = C[0, 1] - C[1, 0]
        B = C[0, 0] + C[1, 1]
        theta = (np.pi / 2) - torch.atan2(B, A)
        return theta

    @classmethod
    def rot_z(cls, theta:torch.Tensor) -> np.ndarray:
        R = cls.rotation_matrix(theta.cpu().numpy(), [0, 0, 1])
        R = R[0:3, 0:3]
        return R

    @classmethod
    def align_umeyama(cls, model:torch.Tensor, data:torch.Tensor,
                    known_scale=False, yaw_only=False) \
                    -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Implementation of the paper, ["Least-squares estimation of transformation parameters between two point patterns"](https://ieeexplore.ieee.org/search/searchresult.jsp?action=search&newsearch=true&queryText=%22DOI%22:10.1109%2F34.88573&SID=google), Shinji Umeyama, IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

        ### Parameters

        - np.ndarray `model`: First trajectory (nx3)
        - np.ndarray `data`
        - bool(opt) `known_scale`(False): by default False
        - bool(opt) `yaw_only`(False): by default

        ### Returns
        - float `s`: scale
        - ndarray `R`: 3x3
        - ndarray `t`: 3x1
        - ndarray `t_error`: translational error per point (1xn)
        """
        assert (model.size(0) == data.size(0)), 'must be same size'

        if cls.cudable:
            model = model.cuda()
            data = data.cuda()

        # substract mean
        mu_M = model.mean(0)
        mu_D = data.mean(0)
        model_zerocentered = model - mu_M
        data_zerocentered = data - mu_D
        # print('model_zerocentered:', model_zerocentered.shape)
        # print('data_zerocentered:', data_zerocentered.shape)

        N = model.size(0)

        # correlation
        C = (1.0 / N) * torch.matmul(model_zerocentered.transpose(0, 1), data_zerocentered)
        # print('C:', C.shape)
        sigma2 = (1.0 / N) * (data_zerocentered**2).sum()
        # print('sigma2:', sigma2.shape)

        U_svd, D_svd, V_svd = torch.linalg.svd(C)
        D_svd = torch.diag(D_svd)
        V_svd = torch.transpose(V_svd, 0, 1)
        S = torch.eye(3, dtype=torch.float64)

        if cls.cudable:
            U_svd = U_svd.cuda()
            D_svd = D_svd.cuda()
            V_svd = V_svd.cuda()
            S = S.cuda()

        if (torch.det(U_svd) * torch.det(V_svd) < 0):
            S[2, 2] = -1.0

        if yaw_only:
            rot_C = torch.matmul(data_zerocentered.transpose(0, 1), model_zerocentered)
            theta = cls.get_best_yaw(rot_C)
            R = cls.rot_z(theta)
            R = torch.from_numpy(R)
            if cls.cudable:
                R = R.cuda()
        else:
            print(U_svd, D_svd, V_svd, S)
            tmp = torch.matmul(S, V_svd.transpose(0, 1))
            R = torch.matmul(U_svd, tmp)

        if known_scale:
            s = 1.0
        else:
            s = (1.0/sigma2) * torch.trace(torch.matmul(D_svd, S)).item()

        t = mu_M - s * torch.matmul(R, mu_D)
        t = t.cpu().numpy()
        R = R.cpu().numpy()
        return s, R, t

    @classmethod
    def alignPositionYaw(cls, t_est:torch.Tensor, t_gt:torch.Tensor):

        _, R, t = cls.align_umeyama(t_gt, t_est, known_scale=True,
                                    yaw_only=True)  # note the order
        R = torch.tensor(R).squeeze()
        t = torch.tensor(t).squeeze()
        return R, t

    @classmethod
    def alignSE3(cls, t_est:torch.Tensor, t_gt:torch.Tensor):
        '''
        Calculate SE3 transformation R and t so that:
            gt = R * est + t
        '''
        _, R, t = cls.align_umeyama(t_gt, t_est, known_scale=True)
        R = torch.tensor(R).squeeze()
        t = torch.tensor(t).squeeze()
        print('R:', R.shape)
        print('t:', t.shape)
        return R, t

    # align by similarity transformation
    # def alignSIM3(p_es, p_gt, q_es, q_gt, n_aligned=-1):
    #     '''
    #     calculate s, R, t so that:
    #         gt = R * s * est + t
    #     '''
    #     idxs = _getIndices(n_aligned, p_es.shape[0])
    #     est_pos = p_es[idxs, 0:3]
    #     gt_pos = p_gt[idxs, 0:3]
    #     s, R, t = align.align_umeyama(gt_pos, est_pos)  # note the order
    #     return s, R, t


    # # a general interface
    # def alignTrajectory(p_es, p_gt, q_es, q_gt, method, n_aligned=-1):
    #     '''
    #     calculate s, R, t so that:
    #         gt = R * s * est + t
    #     method can be: sim3, se3, posyaw, none;
    #     n_aligned: -1 means using all the frames
    #     '''
    #     assert p_es.shape[1] == 3
    #     assert p_gt.shape[1] == 3
    #     assert q_es.shape[1] == 4
    #     assert q_gt.shape[1] == 4

    #     s = 1
    #     R = None
    #     t = None
    #     if method == 'sim3':
    #         assert n_aligned >= 2 or n_aligned == -1, "sim3 uses at least 2 frames"
    #         s, R, t = alignSIM3(p_es, p_gt, q_es, q_gt, n_aligned)
    #     elif method == 'se3':
    #         R, t = alignSE3(p_es, p_gt, q_es, q_gt, n_aligned)
    #     elif method == 'posyaw':
    #         R, t = alignPositionYaw(p_es, p_gt, q_es, q_gt, n_aligned)
    #     elif method == 'none':
    #         R = np.identity(3)
    #         t = np.zeros((3, ))
    #     else:
    #         assert False, 'unknown alignment method'

    #     return s, R, t

    @classmethod
    def rotation_matrix(angle, direction, point=None):
        """Return matrix to rotate about axis defined by point and direction.

        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True
        >>> I = numpy.identity(4, numpy.float64)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True
        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

        """
        import math
        def unit_vector(data, axis=None, out=None):
            """Return ndarray normalized by length, i.e. eucledian norm, along axis.

            >>> v0 = numpy.random.random(3)
            >>> v1 = unit_vector(v0)
            >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
            True
            >>> v0 = numpy.random.rand(5, 4, 3)
            >>> v1 = unit_vector(v0, axis=-1)
            >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
            >>> numpy.allclose(v1, v2)
            True
            >>> v1 = unit_vector(v0, axis=1)
            >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
            >>> numpy.allclose(v1, v2)
            True
            >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float64)
            >>> unit_vector(v0, axis=1, out=v1)
            >>> numpy.allclose(v1, v2)
            True
            >>> list(unit_vector([]))
            []
            >>> list(unit_vector([1.0]))
            [1.0]

            """
            if out is None:
                data = np.array(data, dtype=np.float64, copy=True)
                if data.ndim == 1:
                    data /= math.sqrt(np.dot(data, data))
                    return data
            else:
                if out is not data:
                    out[:] = np.array(data, copy=False)
                data = out
            length = np.atleast_1d(np.sum(data*data, axis))
            np.sqrt(length, length)
            if axis is not None:
                length = np.expand_dims(length, axis)
            data /= length
            if out is None:
                return data


        sina = math.sin(angle)
        cosa = math.cos(angle)
        direction = unit_vector(direction[:3])
        # rotation matrix around unit vector
        R = np.array(((cosa, 0.0,  0.0),
                        (0.0,  cosa, 0.0),
                        (0.0,  0.0,  cosa)), dtype=np.float64)
        R += np.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        R += np.array(((0.0,         -direction[2],  direction[1]),
                        (direction[2], 0.0,          -direction[0]),
                        (-direction[1], direction[0],  0.0)),
                        dtype=np.float64)
        M = np.identity(4)
        M[:3, :3] = R
        if point is not None:
            # rotation not around origin
            point = np.array(point[:3], dtype=np.float64, copy=False)
            M[:3, 3] = point - np.dot(R, point)
        return M

# if __name__ == '__main__':
#     TrajectoryParser.parse_trajectory("", "")
