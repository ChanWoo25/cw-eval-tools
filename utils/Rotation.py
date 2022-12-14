import numpy as np
import torch
from typing import Tuple

import os
import sys
# print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch_utils import bmtm

# The lists of available rotation parameterization
ROT_PARAM_TYPES = [
    'SO3',  # SO(3) Rotation Matrix
    'so3',  # so(3) axis angle
    'quat', # Quaternion with 'xyzw' or 'wxyz' ordering
    'rpy'   # Roll-Pitch-Yaw, Note that Ordering is matter
]

def outer_product(vec1:torch.Tensor, vec2:torch.Tensor):
    """single or batch-based outer product"""
    assert (len(vec1.shape) == 2 and len(vec2.shape) == 2), "vector's shape is unvalid to compute outer product"
    return torch.einsum('bi, bj -> bij', vec1, vec2)

def sinc(x:torch.Tensor) -> torch.Tensor:
    return x.sin() / x

class Rotation:
    """
    Rotation ndarray variable. This can be one of [SO3, so3, quat].
    - if SO3(Rotation matrix), shape must be [n, 3, 3] or [3, 3]
    - if so3(axis angle), shape must be [n, 3] or [3]
    - if quat(Quaternion), shape must be [n, 4] or [4]
    """
    #  tolerance criterion
    device_ = torch.device('cuda:%d'%(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu')


    def __init__(self, data:torch.Tensor, param_type:str, qtype:str=None) -> None:
        self.param_type:str
        self.single:bool
        self.data:torch.Tensor = data
        self.cudable:bool = torch.cuda.is_available()
        self.device:torch.device = torch.device('cuda:%d'%(torch.cuda.current_device()) if self.cudable else 'cpu')
        self.param_type = param_type
        self.qtype:str = None

        assert (type(data) is torch.Tensor), ('data argument must be torch.Tensor')
        assert (param_type in ROT_PARAM_TYPES), ('Unvalid Rotation parameterization!')
        unvalid_shape_err:AssertionError = AssertionError('Unvalid shape for %s parameterization'%self.param_type)

        if self.param_type is 'SO3':
            if data.shape[-1] != 3 or data.shape[-2] != 3:
                raise unvalid_shape_err
            else:
                if len(data.shape) == 2:
                    self.single = True
                elif data.shape[0] > 1:
                    self.single = False
                else:
                    raise unvalid_shape_err
        elif self.param_type is 'so3':
            if data.shape[-1] != 3:
                raise unvalid_shape_err
            else:
                if len(data.shape) == 1:
                    self.single = True
                elif data.shape[0] > 1:
                    self.single = False
                else:
                    raise unvalid_shape_err
        elif self.param_type is 'quat':
            assert (qtype in ['xyzw', 'wxyz']), 'If quaternion, must be specified xyzw or wxyz ordering'
            if data.shape[-1] != 4:
                raise unvalid_shape_err
            else:
                self.qtype:str = qtype
                if len(data.shape) == 1:
                    self.single = True
                elif data.shape[0] > 1:
                    self.single = False
                else:
                    raise unvalid_shape_err
        elif self.param_type is 'rpy':
            raise AssertionError('Not fully checked yet') #TODO
            if data.shape[-1] != 3:
                raise unvalid_shape_err
            else:
                if len(data.shape) == 1:
                    self.single = True
                elif data.shape[0] > 1:
                    self.single = False
                else:
                    raise unvalid_shape_err

        if self.cudable:
            self.data = self.data.cuda()

    def clone(self) -> 'Rotation':
        clone_data = self.data.clone()
        return Rotation(clone_data, param_type=self.param_type, qtype=self.qtype)

    def numpy(self) -> Tuple[np.ndarray, str]:
        if self.data.is_cuda:
            self.data = self.data.cpu()
        return self.data.numpy(), self.param_type

    def __sub__(self, other) -> 'Rotation':
        if self.param_type == 'SO3':
            if isinstance(other, torch.Tensor) and other.shape == (3, 3):
                other = other.expand(self.data.shape[0], 3, 3)
                new_data = bmtm(other, self.data)
                return Rotation(new_data, 'SO3')
            elif isinstance(other, Rotation):
                assert other.single == False
                new_data = bmtm(other.data, self.data)
                return Rotation(new_data, 'SO3')
        else:
            raise NotImplementedError()

    def __add__(self, other) -> 'Rotation':
        if self.param_type == 'SO3':
            if isinstance(other, torch.Tensor) and other.shape == (3, 3):
                other = other.expand(self.data.shape[0], 3, 3)
                new_data = torch.bmm(other, self.data)
                return Rotation(new_data, 'SO3')
            elif isinstance(other, Rotation):
                assert other.single == False
                new_data = torch.bmm(other.data, self.data)
                return Rotation(new_data, 'SO3')
        else:
            raise NotImplementedError()

    @property
    def length(self):
        if self.single:
            return 1
        else:
            return self.data.size(0)

    def shape(self):
        return self.data.shape

    def wedge(self, data):
        if self.single:
            pass
        else:
            n = data.shape[0]
            zero = data.new_zeros(n)
            ret = torch.stack((zero,       -data[:, 2],  data[:, 1],
                               data[:, 2] , zero      , -data[:, 0],
                               -data[:, 1], data[:, 0],  zero), 1)
            return ret.view(n, 3, 3)


    def trace(self, mat):
        """single or batch-based trace"""
        if self.single:
            pass
        else:
            return torch.einsum('bii -> b', mat)

    def vee(self, phi):
        """extract (w1,w2,w3)[N x R^3] from skew symmetric[N x R^3x3] form matrix"""
        if self.single:
            pass
        else:
            return torch.stack((phi[:, 2, 1],
                                phi[:, 0, 2],
                                phi[:, 1, 0]), dim=1)

    def so3_to_SO3(self) -> 'Rotation':
        """Exponential Mapping - Convert 'so3 Axis angle vector' to 'SO3 Rotation matrix'
            - Korean Reference -- [LINK](https://edward0im.github.io/mathematics/2020/05/01/lie-theory/#org608a5f4)"""
        assert(self.param_type is 'so3'), 'Unallowed operation, caller must be so3 param_type'

        if self.single:
            pass
        else:
            angle = self.data.norm(dim=1, keepdim=True)
            mask = angle[:, 0] < 1e-8
            n = self.data.shape[0]
            I3 = self.data.new_ones(3).diagflat().expand(n, 3, 3)

            axis = self.data[~mask] / angle[~mask]
            c = angle[~mask].cos().unsqueeze(2)
            s = angle[~mask].sin().unsqueeze(2)

            SO3 = self.data.new_empty(n, 3, 3)
            SO3[mask] = I3[mask] + self.wedge(self.data[mask])
            outer = torch.einsum('bi, bj -> bij', axis, axis)
            SO3[~mask] = c*I3[~mask] + (1-c)*outer + s*self.wedge(axis)

            self.data = SO3
            self.param_type = 'SO3'
            return self

    def quat_to_SO3(self) -> 'Rotation':
        """Form a rotation matrix from a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'.
        """
        assert(self.param_type is 'quat'), 'Unallowed operation, caller must be quat param_type'

        if self.qtype == 'xyzw':
            qx = self.data[:, 0]
            qy = self.data[:, 1]
            qz = self.data[:, 2]
            qw = self.data[:, 3]
        elif self.qtype == 'wxyz':
            qw = self.data[:, 0]
            qx = self.data[:, 1]
            qy = self.data[:, 2]
            qz = self.data[:, 3]

        # Form the matrix
        if self.single:
            pass
        else:
            n = self.data.shape[0]
            R = self.data.new_empty(n, 3, 3)

            qx2 = qx * qx
            qy2 = qy * qy
            qz2 = qz * qz

            R[:, 0, 0] = 1. - 2. * (qy2 + qz2)
            R[:, 0, 1] = 2. * (qx * qy - qw * qz)
            R[:, 0, 2] = 2. * (qw * qy + qx * qz)

            R[:, 1, 0] = 2. * (qw * qz + qx * qy)
            R[:, 1, 1] = 1. - 2. * (qx2 + qz2)
            R[:, 1, 2] = 2. * (qy * qz - qw * qx)

            R[:, 2, 0] = 2. * (qx * qz - qw * qy)
            R[:, 2, 1] = 2. * (qw * qx + qy * qz)
            R[:, 2, 2] = 1. - 2. * (qx2 + qy2)
            self.data = R
            self.param_type = 'SO3'
            self.qtype = None
            return self

    def SO3_to_so3(self) -> 'Rotation':
        """
            Logarithm Mapping - Convert 'SO3 Rotation matrix' to 'so3 Axis angle vector'\n
            - Korean Reference -- [LINK](https://edward0im.github.io/mathematics/2020/05/01/lie-theory/#org608a5f4)
            - English implementation Reference/sec2. SO(3) logarithm map -- [LINK](https://vision.in.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf)
        """
        assert(self.param_type is 'SO3'), 'Unallowed operation, caller must be SO3 param_type'

        if self.single:
            pass
        else:
            n = self.data.shape[0]
            I3 = self.data.new_ones(3).diagflat().expand(n, 3, 3)
            R = self.data

            cos_angle = (0.5 * self.trace(R) - 0.5).clamp(-1., 1.)
            # Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
            angle = cos_angle.acos()
            mask = angle < 1e-8

            if mask.sum() == 0:
                angle = angle.unsqueeze(1).unsqueeze(1)
                phi = self.vee((0.5 * angle / angle.sin())*(R - R.transpose(1, 2)))
                return Rotation(phi, 'so3')
            elif mask.sum() == n:
                # If angle is close to zero, use first-order Taylor expansion
                phi = self.vee(R - I3)
                return Rotation(phi, 'so3')

            phi = self.vee(R - I3)
            phi[~mask] = self.vee((0.5 * angle[~mask]/angle[~mask].sin()).unsqueeze(1).unsqueeze(2)*(R[~mask] - R[~mask].transpose(1, 2)))
            self.data = phi
            self.param_type = 'so3'
            return self

    def quat_to_so3(self) -> 'Rotation':
        """ Convert quaternion to so3 axis angle"""
        assert (self.param_type == 'quat'), 'Unallowed operation, caller must be quat param_type'

        if self.single:
            pass
        else:
            if self.qtype == 'xyzw':
                qw = self.data[:, 3]
                qxyz = self.data[:, 0:3]
            elif self.qtype == 'wxyz':
                qw = self.data[:, 0]
                qxyz = self.data[:, 1:4]

            n = self.data.shape[0]
            qw = qw.reshape((n, 1))
            qxyz = qxyz.reshape((n, 3))

            norm_ = 0.5 * torch.norm(qxyz, p=2, dim=1, keepdim=True)
            norm_ = torch.clamp(norm_, min=1e-8)
            q = qxyz * torch.acos(torch.clamp(qw, min=-1.0, max=1.0))
            self.data = q / norm_
            self.param_type = 'so3'
            self.qtype = None
            return self

    def SO3_to_quat(self, qtype='xyzw'):
        """Convert a rotation matrix to a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'.
        """
        assert(self.param_type is 'SO3'), 'Unallowed operation, caller must be SO3 param_type'

        if self.single:
            pass
        else:
            R = self.data
            tmp = 1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
            tmp[tmp < 0] = 0
            qw = 0.5 * torch.sqrt(tmp)
            qx = qw.new_empty(qw.shape[0])
            qy = qw.new_empty(qw.shape[0])
            qz = qw.new_empty(qw.shape[0])

            near_zero_mask = qw.abs() < 1e-8

            if near_zero_mask.sum() > 0:
                cond1_mask = near_zero_mask * \
                    (R[:, 0, 0] > R[:, 1, 1])*(R[:, 0, 0] > R[:, 2, 2])
                cond1_inds = cond1_mask.nonzero()

                if len(cond1_inds) > 0:
                    cond1_inds = cond1_inds.squeeze()
                    R_cond1 = R[cond1_inds].view(-1, 3, 3)
                    d = 2. * torch.sqrt(1. + R_cond1[:, 0, 0] -
                        R_cond1[:, 1, 1] - R_cond1[:, 2, 2]).view(-1)
                    qw[cond1_inds] = (R_cond1[:, 2, 1] - R_cond1[:, 1, 2]) / d
                    qx[cond1_inds] = 0.25 * d
                    qy[cond1_inds] = (R_cond1[:, 1, 0] + R_cond1[:, 0, 1]) / d
                    qz[cond1_inds] = (R_cond1[:, 0, 2] + R_cond1[:, 2, 0]) / d

                cond2_mask = near_zero_mask * (R[:, 1, 1] > R[:, 2, 2])
                cond2_inds = cond2_mask.nonzero()

                if len(cond2_inds) > 0:
                    cond2_inds = cond2_inds.squeeze()
                    R_cond2 = R[cond2_inds].view(-1, 3, 3)
                    d = 2. * torch.sqrt(1. + R_cond2[:, 1, 1] -
                                    R_cond2[:, 0, 0] - R_cond2[:, 2, 2]).squeeze()
                    tmp = (R_cond2[:, 0, 2] - R_cond2[:, 2, 0]) / d
                    qw[cond2_inds] = tmp
                    qx[cond2_inds] = (R_cond2[:, 1, 0] + R_cond2[:, 0, 1]) / d
                    qy[cond2_inds] = 0.25 * d
                    qz[cond2_inds] = (R_cond2[:, 2, 1] + R_cond2[:, 1, 2]) / d

                cond3_mask = near_zero_mask & cond1_mask.logical_not() & cond2_mask.logical_not()
                cond3_inds = cond3_mask

                if len(cond3_inds) > 0:
                    R_cond3 = R[cond3_inds].view(-1, 3, 3)
                    d = 2. * \
                        torch.sqrt(1. + R_cond3[:, 2, 2] -
                        R_cond3[:, 0, 0] - R_cond3[:, 1, 1]).squeeze()
                    qw[cond3_inds] = (R_cond3[:, 1, 0] - R_cond3[:, 0, 1]) / d
                    qx[cond3_inds] = (R_cond3[:, 0, 2] + R_cond3[:, 2, 0]) / d
                    qy[cond3_inds] = (R_cond3[:, 2, 1] + R_cond3[:, 1, 2]) / d
                    qz[cond3_inds] = 0.25 * d

            far_zero_mask = near_zero_mask.logical_not()
            far_zero_inds = far_zero_mask
            if len(far_zero_inds) > 0:
                R_fz = R[far_zero_inds]
                d = 4. * qw[far_zero_inds]
                qx[far_zero_inds] = (R_fz[:, 2, 1] - R_fz[:, 1, 2]) / d
                qy[far_zero_inds] = (R_fz[:, 0, 2] - R_fz[:, 2, 0]) / d
                qz[far_zero_inds] = (R_fz[:, 1, 0] - R_fz[:, 0, 1]) / d

            # Check ordering last
            if qtype == 'xyzw':
                self.data = torch.stack([qx, qy, qz, qw], dim=1)
                self.qtype = qtype
            elif qtype == 'wxyz':
                self.data = torch.stack([qw, qx, qy, qz], dim=1)
                self.qtype = qtype
            self.param_type = 'quat'
            return self

    def so3_to_quat(self, qtype='xyzw') -> 'Rotation':
        """ Convert so3 axis angle to quaternion (default='xyzw)"""
        assert (self.param_type == 'SO3'), 'Unallowed operation, caller must be SO3 param_type'

        if self.single:
            pass
        else:
            n = self.data.shape[0]
            theta = self.data.norm(dim=1, keepdim=True)
            w = (0.5 * theta).cos()
            xyz = 0.5 * sinc(0.5 * theta / np.pi) * self.data

            if qtype == 'xyzw':
                self.data = torch.cat((xyz, w), 1)
            elif qtype == 'wxyz':
                self.data = torch.cat((w, xyz), 1)
            else:
                raise AssertionError('Wrong qtype')

            self.qtype = qtype
            self.param_type = 'quat'
            return self

    def SO3(self) -> 'Rotation':
        if self.param_type is 'so3':
            return self.so3_to_SO3()
        elif self.param_type is 'quat':
            return self.quat_to_SO3()
        elif self.param_type is 'rpy':
            pass
        else:
            return self

    def so3(self) -> 'Rotation':
        if self.param_type is 'SO3':
            return self.SO3_to_so3()
        elif self.param_type is 'quat':
            return self.quat_to_so3()
        elif self.param_type is 'rpy':
            pass
        else:
            return self

    def quat(self) -> 'Rotation':
        if self.param_type is 'so3':
            return self.so3_to_quat()
        elif self.param_type is 'SO3':
            return self.SO3_to_quat()
        elif self.param_type is 'rpy':
            pass
        else:
            return self

    def normalize(self):
        assert (self.param_type in ['quat', 'SO3']), 'Normalization is valid for quat and SO3 type only'

        if self.param_type is 'SO3':
            if self.single:
                pass
            else:
                n = self.data.shape[0]
                U, _, V = torch.svd(self.data)
                S = self.data.new_ones(3).diagflat().expand(n, 3, 3)
                S[:, 2, 2] = torch.det(U) * torch.det(V)
                self.data = U.bmm(S).bmm(V.transpose(1, 2))
                return self
        elif self.param_type is 'quat':
            if self.single:
                pass
            else:
                self.data = self.data / self.data.norm(dim=1, keepdim=True)
                return self
        else:
            return self

    def convert_qtype(self, to='xyzw'):
        assert (self.param_type is 'quat'), 'Caller must be quat param_type'
        if self.qtype == to:
            return self
        elif self.qtype == 'xyzw' and to == 'wxyz':
            tmp = torch.zeros_like(self.data)
            tmp[:, 0] = self.data[:, 3]
            tmp[:, 1:4] = self.data[:, 0:3]
            self.data = tmp
            self.qtype = 'wxyz'
            return self
        elif self.qtype == 'wxyz' and to == 'xyzw':
            tmp = torch.zeros_like(self.data)
            tmp[:, 3] = self.data[:, 0]
            tmp[:, 0:3] = self.data[:, 1:4]
            self.data = tmp
            self.qtype = 'xyzw'
            return self
        else:
            raise AssertionError('Something wrong with qtype match')

    def quat_inverse(self):
        assert (self.param_type == 'quat'), 'Unallowed operation, caller must be quat param_type'

        inv = torch.empty_like(self.data)
        if self.qtype == 'xyzw':
            inv[:, :3] = -self.data[:, :3]
            inv[:, 3]  =  self.data[:, 3]
        if self.qtype == 'wxyz':
            inv[:, 1:4] = -self.data[:, 1:4]
            inv[:, 0]   =  self.data[:, 0]

        self.data = inv
        return self


    @classmethod
    def qinterp(cls, qs, t, t_int) -> torch.Tensor:
        device = torch.device('cuda:%d'%(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu')

        idxs = np.searchsorted(t, t_int)
        idxs0 = idxs-1
        idxs0[idxs0 < 0] = 0
        idxs1 = idxs
        idxs1[idxs1 == t.shape[0]] = t.shape[0] - 1
        q0 = qs[idxs0]
        q1 = qs[idxs1]
        tau = np.zeros_like(t_int)
        dt = (t[idxs1]-t[idxs0])[idxs0 != idxs1]
        tau[idxs0 != idxs1] = (t_int-t[idxs0])[idxs0 != idxs1]/dt
        tau = torch.from_numpy(tau).to(device)
        return cls.slerp(q0, q1, tau)


    @staticmethod
    def slerp(ql:torch.Tensor, qr:torch.Tensor, tau:torch.Tensor):
        """
            slerp == Spherical Linear Interpolation
            - Reference: https://en.wikipedia.org/wiki/Slerp
            - cos(sigma) = ql * qr = dot
            - t(in wiki) := tau

            #### Comparison with slerp_old()
            - On result_00,
            - interpolated_q = Rotation.slerp(ql, qr, tau)
            - interpolated_q2 = Rotation.slerp2(ql, qr, tau)
            - rest = interpolated_q - interpolated_q2
            - less than 1e-6 error : 100%
            - less than 1e-8 error : 618/620
        """
        dot = (ql * qr).sum(dim=1)
        qr[dot < 0] = -qr[dot < 0]
        dot = (ql * qr).sum(dim=1)
        if torch.any(dot.data.isnan()):
            print('find dot\'s NaN!', dot.shape)
            for i in range(dot.data.shape[0]):
                    if dot.data[i].isnan():
                        print('(%d): %1.8f' % (i, dot[i]))

        dot[dot>=1.0] -= float(1e-6)
        sigma = dot.acos()
        # In this break point, sigma cause nan, due to
        # if dot's element is 1.0 or -1.0, dot.acos() is nan.
        print(dot[dot >= 1.0].sum())
        # print(dot.cpu().tolist())
        if torch.any(sigma.data.isnan()):
            print('find sigma\'s NaN!', sigma.shape)
            for i in range(sigma.data.shape[0]):
                    if sigma.data[i].isnan():
                        print('(%d): %1.8f,  %1.8f' % (i, sigma[i], dot[i]))


        sin_sigma = torch.sin(sigma)
        if torch.any(sin_sigma.data.isnan()):
            print('find sin_sigma\'s NaN!', sin_sigma.shape)
            for i in range(sin_sigma.data.shape[0]):
                    if sin_sigma.data[i].isnan():
                        print('(%d): %1.8f' % (i, sin_sigma[i]))

        sl = torch.sin((1.0 - tau) * sigma) / sin_sigma
        if torch.any(sl.data.isnan()):
            print('find sl\'s NaN!', sl.shape)
            for i in range(sl.data.shape[0]):
                    if sl.data[i].isnan():
                        print('(%d)' % (i))

        sr = torch.sin(tau * sigma) / sin_sigma
        if torch.any(sr.data.isnan()):
            print('find sr\'s NaN!', sr.shape)
            for i in range(sr.data.shape[0]):
                    if sr.data[i].isnan():
                        print('(%d)' % (i))

        q = sl.unsqueeze(1) * ql + sr.unsqueeze(1) * qr
        q = q / q.norm(dim=1, keepdim=True)
        return q

    @staticmethod
    def slerp_old(ql:torch.Tensor, qr:torch.Tensor, tau:torch.Tensor, DOT_THRESHOLD=0.9995):
        """
            slerp == Spherical Linear Interpolation
            - Reference: https://en.wikipedia.org/wiki/Slerp
            - cos(sigma) = ql * qr = dot
            - t(in wiki) := tau
        """
        dot = (ql * qr).sum(dim=1)
        qr[dot < 0] *= -1.0
        dot = (ql * qr).sum(dim=1)

        q = torch.zeros_like(ql)

        # when sigma -> 0 (== dot >= 0.9995), linear interpolation is ok
        case1 = ql + tau.unsqueeze(1) * (qr - ql)
        q[dot >= DOT_THRESHOLD] = case1[dot >= DOT_THRESHOLD]

        theta_0 = dot.acos()
        sin_theta_0 = theta_0.sin()
        theta = theta_0 * tau
        sin_theta = theta.sin()
        s0 = (theta.cos() - dot * sin_theta / sin_theta_0).unsqueeze(1)
        s1 = (sin_theta / sin_theta_0).unsqueeze(1)
        case2 = s0 * ql + s1 * qr
        q[dot < DOT_THRESHOLD] = case2[dot < DOT_THRESHOLD]

        q = q / q.norm(dim=1, keepdim=True)
        return q

    @staticmethod
    def qmul(q1:'Rotation', q2:'Rotation', out_qtype='xyzw') -> 'Rotation':
        """
            Compute quaternion multiplication
            this function
        """
        assert (q1.param_type == 'quat' and q2.param_type == 'quat'), 'both rotation must be quaternion type'

        q1_wxyz = q1.convert_qtype(to='wxyz')
        q2_wxyz = q2.convert_qtype(to='wxyz')

        terms = outer_product(q2_wxyz.data, q1_wxyz.data)
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        xyz = torch.stack((x, y, z), dim=1)
        xyz[w < 0] *= -1
        w[w < 0] *= -1

        if out_qtype == 'xyzw':
            data = torch.cat((xyz, w.unsqueeze(1)), dim=1)
        elif out_qtype == 'wxyz':
            data = torch.cat((w.unsqueeze(1), xyz), dim=1)
        else:
            raise AssertionError('[qmul] -- Ordring error')

        return Rotation(data, 'quat', out_qtype).normalize()



    # @classmethod
    # def from_rpy(cls, roll, pitch, yaw):
    #     return cls.rotz(yaw).bmm(cls.roty(pitch).bmm(cls.rotx(roll)))

    # @classmethod
    # def rotx(cls, angle_in_radians):
    #     c = angle_in_radians.cos()
    #     s = angle_in_radians.sin()
    #     mat = c.new_zeros((c.shape[0], 3, 3))
    #     mat[:, 0, 0] = 1
    #     mat[:, 1, 1] = c
    #     mat[:, 2, 2] = c
    #     mat[:, 1, 2] = -s
    #     mat[:, 2, 1] = s
    #     return mat

    # @classmethod
    # def roty(cls, angle_in_radians):
    #     c = angle_in_radians.cos()
    #     s = angle_in_radians.sin()
    #     mat = c.new_zeros((c.shape[0], 3, 3))
    #     mat[:, 1, 1] = 1
    #     mat[:, 0, 0] = c
    #     mat[:, 2, 2] = c
    #     mat[:, 0, 2] = s
    #     mat[:, 2, 0] = -s
    #     return mat

    # @classmethod
    # def rotz(cls, angle_in_radians):
    #     c = angle_in_radians.cos()
    #     s = angle_in_radians.sin()
    #     mat = c.new_zeros((c.shape[0], 3, 3))
    #     mat[:, 2, 2] = 1
    #     mat[:, 0, 0] = c
    #     mat[:, 1, 1] = c
    #     mat[:, 0, 1] = -s
    #     mat[:, 1, 0] = s
    #     return mat

    # @classmethod
    # def isclose(cls, x, y):
    #     return (x-y).abs() < cls.eps_

    def SO3_to_euler_rzyx(self) -> 'Rotation':
        assert (self.param_type == 'SO3'), 'This operation need SO3'
        assert (self.data.ndim == 3), 'This operation need dim(3) [n,3,3]'

        firstaxis, parity, repetition, frame = (0, 0, 0, 1)
        i = 0
        j = 1
        k = 2

        M = self.data
        eps = np.finfo(float).eps * 4.0

        if repetition:
            sy = (M[:, i, j]*M[:, i, j] + M[:, i, k]*M[:, i, k])
            sy = sy.sqrt()

            ax = M.new_empty((M.size(0)))
            ay = M.new_empty((M.size(0)))
            az = M.new_empty((M.size(0)))

            ax[sy>eps]  = torch.atan2(M[sy>eps][:, i, j],  M[sy>eps][:, i, k])
            ay[sy>eps]  = torch.atan2(sy[sy>eps]        ,  M[sy>eps][:, i, i])
            az[sy>eps]  = torch.atan2(M[sy>eps][:, j, i], -M[sy>eps][:, k, i])

            ax[sy<=eps] = torch.atan2(-M[sy<=eps][:, j, k],  M[sy<=eps][:, j, j])
            ay[sy<=eps] = torch.atan2(sy[sy<=eps]        ,  M[sy<=eps][:, i, i])
            az[sy<=eps] = 0.0
        else:
            cy = (M[:, i, i]**2 + M[:, j, i]**2).sqrt()

            ax = M.new_empty((M.size(0)))
            ay = M.new_empty((M.size(0)))
            az = M.new_empty((M.size(0)))

            ax[cy>eps]  = torch.atan2( M[cy>eps][:, k, j], M[cy>eps][:, k, k])
            ay[cy>eps]  = torch.atan2(-M[cy>eps][:, k, i], cy[cy>eps])
            az[cy>eps]  = torch.atan2( M[cy>eps][:, j, i], M[cy>eps][:, i, i])

            ax[cy<=eps] = torch.atan2(-M[cy<=eps][:, j, k],  M[cy<=eps][:, j, j])
            ay[cy<=eps] = torch.atan2(-M[cy<=eps][:, k, i],  cy[cy<=eps])
            az[cy<=eps] = 0.0

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax

        ypr = torch.stack((ax, ay, az), dim=1)
        return ypr


    @classmethod
    def to_rpy(cls, Rots):
        """Convert a rotation matrix to RPY Euler angles."""

        pitch = torch.atan2(-Rots[:, 2, 0],
            torch.sqrt(Rots[:, 0, 0]**2 + Rots[:, 1, 0]**2))
        yaw = pitch.new_empty(pitch.shape)
        roll = pitch.new_empty(pitch.shape)

        near_pi_over_two_mask = cls.isclose(pitch, np.pi / 2.)
        near_neg_pi_over_two_mask = cls.isclose(pitch, -np.pi / 2.)

        remainder_inds = ~(near_pi_over_two_mask | near_neg_pi_over_two_mask)

        yaw[near_pi_over_two_mask] = 0
        roll[near_pi_over_two_mask] = torch.atan2(
            Rots[near_pi_over_two_mask, 0, 1],
            Rots[near_pi_over_two_mask, 1, 1])

        yaw[near_neg_pi_over_two_mask] = 0.
        roll[near_neg_pi_over_two_mask] = -torch.atan2(
            Rots[near_neg_pi_over_two_mask, 0, 1],
            Rots[near_neg_pi_over_two_mask, 1, 1])

        sec_pitch = 1/pitch[remainder_inds].cos()
        remainder_mats = Rots[remainder_inds]
        yaw = torch.atan2(remainder_mats[:, 1, 0] * sec_pitch,
                          remainder_mats[:, 0, 0] * sec_pitch)
        roll = torch.atan2(remainder_mats[:, 2, 1] * sec_pitch,
                           remainder_mats[:, 2, 2] * sec_pitch)
        rpys = torch.cat([roll.unsqueeze(dim=1),
                        pitch.unsqueeze(dim=1),
                        yaw.unsqueeze(dim=1)], dim=1)
        return rpys

if __name__ == '__main__':

    q1 = Rotation(torch.randn(2, 4), 'quat', 'xyzw').normalize()
    q2 = Rotation(torch.randn(2, 4), 'quat', 'xyzw').normalize()
    print(q1.data)
    print(q2.data)


