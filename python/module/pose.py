import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

class Pose:
    @classmethod
    def kitti2tum(cls, pose_kitti: np.ndarray):
        if (pose_kitti.ndim == 1):
            assert (pose_kitti.shape[0] == 12)
            rotmat = np.array([pose_kitti[0:3],
                               pose_kitti[4:7],
                               pose_kitti[8:11]])
            R = Rotation.from_matrix(rotmat)
            quat = R.as_quat()
            tran = pose_kitti[3::4]
            pose_tum = np.concatenate([tran, quat], axis=0)
        elif (pose_kitti.ndim == 2):
            assert (pose_kitti.shape[1] == 12)
            N = pose_kitti.shape[0]
            rotmats = np.stack([pose_kitti[:,0:3],
                                pose_kitti[:,4:7],
                                pose_kitti[:,8:11]],
                                axis=1)
            Rs = Rotation.from_matrix(rotmats)
            quats = Rs.as_quat()
            trans = pose_kitti[:,3::4]
            pose_tum = np.concatenate([trans, quats], axis=1)
        else:
            print("Shape unmatched")
            exit(1)

        return pose_tum

    @classmethod
    def tum2transformation(cls, pose_tum: np.ndarray):
        if (pose_tum.ndim == 1):
            assert (pose_tum.shape[0] == 7)
            quat = pose_tum[3:7]
            R = Rotation.from_quat(quat)
            T = np.identity(4, dtype=float)
            T[0:3,0:3] = R.as_matrix()
            T[0:3,3] = pose_tum[0:3]
            return T
        elif (pose_tum.ndim == 2):
            assert (pose_tum.shape[1] == 7)
            N = pose_tum.shape[0]
            Ts = np.zeros((N,4,4), dtype=float)
            quats = pose_tum[:,3:7]
            Rs = Rotation.from_quat(quats)
            Ts[:,0:3,0:3] = Rs.as_matrix()
            Ts[:,0:3,3] = pose_tum[:,0:3]
            Ts[:,3,3] = 1.0
            return Ts
        else:
            print("Shape unmatched")
            exit(1)

    @classmethod
    def transformation2tum(cls, pose_T: np.ndarray):
        if (pose_T.ndim == 2):
            assert (pose_T.shape[0] == 4)
            assert (pose_T.shape[1] == 4)
            rotmat = pose_T[0:3,0:3]
            R = Rotation.from_matrix(rotmat)
            quat = R.as_quat()
            tran = pose_T[0:3,3]
            pose_tum = np.concatenate([tran, quat], axis=0)
            return pose_tum
        elif (pose_T.ndim == 3):
            assert (pose_T.shape[1] == 4)
            assert (pose_T.shape[2] == 4)
            N = pose_T.shape[0]
            pose_tum = np.zeros((N,7), dtype=float)
            Rs = Rotation.from_matrix(pose_T[:,0:3,0:3])
            quats = Rs.as_quat()
            trans = pose_T[:,0:3,3]
            pose_tum[:,0:3] = trans
            pose_tum[:,3:7] = quats
            return pose_tum
        else:
            print("Shape unmatched")
            exit(1)

    @classmethod
    def get_rotmat_for_z_up(cls, trans: np.ndarray):
        assert (trans.ndim == 2)
        assert (trans.shape[1] == 3)
        N = trans.shape[0]
        pca = PCA(n_components=3)
        pca.fit(trans)

        # print(pca.explained_variance_ratio_)
        # print(pca.components_.shape)
        R = pca.components_

        # z_up_trans = np.dot(trans, R.T)
        return R

