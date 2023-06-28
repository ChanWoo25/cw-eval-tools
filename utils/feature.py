from pathlib import Path
import numpy as np
import torch
from Rotation import Rotation
from torch_utils import bmtm

out_fn = Path('/data/results/event_feature_tracking/case_a/CASE_A_230629/0.csv')
out = np.genfromtxt(out_fn, delimiter=',')
print("out.shape: ", out.shape)

gt_fn = Path('/data/test3.csv')
gt = np.genfromtxt(gt_fn, delimiter=',')
print("gt.shape: ", gt.shape)

out = torch.from_numpy(out)
gt = torch.from_numpy(gt)
gt[:,0] /= 1e9

gt_ts  = gt[:,0]
out_ts = out[:,0]

t0 = torch.max(out_ts[0], gt_ts[0]).item()
gt_i0 = torch.searchsorted(gt_ts, t0, right=True) - 1
est_i0 = torch.searchsorted(out_ts, t0, right=True)
assert (gt_ts[gt_i0].item() <= out_ts[est_i0].item())

tn = torch.min(out_ts[-1], gt_ts[-1]).item()
gt_in = torch.searchsorted(gt_ts, tn, right=False)
est_in = torch.searchsorted(out_ts, tn, right=True) - 1
assert (out_ts[est_in].item() <= gt_ts[gt_in].item())

print('t0 ~ tn : %.4f ~ %.4f' % (t0, tn))

out_ts = out[est_i0:est_in, 0]
out_fs = out[est_i0:est_in, 1:3]
gt_ts = gt[gt_i0:gt_in, 0]
gt_ps = gt[gt_i0:gt_in, 1:4]
gt_qs = gt[gt_i0:gt_in, 4:8]


t0 = torch.max(out_ts[0], gt_ts[0]).item()
tn = torch.min(out_ts[-1], gt_ts[-1]).item()

idxs_left  = torch.searchsorted(gt_ts, out_ts, right=True) - 1
idxs_right = torch.searchsorted(gt_ts, out_ts, right=True)

ql = gt_qs[idxs_left]
qr = gt_qs[idxs_right]
pl = gt_ps[idxs_left]
pr = gt_ps[idxs_right]


dt = gt_ts[idxs_right] - gt_ts[idxs_left]
tau = (out_ts - gt_ts[idxs_left]) / dt

interpolated_q = Rotation.slerp(ql, qr, tau) # ok
interpolated_t = torch.lerp(pl, pr, tau.unsqueeze(1)) # ok
interpolated_time = out_ts.unsqueeze(1)
print("interpolated_q.shape: ", interpolated_q.shape)
print(interpolated_q[:5])
print("interpolated_t.shape: ", interpolated_t.shape)
print(interpolated_t[:5])
print("interpolated_time.shape: ", interpolated_time.shape)
print(interpolated_time[:5])

##################################################################
from scipy.spatial.transform import Rotation

fx = 400.0
fy = 400.0
cx = 240.0
cy = 180.0

out_fs = out_fs.numpy()
out_ts = out_ts.numpy()
gt_ts = interpolated_time.numpy()
gt_ps = interpolated_t.numpy()
gt_qs = interpolated_q.numpy()

x1 = out_fs[1,0]
y1 = out_fs[1,1]

X1 = (x1 - cx) / fx
Y1 = (y1 - cy) / fy
Z1 = 1.0

X1 *= 5.0
Y1 *= 5.0
Z1 *= 5.0

pts_cam = np.array([X1, Y1, Z1], dtype=float).reshape([3, 1])
print("pts_cam.shape: ", pts_cam.shape)

print("X1:%.3f, Y1:%.3f, Z1:%.3f" % (X1, Y1, Z1))

print("out t1:", out_ts[1])
print("gt  t1:", gt_ts[1])

# rotation = Rotation.from_quat(gt_qs[1]) # xyzw 확인
# rot_mat = rotation.inv().as_matrix()
rotation = Rotation.from_quat([1.0, 0.0, 0.0, 0.0]) # xyzw 확인
rot_mat = rotation.as_matrix()
print(rot_mat)
print(pts_cam)
print(gt_ps[1])

# pts_world = np.dot(rot_mat, pts_cam) - np.dot(rot_mat, gt_ps[1].reshape([3, 1]))
pts_world = np.dot(rot_mat, pts_cam) + gt_ps[1].reshape([3, 1])
print(pts_world)

for i in range(2, 5):
    camera_rot = Rotation.from_quat([1.0, 0.0, 0.0, 0.0]).inv().as_matrix()
    camera_tr  = gt_ps[i].reshape([3, 1])
    # pts_cam = np.dot(camera_rot, pts_world) + camera_tr
    pts_cam = np.dot(camera_rot, pts_world) - np.dot(camera_rot, camera_tr)
    xi = (pts_cam[0] / pts_cam[2]) * fx + cx
    yi = (pts_cam[1] / pts_cam[2]) * fy + cy
    print("projected xi, yi = [%.3f, %.3f]" % (xi, yi))
    print("outputted xi, yi = [%.3f, %.3f]" % (out_fs[i,0], out_fs[i,1]))


# # Example quaternion
# quaternion = [0.707, 0.0, 0.707, 0.0]

# # Create a Rotation object from the quaternion

# # Get the rotation matrix
# rotation_matrix = rotation.as_matrix()

# # Print the rotation matrix
# print(rotation_matrix)

# est_ts = estimate.times.data
# est_t = estimate.poses.t_data.data
# est_R = estimate.poses.R_data.quat().data
# gt_ts = gt.times.data
# gt_t = gt.poses.t_data.data
# gt_R = gt.poses.R_data.quat().data

# t0 = torch.max(est_ts[0], gt_ts[0]).item()
# gt_i0 = torch.searchsorted(gt_ts, t0, right=True) - 1
# est_i0 = torch.searchsorted(est_ts, t0, right=True)
# assert (gt_ts[gt_i0].item() <= est_ts[est_i0].item())

# tn = torch.min(est_ts[-1], gt_ts[-1]).item()
# gt_in = torch.searchsorted(gt_ts, tn, right=False)
# est_in = torch.searchsorted(est_ts, tn, right=True) - 1
# assert (est_ts[est_in].item() <= gt_ts[gt_in].item())

# assert (gt_ts[gt_i0].item() <= gt_ts[gt_in].item())
# assert (est_ts[est_i0].item() <= est_ts[est_in].item())

# est_ts = est_ts[est_i0:est_in]
# est_t = est_t[est_i0:est_in, :]
# est_R = est_R[est_i0:est_in, :]
# gt_ts = gt_ts[gt_i0:gt_in]
# gt_t = gt_t[gt_i0:gt_in, :]
# gt_R = gt_R[gt_i0:gt_in, :]

# t0 = torch.max(est_ts[0], gt_ts[0]).item()
# tn = torch.min(est_ts[-1], gt_ts[-1]).item()

# idxs_left  = torch.searchsorted(gt_ts, est_ts, right=True) - 1
# idxs_right = torch.searchsorted(gt_ts, est_ts, right=True)

# ql = gt_R[idxs_left]
# qr = gt_R[idxs_right]
# tl = gt_t[idxs_left]
# tr = gt_t[idxs_right]

# dt = gt_ts[idxs_right] - gt_ts[idxs_left]
# tau = (est_ts - gt_ts[idxs_left]) / dt

# interpolated_q = Rotation.slerp(ql, qr, tau) # ok
# interpolated_t = torch.lerp(tl, tr, tau.unsqueeze(1)) # ok
# interpolated_time = est_ts.unsqueeze(1)

# est_t = est_t - est_t[0, :]
# est_t = est_t + interpolated_t[0, :]

# est_R = Rotation(est_R, param_type='quat', qtype='xyzw')
# gt_R = Rotation(interpolated_q, param_type='quat', qtype='xyzw')
# est_R = est_R.SO3()
# gt_R = gt_R.SO3()

# est_R = est_R - est_R.data[0]
# est_R = est_R + gt_R.data[0]
# est_q = est_R.quat().data

# interpolated_gt = torch.cat((interpolated_time,
#                               interpolated_t,
#                               interpolated_q), dim=1)
# interpolated_est = torch.cat((est_ts.unsqueeze(1),
#                               est_t,
#                               est_q), dim=1)
# return interpolated_est, interpolated_gt






################## project_points_3d_to_image
# import numpy as np

# def project_points_3d_to_image(points_3d, intrinsic_matrix):
#     # Add homogeneous coordinates to 3D points
#     points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

#     # Apply camera projection
#     points_2d_hom = intrinsic_matrix @ points_3d_hom.T

#     # Normalize homogeneous coordinates
#     points_2d_norm = points_2d_hom[:2] / points_2d_hom[2]

#     # Transpose and return the projected 2D points
#     return points_2d_norm.T

# # Intrinsic parameters (example values)
# focal_length_x = 500
# focal_length_y = 500
# principal_point_x = 320
# principal_point_y = 240

# # Intrinsic matrix
# intrinsic_matrix = np.array([
#     [focal_length_x, 0, principal_point_x],
#     [0, focal_length_y, principal_point_y],
#     [0, 0, 1]
# ])

# # Example 3D points
# points_3d = np.array([
#     [0, 0, 0],
#     [1, 0, 5],
#     [0, 1, 10],
#     [1, 1, 15]
# ])

# # Project 3D points to the image plane
# points_2d = project_points_3d_to_image(points_3d, intrinsic_matrix)

# # Print the projected 2D points
# for point_2d in points_2d:
#     print(f"Projected 2D Point: ({point_2d[0]:.2f}, {point_2d[1]:.2f})")
