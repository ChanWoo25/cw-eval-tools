from pathlib import Path
import numpy as np
import torch
from Rotation import Rotation as Rot
import matplotlib.pyplot as plt
import os

# out_fn = Path('/data/results/event_feature_tracking/case_a/CASE_A_230629/0.csv')
# out_fn = Path('/data/results/event_feature_tracking/case_a/E230710/0.csv')
# out_fn = Path('/data/results/event_feature_tracking/case_a/230712_A/0.csv')
# out_fn = Path('/data/results/event_feature_tracking/case_a/230714_A/0.csv')

means = []
stds = []

def eval_n_plot(out_fn: Path, gt_fn:Path, png_fn: Path, save_plot:bool=True):
    out = np.genfromtxt(out_fn, delimiter=',')
    # print("out.shape: ", out.shape)
    gt = np.genfromtxt(gt_fn, delimiter=',')
    # print("gt.shape: ", gt.shape)
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

    # print('t0 ~ tn : %.4f ~ %.4f' % (t0, tn))

    out_ts = out[est_i0:est_in, 0]
    out_fs = out[est_i0:est_in, 1:3]
    gt_ts = gt[gt_i0:gt_in, 0]
    gt_ps = gt[gt_i0:gt_in, 1:4]
    gt_qs = gt[gt_i0:gt_in, 4:8]
    # print("out_ts.shape: ", out_ts.shape)
    # print("out_fs.shape: ", out_fs.shape)
    # print("gt_ts.shape: ", gt_ts.shape)
    # print("gt_ps.shape: ", gt_ps.shape)
    # print("gt_qs.shape: ", gt_qs.shape)


    t0 = torch.max(out_ts[0], gt_ts[0]).item()
    tn = torch.min(out_ts[-1], gt_ts[-1]).item()

    idxs_left  = torch.searchsorted(gt_ts, out_ts, right=True) - 1
    idxs_right = torch.searchsorted(gt_ts, out_ts, right=True)
    n_remove = 0
    while idxs_right[-1 - n_remove] >= len(gt_ts):
        n_remove += 1
    # print("n_remove: ", n_remove)
    if n_remove > 0:
        idxs_left = idxs_left[:-n_remove]
        idxs_right = idxs_right[:-n_remove]
        out_ts = out_ts[:-n_remove]
        out_fs = out_fs[:-n_remove]

    ql = gt_qs[idxs_left]
    qr = gt_qs[idxs_right]
    pl = gt_ps[idxs_left]
    pr = gt_ps[idxs_right]


    dt = gt_ts[idxs_right] - gt_ts[idxs_left]
    tau = (out_ts - gt_ts[idxs_left]) / dt

    interpolated_q = Rot.slerp(ql, qr, tau) # ok
    interpolated_t = torch.lerp(pl, pr, tau.unsqueeze(1)) # ok
    interpolated_time = out_ts.unsqueeze(1)
    # print("interpolated_q.shape: ", interpolated_q.shape)
    # print(interpolated_q[:5])
    # print("interpolated_t.shape: ", interpolated_t.shape)
    # print(interpolated_t[:5])
    # print("interpolated_time.shape: ", interpolated_time.shape)
    # print(interpolated_time[:5])

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

    for ref in range(0, 1):
        x1 = out_fs[ref,0]
        y1 = out_fs[ref,1]

        X1 = (x1 - cx) / fx
        Y1 = (y1 - cy) / fy
        Z1 = 1.0

        X1 *= 5.0
        Y1 *= 5.0
        Z1 *= 5.0

        pts_cam = np.array([X1, Y1, Z1], dtype=float).reshape([3, 1])
        # print("pts_cam.shape: ", pts_cam.shape)
        # print("X1:%.3f, Y1:%.3f, Z1:%.3f" % (X1, Y1, Z1))
        # print("out t1:", out_ts[1])
        # print("gt  t1:", gt_ts[1])

        # rotation = Rotation.from_quat(gt_qs[1]) # xyzw 확인
        # rot_mat = rotation.inv().as_matrix()
        rotation = Rotation.from_quat([1.0, 0.0, 0.0, 0.0]) # xyzw 확인
        rot_mat = rotation.as_matrix()
        # print(rot_mat)
        # print(pts_cam)
        # print(gt_ps[1])

        # pts_world = np.dot(rot_mat, pts_cam) - np.dot(rot_mat, gt_ps[1].reshape([3, 1]))
        pts_world = np.dot(rot_mat, pts_cam) + gt_ps[ref].reshape([3, 1])
        # print(pts_world)

        errors = []
        errors_xs = []
        errors_ys = []
        gt_xs = []
        gt_ys = []
        out_XS = []
        out_YS = []
        out_ZS = []
        gt_XS = []
        gt_YS = []
        gt_ZS = []
        for i in range(0, out_fs.shape[0]):
            # 2D Error
            camera_rot = Rotation.from_quat([1.0, 0.0, 0.0, 0.0]).inv().as_matrix()
            camera_tr  = gt_ps[i].reshape([3, 1])
            pts_cam = np.dot(camera_rot, pts_world) - np.dot(camera_rot, camera_tr)
            xi = (pts_cam[0] / pts_cam[2]) * fx + cx
            yi = (pts_cam[1] / pts_cam[2]) * fy + cy
            gt_xs.append(xi)
            gt_ys.append(yi)
            # print("projected xi, yi = [%.3f, %.3f]" % (xi, yi))
            # print("outputted xi, yi = [%.3f, %.3f]" % (out_fs[i,0], out_fs[i,1]))
            error = np.sqrt(  (xi-out_fs[i,0])**2
                            + (yi-out_fs[i,1])**2)
            # print("t:%.4f,  error: %.4f" % (out_ts[i], error))
            errors.append(error)
            errors_xs.append(xi-out_fs[i,0])
            errors_ys.append(yi-out_fs[i,1])

            # 3D Error
            # X = 5.0 * (out_fs[i,0] - cx) / fx
            # Y = 5.0 * (out_fs[i,1] - cy) / fy
            # Z = 5.0 * 1.0
            # pi_cam = np.array([X, Y, Z], dtype=float).reshape([3, 1])
            # rotation = Rotation.from_quat([1.0, 0.0, 0.0, 0.0]) # xyzw 확인
            # rot_mat = rotation.as_matrix()
            # pi_world = np.dot(rot_mat, pi_cam) + gt_ps[i].reshape([3, 1])
            # out_XS.append(pi_world[0])
            # out_YS.append(pi_world[1])
            # out_ZS.append(pi_world[2])
            # gt_XS.append(pts_world[0])
            # gt_YS.append(pts_world[1])
            # gt_ZS.append(pts_world[2])
            # error_3d = np.sqrt(  (pi_world[0] - pts_world[0])**2
            #                    + (pi_world[1] - pts_world[1])**2
            #                    + (pi_world[2] - pts_world[2])**2)
            # print("t:%.4f,  error_3d: %.4f" % (out_ts[i], error_3d))
            # errors.append(error_3d)

        errors = np.array(errors, dtype=float)
        errors_xs = np.array(errors_xs, dtype=float)
        errors_ys = np.array(errors_ys, dtype=float)

        ## 2D Plot
        if save_plot:
            # print("mean: %.4f" % errors.mean())
            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9), dpi=200)
            fig.suptitle('Feature Track Error [px]', fontsize=10)
            axs[0].set_title('x-axis [px]', {'fontsize':8,'fontweight':2})
            axs[0].set_ylabel("estimated 'x' [px]", color='b')
            handle_0_0, = axs[0].plot(gt_ts ,gt_xs, 'k', linewidth=0.6, label="GT")
            handle_0_1, = axs[0].plot(gt_ts, out_fs[:,0],  'b', linewidth=0.6, label="estimate")
            axs0_cl = axs[0].twinx()
            handle_0_2, = axs0_cl.plot(gt_ts, errors_xs,  'r', linewidth=0.6, label="error")
            axs0_cl.axhline(y=0.0, xmin=0, xmax=1, color='r', linewidth=0.6)
            axs0_cl.set_ylabel('2d error [px]', color='r')
            axs0_cl.tick_params('y', colors='r')
            axs[0].legend(handles=[handle_0_0, handle_0_1, handle_0_2], loc='best', fontsize='small')

            axs[1].set_title('y-axis [px]', {'fontsize':8,'fontweight':2})
            axs[1].set_ylabel("estimated 'y' [px]", color='b')
            handle_1_0, = axs[1].plot(gt_ts ,gt_ys, 'k', linewidth=0.6, label="GT")
            handle_1_1, = axs[1].plot(gt_ts, out_fs[:,1],  'b', linewidth=0.6, label="estimate")
            axs1_cl = axs[1].twinx()
            handle_1_2, = axs1_cl.plot(gt_ts, errors_ys,  'r', linewidth=0.6, label="error")
            axs1_cl.axhline(y=0.0, xmin=0, xmax=1, color='r', linewidth=0.6)
            axs1_cl.set_ylabel('2d error [px]', color='r')
            axs1_cl.tick_params('y', colors='r')
            axs[1].legend(handles=[handle_1_0, handle_1_1, handle_1_2], loc='lower left', fontsize='small')

            axs[2].set_title('Error [px]', {'fontsize':8,'fontweight':2})
            axs[2].plot(gt_ts, errors,  'k', linewidth=1, label="GT")
            axs[2].legend(loc='best')

            # fn = out_dir / 'reprojection_error_%s.png' % out_index
            # plt.show()
            # plt.close(fig)
            fig.tight_layout()
            fig.savefig(png_fn)

        return errors.mean(), errors.std(), errors.max()

        ## 3D Plot
        # out_XS = np.array(out_XS, dtype=float)
        # out_YS = np.array(out_YS, dtype=float)
        # out_ZS = np.array(out_ZS, dtype=float)
        # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9), dpi=200)
        # fig.suptitle('3D Comparison', fontsize=12)
        # axs[0].set_title('X-axis [m]', {'fontsize':10,'fontweight':2})
        # axs[0].plot(gt_ts ,gt_XS, 'k', linewidth=1, label="GT")
        # axs[0].plot(gt_ts, out_XS,  'b', linewidth=1, label="est.")
        # axs[0].legend(loc='best')
        # axs[1].set_title('Y-axis [m]', {'fontsize':10,'fontweight':2})
        # axs[1].plot(gt_ts ,gt_YS, 'k', linewidth=1, label="GT")
        # axs[1].plot(gt_ts, out_YS,  'b', linewidth=1, label="est.")
        # axs[1].legend(loc='best')
        # axs[2].set_title('Z-axis [m]', {'fontsize':10,'fontweight':2})
        # axs[2].plot(gt_ts ,gt_ZS, 'k', linewidth=1, label="GT")
        # axs[2].plot(gt_ts, out_ZS,  'b', linewidth=1, label="est.")
        # axs[2].legend(loc='best')
        # fig.tight_layout()
        # plt.show()



    # Refine 하기 전으로 측정했을 때: mean: 3.1546
    # Refine 한 후로 측정했을 때: mean: 3.1907

if __name__ == '__main__':

    ##########
    # Case A #
    ##########
    # gt_fn = Path('/data/test3.csv')
    # for method in ['EFTv0']:
    #     for dataset, ids in {'case_a': ['230718_110256']}.items():
    #         for id in ids:
    #             out_fn = Path('/data/results/event_feature_tracking/case_a/%s/%s/0.csv' % (method, id))
    #             png_dir = Path('./results/%s/%s/%s' % (dataset, method, id))
    #             os.makedirs(png_dir, exist_ok=True)
    #             png_fn = png_dir / 'errors.png'
    #             mean_, std_ = eval_n_plot(out_fn, gt_fn, png_fn)
    #             print("[%s](%s-%s) means:%.4f, std:%.4f" % (method, dataset, id, mean_, std_))

    # case_A_dict = {
    #     'checkerboard_rhombus_dt01': ['230717_103644'],
    #     'checkerboard_rhombus_dt02': ['230717_103610'],
    #     'checkerboard_rhombus_dt03': ['230717_103731'],
    #     'checkerboard_rhombus_dt04': ['230717_103830'],
    #     'checkerboard_rhombus_dt05': ['230717_104034'],
    #     'checkerboard_rhombus_dt06': ['230717_104118'],
    #     'checkerboard_rhombus_dt07': ['230717_104224'],
    #     'checkerboard_rhombus_dt08': ['230717_104835'],
    #     'checkerboard_rhombus_dt09': ['230717_104920'],
    #     'checkerboard_rhombus_dt10': ['230717_105157'],
    # }
    # for method in ['EFTv0']:
    #     for dataset, ids in case_A_dict.items():
    #         gt_fn = '/data/datasets/dataset_ours/dt_%.1f.csv' % (float(dataset[-2:]) / 10.0)
    #         for id in ids:
    #             out_fn = Path('/data/results/event_feature_tracking/%s/%s/%s/0.csv' % (dataset, method, id))
    #             out_vfn = Path('/data/results/event_feature_tracking/%s/%s/%s/0v.csv' % (dataset, method, id))
    #             png_dir = Path('./%s/%s/%s' % (dataset, method, id))
    #             # print("out_fn: ", out_fn)
    #             # print("out_vfn: ", out_vfn)
    #             # print("png_dir: ", png_dir)
    #             os.makedirs(png_dir, exist_ok=True)
    #             png_fn = png_dir / 'errors.png'
    #             mean_, std_ = eval_n_plot(out_fn, gt_fn, png_fn)
    #             means.append(mean_)
    #             stds.append(std_)
    # print("means:\n", means)
    # print("stds:\n", stds)


    ##########
    # Case B #
    ##########
    # for method in ['haste_correlation', 'haste_correlation_star', 'haste_difference', 'haste_difference_star']:
    #     for dataset in ['case_B_dt05']:
    #         gt_fn = Path('/data/datasets/dataset_ours/case_B_dt_05.csv')
    #         out_fn = Path('/data/results/event_feature_tracking/%s/%s/eval.txt' % (dataset, method))
    #         png_dir = Path('./results/%s/%s' % (dataset, method))
    #         os.makedirs(png_dir, exist_ok=True)
    #         png_fn = png_dir / 'errors.png'
    #         eval_n_plot(out_fn, gt_fn, png_fn)

    #     for dataset in ['case_B_dt10']:
    #         gt_fn = Path('/data/datasets/dataset_ours/case_B_dt_10.csv')
    #         out_fn = Path('/data/results/event_feature_tracking/%s/%s/eval.txt' % (dataset, method))
    #         png_dir = Path('./results/%s/%s' % (dataset, method))
    #         os.makedirs(png_dir, exist_ok=True)
    #         png_fn = png_dir / 'errors.png'
    #         eval_n_plot(out_fn, gt_fn, png_fn)

    # for method in ['EFTv0']:
    #     dataset = 'case_B_dt05'
    #     gt_fn = Path('/data/datasets/dataset_ours/case_B_dt_05.csv')
    #     out_fn = Path('/data/results/event_feature_tracking/case_B_dt05/EFTv0/230716_230727/0.csv')
    #     png_dir = Path('./results/%s/%s' % (dataset, method))
    #     os.makedirs(png_dir, exist_ok=True)
    #     png_fn = png_dir / 'errors.png'
    #     eval_n_plot(out_fn, gt_fn, png_fn)

    #     dataset = 'case_B_dt10'
    #     gt_fn = Path('/data/datasets/dataset_ours/case_B_dt_10.csv')
    #     out_fn = Path('/data/results/event_feature_tracking/case_B_dt10/EFTv0/230716_230810/0.csv')
    #     png_dir = Path('./results/%s/%s' % (dataset, method))
    #     os.makedirs(png_dir, exist_ok=True)
    #     png_fn = png_dir / 'errors.png'
    #     eval_n_plot(out_fn, gt_fn, png_fn)

    ##########
    # Case A #
    ##########
    gt_fn = Path('/data/test3.csv')
    case_A_dict = {
        'checkerboard_rhombus_dt01': ['230717_103644'],
        'checkerboard_rhombus_dt02': ['230717_103610'],
        'checkerboard_rhombus_dt03': ['230717_103731'],
        'checkerboard_rhombus_dt04': ['230717_103830'],
        'checkerboard_rhombus_dt05': ['230717_104034'],
        'checkerboard_rhombus_dt06': ['230717_104118'],
        'checkerboard_rhombus_dt07': ['230717_104224'],
        'checkerboard_rhombus_dt08': ['230717_104835'],
        'checkerboard_rhombus_dt09': ['230717_104920'],
        'checkerboard_rhombus_dt10': ['230717_105157'],
    }
    for method in ['EFTv0']:
        for dataset, ids in case_A_dict.items():
            gt_fn = '/data/datasets/dataset_ours/dt_%.1f.csv' % (float(dataset[-2:]) / 10.0)
            for id in ids:
                out_fn = Path('/data/results/event_feature_tracking/%s/%s/%s/0.csv' % (dataset, method, id))
                out_vfn = Path('/data/results/event_feature_tracking/%s/%s/%s/0v.csv' % (dataset, method, id))
                png_dir = Path('./results/%s/%s/%s' % (dataset, method, id))
                # print("out_fn: ", out_fn)
                # print("out_vfn: ", out_vfn)
                # print("png_dir: ", png_dir)
                os.makedirs(png_dir, exist_ok=True)
                png_fn = png_dir / 'errors.png'
                mean_, std_, max_ = eval_n_plot(out_fn, gt_fn, png_fn, save_plot=False)
                print("[%s](%s-%s) means:%.4f, std:%.4f, max:%.4f" % (method, dataset, id, mean_, std_, max_))
                means.append(mean_)
                stds.append(std_)