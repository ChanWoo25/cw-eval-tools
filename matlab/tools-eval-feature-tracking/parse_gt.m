function [ts, ps_xyz, qs_wxyz] = parse_gt(dataset)
%PARSE_GT Load GT and pre-process
gt_fn  = sprintf("/data/datasets/dataset_ecd/%s/groundtruth.txt", dataset);
gt_mat = readmatrix(gt_fn);
ts      = gt_mat(:, 1);
ps_xyz  = gt_mat(:, 2:4);
qs_xyz  = gt_mat(:, 5:7); 
qs_w    = gt_mat(:, 8); 
qs_wxyz = [qs_w, qs_xyz];
end
