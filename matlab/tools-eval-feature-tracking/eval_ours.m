function eval_ours(root_dir, dataset, method, force_overwrite)
% Summary of this function goes here
% dataset='shapes_6dof';
% method='ours_v1';
% root_dir='/data/results/event_feature_tracking';
% force_overwrite=true;

save_dir = sprintf("/home/leecw/results/event_feature_tracking/%s", dataset);
save_fn    = sprintf("%s/%s_errors.csv", save_dir, method);
summary_fn = sprintf("%s/%s_summary.csv", save_dir, method);
status = mkdir(save_dir);
reg = sprintf("%s/%s/%s/*.csv", root_dir, dataset, method);
list = dir(reg);
N = size(list, 1);
if N == 0
    fprintf("Path may have problem.");
end

if isfile(save_fn) && not (force_overwrite) 
    i = 2;
    while isfolder(save_dir)
        save_fn = sprintf("%s/%s/%s_errors_%02d.csv", root_dir, dataset, method, i);
        summary_fn = sprintf("%s/%s/%s_summary_%02d.csv", root_dir, dataset, method, i);
        i = i + 1;
    end
end

csv_list = zeros(N, 1, 'int32');
for i=1:N
    index = str2num(extractBefore(list(i).name, ".csv"));
    csv_list(i) = index;
end
csv_list = sort(csv_list);

% Notice
fprintf("eval_haste [Dataset: %s, Method: %s, Root: %s]\n", dataset, method, root_dir);
fprintf("- error:   %s \n", save_fn);
fprintf("- summary: %s \n", summary_fn);
pixel_threshold = 5.0;
eval_once_per_index = false;
% Summary values
summary_dataset = dataset;
summary_total_feature = 0;
summary_num_inliers = 0;
summary_num_outliers = 0;
% Headers
header_1 = sprintf("# feature_index,detect_time[sec],avg_reprojection_error,life_time[sec],validity");
header_2 = sprintf("# dataset,num_total_features,num_inliers,num_outliers");
writelines(header_1, save_fn);
writelines(header_2, summary_fn);
% Intrinsics
% focal_length   = [230.2097, 231.1228];
focal_length   = [230.0, 230.0];
% principal_point = [121.6862 86.8208];
principal_point = [120.0 90.0];
image_size      = [180 240];
intrinsics = cameraIntrinsics(focal_length, principal_point, image_size, "RadialDistortion", [0.0 0.0]); % distort [-0.4136 0.2042]

%% Parse GT
[gt_ts, gt_ps_xyz, gt_qs_wxyz] = parse_gt(dataset);

for idx=1:N
    index = csv_list(idx);
    if mod(idx, 10) == 0
        fprintf("Precessing %d-th: index %04d\n ", idx, index);
    end
    fmat = read_single_tracks(root_dir, dataset, method, index);
    summary_total_feature = summary_total_feature + 1;
    
    if size(fmat, 1) < 2
        continue;
    end

    ourlier_flag = false;
    for i=1:size(fmat,1)
        if ourlier_flag
            break;
        end

        t     = fmat(i,1);
        px    = fmat(i,2);
        py    = fmat(i,3);
        % fprintf("[%d] t(%.6f), x(%.4f), y(%.4f)\n", index, t, px, py);
        
        if i == 1
            old_t = t;
            old_px = px;
            old_py = py;
            vSet = viewSet;
            viewId = 1;
            viewIds = [viewId];
            viewPoints = [px py];
            viewTimes = [t];
    
            % Interpolate Block
            prev_idx = interp1(gt_ts, 1:length(gt_ts), t, "previous");
            next_idx = prev_idx + 1;
            if (t < gt_ts(prev_idx) || gt_ts(next_idx) < t)
                fprintf("Error A: inter time wrong");
            end
            prev_t = gt_ps_xyz(prev_idx, :);
            next_t = gt_ps_xyz(next_idx, :);
            prev_q_wxyz = gt_qs_wxyz(prev_idx, :);
            next_q_wxyz = gt_qs_wxyz(next_idx, :);
            alpha = (t - gt_ts(prev_idx)) / (gt_ts(next_idx) - gt_ts(prev_idx));
            inter_t = (1-alpha) * prev_t + alpha * next_t;
            prev_q_wxyz = quatnormalize(prev_q_wxyz);
            next_q_wxyz = quatnormalize(next_q_wxyz);
            inter_q_wxyz = quatinterp(prev_q_wxyz, next_q_wxyz, alpha, 'slerp');
            inter_rot = quat2rotm(inter_q_wxyz);
            vSet = addView(vSet, viewId, "Orientation", inter_rot, "Location", inter_t);
            % Interpolate Block
        else
            dist = sqrt((px - old_px) * (px - old_px) + (py - old_py) * (py - old_py));
            if dist >= pixel_threshold
                old_px = px;
                old_py = py;
                viewId = viewId + 1;

                % Interpolate Block
                prev_idx = interp1(gt_ts, 1:length(gt_ts), t, "previous");
                next_idx = prev_idx + 1;
                if (t < gt_ts(prev_idx) || gt_ts(next_idx) < t)
                    fprintf("Error B: inter time wrong");
                end
                prev_t = gt_ps_xyz(prev_idx, :);
                next_t = gt_ps_xyz(next_idx, :);
                prev_q_wxyz = gt_qs_wxyz(prev_idx, :);
                next_q_wxyz = gt_qs_wxyz(next_idx, :);
                alpha = (t - gt_ts(prev_idx)) / (gt_ts(next_idx) - gt_ts(prev_idx));
                inter_t = (1-alpha) * prev_t + alpha * next_t;
                prev_q_wxyz = quatnormalize(prev_q_wxyz);
                next_q_wxyz = quatnormalize(next_q_wxyz);
                inter_q_wxyz = quatinterp(prev_q_wxyz, next_q_wxyz, alpha, 'slerp');
                inter_rot = quat2rotm(inter_q_wxyz);
                vSet = addView(vSet, viewId, "Orientation", inter_rot, "Location", inter_t);
                % Interpolate Block
                
                % Triangulate
                viewIds = [viewIds(:); viewId];
                viewPoints = [viewPoints(:,:); px py];
                viewTimes = [viewTimes(:); t];
                cameraPoses = poses(vSet);
                tracks = pointTrack(viewIds, viewPoints);
                [xyzPoints, error, valid] = triangulateMultiview(tracks, cameraPoses, intrinsics);
                % xyzPoints
                if (error <= 5.0)
                    life_time = t - old_t;
                    % fprintf("# %d - track(%d/%d), error: %.6f, lifetime: %.6f, validity:%d \n", file_index, i, n_track, last_error, life_time, valid);
                    logging_line = sprintf("%d,%.6f,%.6f,%d", index, error, life_time, valid);
                    writelines(logging_line, save_fn, WriteMode="append");
                % else
                    % ourlier_flag = true;
                end
            end
        end
    end
end
end