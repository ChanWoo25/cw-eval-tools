function eval_haste(root_dir, dataset, method, force_overwrite)
% Summary of this function goes here
    input_fn   = sprintf("%s/%s/%s/eval.txt", root_dir, dataset, method);
    save_fn    = sprintf("/home/leecw/results/event_feature_tracking/%s/%s_errors.csv", dataset, method);
    summary_fn = sprintf("/home/leecw/results/event_feature_tracking/%s/%s_summary.csv", dataset, method);
    save_dir = sprintf("/home/leecw/results/event_feature_tracking/%s", dataset);
    status = mkdir(save_dir);

    if ~isfile(input_fn)
        fprintf("Reading '$s' fails. Check the file path.");
    end

    if isfile(save_fn) && not (force_overwrite) 
        i = 2;
        while isfolder(save_dir)
            save_fn = sprintf("%s/%s/%s_errors_%02d.csv", root_dir, dataset, method, i);
            summary_fn = sprintf("%s/%s/%s_summary_%02d.csv", root_dir, dataset, method, i);
            i = i + 1;
        end
    end
    % Notice
    fprintf("eval_haste [Dataset: %s, Method: %s, Root: %s]\n", dataset, method, root_dir);
    fprintf("- input:   %s \n", input_fn);
    fprintf("- error:   %s \n", save_fn);
    fprintf("- summary: %s \n", summary_fn);
    haste_pixel_threshold = 5.0;
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
    % intrinsics = cameraIntrinsics(focal_length, principal_point, image_size, "RadialDistortion", [-0.4136 0.2042]);
    
    %% Parse GT
    [gt_ts, gt_ps_xyz, gt_qs_wxyz] = parse_gt(dataset);

    %% Loop eval.txt
    fid = fopen(input_fn);
    line = fgetl(fid);

    old_idx = 0;
    old_px  = 0.0;
    old_py  = 0.0;
    old_t   = 0.0;
    vSet = viewSet;
    ourlier_flag = false;

    while ischar(line)
        % Function handle to split each row by comma delimiter 
        func = @(input)strsplit(input, ',');
        A     = func(line);
        t     = str2double(cell2mat(A(1, 1)));
        px    = str2double(cell2mat(A(1, 2)));
        py    = str2double(cell2mat(A(1, 3)));
        theta = str2double(cell2mat(A(1, 4)));
        index = str2num(cell2mat(A(1, 5)));
        % fprintf("line: t(%.6f), x(%.4f), y(%.4f), theta(%.4f), index(%d)\n", t, px, py, theta, index);
        
        if old_idx ~= index
            if mod(index, 10) == 0
                fprintf("Precessing: index %04d\n ", index);
            end
            old_idx = index;
            old_t = t;
            old_px = px;
            old_py = py;
            vSet = viewSet;
            viewId = 1;
            viewIds = [viewId];
            viewPoints = [px py];
            viewTimes = [t];
            ourlier_flag = false;

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
            if ourlier_flag
                line = fgetl(fid);
                continue;
            end

            dist = sqrt((px - old_px) * (px - old_px) + (py - old_py) * (py - old_py));
            if dist >= haste_pixel_threshold
                viewId = viewId + 1;
                % fprintf("- Add: view id %d\n", viewId);
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
                [xyzPoints, error, valid] = triangulateMultiview(tracks,cameraPoses,intrinsics);
                % xyzPoints
                if (error <= 5.0)
                    life_time = t - old_t;
                    % fprintf("# %d - track(%d/%d), error: %.6f, lifetime: %.6f, validity:%d \n", file_index, i, n_track, last_error, life_time, valid);
                    logging_line = sprintf("%d,%.6f,%.6f,%d", index, error, life_time, valid);
                    writelines(logging_line, save_fn, WriteMode="append");
                    summary_num_inliers = summary_num_inliers + 1;
                else
                    summary_num_outliers = summary_num_outliers + 1;
                    ourlier_flag = true;
                end
            end
        end

        line = fgetl(fid);
    end

    logging_line = sprintf("%s,%d,%d,%d", summary_dataset, summary_total_feature, summary_num_inliers, summary_num_outliers);
    writelines(logging_line, summary_fn, WriteMode="append");

    return
end
    % %% Parse GT
    % [gt_ts, gt_ps_xyz, gt_qs_wxyz] = parse_gt(dataset);
    % % fprintf("In eof track: 1468941032.22916555\n");
    % % fprintf("gt_ts(1):     %.8f", gt_ts(1));
    % % gt_ts = gt_ts - 1468941032.22916555;
    % 
    % summary_dataset = dataset;
    % summary_total_feature = 0;
    % summary_num_inliers = 0;
    % summary_num_outliers = 0;
    % 
    % header_1 = sprintf("# feature_index,detect_time[sec],avg_reprojection_error,life_time[sec],validity");
    % header_2 = sprintf("# dataset,num_total_features,num_inliers,num_outliers");
    % writelines(header_1, save_fn);
    % writelines(header_2, summary_fn);
    % 
    % gt_s_index = 1;
    % file_index = 1;
    % 
    % focal_length   = [230.2097, 231.1228];
    % principal_point = [121.6862 86.8208];
    % image_size      = [180 240];
    % intrinsics = cameraIntrinsics(focal_length, principal_point, image_size, "RadialDistortion", [-0.4136 0.2042]);
    % 
    % fmat = read_feature_track(dataset, file_index);
    % while not (size(fmat, 1) == 0)
    %     summary_total_feature = summary_total_feature + 1;
    % 
    %     % Find gt start index
    %     while gt_ts(gt_s_index+1, 1) < fmat(1, 1)
    %         gt_s_index = gt_s_index + 1;
    %     end
    % 
    %     % Find gt end index efficiently (from start index)
    %     gt_e_index = gt_s_index + 1;
    %     n_track = size(fmat, 1);
    %     while  gt_ts(gt_e_index, 1) <= fmat(n_track, 1)
    %         gt_e_index = gt_e_index + 1;
    %     end
    % 
    %     % GT interpolate for triangulation
    %     p = zeros(n_track, 4, 'double');
    %     q = zeros(n_track, 4, 'double');
    %     f = zeros(n_track, 1, 'double');
    %     interpolated_ps = zeros(n_track, 3, 'double');
    %     track_index = 1;
    %     for i=gt_s_index:gt_e_index-1
    %         t = fmat(track_index, 1);
    %         while (gt_ts(i) <= t) && (t < gt_ts(i+1))
    %             alpha = (t - gt_ts(i)) / (gt_ts(i+1) - gt_ts(i));
    %             % fprintf("[t:%.6f] alpha: %.4f", t, alpha);
    %             p(track_index, :) = gt_qs_wxyz(i  , :);
    %             q(track_index, :) = gt_qs_wxyz(i+1, :);
    %             f(track_index) = alpha;
    %             interpolated_ps(track_index,:) = (1-alpha) * gt_ps_xyz(i,:) + alpha * gt_ps_xyz(i+1,:);
    %             track_index = track_index + 1;
    % 
    %             if track_index <= n_track
    %                 t = fmat(track_index, 1);
    %             else
    %                 break;
    %             end
    %         end
    %     end
    % 
    %     % invalid track check
    %     if not (track_index == n_track + 1)
    %         fprintf("Track index: %d;", track_index);
    %         fprintf("n_track: %d;", n_track);
    %         fprintf("fmat");
    %         disp(fmat);
    %     end
    % 
    %     p = quatnormalize(p);
    %     q = quatnormalize(q);
    %     interpolated_qs_wxyz = quatinterp(p, q, f, 'slerp');
    %     interpolated_rots = quat2rotm(interpolated_qs_wxyz);
    % 
    %     vSet = viewSet;
    %     first_rot = interpolated_rots(:, :, 1);
    %     first_pos = interpolated_ps(1, :);
    %     vSet = addView(vSet, 1, "Orientation", first_rot, "Location", first_pos);
    %     last_error = 999.0;
    %     life_time = 0.0;
    %     for i=2:n_track
    %         rot = interpolated_rots(:, :, i);
    %         pos = interpolated_ps(i, :);
    %         vSet = addView(vSet, i, "Orientation", rot, "Location", pos);
    %         cameraPoses = poses(vSet);
    % 
    %         view_ids = 1:i;
    %         points = fmat(1:i, 2:3);
    %         tracks = pointTrack(view_ids, points);
    %         [xyzPoints, error, valid] = triangulateMultiview(tracks,cameraPoses,intrinsics);
    %         % xyzPoints
    %         if (error < 5.0)
    %             last_error = error;
    %             life_time = fmat(i, 1) - fmat(1, 1);
    %         else
    %             break;
    %         end
    %         % disp(valid);
    %     end
    % 
    %     if (last_error < 5.0)
    %         summary_num_inliers = summary_num_inliers + 1;
    %         % fprintf("# %d - track(%d/%d), error: %.6f, lifetime: %.6f, validity:%d \n", file_index, i, n_track, last_error, life_time, valid);
    %         logging_line = sprintf("%d,%.6f,%.6f,%d", file_index, last_error, life_time, valid);
    %         writelines(logging_line, save_fn, WriteMode="append");
    %     else
    %         summary_num_outliers = summary_num_outliers + 1;
    %     end
    % 
    %     file_index = file_index + 1;
    %     fmat = read_feature_track(dataset, file_index);
    % end
    % logging_line = sprintf("%s,%d,%d,%d", summary_dataset, summary_total_feature, summary_num_inliers, summary_num_outliers);
    % writelines(logging_line, summary_fn, WriteMode="append");


