function eval_dataset(dataset_name, root_dir, force_overwrite)
%EVAL_DATASET Summary of this function goes here
%   Detailed explanation goes here
    save_dir = sprintf("%s/%s", root_dir, dataset_name);
    if isfolder(save_dir) && not (force_overwrite) 
        i = 2;
        while isfolder(save_dir)
            save_dir = sprintf("%s/%s_%02d", root_dir, dataset_name, i);
            i = i + 1;
        end
    end
    save_error_fn    = sprintf("%s/errors.csv", save_dir);
    save_summary_fn  = sprintf("%s/summary.csv", save_dir);
    mkdir_status     = mkdir(save_dir);
    [gt_ts, gt_ps_xyz, gt_qs_wxyz] = parse_gt(dataset_name);
    % fprintf("In eof track: 1468941032.22916555\n");
    % fprintf("gt_ts(1):     %.8f", gt_ts(1));
    % gt_ts = gt_ts - 1468941032.22916555;
    
    summary_dataset = dataset_name;
    summary_total_feature = 0;
    summary_num_inliers = 0;
    summary_num_outliers = 0;
    
    header_1 = sprintf("# feature_index,detect_time[sec],avg_reprojection_error,life_time[sec],validity");
    header_2 = sprintf("# dataset_name,num_total_features,num_inliers,num_outliers");
    writelines(header_1, save_error_fn);
    writelines(header_2, save_summary_fn);
    
    gt_s_index = 1;
    file_index = 1;
    
    focal_length   = [230.2097, 231.1228];
    principal_point = [121.6862 86.8208];
    image_size      = [180 240];
    intrinsics = cameraIntrinsics(focal_length, principal_point, image_size, "RadialDistortion", [-0.4136 0.2042]);
    
    fmat = read_feature_track(dataset_name, file_index);
    while not (size(fmat, 1) == 0)
        summary_total_feature = summary_total_feature + 1;
        
        % Find gt start index
        while gt_ts(gt_s_index+1, 1) < fmat(1, 1)
            gt_s_index = gt_s_index + 1;
        end
    
        % Find gt end index efficiently (from start index)
        gt_e_index = gt_s_index + 1;
        n_track = size(fmat, 1);
        while  gt_ts(gt_e_index, 1) <= fmat(n_track, 1)
            gt_e_index = gt_e_index + 1;
        end
    
        % GT interpolate for triangulation
        p = zeros(n_track, 4, 'double');
        q = zeros(n_track, 4, 'double');
        f = zeros(n_track, 1, 'double');
        interpolated_ps = zeros(n_track, 3, 'double');
        track_index = 1;
        for i=gt_s_index:gt_e_index-1
            t = fmat(track_index, 1);
            while (gt_ts(i) <= t) && (t < gt_ts(i+1))
                alpha = (t - gt_ts(i)) / (gt_ts(i+1) - gt_ts(i));
                % fprintf("[t:%.6f] alpha: %.4f", t, alpha);
                p(track_index, :) = gt_qs_wxyz(i  , :);
                q(track_index, :) = gt_qs_wxyz(i+1, :);
                f(track_index) = alpha;
                interpolated_ps(track_index,:) = (1-alpha) * gt_ps_xyz(i,:) + alpha * gt_ps_xyz(i+1,:);
                track_index = track_index + 1;
    
                if track_index <= n_track
                    t = fmat(track_index, 1);
                else
                    break;
                end
            end
        end
    
        % invalid track check
        if not (track_index == n_track + 1)
            fprintf("Track index: %d;", track_index);
            fprintf("n_track: %d;", n_track);
            fprintf("fmat");
            disp(fmat);
        end
    
        p = quatnormalize(p);
        q = quatnormalize(q);
        interpolated_qs_wxyz = quatinterp(p, q, f, 'slerp');
        interpolated_rots = quat2rotm(interpolated_qs_wxyz);
        
        vSet = viewSet;
        first_rot = interpolated_rots(:, :, 1);
        first_pos = interpolated_ps(1, :);
        vSet = addView(vSet, 1, "Orientation", first_rot, "Location", first_pos);
        last_error = 999.0;
        life_time = 0.0;
        for i=2:n_track
            rot = interpolated_rots(:, :, i);
            pos = interpolated_ps(i, :);
            vSet = addView(vSet, i, "Orientation", rot, "Location", pos);
            cameraPoses = poses(vSet);
    
            view_ids = 1:i;
            points = fmat(1:i, 2:3);
            tracks = pointTrack(view_ids, points);
            [xyzPoints, error, valid] = triangulateMultiview(tracks,cameraPoses,intrinsics);
            % xyzPoints
            if (error < 5.0)
                last_error = error;
                life_time = fmat(i, 1) - fmat(1, 1);
            else
                break;
            end
            % disp(valid);
        end
    
        if (last_error < 5.0)
            summary_num_inliers = summary_num_inliers + 1;
            % fprintf("# %d - track(%d/%d), error: %.6f, lifetime: %.6f, validity:%d \n", file_index, i, n_track, last_error, life_time, valid);
            logging_line = sprintf("%d,%.6f,%.6f,%d", file_index, last_error, life_time, valid);
            writelines(logging_line, save_error_fn, WriteMode="append");
        else
            summary_num_outliers = summary_num_outliers + 1;
        end
    
        file_index = file_index + 1;
        fmat = read_feature_track(dataset_name, file_index);
    end
    logging_line = sprintf("%s,%d,%d,%d", summary_dataset, summary_total_feature, summary_num_inliers, summary_num_outliers);
    writelines(logging_line, save_summary_fn, WriteMode="append");
end

