function read_mat = read_feature_track(dataset_name, index)
    csv_fn = sprintf('/home/leecw/Reps/event_feature_tracking/EventFeatureTracking/Results/%s/%d.csv', dataset_name, index);
    if isfile(csv_fn)
        read_mat = readmatrix(csv_fn);
    else
        read_mat = [];
    end
end
