% function [outputArg1,outputArg2] = read_feature_track(inputArg1,inputArg2)
% %READ_FEATURE_TRACK Summary of this function goes here
% %   Detailed explanation goes here
% outputArg1 = inputArg1;
% outputArg2 = inputArg2;
% end

function read_mat = read_feature_track(dataset_name, index)
    csv_fn = sprintf('/home/leecw/Reps/event_feature_tracking/EventFeatureTracking/Results/%s/%d.csv', dataset_name, index);
    if isfile(csv_fn)
        read_mat = readmatrix(csv_fn);
    else
        read_mat = [];
    end
end
