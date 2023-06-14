function read_mat = read_single_tracks(root_dir, dataset, method, index)
    csv_fn = sprintf('%s/%s/%s/%d.csv', root_dir, dataset, method, index);
    if isfile(csv_fn)
        read_mat = readmatrix(csv_fn);
    else
        fprintf("'%s' doesn't exist!", csv_fn);
        read_mat = [];
    end
end