function convert2seedfile(dataset, root_dir, save_dir)
%CONVERT2SEEDFILE Summary of this function goes here
%   Detailed explanation goes here

reg = sprintf("%s/%s/*.csv", root_dir, dataset);
strs = dir(reg);
N = size(strs, 1);
seed_content = zeros(N, 5, 'double');

for i=1:N
    fmat = read_feature_track(dataset, i);
    seed_content(i, 1:3) = fmat(1, 1:3);
    seed_content(i, 4) = 0.0;
    seed_content(i, 5) = i; 
    if mod(i, 500) == 0
        disp(seed_content(i, :));
    end
end

save_fn = sprintf("%s/seed_%s.csv", save_dir, dataset);
writematrix(seed_content, save_fn);

end % function

