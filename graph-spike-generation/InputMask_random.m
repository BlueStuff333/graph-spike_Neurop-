function [mask_E, mask_I] = InputMask_random(e_loc, i_loc, Ne_in, Ni_in)

n = length(e_loc);

idx = find(e_loc);
keep = idx(randperm(length(idx), Ne_in));
mask_E = zeros(n, 1);
mask_E(keep) = 1;

idx = find(i_loc);
keep = idx(randperm(length(idx), Ni_in));
mask_I = zeros(n, 1);
mask_I(keep) = 1; 