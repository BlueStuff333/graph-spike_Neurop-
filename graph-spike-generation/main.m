clear all; close all; clc;

%% Setup

% Multithreading
if isempty(gcp('nocreate')) % if no pool, create new
    parpool("threads");   % or parpool(N) for process-based pool
end

% generated network storage directory
batchstamp = string(datetime('now'), "dd-MMM-yyyy_HH-mm-ss");
batchdir = fullfile('networks', batchstamp);
mkdir(batchdir);
mkdir(fullfile('networks', batchstamp, 'train'));
mkdir(fullfile('networks', batchstamp, 'test'));

%% Randomized parameter sets of the following forms:
% K = 1:5, M = 2, p_ij(r) and l_i same
% K = 4, M = 2, R = 3, 4 different sets of initial unit square model parameters
% K = 5, M = 2, R = 3, l_i [(0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.9, 0.1)], p_ij(r) same
% R = 1:4
%% generate n graph-spiking pairs
n_graphs = 1000; % num graphs to make
train_test = 0.8;
n_zeros = ceil(log10(n_graphs));
isDirected = 1; isBinary = 0;
% adj_lists = cell(n_graphs, 1);    % These cause memory issues for large n_graphs
% firings_list = cell(n_graphs, 1); % TODO develop a separate pipeline for visualization from saved .mat files
% randomized P0 bound values [0.2, 0.8]
P0 = 0.1 + 0.6 * rand(2,2);
noise_level = 0.1;
lb = 0.15;             % lower bound per entry
total_mass = 1;        % sum of all 4 entries
free_mass = total_mass - 4 * lb; % mass left to distribute after lower bounds

densities = cell(n_graphs, 1);
tic
parfor i = 1:n_graphs
    % randomize parameters
    n = 200 %+ randi([-100, 100]); % num neurons to generate
    r = 0.8; % proportion of neurons that are E
    Ne = floor(0.8*n); Ni = n-Ne;
    i_scaling = 3; % how much stronger than excitation we want inhibition to be

    %% multifractal parameters
    R = randi(3); % resolution
    % build p_ij matrices
    P = zeros(2,2,R);
    for r_idx = 1:R
        % start near P0 with noise
        P_raw = P0 + noise_level * (rand(2,2) - 0.5);

        % flatten to 4-vector
        p = P_raw(:);

        % enforce lower bound first
        p = max(p, lb);  % each entry >= 0.1

        % project (p - lb) onto simplex of sum free_mass (0.6)
        p_shift = p - lb;
        p_shift = max(p_shift, 0);  % numerical safety
        p_proj = project_to_simplex(p_shift, free_mass);

        % shift back: now entries sum to 1 and are >= lb
        p_final = lb + p_proj;

        % reshape back to 2x2
        P(:,:,r_idx) = reshape(p_final, 2, 2);
    end
    % P = repmat(P, 1, 1, R);
    % randomize side length
    l1 = 0.2 + 0.6 * rand(); % [0.2, 0.8]
    L = [l1, 1-l1];
    M = length(L); 
    K = randi([1,4]);

    % generate the graph
    PK = calcPK_weighted(M, K, P);
    [LK, LKcum] = calcLK(M, K, L);
    [adj] = generateNetworkWMGM(PK, LK, LKcum, n, isDirected, isBinary);
    densities{i} = nnz(adj) / numel(adj);

    % decide which neurons are e and i
    idx = randperm(n, Ni);
    i_locs = zeros(n, 1);
    i_locs(idx) = 1;
    e_locs = ones(n,1) - i_locs;
    adj(:,logical(i_locs)) = -1 * i_scaling * adj(:,logical(i_locs));

    % generate the spikes
    [firings, stl, y] = generateSpikes(adj, e_locs, i_locs);

    % save the graph and its spikes in a .mat file
    filename = sprintf(['graph-spike_%0', num2str(n_zeros), 'd.mat'], i);
    if( i <= n_graphs * train_test)
        out_path = fullfile(batchdir, 'train', filename);
    else
        out_path = fullfile(batchdir, 'test', filename);
    end
    
    S = struct( ...
        'adj', adj, ... 
        'e_locs', e_locs, ... 
        'i_locs', i_locs, ...
        'firings', firings, ...
        ... %%% WMGM params
        'P', P, ...
        'L', L, ...
        'M', M, ...
        'K', K, ...
        'R', R ...
    );
    save(out_path, '-fromstruct', S);

    % free memory
    adj = []; firings = []; e_locs = []; i_locs = [];

    %clc; 
    disp(num2str(i));
    disp("Density: " + num2str(densities{i}));
    disp("WMGM params:");
    disp(['N: ', num2str(n), ', K: ', num2str(K), ', M: ', num2str(M), ', R: ', num2str(R)]);
    disp(['L: ', num2str(L)]);
end
fprintf('Graph spike gen time: %.2f\n', toc);
avg_density = mean(cell2mat(densities));
disp(['Average density: ', num2str(avg_density)]);

% TODO enable/disable visualization at will
% for i = 1:n_graphs
%     % create & save jpg of the graph
%     plotname = sprintf('graph_%d', i);
%     plotsdir = fullfile(batchdir, 'plots'); mkdir (plotsdir);
%     node_colors = saveCliqueColors(adj_lists{i}, plotsdir, plotname);
%     % save the rasters in a jpg
%     raster_dir = fullfile(batchdir, 'rasters'); mkdir(raster_dir);
%     saveRasterPlots(firings_list{i}, node_colors, raster_dir, i);
% end

disp('complete');

function x = project_to_simplex(v, z)
%PROJECT_TO_SIMPLEX Project v onto {x >= 0, sum(x) = z}.
% v: column vector
% z: desired sum (default 1)

    if nargin < 2
        z = 1;
    end

    v = v(:);
    n = numel(v);

    % Sort in descending order
    [u, ~] = sort(v, 'descend');
    cssv = cumsum(u);

    % Find rho
    rho = find(u + (z - cssv) ./ (1:n)' > 0, 1, 'last');
    theta = (cssv(rho) - z) / rho;

    w = max(v - theta, 0);
    x = w;  % already column vector with sum(x) = z
end