function node_colors = saveCliqueColors(A, batchdir, filename)
    A_sym = A | A';
    A_sym = double(A_sym);
    
    node_clique = louvain(A_sym);
    
    colors = hsv(max(node_clique));
    node_colors = colors(node_clique, :);
    
    G_directed = digraph(A);
    fig = figure('Visible', 'off');
    plot(G_directed, 'NodeColor', node_colors, 'MarkerSize', 8);
    
    exportgraphics(fig, fullfile(batchdir, [filename '.jpg']), 'Resolution', 300);
    close(fig);

function communities = louvain(A)
    n = size(A, 1);
    communities = (1:n)';
    m = sum(A(:)) / 2;
    
    if m == 0
        return;
    end
    
    k = sum(A, 2);
    improved = true;
    
    while improved
        improved = false;
        for i = randperm(n)
            ci = communities(i);
            
            neighbors = find(A(i,:) > 0);
            neighbor_comms = unique(communities(neighbors));
            neighbor_comms(neighbor_comms == ci) = [];
            
            best_comm = ci;
            best_delta = 0;
            
            for c = reshape(neighbor_comms, 1, [])
                delta = modularityGain(A, communities, i, c, k, m);
                if delta > best_delta
                    best_delta = delta;
                    best_comm = c;
                end
            end
            
            if best_comm ~= ci
                communities(i) = best_comm;
                improved = true;
            end
        end
    end
    
    [~, ~, communities] = unique(communities);

function delta = modularityGain(A, communities, node, new_comm, k, m)
    old_comm = communities(node);
    
    new_comm_nodes = (communities == new_comm);
    old_comm_nodes = (communities == old_comm);
    old_comm_nodes(node) = false;
    
    k_i_in_new = sum(A(node, new_comm_nodes));
    k_i_in_old = sum(A(node, old_comm_nodes));
    
    sigma_new = sum(k(new_comm_nodes));
    sigma_old = sum(k(old_comm_nodes)) + k(node);
    
    delta = (k_i_in_new - k_i_in_old) / m - ...
            k(node) * (sigma_new - sigma_old + k(node)) / (2 * m^2);