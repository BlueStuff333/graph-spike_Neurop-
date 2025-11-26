function saveRasterPlots(firings, node_colors, raster_dir)
    
    % Plot 1: Color-coded raster
    fig1 = figure('Visible', 'off');
    hold on;
    unique_neurons = unique(firings(:, 2));
    for i = 1:length(unique_neurons)
        neuron_idx = unique_neurons(i);
        mask = firings(:, 2) == neuron_idx;
        color = node_colors(neuron_idx, :);
        plot(firings(mask, 1), firings(mask, 2), '|', ...
             'Color', color, 'MarkerSize', 2);
    end
    xlabel('Time');
    ylabel('Neuron Index');
    title('Raster');
    exportgraphics(fig1, fullfile(raster_dir, 'raster_colored.jpg'), 'Resolution', 300);
    close(fig1);
    
    % Plot 2: Clustered raster
    [~, ~, color_ids] = unique(node_colors, 'rows');
    [~, sorted_idx] = sort(color_ids);
    
    n = size(node_colors, 1);
    neuron_to_new_pos = zeros(n, 1);
    for i = 1:length(sorted_idx)
        neuron_to_new_pos(sorted_idx(i)) = i;
    end
    
    fig2 = figure('Visible', 'off');
    hold on;
    for i = 1:size(firings, 1)
        neuron_idx = firings(i, 2);
        new_idx = neuron_to_new_pos(neuron_idx);
        color = node_colors(neuron_idx, :);
        plot(firings(i, 1), new_idx, '|', 'Color', color, 'MarkerSize', 2);
    end
    xlabel('Time');
    ylabel('Neurons (Clustered by Community)');
    title('Raster');
    exportgraphics(fig2, fullfile(raster_dir, 'raster_clustered.jpg'), 'Resolution', 300);
    close(fig2);