function results = analyze_movement_distances(run_dir)
    % ANALYZE_MOVEMENT_DISTANCES Calculate average movement distances for each agent type
    %
    % Args:
    %   run_dir: Directory containing baseline result CSV files
    %
    % Returns:
    %   results: Struct containing movement statistics
    %
    % Example:
    %   results = analyze_movement_distances('baseline_results/run_20260130_214308');
    
    % Find all CSV files (excluding trajectory-specific files)
    csv_files = dir(fullfile(run_dir, '*.csv'));
    csv_files = csv_files(~contains({csv_files.name}, 'trajectory'));
    
    if isempty(csv_files)
        fprintf('No CSV files found in %s\n', run_dir);
        results = [];
        return;
    end
    
    fprintf('\n%s\n', repmat('=', 1, 80));
    fprintf('MOVEMENT DISTANCE ANALYSIS\n');
    fprintf('%s\n', repmat('=', 1, 80));
    fprintf('Directory: %s\n', run_dir);
    fprintf('Found %d baseline methods\n\n', length(csv_files));
    
    % Initialize results structure
    results = struct();
    
    % Process each baseline method
    for idx = 1:length(csv_files)
        % Extract baseline name
        csv_path = fullfile(csv_files(idx).folder, csv_files(idx).name);
        baseline_name = strrep(csv_files(idx).name, '.csv', '');
        
        % Sanitize field name (remove invalid characters for struct field)
        field_name = baseline_name;
        field_name = strrep(field_name, '-', '_');  % Replace hyphens
        field_name = strrep(field_name, ' ', '_');  % Replace spaces
        field_name = matlab.lang.makeValidName(field_name);  % Ensure valid MATLAB name
        
        % Clean name for display
        clean_name = strrep(baseline_name, '_', ' ');
        
        fprintf('%s\n', repmat('-', 1, 80));
        fprintf('Processing: %s\n', clean_name);
        fprintf('  Field name: %s\n', field_name);
        fprintf('%s\n', repmat('-', 1, 80));
        
        % Read CSV
        data = readtable(csv_path);
        
        % Find UAV columns
        var_names = data.Properties.VariableNames;
        uav_ids = {};
        for i = 1:length(var_names)
            if contains(var_names{i}, '_pos_x')
                uav_id = strrep(var_names{i}, '_pos_x', '');
                uav_ids{end+1} = uav_id;
            end
        end
        
        uav_ids = sort(uav_ids);
        num_uavs = length(uav_ids);
        
        % Calculate distances for each UAV
        uav_distances = zeros(1, num_uavs);
        uav_details = cell(num_uavs, 1);
        
        for uav_idx = 1:num_uavs
            uav_id = uav_ids{uav_idx};
            
            % Extract position columns
            x_col = [uav_id '_pos_x'];
            y_col = [uav_id '_pos_y'];
            z_col = [uav_id '_pos_z'];
            
            x = data.(x_col);
            y = data.(y_col);
            z = data.(z_col);
            
            % Calculate cumulative distance
            total_distance = 0;
            for i = 1:(length(x) - 1)
                dx = x(i+1) - x(i);
                dy = y(i+1) - y(i);
                dz = z(i+1) - z(i);
                
                step_distance = sqrt(dx^2 + dy^2 + dz^2);
                total_distance = total_distance + step_distance;
            end
            
            uav_distances(uav_idx) = total_distance;
            
            % Store detailed info
            uav_details{uav_idx} = struct(...
                'id', uav_id, ...
                'total_distance', total_distance, ...
                'num_steps', length(x), ...
                'avg_step_distance', total_distance / (length(x) - 1), ...
                'start_pos', [x(1), y(1), z(1)], ...
                'end_pos', [x(end), y(end), z(end)] ...
            );
            
            % Print per-UAV statistics
            fprintf('  %s:\n', uav_id);
            fprintf('    Total distance:       %8.2f m\n', total_distance);
            fprintf('    Number of steps:      %8d\n', length(x));
            fprintf('    Avg per step:         %8.2f m\n', total_distance / (length(x) - 1));
            fprintf('    Start position:       (%.1f, %.1f, %.1f)\n', x(1), y(1), z(1));
            fprintf('    End position:         (%.1f, %.1f, %.1f)\n', x(end), y(end), z(end));
            
            % Calculate straight-line distance
            straight_line = sqrt((x(end) - x(1))^2 + (y(end) - y(1))^2 + (z(end) - z(1))^2);
            fprintf('    Straight-line dist:   %8.2f m\n', straight_line);
            fprintf('    Path efficiency:      %8.2f%%\n\n', (straight_line / total_distance) * 100);
        end
        
        % Calculate statistics for this baseline
        avg_distance = mean(uav_distances);
        std_distance = std(uav_distances);
        min_distance = min(uav_distances);
        max_distance = max(uav_distances);
        total_fleet_distance = sum(uav_distances);
        
        % Print baseline summary
        fprintf('  SUMMARY for %s:\n', clean_name);
        fprintf('    Average distance per UAV:  %8.2f m ± %.2f m\n', avg_distance, std_distance);
        fprintf('    Min distance:              %8.2f m\n', min_distance);
        fprintf('    Max distance:              %8.2f m\n', max_distance);
        fprintf('    Total fleet distance:      %8.2f m\n', total_fleet_distance);
        fprintf('    Distance variance:         %8.2f m²\n\n', std_distance^2);
        
        % Store results
        results.(field_name) = struct(...
            'baseline_name', clean_name, ...
            'num_uavs', num_uavs, ...
            'uav_details', {uav_details}, ...
            'avg_distance', avg_distance, ...
            'std_distance', std_distance, ...
            'min_distance', min_distance, ...
            'max_distance', max_distance, ...
            'total_fleet_distance', total_fleet_distance, ...
            'individual_distances', uav_distances ...
        );
    end
    
    % Print comparison table
    fprintf('%s\n', repmat('=', 1, 80));
    fprintf('COMPARISON TABLE\n');
    fprintf('%s\n', repmat('=', 1, 80));
    fprintf('%-30s %12s %12s %12s %15s\n', 'Baseline Method', 'Avg Dist (m)', 'Std Dev', 'Min', 'Max');
    fprintf('%s\n', repmat('-', 1, 80));
    
    baseline_names = fieldnames(results);
    for i = 1:length(baseline_names)
        method = results.(baseline_names{i});
        fprintf('%-30s %12.2f %12.2f %12.2f %15.2f\n', ...
                method.baseline_name, ...
                method.avg_distance, ...
                method.std_distance, ...
                method.min_distance, ...
                method.max_distance);
    end
    fprintf('%s\n', repmat('=', 1, 80));
    
    % Create visualization
    create_distance_plots(results, run_dir);
    
    fprintf('\n✓ Analysis complete!\n\n');
end

function create_distance_plots(results, run_dir)
    % Create visualization plots for movement distances
    
    baseline_names = fieldnames(results);
    num_baselines = length(baseline_names);
    
    if num_baselines == 0
        return;
    end
    
    % Prepare data for plotting
    methods = cell(1, num_baselines);
    avg_distances = zeros(1, num_baselines);
    std_distances = zeros(1, num_baselines);
    
    for i = 1:num_baselines
        method = results.(baseline_names{i});
        methods{i} = method.baseline_name;
        avg_distances(i) = method.avg_distance;
        std_distances(i) = method.std_distance;
    end
    
    % Create figure with two subplots
    fig = figure('Position', [100, 100, 1200, 500]);
    
    % Subplot 1: Bar chart with error bars
    subplot(1, 2, 1);
    hold on;
    
    colors = [0.12, 0.47, 0.71;   % blue
              0.84, 0.15, 0.16;   % red
              0.17, 0.63, 0.17;   % green
              1.00, 0.50, 0.00;   % orange
              0.58, 0.40, 0.74;   % purple
              0.55, 0.34, 0.29];  % brown
    
    x = 1:num_baselines;
    for i = 1:num_baselines
        color_idx = mod(i - 1, size(colors, 1)) + 1;
        bar(x(i), avg_distances(i), 'FaceColor', colors(color_idx, :), ...
            'EdgeColor', 'k', 'LineWidth', 1.5);
    end
    
    % Add error bars
    errorbar(x, avg_distances, std_distances, 'k.', 'LineWidth', 2, ...
             'CapSize', 10);
    
    % Configure subplot
    set(gca, 'XTick', x);
    set(gca, 'XTickLabel', methods);
    xtickangle(45);
    ylabel('Average Distance (m)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Average Movement Distance by Method', 'FontSize', 13, 'FontWeight', 'bold');
    grid on;
    ax = gca;
    ax.GridAlpha = 0.3;
    ax.FontSize = 11;
    hold off;
    
    % Subplot 2: Box plot for each method
    subplot(1, 2, 2);
    hold on;
    
    % Prepare data for box plot
    all_distances = [];
    group_labels = [];
    
    for i = 1:num_baselines
        method = results.(baseline_names{i});
        distances = method.individual_distances;
        
        all_distances = [all_distances, distances];
        group_labels = [group_labels, repmat(i, 1, length(distances))];
    end
    
    % Create box plot
    boxplot(all_distances, group_labels, 'Labels', methods, ...
            'Colors', colors(1:min(num_baselines, size(colors, 1)), :), ...
            'Widths', 0.5);
    
    % Configure subplot
    xtickangle(45);
    ylabel('Distance (m)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Distance Distribution by Method', 'FontSize', 13, 'FontWeight', 'bold');
    grid on;
    ax = gca;
    ax.GridAlpha = 0.3;
    ax.FontSize = 11;
    hold off;
    
    % Save figure
    output_path = fullfile(run_dir, 'movement_distances_analysis.pdf');
    exportgraphics(fig, output_path, 'ContentType', 'vector', 'Resolution', 300);
    fprintf('\n✓ Saved distance analysis plot to %s\n', output_path);
end