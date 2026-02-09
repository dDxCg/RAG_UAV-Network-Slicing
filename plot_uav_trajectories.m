function plot_uav_trajectories(run_dir)
    % PLOT_TRAJECTORY Plot UAV trajectories from saved CSV files with color gradients
    %
    % Args:
    %   run_dir: Directory containing the CSV files with trajectory data
    
    % Find all CSV files (excluding trajectory-specific files)
    csv_files = dir(fullfile(run_dir, '*.csv'));
    csv_files = csv_files(~contains({csv_files.name}, 'trajectory'));
    
    if isempty(csv_files)
        fprintf('No CSV files found in %s\n', run_dir);
        return;
    end
    
    fprintf('Found %d baseline result files\n', length(csv_files));
    
    % Base colors for different UAVs
    base_colors = [0 0 1;      % blue
                   1 0.647 0;  % orange
                   0 0.5 0;    % green
                   1 0 0;      % red
                   0.5 0 0.5;  % purple
                   0.647 0.165 0.165]; % brown
    
    % Create figure with subplots for each baseline
    num_baselines = length(csv_files);
    fig = figure('Position', [100, 100, 700*num_baselines, 600]);
    
    for idx = 1:num_baselines
        % Extract baseline name from filename
        csv_path = fullfile(csv_files(idx).folder, csv_files(idx).name);
        baseline_name = strrep(csv_files(idx).name, '.csv', '');
        baseline_name = strrep(baseline_name, '_', ' ');
        fprintf('\nProcessing: %s\n', baseline_name);
        
        % Read CSV
        data = readtable(csv_path);
        
        % Extract 500 steps from data
        if height(data) > 500
            data = data(1:500, :);
        end
        
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
        fprintf('  Found %d UAVs: %s\n', length(uav_ids), strjoin(uav_ids, ', '));
        
        % Create subplot
        subplot(1, num_baselines, idx);
        hold on;
        
        % Plot each UAV's trajectory with color gradient
        for uav_idx = 1:length(uav_ids)
            uav_id = uav_ids{uav_idx};
            
            % Extract positions
            x_col = [uav_id '_pos_x'];
            y_col = [uav_id '_pos_y'];
            
            x = data.(x_col);
            y = data.(y_col);
            
            % Get base color for this UAV
            color_idx = mod(uav_idx - 1, size(base_colors, 1)) + 1;
            base_color = base_colors(color_idx, :);
            
            % Create color gradient (from pale to strong)
            n_points = length(x);
            alphas = linspace(0.1, 1.0, n_points - 1);
            
            % Plot trajectory with varying alpha
            for i = 1:(n_points - 1)
                alpha_val = alphas(i);
                plot([x(i), x(i+1)], [y(i), y(i+1)], ...
                     'Color', [base_color alpha_val], 'LineWidth', 2, ...
                     'HandleVisibility', 'off');
            end
            
            % Mark start position (pale) - circle
            scatter(x(1), y(1), 200, base_color, 'o', 'filled', ...  % ← Increased from 150 to 200
                   'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'k', ...
                   'LineWidth', 2, 'HandleVisibility', 'off');
            
            % Mark end position (strong) - square
            % scatter(x(end), y(end), 200, base_color, 's', 'filled', ...  % ← Increased from 150 to 200
            %        'MarkerFaceAlpha', 1.0, 'MarkerEdgeColor', 'k', ...
            %        'LineWidth', 2, 'DisplayName', uav_id);
            scatter(x(end), y(end), 200, base_color, 's', 'filled', ...
                   'MarkerFaceAlpha', 1.0, 'MarkerEdgeColor', 'k', ...
                   'LineWidth', 2, 'HandleVisibility', 'off'); % <--- Quan trọng: off để không hiện trong legend cũ
            
            % 2. Vẽ điểm ẢO cho Legend (Tọa độ NaN để không hiện trên hình, nhưng Size cực to)
            scatter(NaN, NaN, 500, base_color, 's', 'filled', ...  % <--- Tăng 200 lên 500 hoặc 600 tùy ý
                   'MarkerFaceAlpha', 1.0, 'MarkerEdgeColor', 'k', ...
                   'LineWidth', 2, 'DisplayName', uav_id);     % <--- Cái này mới hiện tên trong legend
            % --- KẾT THÚC SỬA ---
            
            fprintf('    ✓ Plotted trajectory for %s: %d points\n', uav_id, length(x));
        end
        
        % Configure subplot
        xlabel('X (m)', 'FontSize', 13, 'FontWeight', 'bold');
        ylabel('Y (m)', 'FontSize', 13, 'FontWeight', 'bold');
        title(baseline_name, 'FontSize', 14, 'FontWeight', 'bold');
        
        % ============================================
        % FIXED: Larger legend with better formatting
        % ============================================
        lgd = legend('Location', 'southeast', ...
                     'FontSize', 14, ...           % ← Increased from default (~10)
                     'FontWeight', 'light', ...     % ← Make bold
                     'Box', 'on');            % ← Thicker box
        
        % Optional: Increase marker size in legend
        lgd.ItemTokenSize = [30, 30];   % ← [width, height] of legend markers
        
        grid on;
        ax = gca;
        ax.GridAlpha = 0.3;
        ax.FontSize = 12;  % ← Increase axis tick font size
        
        axis equal;
        xlim([0 2000]);
        ylim([0 2000]);
        hold off;
    end
    
    % Save combined figure
    output_path_combined = fullfile(run_dir, 'trajectories_combined.png');
    print(fig, output_path_combined, '-dpng', '-r150');
    fprintf('\n✓ Saved combined trajectory plot to %s\n', output_path_combined);
    
    % Save each subplot separately
    for idx = 1:num_baselines
        baseline_name = strrep(csv_files(idx).name, '.csv', '');
        
        % Create new figure
        fig_single = figure('Position', [100, 100, 800, 700], 'Visible', 'off');  % ← Larger figure
        
        % Re-read and plot
        csv_path = fullfile(csv_files(idx).folder, csv_files(idx).name);
        data = readtable(csv_path);
        
        if height(data) > 500
            data = data(1:500, :);
        end
        
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
        
        hold on;
        
        % Plot trajectories
        for uav_idx = 1:length(uav_ids)
            uav_id = uav_ids{uav_idx};
            x_col = [uav_id '_pos_x'];
            y_col = [uav_id '_pos_y'];
            
            x = data.(x_col);
            y = data.(y_col);
            
            color_idx = mod(uav_idx - 1, size(base_colors, 1)) + 1;
            base_color = base_colors(color_idx, :);
            
            n_points = length(x);
            alphas = linspace(0.1, 1.0, n_points - 1);
            
            for i = 1:(n_points - 1)
                alpha_val = alphas(i);
                plot([x(i), x(i+1)], [y(i), y(i+1)], ...
                     'Color', [base_color alpha_val], 'LineWidth', 2, ...
                     'HandleVisibility', 'off');
            end
            
            scatter(x(1), y(1), 200, base_color, 'o', 'filled', ...
                   'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'k', ...
                   'LineWidth', 2, 'HandleVisibility', 'off');
            
            scatter(x(end), y(end), 200, base_color, 's', 'filled', ...
                   'MarkerFaceAlpha', 1.0, 'MarkerEdgeColor', 'k', ...
                   'LineWidth', 2, 'DisplayName', uav_id);
        end
        
        % Configure plot
        baseline_title = strrep(baseline_name, '_', ' ');
        xlabel('X (m)', 'FontSize', 13, 'FontWeight', 'bold');
        ylabel('Y (m)', 'FontSize', 13, 'FontWeight', 'bold');
        title(baseline_title, 'FontSize', 14, 'FontWeight', 'bold');
        
        % Large legend for individual plots
        lgd = legend('Location', 'best', ...
                     'FontSize', 14, ...
                     'FontWeight', 'bold', ...
                     'Box', 'on', ...
                     'LineWidth', 1.5);
        lgd.ItemTokenSize = [60, 30];
        
        grid on;
        ax = gca;
        ax.GridAlpha = 0.3;
        ax.FontSize = 12;
        
        axis equal;
        xlim([0 2000]);
        ylim([0 2000]);
        hold off;
        
        % Save
        output_path = fullfile(run_dir, sprintf('trajectory_%s.png', baseline_name));
        print(fig_single, output_path, '-dpng', '-r150');
        fprintf('✓ Saved trajectory plot to %s\n', output_path);
        close(fig_single);
    end
    
    fprintf('\n✓ All trajectory plots generated successfully!\n');
end