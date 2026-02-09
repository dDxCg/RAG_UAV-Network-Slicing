function fig = plot_training_convergence(varargin)
% PLOT_TRAINING_CONVERGENCE Plot training convergence for multiple methods
% Same as plot_baseline_comparison but with different name

    % Simple parameter parsing without inputParser (more robust)
    paths = {};
    names = {};
    smoothing_window = [];
    % plot_title = 'Training Convergence';
    save_path = '';
    
    % Parse arguments
    i = 1;
    while i <= length(varargin)
        if ischar(varargin{i}) || isstring(varargin{i})
            switch lower(char(varargin{i}))
                case 'paths'
                    i = i + 1;
                    paths = varargin{i};
                case 'names'
                    i = i + 1;
                    names = varargin{i};
                case 'smoothing_window'
                    i = i + 1;
                    smoothing_window = varargin{i};
                case 'title'
                    i = i + 1;
                    plot_title = varargin{i};
                case 'save_path'
                    i = i + 1;
                    save_path = varargin{i};
                otherwise
                    error('Unknown parameter: %s', varargin{i});
            end
        end
        i = i + 1;
    end
    
    % Convert single path to cell
    if ischar(paths) || isstring(paths)
        paths = {paths};
    end
    
    % Auto-generate names if not provided
    if isempty(names)
        names = cell(size(paths));
        for j = 1:length(paths)
            [~, fname, ~] = fileparts(paths{j});
            names{j} = strrep(fname, '_', ' ');
        end
    end
    
    % Colors
    colors = [
        0.12, 0.47, 0.71;  % Blue
        0.84, 0.15, 0.16;  % Red
        0.17, 0.63, 0.17;  % Green
        1.00, 0.50, 0.00;  % Orange
        0.58, 0.40, 0.74;  % Purple
        0.55, 0.34, 0.29;  % Brown
        0.89, 0.47, 0.76;  % Pink
        0.50, 0.50, 0.50;  % Gray
    ];
    
    % Load data
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('Loading training data from %d files...\n', length(paths));
    fprintf('%s\n', repmat('=', 1, 70));
    
    data_collection = {};
    
    for j = 1:length(paths)
        fprintf('  [%d/%d] Loading: %s\n', j, length(paths), paths{j});
        
        if ~isfile(paths{j})
            fprintf('    ⚠️  File not found, skipping...\n');
            continue;
        end
        
        try
            data = readtable(paths{j});
            
            % Find step column
            if ismember('step', data.Properties.VariableNames)
                step_col = 'step';
            elseif ismember('steps', data.Properties.VariableNames)
                step_col = 'steps';
            else
                fprintf('    ⚠️  Missing step column, skipping...\n');
                continue;
            end
            
            % Find reward column
            if ismember('reward', data.Properties.VariableNames)
                reward_col = 'reward';
            elseif ismember('rewards', data.Properties.VariableNames)
                reward_col = 'rewards';
            else
                fprintf('    ⚠️  Missing reward column, skipping...\n');
                continue;
            end
            
            % Store data
            item.steps = data.(step_col);
            item.rewards = data.(reward_col);
            item.name = names{j};
            data_collection{end+1} = item;
            
            fprintf('    ✓ Loaded %d steps (Mean: %.4f, Final: %.4f)\n', ...
                    height(data), mean(item.rewards), item.rewards(end));
            
        catch ME
            fprintf('    ⚠️  Error: %s\n', ME.message);
        end
    end
    
    if isempty(data_collection)
        error('No valid data files loaded!');
    end
    
    fprintf('\n✓ Successfully loaded %d files\n', length(data_collection));
    fprintf('%s\n\n', repmat('=', 1, 70));
    
    % Create figure
    fig = figure('Position', [100, 100, 580, 360]);
    hold on;
    
    % Plot each dataset
    for j = 1:length(data_collection)
        data = data_collection{j};
        color_idx = mod(j - 1, size(colors, 1)) + 1;
        color = colors(color_idx, :);
        
        % Smoothing
        if isempty(smoothing_window)
            window = min(1500, floor(length(data.rewards) / 10));
        else
            window = smoothing_window;
        end
        
        if window > 1
            smoothed = movmean(data.rewards, window);
            plot(data.steps, smoothed, 'Color', color, ...
                 'LineWidth', 2.5, 'DisplayName', data.name);
        else
            plot(data.steps, data.rewards, 'Color', color, ...
                 'LineWidth', 2.5, 'DisplayName', data.name);
        end
    end
    
    % Format plot
    xlabel('Training Step', 'FontWeight', 'bold', 'FontSize', 13);
    ylabel('Reward', 'FontWeight', 'bold', 'FontSize', 13);
    % title(plot_title, 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 12);
    grid on;
    ax = gca;
    ax.FontSize = 13;
    alpha(ax, 0.3);
    hold off;
    
    % Save if requested
    if ~isempty(save_path)
        saveas(fig, save_path);
        [pth, nam, ~] = fileparts(save_path);
        exportgraphics(fig, fullfile(pth, [nam, '.pdf']), 'ContentType', 'vector');
        fprintf('✓ Saved to: %s\n', save_path);
    end
    
    % Print statistics
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('SUMMARY\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-30s %12s %12s\n', 'Method', 'Mean Reward', 'Final Reward');
    fprintf('%s\n', repmat('-', 1, 70));
    for j = 1:length(data_collection)
        data = data_collection{j};
        fprintf('%-30s %12.4f %12.4f\n', data.name, ...
                mean(data.rewards), data.rewards(end));
    end
    fprintf('%s\n', repmat('=', 1, 70));
end