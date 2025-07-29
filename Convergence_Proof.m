% === Parameters ===
csv_file_tradi = 'T_0_final_evolution.csv';  
csv_file_multi = 'Multi_Agents_Peak/T_final_evolution.csv';
pairs_to_plot = [0 1; 1 2; 0 2];     
n_agents = 3;                          

% === Load CSV files ===
T_tradi = readtable(csv_file_tradi);
T_multi = readtable(csv_file_multi);

iterations_tradi = T_tradi.Iteration;
iterations_multi = T_multi.Iteration;

% === Plot T(i → j) evolution ===
figure;
hold on;

% Use consistent colors for matching (i, j) pairs
colors = lines(size(pairs_to_plot, 1));

% Plot traditional simulation (dashed lines)
for k = 1:size(pairs_to_plot, 1)
    i = pairs_to_plot(k, 1);
    j = pairs_to_plot(k, 2);
    col_name = sprintf('T_0_%d_%d', i, j);
    
    if any(strcmp(T_tradi.Properties.VariableNames, col_name))
        plot(iterations_tradi, T_tradi.(col_name), '--', ...
            'DisplayName', sprintf('Traditional Market T_{%d→%d}', i + 1, j + 1), ...
            'Color', colors(k,:), ...
            'LineWidth', 2);
    else
        warning('Column %s not found in traditional dataset.', col_name);
    end
end

% Plot multi-agent simulation (solid lines)
for k = 1:size(pairs_to_plot, 1)
    i = pairs_to_plot(k, 1);
    j = pairs_to_plot(k, 2);
    col_name = sprintf('T_%d_%d', i+1, j+1);  % Adjust indexing if needed
    
    if any(strcmp(T_multi.Properties.VariableNames, col_name))
        plot(iterations_multi, T_multi.(col_name), '-', ...
            'DisplayName', sprintf('Multi-agent Market T_{%d→%d}', i, j), ...
            'Color', colors(k,:), ...
            'LineWidth', 2);
    else
        warning('Column %s not found in multi-agent dataset.', col_name);
    end
end

% === Axis labels and styling ===
xlabel('Iteration', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Power exchange T_{ij}', 'FontSize', 14, 'FontWeight', 'bold');
title('Evolution of Energy Exchanges: Traditional vs Multi-agent', ...
       'FontSize', 16, 'FontWeight', 'bold');

legend('Location', 'best', 'FontSize', 12);
grid on;
grid minor;
set(gca, 'FontSize', 12, 'LineWidth', 1.2);
