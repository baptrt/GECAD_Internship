% Load datadata_spade = readtable('Multi_Agents_Peak/P_l_evolution_gamma.csv');
data_rl = readtable('Pl_history_with_signal.csv');
data_no_signal = readtable('Pl_history_without_signal.csv');

% Exclude first iteration
iter_spade = data_spade.Iteration(2:end);
Pl_spade = data_spade.P_l(2:end);

iter_rl = data_rl.Iteration(2:end);
Pl_rl = data_rl.P_l_with_signal(2:end);

iter_no_signal = data_no_signal.Iteration(2:end);
Pl_no_signal = data_no_signal.P_l_without_signal(2:end);

% Plot all three curves with updated colors
figure;
h1 = plot(iter_spade, Pl_spade, 'k-', 'LineWidth', 3); hold on;        % SPADE: black
h2 = plot(iter_rl, Pl_rl, 'b-', 'LineWidth', 3);                       % RL: blue
h3 = plot(iter_no_signal, Pl_no_signal, 'g-', 'LineWidth', 3);         % No signal: green

plot(iter_spade, Pl_spade, 'kx', 'LineWidth', 1.5, 'MarkerSize', 8);   
plot(iter_rl, Pl_rl, 'bx', 'LineWidth', 1.5, 'MarkerSize', 8);         
plot(iter_no_signal, Pl_no_signal, 'gx', 'LineWidth', 1.5, 'MarkerSize', 8); 

% Last values
plot(iter_spade(end), Pl_spade(end), 'ko', 'MarkerFaceColor', 'k');
text(iter_spade(end), Pl_spade(end), sprintf('  %.2f a.u.', Pl_spade(end)), ...
    'Color', 'k', 'FontSize', 24, 'VerticalAlignment', 'bottom');

plot(iter_rl(end), Pl_rl(end), 'bo', 'MarkerFaceColor', 'b');
text(iter_rl(end), Pl_rl(end), sprintf('  %.2f a.u.', Pl_rl(end)), ...
    'Color', 'b', 'FontSize', 24, 'VerticalAlignment', 'bottom');

plot(iter_no_signal(end), Pl_no_signal(end), 'go', 'MarkerFaceColor', 'g');
text(iter_no_signal(end), Pl_no_signal(end), sprintf('  %.2f a.u.', Pl_no_signal(end)), ...
    'Color', 'g', 'FontSize', 24, 'VerticalAlignment', 'bottom');

h4 = yline(3, 'r--', 'LineWidth', 2.5);

% Labels and legend
xlabel('Iterations','FontWeight','bold');
ylabel('Total power exchanged with the DSO (a.u.)','FontWeight','bold');
title('Evolution of Power Exchanged over Iterations');
legend([h1 h2 h3 h4], ...
       {'Multi-Agent Market with Price Signal', ...
        'Traditional Market with Price Signal', ...
        'Traditional Market without Price Signal', ...
        'Threshold P_l^{max} = 3 a.u.'}, ...
       'FontSize', 24, 'Location', 'best');

grid on;
grid minor;
set(gca, 'FontSize', 32);
