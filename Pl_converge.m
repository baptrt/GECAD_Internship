% Load data
data_spade = readtable('Multi_Agents_Peak/P_l_evolution.csv');
data_rl = readtable('Pl_history_with_signal.csv');
data_no_signal = readtable('Pl_history_without_signal.csv');

% Extract values
iter_spade = data_spade.Iteration;
Pl_spade = data_spade.P_l;

iter_rl = data_rl.Iteration;
Pl_rl = data_rl.P_l_with_signal;

iter_no_signal = data_no_signal.Iteration;
Pl_no_signal = data_no_signal.P_l_without_signal;

% Plot all three curves
figure;
plot(iter_spade, Pl_spade, 'b-', 'LineWidth', 1.8); hold on;
plot(iter_rl, Pl_rl, 'g--', 'LineWidth', 1.8);
plot(iter_no_signal, Pl_no_signal, 'r-.', 'LineWidth', 1.8);

% Labels and legend
xlabel('Iterations');
ylabel('Total P_l (Power exchanged with the grid)');
title('Evolution of P_l over Iterations');
legend('SPADE Agent', 'RL with Price Signal', 'No Price Signal');
grid on;
set(gca, 'FontSize', 12);
