% Parameters
P_l_bar = 3.0;                  % Constraint limit (maximum allowed power)
P_opt = 0.8 * P_l_bar;          % Optimal power value (peak of the bell)
sigma = 0.2 * P_l_bar;          % Standard deviation (controls width of the bell)
beta = 1.0;                     % Scaling factor for reward

% Power range (x-axis)
P_l = linspace(0, 1.5 * P_l_bar, 500);

% Gaussian reward function
reward = beta * exp(-((P_l - P_opt).^2) / (2 * sigma^2));
% Plot
figure;
plot(P_l, reward, 'b-', 'LineWidth', 4); hold on;
xline(P_opt, '--k', 'P_{opt}', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 2);
xline(P_l_bar, '--r', 'P_l^{max}', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 2);
xlabel('Power exchanged with the DSO (P_l)');
ylabel('Reward');
title('Gaussian-Based Reward Function Based on Power Exchange');
legend('reward(P_l)', 'P_{opt} (peak)', 'P_l^{max} (constraint)', 'Location', 'best');
grid on;
grid minor;
set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'on') ;