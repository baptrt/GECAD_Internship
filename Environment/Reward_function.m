% Parameters
n_agents = 4;
P_l_bar = 3.0;
beta = 1;
delta = 0.01;

% Simulated gamma vectors
last_gamma = rand(1, n_agents + 1);
prev_gamma = rand(1, n_agents + 1);

% Gamma variation penalty
gamma_delta = norm(last_gamma - prev_gamma) / ((n_agents + 1)^2);

% Power exchange values
P_l_low = linspace(0, P_l_bar, 100);         % P_l ≤ P_l_bar
P_l_high = linspace(P_l_bar, 6, 100);        % P_l > P_l_bar

% Reward computations
reward_low = (beta * (P_l_low.^2)) / P_l_bar^2; %- delta * gamma_delta;
margin_high = P_l_high - P_l_bar;
reward_high = - beta * (margin_high.^2) / P_l_bar^2; %- delta * gamma_delta;

% Plotting
figure;
hold on;

% Plot reward regions
plot(P_l_low, reward_low, 'b', 'LineWidth', 4);    % Blue for P_l <= P_l_bar
plot(P_l_high, reward_high, 'r', 'LineWidth', 4);  % Red for P_l > P_l_bar

% Vertical dotted black line at P_l_bar
yl = ylim; % get current y-limits
plot([P_l_bar P_l_bar], yl, 'k--', 'LineWidth', 2);  % Dotted vertical line

% Styling
grid on;
grid minor;
xlabel('Power Exchange with Grid (P_l)');
ylabel('Reward');
title('Reward Function Based on Power Exchange');
legend('P_l ≤ P_l_{bar}', 'P_l > P_l_{bar}', 'P_l_{bar} Threshold', 'Location', 'best');

% Axis limits (reset y-limits to avoid line clipping)
xlim([0 6]);
ylim(yl); % restore original y-limits after adding line

