% Paramètres
P_l_bar = 3.0;           % Limite maximale autorisée
beta = 1.0;              % Pondération de l'énergie
epsilon = 1.0;           % Marge de tolérance

% Axe des puissances échangées
P_l = linspace(0, 1.5 * P_l_bar, 500);

% Séparer les domaines
P_safe = P_l(P_l <= P_l_bar - epsilon);
P_penalty = P_l(P_l > P_l_bar - epsilon);

% Calcul de la récompense dans la zone "safe"
reward_safe = beta * (P_safe.^2) / P_l_bar^2;

% Calcul de la récompense dans la zone pénalisée
margin = P_penalty - P_l_bar + epsilon;
reward_penalty = - beta * (margin.^2) / P_l_bar^2;

% Tracé
figure; hold on;

% Courbes disjointes
h1 = plot(P_safe, reward_safe, 'b-', 'LineWidth', 3);       % Avant la discontinuité
h2 = plot(P_penalty, reward_penalty, 'b-', 'LineWidth', 3); % Après la discontinuité

% Lignes verticales de repère (avec handles)
h3 = xline(P_l_bar - epsilon, '--k', 'P_l^{max} - \epsilon', ...
    'LabelVerticalAlignment', 'bottom', 'LineWidth', 2);
h4 = xline(P_l_bar, '--r', 'P_l^{max}', ...
    'LabelVerticalAlignment', 'bottom', 'LineWidth', 2);

% Mise en forme
xlabel('Power exchanged with the DSO (P_l)');
ylabel('Reward');
title('Piecewise Reward Function with Discontinuity');
legend([h1 h3 h4], {'reward(P_l)', 'Safe zone limit', 'Constraint limit'}, 'Location', 'best');

grid on;
grid minor;
set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'on');
