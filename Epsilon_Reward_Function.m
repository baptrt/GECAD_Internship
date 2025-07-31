% Paramètres
P_l_bar = 3.0;           % Limite de puissance autorisée
beta = 1.0;              % Pondération de l'énergie
epsilon = 1.0;           % Zone de transition douce près de la contrainte

% Axe des puissances échangées
P_l = linspace(0, 1.5 * P_l_bar, 500);

% Initialisation de la récompense
reward = zeros(size(P_l));

% Calcul de la récompense selon la règle par morceaux
for i = 1:length(P_l)
    P = P_l(i);
    if P <= P_l_bar - epsilon
        reward(i) = beta * (P^2) / P_l_bar^2;
    elseif P <= P_l_bar
        x = (P_l_bar - P) / epsilon;  % x ∈ (0,1)
        reward(i) = beta * (P^2) / P_l_bar^2 * x^2;
    else
        margin = P - P_l_bar;
        reward(i) = - beta * (margin^2) / P_l_bar^2;
    end
end

% Tracer la fonction
figure;
plot(P_l, reward, 'b-', 'LineWidth', 3); hold on;
xline(P_l_bar - epsilon, '--g', 'P_l^{max} - \epsilon', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 2);
xline(P_l_bar, '--r', 'P_l^{max}', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 2);

xlabel('Power exchanged with the DSO (P_l)');
ylabel('Reward');
title('Epsilon-Based Reward Function for Power Exchange');
legend('reward(P_l)', 'Transition zone start', 'Constraint limit', 'Location', 'best');

grid on;
grid minor;
set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'on');
