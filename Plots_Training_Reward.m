clc;
clear;

% Récupérer le chemin du script courant
scriptDir = fileparts(mfilename('fullpath'));

% Noms des fichiers
file1 = 'reward_history_Classic_100k.csv';
file2 = 'reward_history_Gauss_100k.csv';

% Lecture des fichiers
data1 = readtable(fullfile(scriptDir, file1));
data2 = readtable(fullfile(scriptDir, file2));

% Extraire les colonnes
steps1 = data1.step;
reward1 = data1.reward;

steps2 = data2.step;
reward2 = data2.reward;

% Moyennes mobiles
window = 50;
reward1_avg = movmean(reward1, window);
reward2_avg = movmean(reward2, window);

% Créer la figure
figure;
hold on;

% Tracer Reward 1 avec transparence réelle
h1 = line(steps1, reward1);
h1.Color = [0.2 0.6 1.0 0.3];  % RGBA avec transparence à 30%
h1.DisplayName = 'Reward - Discontinuous Reward';

% Tracer Reward 2 avec transparence réelle
h2 = line(steps2, reward2);
h2.Color = [1.0 0.4 0.4 0.3];  % RGBA avec transparence à 30%
h2.DisplayName = 'Reward - Continuous Reward';

% Tracer les moyennes opaques
plot(steps1, reward1_avg, 'Color', [0 0 0.5], 'LineWidth', 2, 'DisplayName', 'Moving Avg - Discontinuous Reward');
plot(steps2, reward2_avg, 'Color', [0.5 0 0], 'LineWidth', 2, 'DisplayName', 'Moving Avg - Continuous Reward');

% Ajustements finaux
xlabel('Training steps');
ylabel('Reward');
title('Evolution of the Reward During the Training Process');
legend('Location', 'best');
grid on;
hold off;