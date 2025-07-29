% === Paramètres ===
csv_file_tradi = 'T_0_final_evolution.csv';  
csv_file_multi = 'Multi_Agents_Peak/T_final_evolution.csv';
pairs_to_plot = [0 1; 1 2; 0 2];     
n_agents = 3;                          

% === Lecture des fichiers CSV ===
T_tradi = readtable(csv_file_tradi);
T_multi = readtable(csv_file_multi);

iterations_tradi = T_tradi.Iteration;
iterations_multi = T_multi.Iteration;

% === Tracé des échanges T_0(i,j) ===
figure;
hold on;

% Couleurs pour uniformiser les courbes correspondantes
colors = lines(size(pairs_to_plot, 1));

% Tracé simulation traditionnelle (pointillés)
for k = 1:size(pairs_to_plot, 1)
    i = pairs_to_plot(k, 1);
    j = pairs_to_plot(k, 2);
    col_name = sprintf('T_0_%d_%d', i, j);
    
    if any(strcmp(T_tradi.Properties.VariableNames, col_name))
        plot(iterations_tradi, T_tradi.(col_name), '--', ...
            'DisplayName', sprintf('Traditionnel T_{%d→%d}', i, j), ...
            'Color', colors(k,:));
    else
        warning('Colonne %s non trouvée dans le fichier traditionnel.', col_name);
    end
end

% Tracé simulation multi-agents (trait plein)
for k = 1:size(pairs_to_plot, 1)
    i = pairs_to_plot(k, 1);
    j = pairs_to_plot(k, 2);
    col_name = sprintf('T_%d_%d', i+1, j+1);  % même nom attendu
    
    if any(strcmp(T_multi.Properties.VariableNames, col_name))
        plot(iterations_multi, T_multi.(col_name), '-', ...
            'DisplayName', sprintf('Multi-agents T_{%d→%d}', i, j), ...
            'Color', colors(k,:));
    else
        warning('Colonne %s non trouvée dans le fichier multi-agents.', col_name);
    end
end

xlabel('Itération');
ylabel('Échange T_{ij}');
title('Comparaison des échanges d''énergie : traditionnel vs multi-agents');
legend('Location', 'best');
grid on;
