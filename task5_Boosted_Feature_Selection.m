function [selected, w_selected, cost_history] = boostedFeatureSelection(X,y,M)

    % << IMPLEMENT THE FUNCTION BODY >>
    
    % Initialize variables
    [P, N] = size(X);
    
    
    selected = zeros(M + 1, 1);
    w_selected = zeros(M + 1, 1);
    cost_history = zeros(M + 1, 1);
    
    
    X_aug = [ones(P, 1), X];
    
    
    available_features = 2:(N + 1);
    % Initially select only the bias and solve the corresponding weight
    x_bias = X_aug(:, 1);
    w0 = x_bias \ y; 
    
    selected(1) = 1;     
    w_selected(1) = w0;
    
    % Current model prediction
    y_pred = x_bias * w0;
    
    % LS cost: 1/P * sum((y_pred - y)^2)
    cost_history(1) = mean((y_pred - y).^2);
    % Perform M rounds of boosting
    for i = 1:M
        
        % Form the residual for the previous round model
        r = y - y_pred;
        
        best_cost = inf;
        best_w = 0;
        best_idx = 0;
        % Compare all unselected feature candidates one-by-one:
        for j = 1:length(available_features)
            cand_idx = available_features(j);
            x_cand = X_aug(:, cand_idx);
            % Fit it to residual to determine the optimal weight, and the resulting cost
            w_cand = x_cand \ r;
            
            % The resulting cost if we were to add this feature
            cand_cost = mean((x_cand * w_cand - r).^2);
            
            % If this candidate reduces the cost the most, save it
            % Using strictly `<` ensures we pick the first one from left-to-right on a tie
            if cand_cost < best_cost
                best_cost = cand_cost;
                best_w = w_cand;
                best_idx = cand_idx;
            end
        end
        % After going through all the candidates:
        
            % Find and select the best from candidates
                available_features(available_features == best_idx) = [];
            % Store the cost of the selected
               cost_history(i + 1) = best_cost; 
            % Store the weight for the selected
        w_selected(i + 1) = best_w;
        selected(i + 1) = best_idx;
        
        % Update the current model's predictions for the next boosting round
        y_pred = y_pred + X_aug(:, best_idx) * best_w;
    end    

end
%code to call the function
% Load data
A = load('boston_housing.csv');

% Extract features and normalize them
X = A(1:end-1,:)';
X = normalize(X);

% Extract output variable and normalize it
y = A(end,:)';
y = normalize(y);

% Number of boosting rounds
M = 5;

[selected, w_selected, cost_history] = boostedFeatureSelection(X,y,M);



% Plot the result (not mandatory, but beneficial)
figure

subplot(211)
plot( 0:M, cost_history )
xticks( 0:M )
xticklabels( selected - 1 )
xlim([-0.5 M+0.5])
ylim([0 1.1*max(cost_history)])
title('LS cost at each round of boosting')
xlabel('Selected feature')


subplot(212)
bar( 0:M, w_selected )
xticks( 0:M )
xticklabels( selected - 1 )
xlim([-0.5 M+0.5])
ylim([min(w_selected)-0.1 max(w_selected)+0.1])
title('Weight values learnt by boosting')
xlabel('Selected feature')
