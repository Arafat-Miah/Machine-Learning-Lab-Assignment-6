function [ W, best_lambda_ind ] = regularizedL1CrossValidation( y_train, y_validate, lambdas )

    % Initialize variables
    L = length(lambdas);
    N = length(y_train);
    
    % W stores the trained model weights for each lambda.
    W = zeros(L, N);
    cv_costs = zeros(L, 1);

    
    parfor i = 1:L
        % Train the model with ith value in lambdas
        [w_trained, ~] = regularizedL1Fitting(y_train, lambdas(i));
        W(i, :) = w_trained;
        
        % Compare each model to y_validate in LS cost sense
        cv_costs(i) = (1/N) * sum((w_trained - y_validate).^2);
    end

    % Determine the best crossvalidation cost and select that model
    [~, best_lambda_ind] = min(cv_costs);
    
end

function [w, cost] = regularizedL1Fitting(y_train, lambda)
    
    % Use output value y as a starting point
    w0 = y_train;
    N = length(y_train);

    % Initialize other parameters
    ALPHA = 0.05;      
    MAX_ITER = 2000;   
    
    % --- Construct the Delta^2 Matrix ---
    
    D2 = zeros(N-2, N);
    for i = 1:(N-2)
        D2(i, i)   = 1;
        D2(i, i+1) = -2;
        D2(i, i+2) = 1;
    end
    
    % Perform gradient descent
    [~, w_best, cost_history, ~] = gradientDescentAD(@costfun, w0, ALPHA, MAX_ITER);
    
    % Pick the best value from the history
    w = w_best;
    cost = cost_history(end);
        
    % The cost function
    function c = costfun(w)
        
        % 1. Least Squares (LS) cost
        ls_cost = (1/N) * sum((w - y_train).^2);
        
        % 2. L1 norm of Delta^2 w using strictly supported matrix multiplication
        
        second_diffs = D2 * w';
        reg_penalty = lambda * sum(abs(second_diffs));
        
        % Compute total cost
        c = ls_cost + reg_penalty;
        
    end
%code to call the function
    % Load the data
load('problem4.mat')

% Set up a range of lambda values
LAMBDAS = 10.^(-3:.25:-1.25);

% Find the best model across LAMBDAS
[ W, best_lambda_ind ] = regularizedL1CrossValidation( y_noisy, y_validate, LAMBDAS );


% Plot the result (not mandatory, but beneficial)
figure

subplot(211)
plot( x, y_true )
hold on;
plot( x, y_noisy, 'x' )
plot( x, y_validate, 'd')
plot( x, W' )
title('All the models')

subplot(212)
plot( x, y_true )
hold on;
plot( x, y_noisy, 'x' )
plot( x, y_validate, 'd')
plot( x, W(best_lambda_ind,:)', 'LineWidth', 3 )
title('The selected model')


end
