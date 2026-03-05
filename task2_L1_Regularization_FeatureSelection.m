function [W, cost_history] = regularizedFeatureSelection(X, y, lambdas)
    % Initialize variables
    [P, N] = size(X);
    L = length(lambdas);
    
    % W stores the weights for each lambda. Dimensions: L x (N+1)
    W = zeros(L, N + 1);
    % cost_history stores the final training cost for each lambda
    cost_history = zeros(L, 1);
    
    % Go through all the lambdas
    for i = 1:L
        % Initialize weights to zero for EACH independent lambda run
        w0 = zeros(1, N + 1);
        
        % Call the local function
        [w_best, c_best] = trainPerceptronL1(X, y, lambdas(i), w0);
        
        % Return trained weights in the matrix W
        W(i, :) = w_best;
        
        % Return the training cost in the cost_history
        cost_history(i) = c_best;
    end
    
end

function [w, cost] = trainPerceptronL1(X, y, lambda, w0)
    % Initialize problem using the exact hints from the assignment
    alpha = 0.1;      
    max_iter = 1000;  
    
    % Perform gradient descent 
    [~, w_best, c_hist, ~] = gradientDescentAD(@cost_softmax, w0, alpha, max_iter);
    
    % Return the best weights and the associated regularized cost
    w = w_best;
    cost = c_hist(end);
        
    % L1-regularized Softmax cost function
    function c = cost_softmax(w)
        [P, ~] = size(X);
        
        % 1. Augment data with a bias term
        X_aug = [ones(P, 1), X];
        
        % 2. Compute linear scores (X_hat * w^T)
        scores = X_aug * w';
        
        % 3. Data Loss (Softmax Cost / Logistic Loss)
        
        data_loss = (1/P) * sum(log(1 + exp(-y .* scores)));
        
        % 4. L1 Regularization Penalty
        
        l1_penalty = lambda * sum(abs(w(2:end)));
        
        % Total Cost
        c = data_loss + l1_penalty;
    end

end
