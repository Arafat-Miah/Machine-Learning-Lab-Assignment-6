function [ w, c, W, C ] = fitKFoldPolynomial( x, y, K, M )

    % Initialize variables 
    P = length(x);
    W = zeros(M + 1, M + 1, K); % Weights: (M+1) x (M+1) x K
    C = zeros(M + 1, K);        % Costs: (M+1) x K
    
    fold_sizes = zeros(1, K);
    base_fold_size = floor(P / K);
    % For all folds
    for k = 1:K
        
        % Select the fold data: validation and training
        start_idx = (k - 1) * base_fold_size + 1;
        if k == K
            % The last fold takes all remaining samples
            end_idx = P;
        else
            end_idx = k * base_fold_size;
        end
        
        val_idx = start_idx:end_idx;
        % Training data is everything that is NOT in the validation set
        train_idx = setdiff(1:P, val_idx); 
        
        fold_sizes(k) = length(val_idx);
        
        x_train = x(train_idx);
        y_train = y(train_idx);
        x_val = x(val_idx);
        y_val = y(val_idx);
        % For all polynomial orders
        for m = 0:M
            
            % Train polynomial model on the training data
            % 1. Create the design matrix for training: [x^0, x^1, ..., x^m]
            X_train = x_train .^ (0:m);
            
            % 2. Solve for weights using the Pseudoinverse (closed-form solution)
            w_train = pinv(X_train) * y_train;
            % Calculate LS cost of the trained model on the validation data
            X_val = x_val .^ (0:m);
            y_pred_val = X_val * w_train;
            c_val = (1 / length(y_val)) * sum((y_pred_val - y_val).^2);
            % Store weights and costs
            W(m + 1, 1:m + 1, k) = w_train'; 
            C(m + 1, k) = c_val;
        end
    end
    
    % Calculate average cost over folds taking into account last fold size
    avg_C = zeros(M + 1, 1);
    for m = 0:M
        avg_C(m + 1) = sum(C(m + 1, :) .* fold_sizes) / P;
    end
    % Choose the polynomial order with least average LS cost
    [~, best_idx] = min(avg_C);
    best_m = best_idx - 1;
    % Retrain model with all data, and evaluate the cost
    X_full = x .^ (0:best_m);
    w_final = pinv(X_full) * y;
    
    w = w_final';
    y_pred_full = X_full * w_final;
    c = (1 / P) * sum((y_pred_full - y).^2);
    % Define nested helper functions below, if necessary
    
end

%Code to call your function
% Load data
A = load('galileo_ramp_data.csv');

% The first row contains the x-values
x = A(1,:)';

% The second row contains the y-values
y = A(2,:)';

% Number of folds
K = 6;

% Test polynomials upto order
M = 6;

% Call your function
[ w, c, W, C ] = fitKFoldPolynomial( x, y, K, M )




