% File 'cifar_data.mat' contains matrix A and vector (uint8 data type)
load('cifar_data.mat');

% Convert to double data type for computations
X = double(A);
x = double(a);

% Number of samples
P = size(X,1);

% Number of features
N = size(X,2);

% Find data center
mX = mean(X, 1);

% Compute the centered data
X0 = X - mX;

% Compute whitening transformation W so that Xw = X0 * W
% 1. Calculate the covariance matrix of the centered data
C = cov(X0);
% 2. Perform eigenvalue decomposition (C = V * D * V')
[V, D] = eig(C);
% 3. Calculate W = V * D^(-1/2)
W = V * (D^(-0.5));

% Compute whitened data matrix Xw so that cov( Xw ) = eye( N )
Xw = X0 * W;
    

% Transform the new sample x into whitened feature space
xw = (x - mX) * W;



% Plot the result (not mandatory, but beneficial)
% Show first MxM images
M = 5;

figure
for i = 1:M^2
    subplot(M,M,i)
    imshow( reshape( X(i,:), 32, 32 )', [0 255] )
end
sgtitle('Original data')

range = max(abs(Xw(:)));
figure
for i = 1:M^2
    subplot(M,M,i)
    imshow( reshape( Xw(i,:), 32, 32 )', [-range range] )
end
sgtitle('Whitened data')

figure
subplot(121)
imshow( reshape( x, 32, 32 )', [0 255] )
title('Original new sample')
subplot(122)
imshow( reshape( xw, 32, 32 )', [-range range] )
title('Whitened new sample')

