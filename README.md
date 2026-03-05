### Task 1: PCA-Sphering (Whitening)
Implemented feature scaling using PCA-sphering (whitening) to prepare high-dimensional image data (CIFAR) for machine learning models. 
- **Objective:** Transform the raw feature space so that the resulting features are completely uncorrelated and have a unit variance.
- **Process:** 1. Centered the data by subtracting the mean ($\mu$).
  2. Computed the covariance matrix and performed eigenvalue decomposition to find the eigenvectors and eigenvalues.
  3. Calculated the whitening matrix $W = V D^{-1/2}$.
  4. Transformed both the training data and new test samples into the whitened feature space.
- **Result:** The transformed data yields a covariance matrix equal to the Identity matrix ($I$), which drastically improves optimization speed and stability for gradient descent algorithms.
