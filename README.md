# Machine Learning Lab - Assignment 6: Feature Engineering, Selection, and Validation

This repository contains my MATLAB solutions for **Assignment 6** of the Machine Learning course (521289S). This module focuses on the critical data preprocessing, model selection, and validation steps required to build robust and generalizable machine learning models. 

Topics covered include PCA-sphering (whitening), automated feature selection (L1 Regularization and Boosting), and empirical validation techniques (K-Fold Cross-Validation).

## 📂 Repository Structure

The solutions are organized into separate MATLAB scripts and functions as required by the assignment templates:

| File Name | Description |
| :--- | :--- |
| `task1_PCA_Sphering_Whitening.m` | Implementation of feature scaling using PCA-sphering to remove correlations and achieve unit variance. |
| `task2_L1_Regularization_FeatureSelection.m` | Automated feature selection using an $L_1$-regularized Softmax cost function. |
| `task3_Regularized_L1_CrossValidation.m` | Parallelized cross-validation pipeline to denoise a piece-wise linear signal using a second-order difference $L_1$ penalty. |
| `task4_KFold_CrossValidation_Polynomial.m` | Implementation of $K$-fold cross-validation from scratch to determine the optimal degree for a polynomial regression model. |
| `task5_Boosted_Feature_Selection.m` | A greedy forward-selection algorithm (boosting) that iteratively adds features to a linear regression model based on residual errors. |

---

## 📝 Task Details

### Task 1: Feature Scaling via PCA-Sphering (Whitening)
Standard normalization scales features individually, but leaves correlations intact. This task implements PCA-sphering (whitening).
- **Process**: The data is mean-centered, and Eigendecomposition is computed on the covariance matrix. The data is then rotated so eigenvectors align with the axes, and scaled by the inverse square root of the eigenvalues ($\sqrt{\lambda_k}$). 
- **Result**: The transformed data cloud becomes perfectly spherical, meaning the new features have strictly zero linear correlation and unit variance, which drastically speeds up gradient descent optimization.

### Task 2: Feature Selection via L1-Regularization
Implemented an automated way to prune redundant features from a Credit Risk dataset.
- **Theory**: By adding an $L_1$ vector norm penalty ($\lambda \sum |w_i|$) to the Softmax cost function, the optimizer makes a compromise between model accuracy and model size.
- **Mechanism**: As the regularization parameter $\lambda$ increases, the weights of less important features are forced to exactly zero, effectively removing them from the model while preserving the most predictive features.

### Task 3: Cross-Validation via Regularization
Denoised a highly noisy 1D signal to recover its underlying piece-wise linear structure.
- **Implementation**: Formulated a custom cost function that combines Least Squares error with an $L_1$ penalty on the second-order differences of the weights ($\lambda \|\Delta^2 \mathbf{w}\|_1$).
- **Validation**: Utilized parallel computing (`parfor`) to train models across a logarithmic spectrum of $\lambda$ values. A hold-out validation set was used to identify the specific $\lambda$ that perfectly balances data fitting with structural smoothing.

### Task 4: K-Fold Cross-Validation for Polynomial Regression
Developed a complete validation pipeline to dynamically select model complexity without overfitting.
- **Process**: Partitioned the dataset into $K$ non-overlapping folds. The model is trained on $K-1$ folds and validated on the remaining fold, rotating until all data has been used for both training and validation.
- **Optimization**: Used the Moore-Penrose Pseudoinverse to instantly compute the exact closed-form Least Squares solution for polynomial design matrices up to degree $M$.
- **Evaluation**: Calculated a weighted average of the validation errors to handle uneven fold sizes and selected the polynomial degree that minimized generalization error.

### Task 5: Feature Selection via Boosting
Implemented a forward boosting algorithm to iteratively build a linear regression model for the Boston Housing dataset.
- **Theory**: The model starts with the simplest possible configuration (only a bias term) and extends it by adding one feature at a time.
- **Mechanism**: At each boosting step, the algorithm tests all unselected features against the *residual error* of the current model. The single feature that maximally decreases the remaining prediction error is permanently added to the model. 

---

## ⚠️ Repository Purpose & Academic Integrity

This repository is created solely to demonstrate the knowledge and practical skills I gained in machine learning optimization and validation during this course.

**The code is:**

* ❌ **Not intended for reuse, redistribution, or submission by others**
* ❌ **Not shared for the purpose of passing coursework or assessments**
* ✅ **Maintained as a personal academic and technical portfolio artifact**

Any use of this material should respect academic integrity policies and course regulations.
