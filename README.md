# Machine Learning Assignments

This repository contains a collection of Machine Learning assignments completed during my Computer Engineering studies at Istanbul Technical University (İTÜ). The projects focus on implementing fundamental ML algorithms, analyzing model performances, extracting mathematical insights, and writing core algorithms entirely from scratch.

## 👨‍💻 Author
**Ahmed Said Gülşen**

## 📚 Assignments Overview

### Assignment 1: Linear & Polynomial Regression via Direct Optimization and Gradient Descent
* **Concepts Learned:** Mean Squared Error (MSE), Normal Equation (Least Squares), Batch Gradient Descent, Polynomial feature expansion, model complexity, underfitting, and overfitting[cite: 3].
* **What I Did:** 
    * Built linear and polynomial regression models entirely from scratch using NumPy, strictly avoiding built-in libraries like `scikit-learn`[cite: 3].
    * Implemented and compared the analytical Direct Optimization method (calculating weights via matrix inversion) against the iterative Batch Gradient Descent approach[cite: 3].
    * Generated polynomial features for degrees 1, 2, 3, 5, and 10 to evaluate how varying model complexities fit synthetic datasets and influence the MSE[cite: 3].
    * Analyzed the computational complexity trade-offs between matrix inversion and iterative multiplication for large-scale datasets[cite: 3].

### Assignment 2: Logistic Regression & L2 Regularization
* **Concepts Learned:** Sigmoid activation function, forward and backward propagation, Mean Squared Error (MSE) loss vs. Cross-Entropy loss, and L2 Regularization (Weight Decay)[cite: 4].
* **What I Did:** 
    * Developed a robust multiclass `LogisticRegression` class from the ground up using fundamental NumPy array operations[cite: 4].
    * Mathematically coded the sigmoid activation and its derivative to compute pure errors and perform gradient descent updates[cite: 4].
    * Trained the classifier using two distinct approaches: one utilizing MSE loss and another utilizing Cross-Entropy loss[cite: 4].
    * Integrated an L2 regularization term into the Cross-Entropy gradient calculations to prevent overfitting and improve the model's generalization capabilities on test sets[cite: 4].

### Assignment 3: Support Vector Machines (SVM) & Hyperparameter Tuning
* **Concepts Learned:** Support Vector Machines, Linear vs. RBF Kernels, Hyperparameter Tuning (GridSearchCV), Data Scaling, and Model Interpretability.
* **What I Did:** 
    * Trained SVM models on the complex, high-dimensional **CIFAR-10** image dataset.
    * Performed extensive hyperparameter tuning using `GridSearchCV` to find the optimal `C` and `gamma` values for the RBF kernel.
    * Overcame severe convergence issues by appropriately standardizing pixel data, which vastly reduced the required training time.
    * Extracted and visualized the learned weights of a Linear SVM, generating "template images" to mathematically interpret how the model recognizes specific features of different vehicle and animal classes.

### Assignment 4: Principal Component Analysis (PCA) & Dimensionality Reduction
* **Concepts Learned:** Dimensionality Reduction, Singular Value Decomposition (SVD), Explained Variance, Image Reconstruction, and the trade-off between data compression and model accuracy.
* **What I Did:** 
    * Implemented the **PCA algorithm entirely from scratch** using NumPy (data centering, SVD application, variance calculation) without relying on built-in `sklearn.decomposition` libraries.
    * Analyzed the `digits` dataset to determine the minimal number of principal components needed to preserve 80%, 90%, and 95% of the cumulative variance.
    * Visualized the step-by-step reconstruction of hand-drawn digits from highly compressed lower-dimensional spaces to observe noise reduction and structural retention.
    * Combined the custom PCA implementation with a Logistic Regression classifier, proving mathematically that dimensionality can be reduced by over 50% (from 64 down to 30 dimensions) while still maintaining a highly robust ~96% classification accuracy.

## 🛠️ Technologies & Libraries Used
* **Python 3**
* **NumPy:** Utilized heavily for complex matrix operations, linear algebra (SVD), and writing mathematical optimization algorithms from scratch without external dependencies[cite: 3, 4].
* **PyTorch:** Included as an allowable alternative for tensor-based mathematical operations and gradient calculations in regression tasks[cite: 3].
* **Scikit-Learn:** Used for baseline model comparisons, evaluation metrics (Confusion Matrix, Classification Report), data scaling tools (`StandardScaler`), and hyperparameter tuning.
* **Matplotlib:** For visualizing high-dimensional data projections, plotting learned SVM weights, tracking loss curves over epochs, and displaying reconstructed image matrices[cite: 3].
* **Jupyter Notebook:** The primary environment for interactive algorithm development, debugging, and comprehensive technical reporting[cite: 3].
