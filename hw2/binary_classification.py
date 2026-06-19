import numpy as np

class LogisticRegression:
    
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Constructor assumes a x_train matrix in which each column contains an instance.
        Vector y_train contains one integer for each instance, indicating the instance's label. 
        
        Constructor initializes the weights W and B, alpha, and a one-vector Y containing the labels
        of the training set. Here we assume there are 10 labels in the dataset. 
        """
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._m = x_train.shape[1]
        
        self._W = np.random.randn(10, 784) * 0.01
        self._B = np.zeros((10, 1))
        self._Y = np.zeros((10, self._m))
        self._alpha = 0.05

        for index, value in enumerate(self._y_train):
            self._Y[value][index] = 1
            
    def sigmoid(self, Z):
        """
            Args:
            - Z (numpy.ndarray): The input array.

            Returns:
            - numpy.ndarray: An array of the same shape as Z, where each element is the sigmoid of the corresponding element in Z.
            
        """
        
        ##############################################################################
        #Computes the sigmoid value for all values in vector Z
        # TODO: # Write the computation of the sigmoid function for a given matrix Z                                                        #
        ##############################################################################
        # Replace "pass" statement with your code
        
        # formula: 1 / (1 + e^-Z)
        sigmoid_val = 1 / (1 + np.exp(-Z))
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return sigmoid_val

    def derivative_sigmoid(self, A):
        """
            Args:
            - A (numpy.ndarray): The input array.

            Returns:
            - numpy.ndarray: An array of the same shape as A, where each element is the derivative of the sigmoid function
            for the corresponding element in A.
        """
 
        ##############################################################################
        #Computes the derivative of the sigmoid for all values in vector A
        # TODO: # Write the derivative of the sigmoid function for a given value A produced by the sigmoid                                                       #
        ##############################################################################
        # Replace "pass" statement with your code
        
        # formula: A * (1 - A)
        deriv_sigmoid_val = A * (1 - A)
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return deriv_sigmoid_val

    def h_theta(self, X):
        """
            Args:
            - X (numpy.ndarray): The input feature matrix.

            Returns:
            - numpy.ndarray: A column vector of predicted values obtained
        """

        ##############################################################################
        #Computes the value of the hypothesis according to the logistic regression rule
        # TODO: # Write the computation of the hypothesis for a given matrix X                                                       #
        ##############################################################################
        # Replace "pass" statement with your code
        
        # Calculate the linear combination Z = W * X + B
        # self._W is (10, 784) and X is (784, m), so np.dot gives (10, m)
        Z = np.dot(self._W, X) + self._B
        
        # Pass Z through the sigmoid function to get probabilities between 0 and 1
        h_theta = self.sigmoid(Z)
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

        return h_theta
    
    def return_weights_of_digit(self, digit):
        """
            Args:
            - digit (int): The digit for which the weights are to be returned.

            Returns:
            - numpy.ndarray: A row vector of weights from the weights matrix corresponding to the given digit.
        """

        ##############################################################################
        # TODO: # Returns the weights of the model for a given digit                                                      #
        ##############################################################################
        # Replace "pass" statement with your code
        
        # self._W has shape (10, 784). Each row corresponds to a digit (0 to 9).
        # We simply return the row at the index 'digit'.
        weights_of_digits = self._W[digit]
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

        return weights_of_digits
    
    def train_mse_loss(self, iterations):
        """
        Performs a number of iterations of gradient descend equals to the parameter passed as input.
        
        Returns a list with the percentage of instances classified correctly in the training and in the test sets.
        """
        classified_correctly_train_list = []
        classified_correctly_test_list = []
        
        for i in range(iterations):
 
            ##############################################################################
            # TODO: #Please write your answers for A, pure_error, W, B , classified_correctly, 
            #         percentage_classified_correctly,test_correct parts      

            # Write the following four lines of code for computing the value produced by the model (A)
            # The pure error for all training instances (pure_error)
            # And adjust the matrices self._W and self._B according to the gradient descent rule
            ####                                      
            ##############################################################################

            # ==========================================
            # 1. Forward pass: Compute the hypothesis (A)
            A = self.h_theta(self._x_train)
            
            # 2. Compute the pure error: E = A - Y
            pure_error = A - self._Y
            
            # 3. Apply Chain Rule for MSE: Multiply pure error by the derivative of sigmoid
            dZ = pure_error * self.derivative_sigmoid(A)
            
            # 4. Calculate gradients (dW and dB)
            # dW = (1/m) * (dZ dot X^T)
            dW = (1 / self._m) * np.dot(dZ, self._x_train.T)
            
            # dB = (1/m) * sum of dZ along columns
            dB = (1 / self._m) * np.sum(dZ, axis=1, keepdims=True)
            
            # 5. Update parameters using Gradient Descent
            self._W = self._W - self._alpha * dW
            self._B = self._B - self._alpha * dB
            # ==========================================

            if i % 100 == 0:
                # ==========================================
                # Calculate training accuracy
                # Find the index of the max probability in each column (predicted digit)
                predictions = np.argmax(A, axis=0)
                classified_correctly = np.sum(predictions == self._y_train)
                percentage_classified_correctly = (classified_correctly / self._m) * 100
                
                classified_correctly_train_list.append(percentage_classified_correctly)
                
                # Calculate test accuracy
                Y_hat_test = self.h_theta(self._x_test)
                test_predictions = np.argmax(Y_hat_test, axis=0)
                test_correct = np.sum(test_predictions == self._y_test)
                
                classified_correctly_test_list.append((test_correct) / len(self._y_test) * 100)
                # ==========================================
                
                print('Accuracy train data: %.2f' % percentage_classified_correctly)

        return classified_correctly_train_list, classified_correctly_test_list


    def train_cross_entropy_loss(self, iterations):
        """
        Performs a number of iterations of gradient descent equal to the parameter passed as input.

        In this version, L2 regularization must also be considered when updating the weights.

        Returns a list with the percentage of instances classified correctly in the training
        and in the test sets.
        """

        classified_correctly_train_list_ce = []
        classified_correctly_test_list_ce = []

        for i in range(iterations):
            # Write the following lines of code for computing the value produced by the model (A)
            # Compute the pure error for all training instances (pure_error)
            # Update the matrices self._W and self._B according to the gradient descent rule
            # IMPORTANT: Include the L2 regularization term when updating self._W.
            # The L2 penalty term is: lambda * ||W||^2

            ##############################################################################
            # TODO: Please write your answers for A, pure_error, W, B , classified_correctly,
            #       percentage_classified_correctly_ce, test_correct parts
            #
            # NOTE:
            # - Implement gradient descent with L2 regularization.
            # - The regularization term should be applied only to the weights (self._W),
            #   not to the bias (self._B).
            #
            # Write the following lines of code:
            # 1) Compute the model output A
            # 2) Compute the pure error (pure_error)
            # 3) Update self._W including the L2 regularization term
            # 4) Update self._B
            #
            ##############################################################################

            # 1. Forward pass: Compute the hypothesis (A)
            A = self.h_theta(self._x_train)
            
            # 2. Compute the pure error for Cross-Entropy: dZ = A - Y
            # We DO NOT multiply by the derivative of sigmoid here
            # The math of Cross-Entropy cancels it out perfectly
            pure_error = A - self._Y
            
            # 3. Calculate gradients (dW and dB) with L2 Regularization
            # We define a small lambda (regularization hyperparameter)
            lambd = 0.01 
            
            # dW = (1/m) * (pure_error dot X^T) + (2 * lambda * W)
            dW = (1 / self._m) * np.dot(pure_error, self._x_train.T) + (2 * lambd * self._W)
            
            # dB = (1/m) * sum of pure_error along columns (Bias is NOT regularized!)
            dB = (1 / self._m) * np.sum(pure_error, axis=1, keepdims=True)
            
            # 4. Update parameters using Gradient Descent
            self._W = self._W - self._alpha * dW
            self._B = self._B - self._alpha * dB
            # ==========================================

            if i % 100 == 0:
                # ==========================================
                # Calculate training accuracy
                predictions_ce = np.argmax(A, axis=0)
                classified_correctly = np.sum(predictions_ce == self._y_train)
                percentage_classified_correctly_ce = (classified_correctly / self._m) * 100
                
                classified_correctly_train_list_ce.append(percentage_classified_correctly_ce)

                # Calculate test accuracy
                Y_hat_test = self.h_theta(self._x_test)
                test_predictions = np.argmax(Y_hat_test, axis=0)
                test_correct = np.sum(test_predictions == self._y_test)
                
                classified_correctly_test_list_ce.append((test_correct) / len(self._y_test) * 100)
                # ==========================================

                print('Accuracy train data: %.2f' % percentage_classified_correctly_ce)

        return classified_correctly_train_list_ce, classified_correctly_test_list_ce




