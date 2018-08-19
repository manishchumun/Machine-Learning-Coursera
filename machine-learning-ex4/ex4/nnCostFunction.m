function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

% output of hidden layer (5000 x 25)
out_hidden_layer = sigmoid(X * Theta1');

% Add ones to the out_hidden_layer matrix (5000 x 26)
out_hidden_layer = [ones(m, 1) out_hidden_layer];

% output the h of x (5000 x 10)
h_theta_x = sigmoid(out_hidden_layer * Theta2');

% perform one-hot encoding for num_labels
y_encoded = zeros(size(h_theta_x));
for i = 1:m
  y_encoded(i, y(i)) = 1;
endfor

% calculate the cost function
J = sum(sum((-y_encoded .* log(h_theta_x)) - ((1 - y_encoded) .* log(1 - h_theta_x)))) / m;

% regularisation
R = (sum(sum(Theta1(:,2:end) .^2)) + sum(sum(Theta2(:,2:end) .^2)))*(lambda/(2*m));

% Regularisation added to the cost function
J = J + R;

% backpropagation without regularisation
delta_3 = h_theta_x .- y_encoded;

delta_2 = (delta_3 * Theta2(:,2:end)) .* sigmoidGradient(X * Theta1');

Theta1_grad = (delta_2' * X)/m;
Theta2_grad = (delta_3' * out_hidden_layer)/m;

% backpropagation with regularisation
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m).* Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m).* Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
