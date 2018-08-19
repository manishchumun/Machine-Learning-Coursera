function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_theta = X * theta;

J = sum((h_theta .- y) .^ 2) / (2 * m) + (lambda/(2*m) * sum(theta(2:end) .^ 2));

grad_theta_0 = X(:,1)' * (h_theta .- y) ./ m;

grad_theta_remaining = ((X(:,2:end)' * (h_theta .- y)) ./ m) + (lambda/m .* theta(2:end));

grad = vertcat(grad_theta_0, grad_theta_remaining);


% =========================================================================

#grad = grad(:);

end
