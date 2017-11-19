function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = lambda/m;

% You need to return the following variables correctly
h = X * theta;
theta_reg = theta;
theta_reg(1) = 0;
J = 1 / (2 * m) * sum((h - y) .* (h - y)) + n / 2 * (theta_reg' * theta_reg);

grad = 1/m * X' * (h - y) + n * theta_reg;

end
