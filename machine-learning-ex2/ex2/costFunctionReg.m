function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h = sigmoid(X * theta);
theta_reg = theta;
theta_reg(1, 1) = 0;
n = lambda/m;

J = 1/m * (-(y)' * log(h) - (1 - y)'*log(1 - h)) + n/2 * (theta_reg' * theta_reg);

grad = 1/m * X' * (h - y) + n * theta_reg;

end
