function [J, grad] = nnCostFunction(nn_params, ...
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
n = lambda / m;
A1 = [ones(m, 1) X]; % (5000, 401)

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:); % (5000, 10)

Z2 = A1 * Theta1';
A2 = [ones(m, 1) sigmoid(Z2)]; % (5000, 26)
h = sigmoid(A2 * Theta2'); % (5000, 10)
J = sum(sum(-y_matrix .* log(h) - (1 - y_matrix) .* log(1 - h))) * 1/m...
    + n/2 * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

d3 = h - y_matrix; % (5000, 10)
d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(Z2); % (5000, 25)

D1 = d2' * A1; % (25, 401)
D2 = d3' * A2; % (10, 26)

Theta1_reg = Theta1;
Theta1_reg(:, 1) = 0;
Theta2_reg = Theta2;
Theta2_reg(:, 1) = 0;

Theta1_grad = 1/m * D1 + n * Theta1_reg;
Theta2_grad = 1/m * D2 + n * Theta2_reg;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
