function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    J_history(iter) = computeCostMulti(X, y, theta);
    theta = theta - alpha * 1/m * (((theta' * X')' - y)' * X)';
    if mod(iter, 100000) == 0
        fprintf('\nComputed J = %f\n', J_history(iter));
    end
end

end
