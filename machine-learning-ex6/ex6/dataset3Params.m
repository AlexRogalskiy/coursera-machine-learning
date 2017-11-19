function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
values = [0.01;0.03;0.1;0.3;1;3;10;30];

C_test = 0;
sigma_test = 0;
pred = inf;
for i = 1:length(values)
    param_C = values(i);
    for j = 1:length(values)
        param_sigma = values(j);
        model = svmTrain(...
            X, y, param_C,...
            @(x1, x2)...
                gaussianKernel(x1, x2, param_sigma)...
        );
        predict = svmPredict(model, Xval);
        predict = double(predict ~= yval);
        err = mean(predict);
        if err < pred
            pred = err;
            C_test = param_C;
            sigma_test = param_sigma;
        end
        fprintf('\t');
    end
    fprintf('\n');
end

C = C_test;
sigma = sigma_test;

end
