function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%C_train = [0.01 0.03 0.1 0.3 1 3 10 30];
%sigma_train = [0.01 0.03 0.1 0.3 1 3 10 30];

%C_train = C_train(:); sigma_train = sigma_train(:);

%min_error = 0;

%x1 = [1 2 1]; x2 = [0 4 -1];

%for i = 1:size(C_train, 1)
%    for j = 1:size(sigma_train, 1)
%        
%        fprintf('C_train = %f and sigma_train = %f', C_train(i), sigma_train(j));
%        
%        model = svmTrain(X, y, C_train(i), @(X, y) gaussianKernel(X, y, sigma_train(j)));
%        predictions = svmPredict(model, Xval);
%
%        error = mean(double(predictions ~= yval));
%        
%        if (i == 1 && j == 1)
%            min_error = error;
%            fprintf('Error = %f \n', error);
%            
%            C = C_train(i);
%            sigma = sigma_train(j);
%        
%        elseif (error < min_error),
%            min_error = error;
%            fprintf('Error = %f \n', error);
%            
%            C = C_train(i);
%            sigma = sigma_train(j);
%        
%        endif
%              
%    endfor
%
%endfor

%fprintf('Model parameters C = %f and sigma = %f \n', C, sigma);

% =========================================================================

end
