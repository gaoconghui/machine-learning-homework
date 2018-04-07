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
sigma = 0.3;

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


maybe = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

error_est = -1;
_C = 0;
_sigma = 0;
for i = 1:8;
	for j = 1:8;
		model= svmTrain(X, y,maybe(i), @(x1, x2) gaussianKernel(x1, x2, maybe(j))); 
		y_predict = svmPredict(model,Xval);
		error = mean(double(y_predict ~= yval))
		if error_est < 0 || error < error_est;
			error_est = error;
			_C = maybe(i);
			_sigma = maybe(j);
		end;
	end;
end;
C = _C;
sigma = _sigma;
% =========================================================================

end
