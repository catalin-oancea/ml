function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


hTheta = sigmoid(X * theta); %compute hypothesis

J = (-1/m) * sum(y.*log(hTheta) + (1-y).*log (1.-hTheta));

% compute the gradient 
for i = 1:m
	% hypothesis = mx1 column vector
	% y = mx1 column vector
	% X = mxn matrix
	grad = grad + ( hTheta(i) - y(i) ) * X(i, :)';
end

% gradient = nx1 column vector
grad = (1/m) * grad;

% =============================================================

end
