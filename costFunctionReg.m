function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% - Cost Function -------------------------------------------
temp1 = 0;
sum_theta2 = 0;
grad_temp1 = 0;
n = size(theta,1);
grad_temp2 = zeros(n-1,1);
theta_temp = theta(2:n);

for i = 1:m
    h(i) = sigmoid(theta'*X'(:,i));
    temp1 = temp1 + ((-y(i))*log(h(i))-(1-y(i))*log(1-h(i)));
  
    grad_temp1 = grad_temp1 + (h(i) - y(i))*X'(1,i);
    grad_temp2 = grad_temp2 + (h(i) - y(i))*X'(2:n,i);
end

grad_temp2 = grad_temp2 + lambda*theta_temp;
sum_theta2 = sum(theta(2:n).^2);

J = temp1/m + sum_theta2*lambda/(2*m);
grad = [grad_temp1/m; grad_temp2/m];


% =============================================================

end
