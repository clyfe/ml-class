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

n = size(X)(2);

sum = 0;
for i=1:m,
  xi = theta' * X(i,:)';
  h = 1 / (1 + e^(-xi));
  sum = sum - y(i) * log(h) - (1-y(i)) * log(1-h);
end
lsum = 0;
for j=2:n
  lsum = lsum + theta(j)^2;
end 
J = (1/m)*sum + lambda/(2*m)*lsum;

s = size(theta);
for j = 1:s(1),
  sum = 0;
  for i=1:m,
    xi = theta' * X(i,:)';
    h = 1 / (1 + e^(-xi));
    sum = sum + (h - y(i))*X(i,j);
  end
  if j == 1
    grad(j) = (1/m)*sum;
  else
    grad(j) = (1/m)*(sum + lambda*theta(j));
  end
end



% =============================================================

end
