function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

s = size(z);

if s == [1 1],
  g(1) = 1 / (1 + e^(-z(1)));
elseif s(1) == 1 && s(2) > 1,
  for i = 1:size(z)(2),
    g(i) = 1 / (1 + e^(-z(i)));
  end
elseif s(1) > 1 && s(2) > 1
  for i = 1:size(z)(1),
    for j = 1:size(z)(2),
      g(i, j) = 1 / (1 + e^(-z(i, j)));
    end
  end
else
  % raise error
end


% =============================================================

end
