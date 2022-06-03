function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = size(X,2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
%     temp = zeros(n,1);
%     for i = 1 : m 
%         
%         for j = 1 : n
%             aux =  ( ( (theta' * X(i,:)')  - y(i) )*X(i,j)); 
%             temp(j,1) = temp(j,1) + aux;
%         end
%         
%         if (i == m)
%             theta = theta - (alpha/m) * temp;
%         end
%         
%     end
   error = (X * theta) - y;
   theta = theta - ((alpha/m) * X'*error);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
