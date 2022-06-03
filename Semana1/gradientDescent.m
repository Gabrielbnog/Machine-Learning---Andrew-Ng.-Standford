function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    var0 = 0;
    var1 = 0;
 
    for i = 1 : m 
        vec = [X(i,1); X(i,2)]; 
        temp0 =  (((theta' * vec) - y(i))*X(i,1)); %Theta zero
        temp1 =  (((theta' * vec) - y(i))*X(i,2)); %Theta um
        var0 = var0 + temp0;
        var1 = var1 + temp1;
    end
    
    thetazero = theta(1,1) - (alpha/m) * var0;
    thetaum = theta(2,1) - (alpha/m) * var1;
 
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    theta = [thetazero;thetaum];

end

end
