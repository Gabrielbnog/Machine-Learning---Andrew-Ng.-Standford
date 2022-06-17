function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n= length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

somalosgc = 0;
for i = 1 : m
    
    %Avaliando a funcao custo J_theta
    h = sigmoid(theta'*X(i,:)');
    temp = y(i)*log(h) + (1-y(i))*log(1-h);
    somalosgc = temp + somalosgc;    
end

%adicionando termo de regularizacao
regtemp = 0;
aux=0;
for j =2 :n 
    aux = theta(j).^2;
    regtemp = aux + regtemp;
end

J =  (-1/m) * somalosgc  + (lambda/(2*m))*regtemp ; %Funcao custo regularizada

%Gradientes da funcao custo
for j =1:n
    aux = 0;
    somagrad = 0;
    %calculando cada um dos gradientes
    for i =1:m
        h = sigmoid(theta'*X(i,:)');
        aux = (h-y(i))*X(i,j);
        somagrad = aux + somagrad;  
    end
    if (j == 1) %Por convencao o primeiro nao é regularizado
        grad(j) = (1/m)*(somagrad);
    end
    
    if (j > 1)
        grad(j) = (1/m)*(somagrad) + (lambda/m)*theta(j);
    end
end


% =============================================================

end
