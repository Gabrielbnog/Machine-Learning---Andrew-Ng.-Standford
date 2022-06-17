function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

%reshape na forma para o theta1: reshape(1:25x401,25,401)
% logo 25 = hidden_layer_size e 
%      401 = input_layer_size + 1

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%%%%%%%%%%%%%%COST FUNCTION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a1 = X;                                 %Primeira camada a1 - Input layer
a1 = [ones(m, 1) a1];                   %Adicionando bias unit
z2 = a1*Theta1';                
a2 =  sigmoid(z2);                      %Segunda camada a2 - Hidden layer  
a2 = [ones(m, 1) a2];                   %Adicionando bias unit
z3  = a2*Theta2';                       %Saida da camda 2
a3 = sigmoid(z3);                       %Output layer h(x)


%Quebrando o vetor y 
yvec = eye(num_labels);
%Matriz de Y em que cada linha corresponde a saida que deve estar de y
Y = zeros(m, num_labels);
%Para cada linha i de Y, terá um vetor do y original.
for i=1:m
  Y(i, :)= yvec(y(i), :);
end

temp  = 0;
for i = 1 : m
    for k = 1: num_labels
      aux = Y(i,k)*log(a3(i,k)) + (1 - Y(i,k))*log( 1 - a3(i,k));
      temp  = temp + aux;
    end
end

%Termo de regularização, note que é regularizado para a camada 2 e 3 sendo que a 1 é
%input. A soma acontece nas duas dimensões das matrizes de theta.
%A primeira coluna de theta não é regularizada devido a ser a bias

%dimensões:
%theta1: 25x401
%theta2: 10x26

regterm = sum(sum(Theta1(:,2:end).* Theta1(:,2:end))) + sum(sum(Theta2(:,2:end).* Theta2(:,2:end)));

J = (-1/m)*temp + (lambda/(2*m)) * regterm;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%Dimensões dos theta:
%theta1: 25x401
%theta2: 10x26

Delta1 = zeros(hidden_layer_size,input_layer_size+1);
Delta2 = zeros(num_labels,hidden_layer_size+1);

%Para cada feature i
for t = 1:m 
    
    %Passo 1 - Forward propagation
    a1 = X(t,:);                    %Input layer, a1:(400x1)
    a1 = [1; a1'];                  %Adicionando bias unit na primeira coluna, a1:(401x1)
    z2 = Theta1*a1;                 %z2: (25x1)
    a2 =  sigmoid(z2);              %Segunda camada a2 - Hidden layer, a2:(25x1)  
    a2 = [1; a2];                   %Adicionando bias unit a2:(26x1)
    z3  = Theta2*a2;                %Saida da camda 2, z3:(10x1)
    a3 = sigmoid(z3);               %Output layer h(x)
    
    %Passo 2 - Cálculo do vetor delta (erro)
    
    delta3 = a3(:) - Y(t,:)'; %erro delta3
    derg2 = a2.*(1 - a2); %derivadada de g'(z2) -->a2.*(1-a2)
    delta2 = (Theta2'*delta3).*derg2;
    
    %Passo 3 - Cálculo do Deltas, que irão resultar nos gradientes
    %retornado delta2_0.Note que isso irá sendo somado para cada feature i
    
    Delta1 = Delta1 + delta2(2:end)*a1';
    Delta2 = Delta2 + delta3*a2';
    
end


for j = 1: input_layer_size+1
    if (j == 1)
        Theta1_grad(:,j) = Delta1(:,j)/m;
    else
        Theta1_grad(:,j) = Delta1(:,j)/m + (lambda/m)*Theta1(:,j);
    end
end

for j = 1: hidden_layer_size+1
    if (j == 1)
        Theta2_grad(:,j) = Delta2(:,j)/m;
    else
        Theta2_grad(:,j) = Delta2(:,j)/m + (lambda/m)*Theta2(:,j);
    end
end



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
