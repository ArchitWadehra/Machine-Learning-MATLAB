function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X];                            % adding bias to input layer

% X -> 5000x401; Theta1 -> 25x401; thus, z_2/a_2 -> 5000x25
z_2 = X * transpose(Theta1);   %(X = a_1)      % computing hidden layer
a_2 = sigmoid(z_2);

a_2 = [ones(m, 1) a_2];                        % adding bias to hidden layer

% a_2 -> 5000x26; Theta2 -> 10x26; thus, z_3/h -> 5000x10
z_3 = a_2 * transpose(Theta2);                 % computing output layer
h = sigmoid(z_3);

% now h contains the probabilities of being a label (0-9) for all 5000 images 

[val, p] = max(h, [], 2);    % 5000x1 (max probability of each row)
% (value, index) -> we only need the index since it denotes our label

% now we will cross-check our prediction of label with 'y'(actual label) to find accuracy of our model

% =========================================================================


end
