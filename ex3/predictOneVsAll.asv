function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% X         -> 5000 X 401 (INPUT : 5000 number images with 401 (400+bias) features(pixels)) 
% all_theta -> 10 X 401   (all 401 parameters(theta) for all 401 features of all 10 labels(0-9))

h = sigmoid(X * transpose(all_theta));   % 5000 X 10 (probabilty of being a label among 10 labels for 5000 input images)

[val, p] = max(h, [], 2);    % 5000 X 1 (max probability of each row)
% (value, index) -> we only need the index since it denotes our label

% now we will cross-check our prediction of label with 'y'(actual label) to find accuracy of our model

% =========================================================================


end
