function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% forming k dimensions features from n dimensions

% X         -> 50x2 (m x n)  [m examples of n dimensions]
% sigma & U -> 2x2  (n x n)
% U_reduce  -> 2x1  (n x k)
% Z         -> 1x1  (m x k)  [m examples of k dimensions]

U_reduce = U(:, 1:K);
Z = X * U_reduce;

% =============================================================

end
