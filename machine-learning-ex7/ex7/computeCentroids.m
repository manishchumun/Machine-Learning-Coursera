function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for k = 1:K
    
    % sum of elements
    sum_ = zeros(1, n);
    % count of the elements with centroid k
    Ck = 0;
    
    % we loop through the idx
    for i = 1:m
        % identify all elements with centroid k
        if idx(i) == k
            % add the element to sum
            sum_ = sum_ .+ X(i, :);
            % increment the count
            Ck = Ck + 1;
        endif
    endfor
   
    % calculate the data point for the centroid 
    centroids(k, :) = sum_ ./ Ck;

endfor

% =============================================================


end

