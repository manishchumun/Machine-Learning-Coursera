function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:size(X, 1)
    for k = 1:K
        
        % we calcuate the distance between X(i) and c(k)
        dis = sumsq(X(i, :) - centroids(k, :)); 
        
        % we check if the distance is the minimum distance and
        % assign the centroid accordingly 
        if k == 1
            min_dis = dis;
            idx(i) = k;
        elseif dis < min_dis
            min_dis = dis;
            idx(i) = k;
        endif
    endfor
    
    % reset the value of min_dis
    min_dis = 0; 

endfor

% =============================================================

end
