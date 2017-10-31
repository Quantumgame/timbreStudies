function [new_data, pc] = pcaGlobal5(data, varThresh)
% this is a global feature transform function (see evalSimilarity)
% transforms a n*p1 data matrix by computing its principal components, 
% keeping the p2 first components such that their cumulated variance > varThresh, 
% and projecting the old data onto the new, smaller-dimension space
% PCA is computed with Halko et al.'s fast truncated SVD
% inputs:
% - data: a n*p1 matrix of n observations in dimension p1
% - varThresh: a float between 0-1 corresponding to the minimum percentage
% of total variance that is preserved in the smaller-dimension space (the
% larger the threshold, the more principal components are kept)
% outputs: 
% - new_data: a n*p2 matrix of n observations (same order as data) in the
% new space (p2 < p1)



MAX_P = 100; 
[n,p]=size(data); 
if p>n
    data = data';
    [n,p]=size(data); 
end
MAX_P = min([MAX_P n p]); 



% subtract off the mean for each dimension
data = data - repmat(mean(data,1),n,1);

% construct the matrix Y
Y = data / sqrt(n-1);
% SVD does it all
[u,S,pc] = fsvd(Y, MAX_P);
% project the original data
% points = transpose(pc' * data');
points = data * pc';
% calculate the variances
S = diag(S);
variances = S .* S;

% find minimum nb of components needed to have var > varThresh
cum_explained = cumsum(variances / sum(variances)); 
where = find(cum_explained>=varThresh); 
if size(where,1) == 0
    p2 = size(points,2) ;
else
    p2 = where(1); 
end
%fprintf(1, 'PCA: dim %d -> %d (%.2f prct variance)\n',p,p2,cum_explained(p2)); 

%project data
new_data = points(:,1:p2);
