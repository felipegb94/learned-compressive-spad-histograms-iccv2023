function [zn_v] = ZeroNorm(v)
% Apply zero norm normalization across the last dimension

last_dim = ndims(v);

% Pre-compute
mean_v = mean(v, last_dim);
zeronorm_v = v - mean_v;  
EPSILON = 1e-8;

zn_v = zeronorm_v ./ (vecnorm(zeronorm_v, 2, last_dim) + EPSILON);
    
end
   

