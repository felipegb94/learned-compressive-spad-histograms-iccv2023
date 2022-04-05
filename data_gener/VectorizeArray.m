function [vec_A] = VectorizeArray(A)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

A_ndims = ndims(A);
last_dim = size(A, A_ndims);
num_elems = round(numel(A) / last_dim);


vec_A = reshape(A, [num_elems, last_dim]);

end