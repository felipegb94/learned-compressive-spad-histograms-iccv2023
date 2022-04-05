function [est_depthbin] = EstDepthBinsArgmax(hist_img)
%EstDepthBinsArgmax Estimate the depth bin using argmax. 
%   Compute argmax across histogram dimensin.
%   Assume that the last dimension is the histogram dimension

last_dim = ndims(hist_img);
[~, est_depthbin] = max(hist_img,[], last_dim); 


end