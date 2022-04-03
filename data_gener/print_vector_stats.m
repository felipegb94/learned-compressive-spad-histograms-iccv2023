function print_vector_stats(v, name)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
fprintf("%s: \n", name);
fprintf("    Dimensions: %f, %f \n", size(v));
fprintf("    Max/Min: %f, %f \n", max(v(:)), min(v(:)));
end