function [v1convv2] = CircXConv(v1,v2)
% Circular convolution of real inputs across last dimension of v1

assert(isequal(size(v1), size(v2)), 'size should match');

last_dim = ndims(v1);
n = size(v1, last_dim);

fft_v1 = fft(v1, n, last_dim);
fft_v2 = fft(v2, n, last_dim);

v1convv2 = real(ifft(fft_v1.*fft_v2, n, last_dim));

end

