function SaveSimulatedSPADImgSUNRGBD(fname, spad, SBR, range_bins, intensity, bin_size, K, num_bins)
save(fname, 'spad', 'SBR', 'range_bins', 'intensity', 'bin_size', 'K', 'num_bins');
end
