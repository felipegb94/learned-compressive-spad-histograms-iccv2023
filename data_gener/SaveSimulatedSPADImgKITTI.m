function SaveSimulatedSPADImgKITTI(fname, spad, SBR, range_bins, intensity, bin_size, num_bins, az, el)
save(fname, 'spad', 'SBR', 'range_bins', 'intensity', 'bin_size', 'num_bins', 'az', 'el');
end
