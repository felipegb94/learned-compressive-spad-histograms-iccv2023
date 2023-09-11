function SaveSimulatedSPADImgSmall(fname, detections, SBR, range_bins, intensity, bin_size, focal_length)
save(fname, 'detections', 'SBR', 'range_bins', 'intensity', 'bin_size', 'focal_length');
end
