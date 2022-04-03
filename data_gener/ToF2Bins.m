function [range_bins] = ToF2Bins(tof, bin_size, num_bins)
%ToF2Bins Convert tof to bin. Truncated to num_bins if bin > num_bins
    % Convert tof to bins
    range_bins = round(tof ./ bin_size);
    if any(reshape(range_bins > num_bins, 1, []))
        fprintf('some photon events out of range\n');
    end
    % Truncated
    range_bins = min(range_bins, num_bins);   
    range_bins = max(range_bins, 1);
end