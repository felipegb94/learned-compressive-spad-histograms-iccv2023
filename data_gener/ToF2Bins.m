function [range_bins] = ToF2Bins(tof, bin_size, num_bins, round_bin)
%ToF2Bins Convert tof to bin. Truncated to num_bins if bin > num_bins
    % Convert tof to bins
    if(round_bin)
        range_bins = round(tof ./ bin_size);
    else
        range_bins = tof ./ bin_size;
    end
        
    if any(reshape(range_bins > num_bins, 1, []))
        fprintf('some photon events out of range\n');
    end
    % BUG: We should not truncate depths that are larger we should wrap
    % them around
    % As of 02-06-2023 we changed the following from truncate to wrap
    % around instead
    % OLD: Truncate
%     range_bins = min(range_bins, num_bins);   
%     range_bins = max(range_bins, 1);
    % NEW: Modulo range bins
    range_bins = mod(range_bins, num_bins);   
    range_bins = max(range_bins, 1);

end