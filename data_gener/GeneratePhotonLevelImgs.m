function [signal_ppp, ambient_ppp] = GeneratePhotonLevelImgs(alpha, intensity, mean_signal_photons, mean_background_photons, dark_img)
%GeneratePhotonLevelImgs 
% Take the alpha and intensity images and scale them with the mean_signal_photons and mean_background_photons to get the
% the signal level and background level images


    % add albedo/range/lighting/dark count effects
    % Set mean signal photons using alpha
    signal_ppp = SetArrayMean(alpha, mean_signal_photons);
%     signal_ppp = alpha ./ mean(alpha(:)) .* mean_signal_photons;

    % make approximately correct ratio between ambient light and dark count
    ambient_ppp = dark_img + SetArrayMean(intensity, mean_background_photons);
%     ambient_ppp = dark_img + mean_background_photons .* intensity ./ mean(intensity(:));
    
    % apply a global scale to both to get the exact desired sbr
    ambient_ppp = SetArrayMean(ambient_ppp, mean_background_photons);
end

function [scaled_img] = SetArrayMean(img, mean_val)
    scaled_img = img ./ mean(img(:)) .* mean_val;
end
