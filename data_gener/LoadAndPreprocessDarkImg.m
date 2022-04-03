function dark_img = LoadAndPreprocessDarkImg(dark_img_param_idx, nr, nc)
%LoadAndPreprocessDarkImg Load a dark image or output an empty one
%   dark_img_param_idx: which type of dark image to load

if dark_img_param_idx == 0
    disp('Loading empty dark_img');
    dark_img = zeros(nr, nc);
else    
    if dark_img_param_idx == 1
        disp('Loading dark_img from TrainData/nyuv2/dark_img.mat');
        load './TrainData/dark_img.mat' dark_img;
        res = size(dark_img, 2);
        assert(nr==res, 'nr should match dark_img res');
        assert(nc==res, 'nc should match dark_img res');
        % Take the mean across rows of dark img, repeat mean and transpose
        % it
        dark_img = repmat(mean(dark_img(:,1:res)),[res,1])';
    else
        error('Bad param_idx value');
    end
end

end