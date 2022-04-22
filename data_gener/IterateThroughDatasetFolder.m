%% Iterate through dataset folder
% Go over all files in the dataset and load each one of them
% Why? If we want to change the structure of the saved files, this makes it
% easy to do, without having to run the simulation again (which can take a
% a couple hours to run for 10000+ files.


clear;

% rel_dirpath = './TestData/middlebury/processed/SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0/spad_*';
rel_dirpath = './TrainData/SimSPADDataset_nr-64_nc-64_nt-1024_tres-80ps_dark-1_psf-1/*/spad_*';

spad_data_files = dir(rel_dirpath);

n_files = numel(spad_data_files);


parfor i = 1:n_files
    fprintf('Loading %s \n', spad_data_files(i).name);
    spad_data_fpath = fullfile(spad_data_files(i).folder, spad_data_files(i).name);
    spad_data = load(spad_data_fpath);
    % do something with the file...

    % Save spad_data with the new function
    % Skip if we already converted it
    if(isfield(spad_data, 'est_range_bins_argmax'))
        fprintf('    Skipping %s \n', spad_data_files(i).name);
    else
        fprintf('    Saving new %s \n', spad_data_files(i).name);
        SaveSimulatedSPADImg(spad_data_fpath, ...
        spad_data.spad, ...
        spad_data.SBR, ...
        spad_data.range_bins,... 
        spad_data.bin_hr, ...
        spad_data.est_range_bins, ...
        spad_data.rates_norm_params, ...
        spad_data.rates, ...
        spad_data.intensity, ...
        spad_data.intensity_hr, ...
        spad_data.mean_signal_photons, ...
        spad_data.mean_background_photons, ...
        spad_data.bin_size)
    end

end