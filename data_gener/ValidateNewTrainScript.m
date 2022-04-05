% go through a folder of simulated files and compare their outputs
clear;

tolerance = 1e-6;
scene_id = 'dining_room_0001a';
% scene_id = 'dining_room_0001b';
% scene_id = 'dining_room_0002';
% scene_id = 'living_room_0042b';
% scene_id = 'living_room_0043a';

visualize_data = true;

old_dataset_dirpath = ['./TrainData/processed/', scene_id];
new_dataset_dirpath = ['./TrainData/SimSPADDataset_nr-64_nc-64_nt-1024_tres-80ps_dark-1_psf-1/', scene_id];

old_fnames = ls([old_dataset_dirpath '/spad_*']);
old_fnames = regexp(old_fnames, 'spad_\d\d\d\d_p1.mat', 'match');
old_fnames = sort(old_fnames);
old_fname_nums = regexp(old_fnames,'\d\d\d\d','match');
old_fname_nums = [old_fname_nums{:}];

new_fnames = ls([new_dataset_dirpath '/spad_*']);
new_fnames = regexp(new_fnames, 'spad_\d\d\d\d_p1.mat', 'match');
new_fnames = sort(new_fnames);
new_fname_nums = regexp(new_fnames,'\d\d\d\d','match');
new_fname_nums = [new_fname_nums{:}];

assert(numel(new_fname_nums) == numel(old_fname_nums), 'Number of files does not match')

for ii = 1 : 1 : numel(new_fnames)
% for ii = 1 : 1 : numel(2)

    fprintf('Old Fname: %s \n', old_fnames{ii});
    fprintf('New Fname: %s \n', new_fnames{ii});

    old_spad_out = load(sprintf('%s/%s', old_dataset_dirpath, old_fnames{ii}));
    new_spad_out = load(sprintf('%s/%s', new_dataset_dirpath, new_fnames{ii}));

    old_sbr = old_spad_out.SBR;
    old_mean_signal_photons = old_spad_out.photons;
    old_mean_background_photons = old_mean_signal_photons / old_sbr;
    
    new_sbr = new_spad_out.SBR;
    new_mean_signal_photons = new_spad_out.mean_signal_photons;
    new_mean_background_photons = new_spad_out.mean_background_photons;



    if((old_mean_background_photons == new_mean_background_photons) && (old_mean_signal_photons == new_mean_signal_photons))
        fprintf('Old PhiSig: %f, Old PhiBkg: %f\n', old_mean_signal_photons, old_mean_background_photons);
        fprintf('New PhiSig: %f, New PhiBkg: %f\n', new_mean_signal_photons, new_mean_background_photons);
        disp('Match!!!');
        
        old_rates = old_spad_out.rates;
        old_spad = old_spad_out.spad;
        old_range_bins = old_spad_out.bin;
        old_range_bins_hr = old_spad_out.bin_hr;

        new_rates = new_spad_out.rates;
        new_unorm_rates = InverseNormalizePhotonRates(new_rates, new_spad_out.rates_norm_params.rates_offset, new_spad_out.rates_norm_params.rates_scaling);
        new_spad = new_spad_out.spad;
        new_range_bins = new_spad_out.range_bins;
        new_range_bins_hr = new_spad_out.range_bins_hr;

        if(isequal(new_rates, old_rates))
            disp('    [PASSED] Photon Flux Rates are EXACTLY equal');
        else
            disp('    [FAIL] Photon Flux Rates NOT EXACTLY equal');
        end

        if(ismembertol(new_rates, old_rates, tolerance))
            disp('    [PASSED] Photon Flux Rates are APPROX equal');
        else
            disp('    [FAIL] Photon Flux Rates NOT APPROX equal');
            error('Bug in new dataset')
        end

        if(isequal(new_range_bins, old_range_bins))
            disp('    [PASSED] Resized range bins are equal');
        else
            disp('    [FAIL] Resized range bins  NOT equal');
            error('Bug in new dataset')
        end

        if(isequal(new_range_bins_hr, old_range_bins_hr))
            disp('    [PASSED] HRes range bins are equal');
        else
            disp('    [FAIL] HRes range bins  NOT equal');
            error('Bug in new dataset')
        end

        if(visualize_data)
            nr = size(new_range_bins,1);
            nc = size(new_range_bins,2);
            nt = size(old_rates,3);
    
            old_spad_full = full(old_spad);
            new_spad_full = full(new_spad);
            old_spad_full = reshape(old_spad_full, nr, nc, nt);
            new_spad_full = reshape(new_spad_full, nr, nc, nt);
            clf;
            subplot(3,2,1);
            imagesc(squeeze(sum(old_spad_full,3))); title('Old SPAD');colorbar;
            subplot(3,2,2);
            imagesc(squeeze(sum(new_spad_full,3))); title('New SPAD');colorbar;
            subplot(3,3,4);
            imagesc(squeeze(sum(old_rates,3))); title('Old Norm Rates');colorbar;
            subplot(3,3,5);
            imagesc(squeeze(sum(new_rates,3))); title('New Norm Rates');colorbar;
            subplot(3,3,6);
            imagesc(squeeze(sum(new_unorm_rates,3))); title('New Rates');colorbar;

            subplot(3,1,3);
            r = randi(nr); c = randi(nc);
            plot(squeeze(old_rates(r,c,:)), '--', 'linewidth', 3); hold on;
            plot(squeeze(new_rates(r,c,:)), '--', 'linewidth', 1.5);
            max_old_spad_full = max(squeeze(old_spad_full(r,c,:)));
            max_new_spad_full = max(squeeze(new_spad_full(r,c,:)));
            plot(squeeze(old_spad_full(r,c,:))./max_old_spad_full, '-', 'linewidth', 1); hold on;
            plot(squeeze(new_spad_full(r,c,:))./max_new_spad_full, '-', 'linewidth', 1);
            xlim([new_range_bins(r,c)-40, new_range_bins(r,c)+40]);
            legend('Old Rates', 'New Rates', 'Old SPAD Hist', 'New SPAD Hist');
            pause(0.1)
        end
        

    end

end
