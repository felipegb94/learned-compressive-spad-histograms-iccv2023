function [sim_param_str] = ComposeSimParamString(nr, nc, nt, bin_size, dark_img_param_idx, psf_img_param_idx)
    
    nr_str = sprintf('nr-%d', nr);
    nc_str = sprintf('nc-%d', nc);
    nt_str = sprintf('nt-%d', nt);
    tres_str = sprintf('tres-%dps', round(bin_size*1e12));
    dark_img_param_str = sprintf('dark-%d', dark_img_param_idx);
    psf_img_param_str = sprintf('psf-%d', psf_img_param_idx);

    sim_param_str = sprintf('%s_%s_%s_%s_%s_%s', nr_str, nc_str, nt_str, tres_str, dark_img_param_str, psf_img_param_str);

end