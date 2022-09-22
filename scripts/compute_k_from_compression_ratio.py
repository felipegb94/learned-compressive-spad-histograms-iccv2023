'''
    This script computes k (number of encoding kernels/filters) from a desired compression ratio and a CSPH size

        compression_ratio = hist_img_size / (num_blocks*k)
        so, k = hist_img_size / (compression_ratio * num_blocks )

    where,
        - hist_img_size is the size of the full-resolution histogram image
        - num_blocks is the number of blocks that we divide the hist_img that we are compressing

'''

#### Standard Library Imports

#### Library imports

#### Local imports

if __name__=='__main__':

    ## Set full-resolution CSPH dimensions
    (nt, nr, nc) = (1024, 32, 32)
    hist_img_size = nt*nr*nc

    ## Set desired compression levels
    compression_levels = [32, 64, 128]

    ## Set encoding kernel dimensions we will use
    encoding_kernel_dims = [(1024,1,1), (1024,2,2), (1024,4,4), (256,4,4)]

    for encoding_kernel_dim in encoding_kernel_dims:
        print("----------------------------")
        print("encoding_kernel_dims = {}:".format(encoding_kernel_dim))
        assert((nt % encoding_kernel_dim[0]) == 0), "tdim not divisible"
        assert((nr % encoding_kernel_dim[1]) == 0), "rdim not divisible"
        assert((nc % encoding_kernel_dim[2]) == 0), "cdim not divisible"
        num_t_blocks = int(nt / encoding_kernel_dim[0])
        num_r_blocks = int(nr / encoding_kernel_dim[1])
        num_c_blocks = int(nc / encoding_kernel_dim[2])
        num_blocks = num_t_blocks*num_r_blocks*num_c_blocks
        for compression_ratio in compression_levels:
            assert(hist_img_size % (compression_ratio*num_blocks) == 0), "params do not lead to an integer k"
            k = int(hist_img_size / (compression_ratio*num_blocks))
            print("    compression_ratio = {} --> k = {}".format(compression_ratio, k))
        print("----------------------------")
