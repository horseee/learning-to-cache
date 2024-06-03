from tools.fid_score import calculate_fid_given_paths
import sys

if __name__ == '__main__':
    sample_npz_path = sys.argv[1]
    res = sys.argv[2]

    if res == '256':
        ref_path = 'assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz'
    elif res == '512':
        ref_path = 'assets/fid_stats/fid_stats_imagenet512_guided_diffusion.npz'
    else:
        raise NotImplementedError
    fid_value = calculate_fid_given_paths([ref_path, sample_npz_path], batch_size=1000)
    print(fid_value)

