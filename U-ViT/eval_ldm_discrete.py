from tools.fid_score import calculate_fid_given_paths
import ml_collections
import torch
import numpy as np
from torch import multiprocessing as mp
import accelerate
import utils
from datasets import get_dataset
import tempfile
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import builtins
import libs.autoencoder


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()

    if 'dynamic' in config.sample:
        # Get Timestep Mapping
        t_0 = 1. / 1000
        t_T = 1.0
        order_value = 2
        N_steps = config.nfe // order_value
        timesteps = torch.linspace(t_T, t_0, N_steps + 1).cpu().numpy()
        #timesteps = timesteps.numpy()
        timestep_mapping = np.round(timesteps * 1000)
        #accelerator.unwrap_model(nnet).set_timestep_map(timestep_mapping)
        
        accelerator.unwrap_model(nnet).load_ranking(config.router, config.nfe, timestep_mapping, config.thres)
    elif 'rank' in config.sample:
        # Get Timestep Mapping
        t_0 = 1. / 1000
        t_T = 1.0
        order_value = 2
        N_steps = config.nfe // order_value
        timesteps = torch.linspace(t_T, t_0, N_steps + 1).cpu().numpy()
        #timesteps = timesteps.numpy()
        timestep_mapping = np.round(timesteps * 1000)
        #accelerator.unwrap_model(nnet).set_timestep_map(timestep_mapping)
        
        accelerator.unwrap_model(nnet).load_ranking(config.nfe, config.thres)
    
    elif 'topk' in config.sample or 'random' in config.sample:
        accelerator.unwrap_model(nnet).load_ranking(config.sample.topk, config.sample.reverse, config.sample.random)


    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def decode_large_batch(_batch):
        decode_mini_batch_size = 50  # use a small batch size since the decoder is large
        xs = []
        pt = 0
        for _decode_mini_batch_size in utils.amortize(_batch.size(0), decode_mini_batch_size):
            x = decode(_batch[pt: pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs

    if 'cfg' in config.sample and config.sample.cfg and config.sample.scale > 0:  # classifier free guidance
        logging.info(f'Use classifier free guidance with scale={config.sample.scale}')
        def cfg_nnet(x, timesteps, y):
            _cond = nnet(x, timesteps, y=y)
            _uncond = nnet(x, timesteps, y=torch.tensor([dataset.K] * x.size(0), device=device))
            return _cond + config.sample.scale * (_cond - _uncond)
    else:
        def cfg_nnet(x, timesteps, y):
            _cond = nnet(x, timesteps, y=y)
            return _cond

    logging.info(config.sample)
    assert os.path.exists(dataset.fid_stat)
    logging.info(f'sample: n_samples={config.sample.n_samples}, mode={config.train.mode}, mixed_precision={config.mixed_precision}')

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    def sample_z(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)

        if config.sample.algorithm == 'dpm_solver':
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

            def model_fn(x, t_continuous):
                t = t_continuous * N
                eps_pre = cfg_nnet(x, t, **kwargs)
                return eps_pre

            dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
            _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / N, T=1., order=2, method='singlestep')

        else:
            raise NotImplementedError

        return _z

    def sample_fn(_n_samples):
        if config.train.mode == 'uncond':
            kwargs = dict()
        elif config.train.mode == 'cond':
            kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
        else:
            raise NotImplementedError
        _z = sample_z(_n_samples, _sample_steps=config.nfe, **kwargs)
        return decode_large_batch(_z)

    with tempfile.TemporaryDirectory() as temp_path:
        path = config.sample.path or temp_path
        #if accelerator.is_main_process:
        #    os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        #utils.sample2dir(accelerator, path, config.sample.n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)
        utils.sample2npz(accelerator, path, config.sample.n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess, reset_fn=accelerator.unwrap_model(nnet).reset)

        if accelerator.is_main_process:
            torch.cuda.empty_cache()
            fid = calculate_fid_given_paths((dataset.fid_stat, f"{path}.npz"), batch_size=1000)
            log_path = path.replace('manual_samples/', 'log/')
            with open(f"{log_path}.log", "a") as f:
                f.write(f"npz_path={path}.npz, fid={fid}")
            logging.info(f'npz_path={path}.npz, fid={fid}')


from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output log.")
flags.DEFINE_string("nfe", None, "NFE")
flags.DEFINE_string("router", None, "path of router")
flags.DEFINE_string("thres", "0", "threshold of router")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.nfe = int(FLAGS.nfe)
    config.thres = float(FLAGS.thres)
    config.router = FLAGS.router

    evaluate(config)


if __name__ == "__main__":
    app.run(main)
