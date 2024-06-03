import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from accelerate import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder
import numpy as np


def format_image_to_wandb(num_router, router_size, router_scores):
    image = np.zeros((num_router, router_size, 3), dtype=np.float32)
    ones = np.ones((3), dtype=np.float32)
    for idx, score in enumerate(router_scores):
        mask = score.cpu().detach()
        for pos in range(router_size):
            image[idx, pos] = ones * mask[pos].item()
    return image

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)

def sos(a, start_dim=1):  # sum of square
    e = a.pow(2).flatten(start_dim=start_dim)
    return e.sum(dim=-1)


class Schedule(object):  # discrete time
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0., _betas)
        self.alphas = 1. - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0):  # sample from q(xn|x0), where n is uniform
        n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
        eps = torch.randn_like(x0)
        xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        return torch.tensor(n, device=x0.device), eps, xn
    
    def get_xn(self, x0, n):
        eps = torch.randn_like(x0)
        xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        return torch.tensor(n, device=x0.device), eps, xn

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'


def LSimple(x0, nnet, schedule, **kwargs):
    
    n, eps, xn = schedule.sample(x0)  # n in {1, ..., 1000}
    eps_pred = nnet(xn, n, **kwargs)
    return mos(eps - eps_pred)


def LRouter(x0, nnet, schedule,  order=None, timesteps=None, dpm_solver=None, **kwargs):
    #print(x0.shape)
    #print(order, timesteps)

    def model_fn(x, t_continuous):
        t = t_continuous * 1000
        eps_pre = nnet(x, t, **kwargs)
        return eps_pre
    dpm_solver.model = model_fn
    nnet.module.reset_cache_features()
    random_step = np.random.randint(0, len(order)-1)
    random_t = np.round(timesteps[random_step] * 1000).astype(int).repeat(x0.shape[0])
    
    #print(random_t)
    _, _, xn = schedule.get_xn(x0, random_t)
    vec_s = torch.ones((xn.shape[0],)).to(xn.device) * timesteps[random_step]
    vec_t = torch.ones((xn.shape[0],)).to(xn.device) * timesteps[random_step + 1]
    with torch.no_grad():
        xn_minus_1 = dpm_solver.dpm_solver_second_update(xn, vec_s, vec_t, return_noise=False, solver_type='dpm_solver')
    
    random_t_minus_1 = np.round(timesteps[random_step + 1] * 1000).astype(int).repeat(x0.shape[0])
    random_t_minus_1 = torch.tensor(random_t_minus_1).to(xn_minus_1.device)

    # Teacher
    nnet.module.set_activate_cache(False)
    nnet.module.set_record_cache(False)
    t_pred = nnet(xn_minus_1, random_t_minus_1, **kwargs)

    # Student
    nnet.module.set_activate_cache(True)
    
    s_pred, l1_loss = nnet(xn_minus_1, random_t_minus_1, **kwargs)

    nnet.module.set_activate_cache(False)
    nnet.module.set_record_cache(True)
    
    return sos(t_pred - s_pred), l1_loss
    

def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])
    #accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train')#, mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    # Load Dataset
    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond')
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    # Load Model and Optimizer
    train_state = utils.initialize_train_state(config, device)
    train_state.nnet.add_router(config.nfe)
    router_optim = torch.optim.AdamW(
        [param for name, param in train_state.nnet.named_parameters() if "routers" in name], 
        lr=config.router_lr, weight_decay=0
    )
    train_state.update_optimizer(router_optim)
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)
    logging.info(f'load nnet from {config.nnet_path}')
    msg = accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'), strict=False)
    logging.info(f'load nnet messgae = {config.nnet_path}')
    

    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    # Load Autoencoder
    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    # Setup DPM Solver
    _betas = stable_diffusion_beta_schedule()
    _schedule = Schedule(_betas)
    logging.info(f'use {_schedule}')

    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())
    dpm_solver = DPM_Solver(None, noise_schedule, predict_x0=True, thresholding=False)
    t_0 = 1. / _schedule.N
    t_T = 1.0
    order_value = 2
    N_steps = config.nfe // order_value
    order = [order_value,] * N_steps
    timesteps = dpm_solver.get_time_steps(
        skip_type='time_uniform', t_T=t_T, t_0=t_0, N=N_steps, device=device
    )
    timesteps = timesteps.cpu().numpy()
    timestep_mapping = np.round(timesteps * 1000)
    accelerator.unwrap_model(nnet).set_timestep_map(timestep_mapping)

    @ torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @ torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    
    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        if config.train.mode == 'uncond':
            _z = autoencoder.sample(_batch) if 'feature' in config.dataset.name else encode(_batch)
            data_loss, l1_loss = LRouter(_z, nnet, _schedule, order=order, timesteps=timesteps, dpm_solver=dpm_solver, l1_weight=config.l1_weight)
        elif config.train.mode == 'cond':
            #print("Label = ", _batch[1])
            _z = autoencoder.sample(_batch[0]) if 'feature' in config.dataset.name else encode(_batch[0])
            data_loss, l1_loss = LRouter(_z, nnet, _schedule, y=_batch[1], order=order, timesteps=timesteps, dpm_solver=dpm_solver)
            loss = data_loss + config.l1_weight * l1_loss
        else:
            raise NotImplementedError(config.train.mode)
        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        _metrics['data_loss'] = accelerator.gather(data_loss.detach()).mean()
        _metrics['l1_loss'] = accelerator.gather(l1_loss.detach()).mean()


        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.step += 1

        #print("Router 0:", nnet.module.routers[0].prob.data)
        #print("Router 1:", nnet.module.routers[1].prob.data)
        #print()
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)
   
    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')
    
    loss_metrics = 0
    data_loss_metrics = 0
    l1_loss_metrics = 0
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        if accelerator.is_main_process:
            loss_metrics += metrics['loss']
            data_loss_metrics += metrics['data_loss']
            l1_loss_metrics += metrics['l1_loss']

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            scores = [nnet.module.routers[idx]() for idx in range(1, config.nfe//2)]
            mask = format_image_to_wandb(config.nfe//2-1, nnet.module.depth*2, scores)
            mask = wandb.Image(
                mask,
            )
            metrics['loss'] = loss_metrics / config.train.log_interval
            metrics['data_loss'] = data_loss_metrics / config.train.log_interval
            metrics['l1_loss'] = l1_loss_metrics / config.train.log_interval
            final_score = [sum(score) for score in scores]
            metrics['non_zero'] = sum(final_score)  / (len(final_score) * len(scores[0]))

            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            metrics['router'] = mask
            #logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)
            loss_metrics, data_loss_metrics, l1_loss_metrics = 0, 0, 0

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            #fid = eval_step(n_samples=10000, sample_steps=50)  # calculate fid of the saved checkpoint
            #step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    


from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("nfe", None, "NFE")
flags.DEFINE_string("router_lr", None, "learning rate for router")
flags.DEFINE_string("l1_weight", None, "l1 weight for router loss")
flags.DEFINE_string("nnet_path", None, "l1 weight for router loss")



def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.nfe = int(FLAGS.nfe)
    config.router_lr = float(FLAGS.router_lr)
    config.l1_weight = float(FLAGS.l1_weight)
    config.nnet_path = FLAGS.nnet_path
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
