# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import click
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import json
from pathlib import Path
from tqdm import tqdm
from torch_utils import distributed as dist
from train import parse_int_list


def heun_step(net, x_cur, t_cur, t_next, i, class_labels, num_steps, S_churn, S_min, S_max, S_noise, randn_like,
              high_precision, dtype, cfg_weight=0):
    """Executes one Heun step, as detailed in Algorithm 2 of the EDM paper."""
    if S_min <= t_cur <= S_max:
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
    else:
        x_hat = x_cur
        t_hat = t_cur

    # Euler step.
    d_cur = compute_score(net, x_hat, t_hat, class_labels, high_precision, dtype, cfg_weight)
    x_eul = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        d_prime = compute_score(net, x_eul, t_next, class_labels, high_precision, dtype, cfg_weight)
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    else:
        x_next = x_eul
        d_prime = None

    return x_next, d_cur, d_prime, t_hat


def compute_score(net, x, t, class_labels, high_precision, dtype, cfg_weight=0):
    """Estimates the score, optionally with classifier-free guidance. Algorithm 2, line 7 in the EDM paper."""
    denoised = net(x, t, class_labels, force_fp32=high_precision).to(dtype)
    if cfg_weight:  # classifier-free guidance
        denoised = ((1 + cfg_weight) * denoised - cfg_weight * net(x, t, class_labels=None, force_fp32=high_precision).to(dtype))
    d_cur = (x - denoised) / t  # score = force / t^2
    return d_cur


def edm_sampler(
    net, latents, num_steps, sigma_min, sigma_max, rho, S_churn, S_min, S_max, S_noise,
    class_labels=None, randn_like=torch.randn_like, high_precision=True, cfg_weight=0, **kwargs,
):
    """EDM sampler, Algorithm 2 in EDM paper."""
    dtype = torch.float64 if high_precision else torch.float32

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    heun_kwargs = {'num_steps': num_steps, 'S_churn': S_churn, 'S_min': S_min,
                   'S_max': S_max, 'S_noise': S_noise, 'randn_like': randn_like,
                   'high_precision': high_precision, 'dtype': dtype, 'cfg_weight': cfg_weight}

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(dtype) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        x_next, d_cur, d_prime, t_hat = heun_step(net, x_cur, t_cur, t_next, i, class_labels, **heun_kwargs)

    return x_next


class StackedRandomGenerator:
    """Wrapper for torch.Generator that allows specifying a different random seed for each sample in a minibatch."""
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

    def multinomial(self, input, replacement=False, device='cuda', **kwargs):
        return torch.stack([torch.multinomial(input, 1, replacement, generator=gen, **kwargs).to(device) for gen in self.generators])


def get_labels(rnd, label_dim, cluster_freq, batch_idx, batch_size, device, raw_seeds_batch=None):
    """Samples pseudo-labels, either from a uniform distribution or given weights in cluster_freq."""
    # Uniform sampling
    if cluster_freq is None:
        labels = rnd.randint(label_dim, size=[batch_size], device=device)  # uniform label-distribution
        return labels

    # Weighted sampling using weights stored in cluster_freq
    if cluster_freq.dim() == 1:
        cluster_freq = cluster_freq.to(device).to(torch.float32)
        labels = rnd.multinomial(cluster_freq, replacement=True, device=device).squeeze()
    elif cluster_freq.dim() == 2:
        if raw_seeds_batch is not None:
            seeds_tensor = cluster_freq[batch_idx, 0]
            assert torch.all(seeds_tensor == raw_seeds_batch), \
                f'Seeds do not match: seeds_tensor: {seeds_tensor}, raw_seeds_batch: {raw_seeds_batch}'
        labels = cluster_freq[batch_idx, -1]
    else:
        raise NotImplementedError
    return labels.long()


@click.command()
# Inference kwargs
@click.option('--network', 'network_pkl',   help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                    help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0', show_default=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=16, show_default=True)
@click.option('--cfg_weight',               help="Classifier-free guidance weight. 0 = no guidance",                 type=float, default=0.)
@click.option('--load_seeds',               help="Load predefined seeds from this path",                             type=str, default=None)
@click.option('--freq_path',                help="Load class/cluster frequencies from this path (p(c))",             type=str, default=None)
# Sampler kwargs
@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=21, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True), default=2e-3)
@click.option('--sigma_max',                help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True), default=80)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=-1000000, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, cfg_weight, load_seeds, freq_path,
         device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models" and
    "Rethinking cluster-conditioned diffusion models".

    Examples with pre-trained EDM model:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # Load cherry-picked seeds
    if load_seeds is not None:
        raw_seeds = torch.load(load_seeds)
        # Fill up seeds to 50k
        raw_seeds = raw_seeds[raw_seeds != -1]
        raw_seeds = raw_seeds[:len(seeds)]
        unique = raw_seeds.unique()
        dist.print0(f'Loaded {unique.shape[0]} seeds from {load_seeds}')
        raw_seeds = torch.zeros(len(seeds), dtype=torch.int64)
        raw_seeds[-len(unique):] = unique
        seed = 0
        for i in range(len(seeds) - len(unique)):
            while seed in unique:
                seed += 1
            raw_seeds[i] = seed
            seed += 1
        seeds = list(range(len(seeds)))
    else:
        raw_seeds = torch.tensor(seeds)
        assert torch.all(raw_seeds.unique() == raw_seeds), 'Non-unique seeds are not supported'

    num_batches = ((len(seeds) - 1) // (max_batch_size * world_size) + 1) * world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[rank:: world_size]  # seeds, batched for each GPU
    idx_batches = torch.arange(len(seeds)).tensor_split(num_batches)[rank:: world_size]  # indices, batched for each GPU

    # Rank 0 goes first.
    if rank != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(rank == 0)) as f:
        data = pickle.load(f)
        net = data['ema'].to(device)

    # Other ranks follow.
    if rank == 0:
        torch.distributed.barrier()

    method_str = 'classifier-free guidance' if cfg_weight > 0 else 'EDM inference'
    dist.print0(f'Generating {len(seeds)} images to "{outdir}" with {method_str}...')

    dist.print0(f'Sampler kwargs: {sampler_kwargs}')
    total_counter = 0
    if freq_path and not net.label_dim:
        dist.print0("--freq_path was specified, but the network does not have a label_dim. Ignoring --freq_path.")

    if net.label_dim:
        if freq_path:
            cluster_freq = torch.load(freq_path)  # [num_clusters,]
            assert net.label_dim == cluster_freq.shape[0], f'Label dim {net.label_dim} does not match cluster_freq shape {cluster_freq.shape}.'
        else:
            cluster_freq = None
        if rank == 0:
            label_dict_path = os.path.join(outdir, 'label_dict.pt')
            if os.path.exists(label_dict_path):
                dist.print0(f'Loading labels from {label_dict_path}')
                label_dict = torch.load(label_dict_path)
            else:
                label_dict = {}

    # Save arguments to JSON file
    all_arguments = click.get_current_context().params
    all_arguments["seeds"] = len(seeds) 
    os.makedirs(outdir, exist_ok=True)
    json_filename = Path(outdir) / 'generate_args.json'
    with open(json_filename, 'w') as jsonfile:
        json.dump(all_arguments, jsonfile, indent=4)

    # Loop over batches.
    for c, (batch_seeds, batch_idx) in enumerate(tqdm(zip(rank_batches, idx_batches), unit='batch', disable=(rank != 0),
                                                 ascii=True, total=len(rank_batches))):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)

        if batch_size == 0:
            continue

        # Number of generated images across all GPUs in this batch
        total_batch_size = 0
        for r in range(world_size):
            total_batch_size += all_batches[world_size * c: world_size * (c + 1)][r].shape[0]
        total_counter += total_batch_size

        # Pick latents and labels.
        raw_seeds_batch = raw_seeds[batch_idx]
        rnd = StackedRandomGenerator(device, raw_seeds_batch)  # Controllable randomization
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None

        # Set up label condition
        if net.label_dim:
            labels = get_labels(rnd, net.label_dim, cluster_freq, batch_idx, batch_size, device, raw_seeds_batch).to(device)
            class_labels = torch.eye(net.label_dim, device=device)[labels]

            # collect labels across GPUs
            batch_labels = torch.zeros(total_batch_size, dtype=torch.int32, device=device)
            batch_labels[rank::world_size] = labels.to(torch.int32)
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1


        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        sampler_kwargs.update(randn_like=rnd.randn_like, class_labels=class_labels)
        images = edm_sampler(net, latents, cfg_weight=cfg_weight, **sampler_kwargs)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for i, (seed, image_np) in enumerate(zip(batch_seeds, images_np)):
            # Save samples
            image_dir = os.path.join(outdir, f'{seed-seed%50000:07d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:07d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        # Save labels.
        if net.label_dim:
            torch.distributed.reduce(batch_labels, dst=0, op=torch.distributed.ReduceOp.SUM)  # Collect all labels on GPU 0
            if rank == 0:
                for r in range(world_size):
                    # Iterate the batch_labels for this rank and the seeds for this rank
                    for label, seed in zip(batch_labels[r::world_size].cpu(), all_batches[world_size * c + r]):
                        label_dict[seed.item()] = label.item()
                torch.save(label_dict, os.path.join(outdir, 'label_dict.pt'))

    # Done.
    torch.distributed.barrier()
    
    dist.print0('Done.')


if __name__ == "__main__":
    main()
