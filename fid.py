# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import dnnlib
from torch_utils import distributed as dist
from training import dataset
import pandas as pd
from pathlib import Path


@torch.no_grad()
def calculate_inception_stats(
    image_path, num_expected=0, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'), subclass=-1, subset_ids=None
):
    # Rank 0 goes first.
    if dist.get_world_size() > 1 and dist.get_rank() != 0:
        torch.distributed.barrier()

    # Device to store features. CPU is better but not supported by torch.distributed.
    tmp_device = device if dist.get_world_size() > 1 else torch.device('cpu')

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    if subset_ids is not None:
        indices = torch.load(subset_ids)
        if isinstance(indices, list) or torch.is_tensor(indices):
            assert len(indices) == 50000, f"subset_ids should have 50000 indices, but got {len(indices)}"
            dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=0, random_seed=seed, subclass=subclass)
            dataset_obj = torch.utils.data.Subset(dataset_obj, indices)
            dist.print0("\n\n Warning:SubSet loaded!", len(dataset_obj))
        # TODO test this in the future if needed
        elif isinstance(indices, dict):
            datasets = []   
            for path, indeces_subset in indices.items():
                _dataset = dataset.ImageFolderDataset(path=path, max_size=0, random_seed=seed, subclass=subclass)
                datasets.append(torch.utils.data.Subset(_dataset, indeces_subset))
            dataset_obj = torch.utils.data.ConcatDataset(datasets)
            dist.print0("\n\n Warning:ConcatDataset loaded!", len(dataset_obj))
            assert len(dataset_obj) == 50000, f"subset_ids should have 50000 indices, but got {len(dataset_obj)}"       
    else:
        dist.print0(f'Loading images from "{image_path}"...')
        dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed, subclass=subclass)

    assert len(dataset_obj) % dist.get_world_size() == 0, f"dataset size {len(dataset_obj)} is not divisible by {dist.get_world_size()} (n GPUs)."

    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_world_size() > 1 and dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers,
                                              prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    logits_out = []
    features_out = []
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0), ascii=True):
        if dist.get_world_size() > 1:
            torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs)
        features_out.append(features.to(tmp_device))
        logits_out.append(features.mm(detector_net.output.weight.T).to(tmp_device))
        features = features.to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features
    logits_out = torch.cat(logits_out, dim=0)
    features_out = torch.cat(features_out, dim=0)

    # Calculate grand totals.
    logits_out = _gather_cat(logits_out)
    features_out = _gather_cat(features_out)

    if dist.get_world_size() > 1:
        torch.distributed.all_reduce(mu)
        torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy(), logits_out.cpu(), features_out.cpu()


def _gather_cat(x):
    if dist.get_world_size() > 1:
        x_tmp = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(x_tmp, x)
        x = torch.cat(x_tmp, dim=0)
    return x


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


@click.group()
def main():
    """Calculate Frechet Inception Distance (FID).

    Examples:

    \b
    # Generate 50000 images and save them as fid-tmp/*/*.png
    torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Calculate FID
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

    \b
    # Compute dataset reference statistics
    python fid.py ref --data=datasets/my-dataset.zip --dest=fid-refs/my-dataset.npz
    """


@main.command()
@click.option('--images', 'image_path', help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--ref', 'ref_path',      help='Dataset reference statistics ', metavar='NPZ|URL',    type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--subset_ids', type=str, default=None)
@click.option('--calc_is', is_flag=True, show_default=True, default=False)
@click.option('--calc_mss', is_flag=True, show_default=True, default=False)
@click.option('--calc_vendi', is_flag=True, show_default=True, default=False)
def calc(image_path, ref_path, num_expected, seed, batch, subset_ids, calc_is, calc_mss, calc_vendi):
    """Calculate FID for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))

    mu, sigma, logits, feats = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch, subset_ids=subset_ids)
    if dist.get_rank() == 0:
        img_path_out = Path(image_path)
        out_path = img_path_out.parent / f'fid_seeds_{img_path_out.name}.csv' 
        if subset_ids is not None:
            subset_ids = subset_ids.replace('.pt', '')
            out_path = img_path_out / f"fid_subset_{Path(subset_ids).name}.csv"

        dist.print0('Calculating FID...')
        fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])

        df = pd.DataFrame({'FID': [fid]})
        df.to_csv(out_path, index=False)
        print(df.round(3).to_string(index=False))
    torch.distributed.barrier()


@main.command()
@click.option('--data', 'dataset_path', help='Path to the dataset', metavar='PATH|ZIP',         type=str, required=True)
@click.option('--dest', 'dest_path',    help='Destination .npz file', metavar='NPZ',            type=str, required=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',               type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--subclass',             help='Compute stats only for one class', metavar='INT', type=click.IntRange(min=-1, max=9), default=-1, show_default=True)
def ref(dataset_path, dest_path, batch, subclass):
    """Calculate dataset reference statistics needed by 'calc'."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    mu, sigma = calculate_inception_stats(image_path=dataset_path, max_batch_size=batch, subclass=subclass, num_expected=0)
    dist.print0(f'Saving dataset reference statistics to "{dest_path}"...')
    if dist.get_rank() == 0:
        if os.path.dirname(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.savez(dest_path, mu=mu, sigma=sigma)

    torch.distributed.barrier()
    dist.print0('Done.')


if __name__ == "__main__":
    main()

