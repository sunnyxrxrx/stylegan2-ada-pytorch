# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate a horizontal W-space interpolation strip using a pretrained network."""

import os
import re
from typing import List, Optional

import click
import numpy as np
import PIL.Image
import torch

from torch_utils.checkpoint import load_network_checkpoint


def num_range(s: str) -> List[int]:
    """Accept either a comma separated list 'a,b,c' or a range 'a-c'."""

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


def make_label(G, device: torch.device, class_idx: Optional[int]) -> torch.Tensor:
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim == 0:
        if class_idx is not None:
            print('warn: --class is ignored for an unconditional network')
        return label
    if class_idx is None:
        raise click.ClickException('Must specify --class when using a conditional network')
    label[:, class_idx] = 1
    return label


def tensor_to_uint8_image(image: torch.Tensor) -> np.ndarray:
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return image[0].cpu().numpy()


@click.command()
@click.option('--network', 'network_pkl', help='Network checkpoint filename', required=True)
@click.option('--seed-a', type=int, required=True, help='Starting seed')
@click.option('--seed-b', type=int, required=True, help='Ending seed')
@click.option('--layers', type=num_range, default='0-6', show_default=True, help='W layers to interpolate')
@click.option('--steps', type=int, default=9, show_default=True, help='Number of interpolation frames, including endpoints')
@click.option('--trunc', 'truncation_psi', type=float, default=1, show_default=True, help='Truncation psi')
@click.option('--class', 'class_idx', type=int, help='Class label for conditional networks')
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=str, required=True)
def generate_interpolation(
    network_pkl: str,
    seed_a: int,
    seed_b: int,
    layers: List[int],
    steps: int,
    truncation_psi: float,
    class_idx: Optional[int],
    noise_mode: str,
    outdir: str,
):
    """Generate a single-row interpolation strip in W space."""

    if steps < 2:
        raise click.ClickException('--steps must be at least 2')

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    G = load_network_checkpoint(network_pkl)['G_ema'].to(device)  # type: ignore
    G.eval().requires_grad_(False)

    valid_layers = sorted(set(layers))
    if not valid_layers:
        raise click.ClickException('No interpolation layers were provided')
    if min(valid_layers) < 0 or max(valid_layers) >= G.num_ws:
        raise click.ClickException(f'Layer range must stay within [0, {G.num_ws - 1}]')

    os.makedirs(outdir, exist_ok=True)

    label = make_label(G, device, class_idx)

    print('Computing endpoint W vectors...')
    z_a = torch.from_numpy(np.random.RandomState(seed_a).randn(1, G.z_dim)).to(device)
    z_b = torch.from_numpy(np.random.RandomState(seed_b).randn(1, G.z_dim)).to(device)
    w_a = G.mapping(z_a, label, truncation_psi=truncation_psi)[0]
    w_b = G.mapping(z_b, label, truncation_psi=truncation_psi)[0]

    print('Generating interpolation frames...')
    ts = np.linspace(0.0, 1.0, steps)
    frame_images = []
    for idx, t in enumerate(ts):
        w_interp = w_a.clone()
        w_interp[valid_layers] = w_a[valid_layers].lerp(w_b[valid_layers], float(t))
        image = G.synthesis(w_interp.unsqueeze(0), noise_mode=noise_mode)
        image_np = tensor_to_uint8_image(image)
        frame_images.append(image_np)
        PIL.Image.fromarray(image_np, 'RGB').save(f'{outdir}/step_{idx:03d}.png')

    print('Saving interpolation grid...')
    width = G.img_resolution
    height = G.img_resolution
    canvas = PIL.Image.new('RGB', (width * steps, height), 'black')
    for idx, image_np in enumerate(frame_images):
        canvas.paste(PIL.Image.fromarray(image_np, 'RGB'), (width * idx, 0))
    canvas.save(f'{outdir}/grid.png')

    print(f'Saved {steps} interpolation steps to "{outdir}"')
    print(f'Seeds: {seed_a} -> {seed_b}')
    print(f'Interpolated layers: {valid_layers}')


if __name__ == "__main__":
    generate_interpolation()  # pylint: disable=no-value-for-parameter
