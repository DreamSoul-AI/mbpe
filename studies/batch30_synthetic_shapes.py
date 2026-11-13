"""Larger synthetic shape masks for batch 30. No downloads.

Each image is a 64x64 grid. With seed 20261111, every image draws 4 to 8
axis-aligned rectangles and 2 to 5 disks. Binary masks paint them as 1 on a
0 background. Four-level masks paint each shape with a label from 1 to 3.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMBA_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
from pathlib import Path

import numpy as np

from batch30_mask_baselines import blobs, run_suite


def make_images(count, side, alphabet, seed):
    rng = np.random.default_rng(seed)
    images = []
    yy, xx = np.ogrid[:side, :side]
    for _ in range(count):
        grid = np.zeros((side, side), dtype=np.int64)
        for _rect in range(int(rng.integers(4, 9))):
            y = int(rng.integers(0, side))
            x = int(rng.integers(0, side))
            height = int(rng.integers(4, max(5, side // 3)))
            width = int(rng.integers(4, max(5, side // 3)))
            value = 1 if alphabet == 2 else int(rng.integers(1, alphabet))
            grid[y:y + height, x:x + width] = value
        for _disk in range(int(rng.integers(2, 6))):
            cy = int(rng.integers(0, side))
            cx = int(rng.integers(0, side))
            radius = int(rng.integers(4, max(5, side // 4)))
            value = 1 if alphabet == 2 else int(rng.integers(1, alphabet))
            grid[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius] = value
        images.append(grid)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--alphabet', type=int, choices=(2, 4), required=True)
    parser.add_argument('--side', type=int, default=64)
    parser.add_argument('--train', type=int, default=64)
    parser.add_argument('--test', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261111)
    args = parser.parse_args()
    kind = 'bin128' if args.alphabet == 2 else '4level'
    images = make_images(args.train + args.test, args.side, args.alphabet, args.seed)
    train = images[:args.train]
    test = images[args.train:]
    flat, byte_blob, packed = blobs(test, args.alphabet)
    run_suite('synthetic', train, {'synthetic': (test, flat, byte_blob, packed)}, kind,
              args.out / 'p20-synthetic' / f'side{args.side}-a{args.alphabet}.json')
    print('P20_DONE', flush=True)


if __name__ == '__main__':
    main()
