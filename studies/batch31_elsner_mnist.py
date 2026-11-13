"""Official Elsner notebook on MNIST bin128. The notebook file is not modified.

The in-memory copy uses 28x28. New ids still start at max(mask)+1.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMBA_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
from pathlib import Path

import batch30_elsner_mask as official
import batch31_mnist as mnist


_prepare_setup = official.prepare_setup


def prepare_setup(source, images, rules, alphabet):
    text = _prepare_setup(source, images, rules, alphabet)
    return text.replace('WIDTH, HEIGHT = 32, 32', 'WIDTH, HEIGHT = 28, 28', 1)


official.prepare_setup = prepare_setup


def grids(kind, train_count, test_count):
    train = mnist.load_idx(mnist.RAW / 'train-images-idx3-ubyte')
    test = mnist.load_idx(mnist.RAW / 't10k-images-idx3-ubyte')
    chosen = mnist.choose_indices(len(train), 1000)
    train_grids = [mnist.mask_image(train[index], kind) for index in chosen[:train_count]]
    test_grids = [mnist.mask_image(test[index], kind) for index in range(test_count)]
    return train_grids, test_grids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-images', type=int, required=True)
    parser.add_argument('--rules', type=int, required=True)
    parser.add_argument('--test-images', type=int, required=True)
    args = parser.parse_args()
    train, test = grids('bin128', args.train_images, args.test_images)
    tag = f'bin128-train{args.train_images}-r{args.rules}-test{args.test_images}'
    root = args.out / 'p22-elsner' / tag
    official.learn(root / 'train', train, args.rules, 2)
    official.apply_test(root / 'test', root / 'train', test, args.rules, 2)
    print('P22_DONE', tag, flush=True)


if __name__ == '__main__':
    main()
