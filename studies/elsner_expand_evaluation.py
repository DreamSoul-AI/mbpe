"""Larger held-out evaluation of existing rules using the official notebook loop."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'tmp' / 'elsner-expanded-evaluation'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='8'
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp' / 'matplotlib'))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
assert psutil.Process().nice()==psutil.IDLE_PRIORITY_CLASS
import hashlib
import argparse
import json
import pickle
import tarfile
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
import elsner_svhn_celeba as runner

def gray(rgb):
    return np.asarray(Image.fromarray(rgb).convert('L'),dtype=np.uint8)


def select_per_class(labels, count=10):
    # Includes the earlier one-per-class test samples, for paired reporting.
    return [int(i) for c in range(10) for i in np.flatnonzero(labels==c)[:count]]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['all','celeba'],default='all')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    notebook=runner.NOTEBOOK
    digest=hashlib.sha256(notebook.read_bytes()).hexdigest()
    jobs=[]
    with tarfile.open(ROOT/'data/cifar-10-python.tar.gz') as arc:
        batch=pickle.load(arc.extractfile('cifar-10-batches-py/test_batch'),encoding='bytes')
        indices=select_per_class(np.asarray(batch[b'labels']))
        rgbs=batch[b'data'][indices].reshape(-1,3,32,32).transpose(0,2,3,1)
        jobs.append(('cifar10',[gray(x) for x in rgbs],indices))
    sv=loadmat(ROOT/'data/test_32x32.mat')
    indices=select_per_class(sv['y'].ravel()%10)
    jobs.append(('svhn',[gray(sv['X'][:,:,:,i]) for i in indices],indices))
    paths=sorted((ROOT/'data/celeba/partial_jpg').glob('*.jpg'))[10:100]
    pictures=[]
    for path in paths:
        with Image.open(path) as image:
            # Preserve the earlier experiment: ToTensor then notebook Resize.
            # PIL resizing first rounds differently and is not equivalent.
            pictures.append(np.asarray(image.convert('L')))
    jobs.append(('celeba',pictures,[p.name for p in paths]))
    reports=[]
    for name,images,indices in jobs:
        if args.dataset!='all' and name!=args.dataset:
            continue
        work=output/name/'test'
        if work.exists():
            raise FileExistsError(work)
        rules=ROOT/'results/2026-11-10-elsner-color-modes'/name/'gray/train'
        meta={'dataset':name,'count':len(images),'indices':indices,'notebook_sha256':digest,
              'rules_sha256':hashlib.sha256((rules/'rules.dat').read_bytes()).hexdigest(),
              'rules_source':str(rules),'rules_learned_images':10,'mode':'gray',
              'new_merges':256,'workers':0,'torch_threads':torch.get_num_threads(),
              'priority':str(psutil.Process().nice()),'overlap_prior_test':10,
              'pixels':'PIL L; official notebook ToTensor then Resize 32x32; CelebA partial nonofficial split'}
        work.parent.mkdir(parents=True,exist_ok=True)
        (work.parent/'input.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        print('START',name,len(images),flush=True)
        runner.PREPARED_GRIDS=images
        summary=runner._apply_test(work,rules,'grids',1,len(images),32,32,256)
        assert hashlib.sha256(notebook.read_bytes()).hexdigest()==digest
        reports.append(dict(meta=meta,summary=summary))
        (output/'reports.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print('FINISHED',name,json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
