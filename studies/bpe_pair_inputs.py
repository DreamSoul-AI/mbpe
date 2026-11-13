"""Freeze shared raw CIFAR inputs for both unmodified public BPE baselines."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='8'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import tarfile
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
import numpy as np
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    with tarfile.open(root/'data/cifar-10-python.tar.gz') as arc:
        train=pickle.load(arc.extractfile('cifar-10-batches-py/data_batch_1'),encoding='bytes')
        test=pickle.load(arc.extractfile('cifar-10-batches-py/test_batch'),encoding='bytes')
    train_labels=np.asarray(train[b'labels']);test_labels=np.asarray(test[b'labels'])
    train_indices=sorted(int(np.flatnonzero(train_labels==c)[0]) for c in range(10))
    test_indices=[int(i) for c in range(10) for i in np.flatnonzero(test_labels==c)[:100]]
    paired=[c*100+j for c in range(10) for j in range(10)]
    train_rgb=train[b'data'][train_indices].reshape(10,3,32,32)
    test_rgb=test[b'data'][test_indices].reshape(1000,3,32,32)
    report={'dataset':'CIFAR10 raw uint8 pixels, channels separate','train_indices':train_indices,
            'test_indices':test_indices,'paired_positions_in_1000':paired,
            'train_images':10,'large_test_images':1000,'paired_test_images':100,'channels':{}}
    for channel,mode in enumerate(('red','green','blue')):
        old=root/'results/2026-11-10-elsner-color-modes/cifar10'/mode/'train'
        files=sorted((old/'transcribed_data_og').rglob('*.dat'),key=lambda p:int(p.stem))
        original=np.stack([torch.load(p,map_location='cpu',weights_only=False).long().numpy().astype(np.int32) for p in files])
        assert np.array_equal(original,train_rgb[:,channel].astype(np.int32)),mode
        large=np.ascontiguousarray(test_rgb[:,channel],dtype=np.int32)
        arrays={'train':original,'test1000':large,'paired100':large[paired]}
        meta={'elsner_rules':str(old/'rules.dat'),
              'elsner_rules_sha256':hashlib.sha256((old/'rules.dat').read_bytes()).hexdigest(),
              'training_input_exact_equal':True,'files':{}}
        for label,array in arrays.items():
            path=args.out/f'{mode}-{label}.npy';np.save(path,array)
            meta['files'][label]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'shape':list(array.shape),'pixel_sha256':hashlib.sha256(array.tobytes()).hexdigest()}
        report['channels'][mode]=meta
    (args.out/'input.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Frozen 10 training / 1000 test / paired 100 per RGB channel; old official learning inputs exactly match.')


if __name__=='__main__':main()
