"""Real raw-pixel coverage of the paired mechanism, no official source changes."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='8'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np
from nd_bpe_probe import AxisBPE
from nd_bpe_graph_probe import run
from nd_bpe_incremental_probe import Index, IndexedState


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rgb',action='store_true')
    args=parser.parse_args()
    if os.name=='nt':
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    root=Path(__file__).resolve().parents[1]
    datasets=[]
    indices=[]
    with tarfile.open(root/'data/cifar-10-python.tar.gz') as arc:
        for name,per_class in [('data_batch_1',1 if args.rgb else 2),('test_batch',1)]:
            batch=pickle.load(arc.extractfile('cifar-10-batches-py/'+name),encoding='bytes')
            labels=np.asarray(batch[b'labels'])
            chosen=[int(i) for c in range(10) for i in np.flatnonzero(labels==c)[:per_class]]
            rgb=batch[b'data'][chosen].reshape(len(chosen),3,32,32)
            if args.rgb:
                datasets.append([((32,32,3),{(y,x,c):int(image[c,y,x]) for y in range(32) for x in range(32) for c in range(3)}) for image in rgb])
            else:
                datasets.append([((32,32),{(y,x):int(image[0,y,x]) for y in range(32) for x in range(32)}) for image in rgb])
            indices.append(chosen)
    if args.rgb:
        index=Index();states=[IndexedState(*sample,index,i) for i,sample in enumerate(datasets[0])]
        trace=[]
        for step in range(64):
            key,_=index.best();candidates=[]
            for image,a,b in index.occurrences[key]:
                state=states[image]
                if state.regions[a][1]>state.regions[b][1]:a,b=b,a
                candidates.append((state.edges[a][b],image,a,b))
            for _,image,a,b in sorted(candidates):states[image].replace(a,b,256+step)
            trace.append({'rule':key,'token':256+step})
        report={'trace':trace,'rules':len(trace),'axes':'height,width,channel; all 3 axes enabled'}
    else:
        report=run((32,32),False,budget=256,training=datasets[0])
    dictionary={'rank':3 if args.rgb else 2,'base_alphabet':256,
                'rules':[(row['rule'],row['token']) for row in report['trace']]}
    model=AxisBPE.from_dictionary(dictionary)
    lengths=[]
    before=json.dumps(model.dictionary(),sort_keys=True)
    for shape,cells in datasets[1]:
        packet=model.encode(shape,cells)
        clone=AxisBPE.from_dictionary(json.loads(json.dumps(dictionary)))
        assert clone.decode(json.loads(json.dumps(packet)))==(shape,cells)
        compact=model.without_anchors(packet,cells)
        assert clone.decode_without_anchors(compact)==(shape,cells)
        lengths.append(len(packet['tokens']))
    assert before==json.dumps(model.dictionary(),sort_keys=True)
    report.update(dataset='CIFAR10 whole raw RGB' if args.rgb else 'CIFAR10 raw red pixels',indices=indices,
                  train_images=len(datasets[0]),test_images=10,test_lengths=lengths,
                  test_ratio=sum(lengths)/((3072 if args.rgb else 1024)*len(lengths)),lossless_bad=0,
                  includes_official_implementation=False)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('trace','indices')},indent=2))


if __name__=='__main__':main()
