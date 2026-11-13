"""Small CPU validation of the candidate contract and scoring hypothesis."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='8'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import time
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
import numpy as np
from lossless_nd_candidate import LosslessNDCandidate


def serde(model,shape,cells):
    before=json.dumps(model.dictionary(),sort_keys=True)
    packet=model.encode(shape,cells)
    clone=LosslessNDCandidate.from_dictionary(json.loads(before))
    assert clone.decode(json.loads(json.dumps(packet)))==(tuple(shape),cells)
    assert before==json.dumps(model.dictionary(),sort_keys=True)
    return packet


def contracts():
    rng=random.Random(20261111);results=[]
    for rank in (0,1,2,3,4,8):
        training=[]
        for phase in range(3):
            shape=tuple(2+(phase+i)%2 for i in range(rank))
            cells={p:(sum(p)+phase)%3-2 for p in product(*(range(n) for n in shape)) if sum(p)%5!=1}
            training.append((shape,cells))
        model=LosslessNDCandidate(rank);model.learn(training,budget=12)
        shape=tuple(3 if i==0 else 2 for i in range(rank))
        cells={p:(2**70 if rng.random()<.1 else -7) for p in product(*(range(n) for n in shape)) if rng.random()>.2}
        packet=serde(model,shape,cells)
        assert serde(model,shape,{})['tokens']==[]
        results.append({'rank':rank,'test_shape':shape,'effective_cells':len(cells),
                        'unseen_literals':len(packet['literals']),'lossless':True})
    a=((2,2),{(0,0):1,(0,1):2,(1,0):3,(1,1):4})
    b=((2,2),{(0,0):1,(0,1):3,(1,0):2,(1,1):4})
    model=LosslessNDCandidate(2);model.learn([a,b,a,b],budget=1)
    assert serde(model,*a)!=serde(model,*b)
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    report={'contract_checks':contracts(),'paired':[]}
    inputs=json.loads((args.inputs/'input.json').read_text())
    for mode,meta in inputs['channels'].items():
        arrays={}
        for label in ('train','paired100'):
            path=Path(meta['files'][label]['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['files'][label]['sha256']
            arrays[label]=np.load(path)
        def cells(image):return {(y,x):int(image[y,x]) for y in range(32) for x in range(32)}
        samples=[((32,32),cells(image)) for image in arrays['train']]
        for score in ('frequency','realizable'):
            started=time.perf_counter();model=LosslessNDCandidate(2)
            train_lengths=model.learn(samples,budget=64,score=score)
            dictionary=model.dictionary();lengths=[];packet_bytes=[];literal_counts=[]
            for image in arrays['paired100']:
                packet=serde(model,(32,32),cells(image))
                lengths.append(len(packet['tokens']))
                packet_bytes.append(len(json.dumps(packet,separators=(',',':')).encode()))
                literal_counts.append(len(packet['literals']))
            row={'channel':mode,'score':score,'train_images':10,'test_images':100,'budget':64,
                 'actual_rules':len(model.rules),'base_alphabet':len(model.base),
                 'train_ratio':sum(train_lengths)/(10*1024),'test_ratio':sum(lengths)/(100*1024),
                 'lossless_bad':0,'all_packet_and_dictionary_json_roundtrips':True,
                 'dictionary_bytes':len(json.dumps(dictionary,separators=(',',':')).encode()),
                 'packet_bytes_total':sum(packet_bytes),'raw_uint8_bytes':100*1024,
                 'literal_counts':literal_counts,'test_lengths':lengths,
                 'input_pixel_sha256':meta['files']['paired100']['pixel_sha256'],
                 'seconds':time.perf_counter()-started}
            report['paired'].append(row)
            (args.out/f'{mode}-{score}-dictionary.json').write_text(json.dumps(dictionary),encoding='utf-8')
            (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(mode,score,'train',row['train_ratio'],'test',row['test_ratio'],'lossless',100,flush=True)


if __name__=='__main__':main()
