"""Raw MNIST row BPE, fixed 256-rule budget, independent Huffman archive check."""

import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import struct
import time

import numpy as np
import psutil

from batch30_mask_baselines import learn_apply,merge_one
from canonical_huffman import lengths_from_validation
import grid_bpe_file as archive
from grid_bpe_study import write_checked
from rgb_capacity_study import digest_file,save_json,resource_receipt


def load_images(path):
    data=path.read_bytes();magic,count,rows,cols=struct.unpack('>IIII',data[:16])
    if magic!=2051 or (rows,cols)!=(28,28) or len(data)!=16+count*784:
        raise ValueError('expected raw MNIST image IDX')
    return np.frombuffer(data,dtype=np.uint8,offset=16).reshape(count,28,28)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.raw=args.raw.resolve();args.out=args.out.resolve()
    process=psutil.Process();process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:1]);started=time.perf_counter()
    if psutil.virtual_memory().available<4*1024**3:raise RuntimeError('insufficient free memory')
    train_path=args.raw/'train-images-idx3-ubyte';labels_path=args.raw/'train-labels-idx1-ubyte'
    test_path=args.raw/'t10k-images-idx3-ubyte'
    train=load_images(train_path);test=load_images(test_path)
    label_data=labels_path.read_bytes();magic,count=struct.unpack('>II',label_data[:8])
    if magic!=2049 or count!=len(train) or len(label_data)!=count+8:raise ValueError('bad labels')
    labels=np.frombuffer(label_data,dtype=np.uint8,offset=8)
    # Same first-50-per-class learning selection as the historical Elsner model.
    chosen=Counter();learn=[]
    for index,label in enumerate(labels.tolist()):
        if chosen[label]<50:learn.append(index);chosen[label]+=1
    if len(learn)!=500:raise ValueError('learning classes incomplete')
    seen={sha256(train[index].tobytes()).hexdigest() for index in learn}
    validation=[];counts=Counter();learn_indices=set(learn)
    for index,label in enumerate(labels.tolist()):
        digest=sha256(train[index].tobytes()).hexdigest()
        if index not in learn_indices and digest not in seen and counts[label]<100:
            validation.append(index);counts[label]+=1;seen.add(digest)
    if len(validation)!=1000:raise ValueError('validation classes incomplete')
    sources={path.name:digest_file(path) for path in (train_path,labels_path,test_path)}
    save_json(args.out/'protocol.json',{'scan':'row','rules':256,'base':256,'raw_gray_levels':256,
        'learn_indices':learn,'validation_indices':validation,'test_indices':list(range(len(test))),
        'source_sha256':sources,'huffman_counts':'validation only, one-count smoothing',
        'extra_model_positions':0,'condition':'fixed 28x28 shape, row scan and image boundaries shared',
        'source_sha256_snapshot':{p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')}})
    training=[train[index].ravel().astype(np.int32) for index in learn]
    validating=[train[index].ravel().astype(np.int32) for index in validation]
    rules,saved=learn_apply(training,{'validation':validating},256,256,{256})
    if len(rules)!=256:raise ValueError('rule budget not reached')
    val_tokens,bad=saved['validation'][256]
    if bad:raise ValueError('validation roundtrip failed')
    pairs=tuple((left,right) for left,right,_ in rules)
    spec=archive.Spec('row',2,tuple(range(256)),(pairs,))
    lengths=(lengths_from_validation(val_tokens,513),)
    save_json(args.out/'selected-before-test.json',{'scan':'row','pairs':pairs,
        'huffman_lengths':list(lengths[0]),'protocol_sha256':digest_file(args.out/'protocol.json')})
    shapes={i:(i,) for i in range(256)}
    for left,right,token in rules:shapes[token]=shapes[left]+shapes[right]
    packets=[];counts=[];reference=[];test_hashes=set()
    for number,image in enumerate(test):
        original=image.ravel().astype(np.int32);tokens=original
        for left,right,token in rules:tokens=merge_one(tokens,left,right,token)
        values=tokens.astype(int).tolist()
        restored=[v for token in values for v in shapes[token]]
        if restored!=original.astype(int).tolist():raise ValueError('test pixels differ')
        packets.append(archive.Packet((28,28),None,(),tuple(values)));counts.append(len(values))
        reference.append(archive.grid_bytes((28,28),None,restored))
        test_hashes.add(sha256(image.tobytes()).hexdigest())
        if (number+1)%1000==0:print(f'encoded {number+1}/{len(test)}',flush=True)
        if process.memory_info().rss>2*1024**3 or psutil.virtual_memory().available<4*1024**3:
            raise RuntimeError('memory guard triggered')
    receipt=write_checked(args.out/'mnist-row-256.huffman.mbpg',spec,[0]*len(test),packets,
        'huffman',lengths,b''.join(reference))
    total=sum(counts);raw=len(test)*784
    save_json(args.out/'report.json',{'complete':True,'learning_images':500,'validation_images':1000,
        'test_images':len(test),'raw_positions':raw,'tokens':total,'mean_tokens':total/len(test),
        'remaining_percent':100*total/raw,'positions_saved_percent':100*(raw-total)/raw,
        'embedding_width':192,'dtype':'BF16','per_position_bytes':384,
        'raw_input_bytes':raw*384,'bpe_input_bytes':total*384,
        'learning_rgb_hash_overlap_test':len({sha256(train[i].tobytes()).hexdigest() for i in learn}&test_hashes),
        'archive':receipt,'file_saved_percent':100*(raw-receipt['total_bytes'])/raw,
        'seconds':time.perf_counter()-started,'resource':resource_receipt()})
    print(f'complete: {total} tokens, file {receipt["total_bytes"]} bytes',flush=True)


if __name__=='__main__':main()
