"""CPU byte-grid BPE budget curve, with token positions as the primary metric."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUDA_VISIBLE_DEVICES']=''

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import threading

import numpy as np
import psutil

from rgb_capacity_study import digest_file,read_json,save_json,resource_receipt
from rgb_followup_inputs import load_split
import rgb_input_codec as codec

BUDGETS=(256,1024,4096,16384,65536,100000,150000)


def texts(image):
    return [''.join(chr(256+int(v)) for v in channel.T.ravel()) for channel in image]


def expand(ids,words):
    raw=''.join(words[int(i)] for i in ids)
    if len(raw)!=1024 or any(not 256<=ord(c)<512 for c in raw):
        raise ValueError('invalid channel expansion')
    return np.fromiter((ord(c)-256 for c in raw),dtype=np.uint8).reshape(32,32).T


def decode(args):
    data=read_json(args.out/args.dataset/'report.json');results={}
    identities=read_json(args.inputs/'inputs.json')['datasets'][args.dataset]['splits']['e2-test']['samples']
    for budget,row in data['results'].items():
        path=args.out/args.dataset/row['tokenizer'];packet=args.out/args.dataset/row['packet']
        if digest_file(path)!=row['tokenizer_sha256'] or digest_file(packet)!=row['packet_sha256']:
            raise ValueError('frozen output changed')
        vocab=read_json(path)['model']['vocab'];words={i:w for w,i in vocab.items()}
        with np.load(packet,allow_pickle=False) as payload:
            ids=payload['ids'];counts=payload['counts'];offset=0
            for identity,lengths in zip(identities,counts,strict=True):
                channels=[]
                for n in lengths:
                    channels.append(expand(ids[offset:offset+int(n)],words));offset+=int(n)
                image=codec.restore_difference(np.stack(channels))
                if sha256(image.tobytes()).hexdigest()!=identity['sha256']:raise ValueError('independent RGB decode differs')
            if offset!=len(ids):raise ValueError('unused token payload')
        results[budget]=len(identities)
    save_json(args.out/args.dataset/'independent-decode.json',{'checked':results,'resource':resource_receipt()})


def run(args):
    from tokenizers import Tokenizer,models,trainers
    import tokenizers
    directory=args.out/args.dataset;directory.mkdir(exist_ok=False)
    prior=read_json(args.prior/'inputs.json')['datasets'][args.dataset]['splits']['learn']
    learn_path=args.prior/args.dataset/'learn.npz'
    if digest_file(learn_path)!=prior['array_sha256']:raise ValueError('learning arrays changed')
    with np.load(learn_path,allow_pickle=False) as payload:learn=payload['images']
    if any(sha256(im.tobytes()).hexdigest()!=r['sha256'] for im,r in zip(learn,prior['samples'],strict=True)):
        raise ValueError('learning identities changed')
    sequences=[s for image in learn for s in texts(codec.difference(image))]
    tokenizer=Tokenizer(models.BPE())
    trainer=trainers.BpeTrainer(vocab_size=256+max(BUDGETS),min_frequency=1,
        initial_alphabet=[chr(256+i) for i in range(256)],show_progress=False)
    started=time.perf_counter();tokenizer.train_from_iterator(sequences,trainer=trainer)
    tokenizer.save(str(directory/'maximum-tokenizer.json'))
    maximum=read_json(directory/'maximum-tokenizer.json');merges=maximum['model']['merges']
    vocab=maximum['model']['vocab']
    # Freeze all prefix grammars before opening holdout metrics. No normalization,
    # pretokenizer, word splitting, cross-channel or cross-image merges.
    frozen={}
    for budget in BUDGETS:
        actual=min(budget,len(merges));pairs=merges[:actual]
        words={chr(256+i):vocab[chr(256+i)] for i in range(256)}
        for a,b in pairs:words[a+b]=vocab[a+b]
        model=Tokenizer(models.BPE(vocab=words,merges=[tuple(p) for p in pairs]))
        path=directory/f'rules-{budget}.json';model.save(str(path))
        frozen[str(budget)]={'requested_rules':budget,'actual_rules':actual,'vocabulary_entries':len(words),
            'tokenizer':path.name,'tokenizer_sha256':digest_file(path)}
    save_json(directory/'sealed-before-test.json',{'budgets':frozen,'learn_sha256':prior['array_sha256'],
        'inputs_sha256':digest_file(args.inputs/'inputs.json'),'scan':'column','transform':'left-difference',
        'min_frequency':1,'tokenizers':tokenizers.__version__,'training_seconds':time.perf_counter()-started})
    images,identities=load_split(args.inputs,args.dataset,'e2-test')
    results={}
    for budget,row in frozen.items():
        model=Tokenizer.from_file(str(directory/row['tokenizer']));v=model.get_vocab();inv={i:w for w,i in v.items()}
        totals=[];counts=[];flat=[]
        for image,identity in zip(images,identities,strict=True):
            channels=[];ns=[]
            for string in texts(codec.difference(image)):
                ids=model.encode(string,add_special_tokens=False).ids
                channels.append(expand(ids,inv));ns.append(len(ids));flat.extend(ids)
            restored=codec.restore_difference(np.stack(channels))
            if sha256(restored.tobytes()).hexdigest()!=identity['sha256']:raise ValueError('RGB roundtrip differs')
            counts.append(ns);totals.append(sum(ns))
        packet=directory/f'rules-{budget}.npz'
        with packet.open('xb') as handle:np.savez(handle,ids=np.asarray(flat,dtype=np.uint32),counts=np.asarray(counts,dtype=np.uint32))
        total=int(sum(totals));n=len(images);results[budget]={**row,'packet':packet.name,
            'packet_sha256':digest_file(packet),'tokens':total,'per_image_tokens':totals,
            'mean_tokens':total/n,'remaining_percent':100*total/(n*3072),
            'positions_saved_percent':100*(1-total/(n*3072)),'images_at_least_half':sum(t<=1536 for t in totals),
            'mean_input_bytes':384*total/n,'embedding_table_bf16_reference_bytes':len(v)*384,'lossless_images':n}
        print(json.dumps({'dataset':args.dataset,'budget':budget,'actual':row['actual_rules'],
            'mean_tokens':total/n,'saved_percent':results[budget]['positions_saved_percent'],
            'half_images':results[budget]['images_at_least_half']}),flush=True)
        save_json(directory/f'checkpoint-{budget}.json',results[budget])
        if psutil.virtual_memory().available<4*1024**3:raise RuntimeError('free memory below reserve')
    save_json(directory/'report.json',{'complete':True,'results':results,'resource':resource_receipt()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('run','decode'))
    for key in ('out','inputs','prior'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--dataset',choices=('cifar','svhn'),required=True);args=p.parse_args()
    for key in ('out','inputs','prior'):setattr(args,key,getattr(args,key).resolve())
    process=psutil.Process();process.nice(psutil.IDLE_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[:1])
    if psutil.virtual_memory().available<4*1024**3:raise RuntimeError('insufficient free memory')
    def watch():
        while True:
            if process.memory_info().rss>8*1024**3 or psutil.virtual_memory().available<4*1024**3:
                (args.out/(args.dataset+'-resource-failure.json')).write_text(
                    json.dumps({'rss':process.memory_info().rss,'available':psutil.virtual_memory().available}),encoding='utf-8')
                os._exit(75)
            time.sleep(1)
    threading.Thread(target=watch,daemon=True).start()
    globals()[args.stage](args)


if __name__=='__main__':main()
