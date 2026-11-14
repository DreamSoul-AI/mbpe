"""E4: real BF16 batches on frozen new RGB splits, no model forward pass."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np
import psutil

import input_batching as batching
import rgb_input_codec as codec
from lossless_nd_candidate import LosslessNDCandidate
from rgb_capacity_study import digest_file, read_json, save_json, torch_cpu, resource_receipt
from rgb_followup_inputs import load_split


SEEDS=(151201,151202,151203)
METHODS=('raw-fifo','bpe-fifo','bpe-w16','bpe-w64','bpe-w128')


def model_for(args):
    path=args.inputs/args.dataset/'difference-dictionary.json'
    expected=read_json(args.inputs/'inputs.json')['datasets'][args.dataset]['dictionaries']['difference']
    if digest_file(path)!=expected:
        raise ValueError('frozen shared dictionary changed')
    return LosslessNDCandidate.from_dictionary(read_json(path))


def prepare(args):
    args.out.mkdir(parents=True,exist_ok=False)
    source=args.out/'source';source.mkdir()
    files={}
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(path,source/path.name);files[path.name]=digest_file(path)
    save_json(args.out/'protocol.json',{'inputs_sha256':digest_file(args.inputs/'inputs.json'),
        'arrival_seeds':SEEDS,'methods':METHODS,'capacity':192,'batch':16,
        'window_selection':'minimum validation mean model-input plus mask and int32 lengths; tie smaller window',
        'cpu_queue_separate':True,'source_sha256':files})


def encode(args):
    if args.split=='e4-test':
        verify_selection(args)
    images,identities=load_split(args.inputs,args.dataset,args.split)
    directory=args.out/args.dataset/args.split
    directory.mkdir(parents=True,exist_ok=False)
    model=model_for(args);torch=torch_cpu()
    streams={mode:(directory/(mode+'.bf16')).open('xb') for mode in ('raw','bpe')}
    rows=[]
    try:
        for i,(image,identity) in enumerate(zip(images,identities,strict=True)):
            row={'identity':identity,'modes':{}}
            for mode in streams:
                blob=image.tobytes() if mode=='raw' else codec.encode_bpe(codec.difference(image),model)[0]
                tensor=codec.to_bf16(blob,192,torch)
                restored=codec.restore_input(tensor,192,image.shape,model,residual=mode=='bpe',bpe=mode=='bpe',torch=torch)
                if not np.array_equal(restored,image):
                    raise ValueError('input restoration failed')
                handle=streams[mode];offset=handle.tell()
                handle.write(tensor.view(torch.uint16).numpy().tobytes())
                row['modes'][mode]={'offset':offset,'positions':len(tensor),'payload_bytes':len(blob)}
            rows.append(row)
            if (i+1)%200==0:
                print(json.dumps({'encoded':args.dataset,'split':args.split,'images':i+1}),flush=True)
    finally:
        for handle in streams.values():handle.close()
    save_json(directory/'encoding.json',{'complete':True,'rows':rows,
        'files':{mode:digest_file(directory/(mode+'.bf16')) for mode in streams},'resource':resource_receipt()})


def load_tensor(handle,row,mode,torch):
    meta=row['modes'][mode]
    handle.seek(meta['offset']);blob=handle.read(meta['positions']*384)
    if len(blob)!=meta['positions']*384:
        raise ValueError('truncated saved input')
    return torch.frombuffer(bytearray(blob),dtype=torch.bfloat16).reshape(meta['positions'],192)


def evaluate(args):
    if args.split=='e4-test':verify_selection(args)
    directory=args.out/args.dataset/args.split
    encoded=read_json(directory/'encoding.json');rows=encoded['rows']
    expected=read_json(args.inputs/'inputs.json')['datasets'][args.dataset]['splits'][args.split]['samples']
    if not encoded['complete'] or [row['identity'] for row in rows]!=expected:
        raise ValueError('encoded identities differ')
    for mode,digest in encoded['files'].items():
        if digest_file(directory/(mode+'.bf16'))!=digest:raise ValueError('saved input changed')
    torch=torch_cpu();model=model_for(args);results={}
    for seed in SEEDS:
        arrival=np.random.Generator(np.random.PCG64(seed)).permutation(len(rows)).tolist()
        methods={}
        for method in METHODS:
            mode='raw' if method=='raw-fifo' else 'bpe'
            window=int(method.rsplit('w',1)[1]) if '-w' in method else 16
            lengths=[int(row['modes'][mode]['positions']) for row in rows]
            records=[];seen=set();peak_queue=0;max_queue=0
            path=directory/f'{seed}-{method}.batches'
            started=time.perf_counter()
            with (directory/(mode+'.bf16')).open('rb') as source,path.open('xb') as sink:
                for queue,groups in batching.window_groups(lengths,arrival,window,sort='-w' in method):
                    held={i:load_tensor(source,rows[i],mode,torch) for i in queue}
                    peak_queue=max(peak_queue,sum(t.numel()*t.element_size() for t in held.values()))
                    max_queue=max(max_queue,len(held))
                    for indices in groups:
                        packet=batching.make_batch([held[i] for i in indices],indices,torch)
                        cost=batching.batch_bytes(*packet[:3]);batching.write_batch(sink,packet)
                        for identity,tensor in batching.unpack_batch(*packet,torch):
                            if identity in seen:raise ValueError('duplicate image in queue')
                            image=codec.restore_input(tensor,192,(3,32,32),model,residual=mode=='bpe',bpe=mode=='bpe',torch=torch)
                            if sha256(image.tobytes()).hexdigest()!=rows[identity]['identity']['sha256']:
                                raise ValueError('batch restoration failed')
                            seen.add(identity)
                        records.append({'identities':indices,**cost})
                    del held
            if seen!=set(range(1000)):raise ValueError('batch identity coverage differs')
            totals={key:sum(record[key] for record in records) for key in ('input_bytes','length_bytes','mask_bytes','allocated_positions')}
            methods[method]={**totals,'model_input_bytes':sum(totals[k] for k in ('input_bytes','length_bytes','mask_bytes')),
                'peak_cpu_queue_payload_bytes':peak_queue,'maximum_queued_images':max_queue,
                'wait_bound_arrivals':window-1,'host_routing_bytes':4000,
                'lossless_images':len(seen),'batches':records,'file':path.name,'sha256':digest_file(path),
                'seconds':time.perf_counter()-started}
            print(json.dumps({'batched':args.dataset,'split':args.split,'seed':seed,'method':method,'bytes':methods[method]['model_input_bytes']}),flush=True)
        if methods['bpe-fifo']['model_input_bytes']!=methods['bpe-w16']['model_input_bytes']:
            raise ValueError('sorting inside existing batches changed allocation')
        results[str(seed)]=methods
    save_json(directory/'batches.json',{'complete':True,'results':results,'resource':resource_receipt()})


def choose(args):
    artifacts={'inputs.json':digest_file(args.inputs/'inputs.json')};choices={}
    for dataset in ('cifar','svhn'):
        path=args.out/dataset/'validation'/'batches.json';data=read_json(path)
        if not data['complete']:raise ValueError('validation incomplete')
        costs={w:sum(data['results'][str(seed)][f'bpe-w{w}']['model_input_bytes'] for seed in SEEDS) for w in (16,64,128)}
        choices[dataset]=min(costs,key=lambda w:(costs[w],w))
        artifacts[f'{dataset}/validation/batches.json']=digest_file(path)
        artifacts[f'{dataset}/difference-dictionary.json']=digest_file(args.inputs/dataset/'difference-dictionary.json')
    save_json(args.out/'selected-before-test.json',{'choices':choices,'artifacts':artifacts,'created_unix':time.time()})


def verify_selection(args):
    selected=read_json(args.out/'selected-before-test.json')
    for name,digest in selected['artifacts'].items():
        base=args.inputs if name=='inputs.json' or name.endswith('dictionary.json') else args.out
        if digest_file(base/name)!=digest:raise ValueError('frozen validation selection changed')
    return selected


def decode(args):
    if args.split=='e4-test':verify_selection(args)
    directory=args.out/args.dataset/args.split
    data=read_json(directory/'batches.json')
    identities=read_json(args.inputs/'inputs.json')['datasets'][args.dataset]['splits'][args.split]['samples']
    torch=torch_cpu();model=model_for(args);checked={}
    for seed,methods in data['results'].items():
        for method,record in methods.items():
            path=directory/record['file']
            if digest_file(path)!=record['sha256']:raise ValueError('batch file changed')
            seen=set();totals={'input_bytes':0,'length_bytes':0,'mask_bytes':0,'allocated_positions':0}
            with path.open('rb') as handle:
                while True:
                    packet=batching.read_batch(handle,torch)
                    if packet is None:break
                    cost=batching.batch_bytes(*packet[:3])
                    for k in totals:totals[k]+=cost[k]
                    for identity,tensor in batching.unpack_batch(*packet,torch):
                        if not 0<=identity<1000 or identity in seen:raise ValueError('bad image identity')
                        bpe=method!='raw-fifo'
                        image=codec.restore_input(tensor,192,(3,32,32),model,residual=bpe,bpe=bpe,torch=torch)
                        if sha256(image.tobytes()).hexdigest()!=identities[identity]['sha256']:
                            raise ValueError('independent batch decode differs')
                        seen.add(identity)
            if seen!=set(range(1000)) or any(totals[k]!=record[k] for k in totals):
                raise ValueError('independent coverage/accounting differs')
            checked[seed+'-'+method]=1000
    save_json(directory/'independent-decode.json',{'checked':checked,'resource':resource_receipt()})


def summarize(args):
    selection=verify_selection(args);result={}
    for dataset in ('cifar','svhn'):
        directory=args.out/dataset/'e4-test';report=read_json(directory/'batches.json')
        checked=read_json(directory/'independent-decode.json')['checked']
        if len(checked)!=15 or set(checked.values())!={1000}:raise ValueError('decode incomplete')
        output={}
        for seed,methods in report['results'].items():
            raw=methods['raw-fifo']['model_input_bytes'];chosen=f"bpe-w{selection['choices'][dataset]}"
            costs={name:{k:row[k] for k in ('input_bytes','length_bytes','mask_bytes','model_input_bytes',
                    'allocated_positions','peak_cpu_queue_payload_bytes','maximum_queued_images','wait_bound_arrivals')}
                   for name,row in methods.items()}
            for name,row in costs.items():row['fewer_input_percent']=100*(raw-row['model_input_bytes'])/raw
            # Common 128-arrival blocks, aligned with all candidate windows.
            def block_totals(name):
                blocks=[0]*8
                arrival=np.random.Generator(np.random.PCG64(int(seed))).permutation(1000).tolist()
                inverse={identity:i//128 for i,identity in enumerate(arrival)}
                for batch in methods[name]['batches']:
                    groups={inverse[i] for i in batch['identities']}
                    if len(groups)!=1:raise ValueError('batch crossed analysis block')
                    blocks[next(iter(groups))]+=batch['input_bytes']+batch['length_bytes']+batch['mask_bytes']
                return np.array(blocks,dtype=np.int64)
            base=block_totals('raw-fifo');candidate=block_totals(chosen)
            indices=np.random.Generator(np.random.PCG64(151301)).integers(0,8,(10000,8))
            ratios=100*(base[indices].sum(axis=1)-candidate[indices].sum(axis=1))/base[indices].sum(axis=1)
            output[seed]={'methods':costs,'chosen':chosen,'gain_block_ci95_percent':np.quantile(ratios,[.025,.975]).tolist(),
                          'analysis_blocks':8,'small_block_count_limit':True}
        result[dataset]=output
    save_json(args.out/'summary.json',{'complete':True,'results':result,'resource':resource_receipt()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','encode','evaluate','choose','decode','summarize'))
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dataset',choices=('cifar','svhn'),default='cifar')
    p.add_argument('--split',choices=('validation','e4-test'),default='validation')
    args=p.parse_args();args.inputs=args.inputs.resolve();args.out=args.out.resolve()
    process=psutil.Process();process.nice(psutil.IDLE_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[:1])
    if psutil.virtual_memory().available<4*1024**3:raise RuntimeError('insufficient memory for bounded CPU study')
    globals()[args.stage](args)


if __name__=='__main__':main()
