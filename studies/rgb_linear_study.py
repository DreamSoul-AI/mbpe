"""E2 fixed-scan RGB controls; validation selects once, E2 holdout stays separate."""

import argparse
from hashlib import sha256
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil

from rgb_capacity_study import digest_file,read_json,save_json,resource_receipt,torch_cpu
from rgb_followup_inputs import load_split
from rgb_linear_codec import LinearRGBGrammar,SCANS,learn_rgb
import rgb_input_codec as codec
from lossless_nd_candidate import LosslessNDCandidate


def prepare(args):
    args.out.mkdir(parents=True,exist_ok=False)
    source=args.out/'source';source.mkdir()
    sources={}
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(path,source/path.name);sources[path.name]=digest_file(path)
    save_json(args.out/'protocol.json',{'inputs_sha256':digest_file(args.inputs/'inputs.json'),
        'learn_inputs_sha256':digest_file(args.prior/'inputs.json'),'scans':SCANS,
        'rules':256,'width':192,'bits':9,'capacity':192,'source_sha256':sources,
        'comparison':'per-image input positions; batch mechanism comparisons use common FIFO only'})


def fit(args):
    meta=read_json(args.prior/'inputs.json')['datasets'][args.dataset]['splits']['learn']
    path=args.prior/args.dataset/'learn.npz'
    if digest_file(path)!=meta['array_sha256']:raise ValueError('E1 learning arrays changed')
    with np.load(path,allow_pickle=False) as data:images=data['images']
    if any(sha256(im.tobytes()).hexdigest()!=row['sha256'] for im,row in zip(images,meta['samples'],strict=True)):
        raise ValueError('learning identities differ')
    if args.transform=='difference':images=np.stack([codec.difference(im) for im in images])
    model=learn_rgb(images,args.scan)
    if len(model.rules)!=256:raise ValueError('fixed rule budget not reached')
    directory=args.out/args.dataset;directory.mkdir(exist_ok=True)
    path=directory/f'{args.transform}-{args.scan}.json'
    save_json(path,model.dictionary())
    save_json(path.with_suffix('.fit.json'),{'images':1000,'channels':3000,
        'learning_split_sha256':meta['array_sha256'],'dictionary_sha256':digest_file(path),'resource':resource_receipt()})
    print(f'fit {args.dataset} {args.transform} {args.scan}',flush=True)


def model_for(args,transform,mode):
    if mode=='nd':
        path=args.inputs/args.dataset/(transform+'-dictionary.json')
        expected=read_json(args.inputs/'inputs.json')['datasets'][args.dataset]['dictionaries'][transform]
        if digest_file(path)!=expected:raise ValueError('E1 ND dictionary changed')
        return LosslessNDCandidate.from_dictionary(read_json(path))
    path=args.out/args.dataset/f'{transform}-{mode}.json'
    receipt=read_json(path.with_suffix('.fit.json'))
    if digest_file(path)!=receipt['dictionary_sha256']:raise ValueError('linear dictionary changed')
    return LinearRGBGrammar.from_dictionary(read_json(path))


def evaluate(args):
    if args.split=='e2-test':
        verify_selection(args)
    images,identities=load_split(args.inputs,args.dataset,args.split)
    directory=args.out/args.dataset/args.split;directory.mkdir(exist_ok=False)
    torch=torch_cpu();results={}
    selected=read_json(args.out/'selected-before-test.json')['choices'][args.dataset] if args.split=='e2-test' else None
    for transform in ('raw','difference'):
        modes=(selected[transform],'nd') if selected else (*SCANS,'nd')
        for mode in modes:
            model=model_for(args,transform,mode);records=[]
            path=directory/f'{transform}-{mode}.bf16'
            with path.open('xb') as handle:
                for i,(image,identity) in enumerate(zip(images,identities,strict=True)):
                    transformed=codec.difference(image) if transform=='difference' else image
                    blob,counts=codec.encode_bpe(transformed,model)
                    tensor=codec.to_bf16(blob,192,torch)
                    restored=codec.restore_input(tensor,192,(3,32,32),model,residual=transform=='difference',bpe=True,torch=torch)
                    if sha256(restored.tobytes()).hexdigest()!=identity['sha256']:
                        raise ValueError('linear/ND BF16 restoration differs')
                    import struct
                    handle.write(struct.pack('<I',len(tensor)));handle.write(tensor.view(torch.uint16).numpy().tobytes())
                    records.append({'identity':identity,'tokens':sum(counts),'payload_bytes':len(blob),'positions':len(tensor)})
            results[transform+'-'+mode]={'records':records,'sha256':digest_file(path),'lossless_images':1000}
            print(f'evaluated {args.dataset} {args.split} {transform} {mode}',flush=True)
    save_json(directory/'report.json',{'complete':True,'results':results,'resource':resource_receipt()})


def choose(args):
    choices={};artifacts={'inputs.json':digest_file(args.inputs/'inputs.json')}
    for dataset in ('cifar','svhn'):
        path=args.out/dataset/'validation/report.json';report=read_json(path)
        if not report['complete']:raise ValueError('validation incomplete')
        choices[dataset]={transform:min(SCANS,key=lambda mode:(
            sum(row['positions'] for row in report['results'][transform+'-'+mode]['records']),SCANS.index(mode)))
            for transform in ('raw','difference')}
        artifacts[f'{dataset}/validation/report.json']=digest_file(path)
        for transform in ('raw','difference'):
            for scan in SCANS:
                name=f'{dataset}/{transform}-{scan}.json';artifacts[name]=digest_file(args.out/name)
    save_json(args.out/'selected-before-test.json',{'choices':choices,'artifacts':artifacts,'created_unix':time.time()})


def verify_selection(args):
    selected=read_json(args.out/'selected-before-test.json')
    for name,digest in selected['artifacts'].items():
        base=args.inputs if name=='inputs.json' else args.out
        if digest_file(base/name)!=digest:raise ValueError('E2 validation choice changed')
    return selected


def decode(args):
    import struct
    selection=verify_selection(args)
    directory=args.out/args.dataset/'e2-test';report=read_json(directory/'report.json')
    identities=read_json(args.inputs/'inputs.json')['datasets'][args.dataset]['splits']['e2-test']['samples']
    torch=torch_cpu();checked={}
    for transform in ('raw','difference'):
        for mode in (selection['choices'][args.dataset][transform],'nd'):
            key=transform+'-'+mode;result=report['results'][key]
            if [r['identity'] for r in result['records']]!=identities:raise ValueError('test identities differ')
            model=model_for(args,transform,mode);path=directory/(key+'.bf16')
            if digest_file(path)!=result['sha256']:raise ValueError('test input file changed')
            with path.open('rb') as handle:
                for identity in identities:
                    header=handle.read(4)
                    if len(header)!=4:raise ValueError('truncated header')
                    length,=struct.unpack('<I',header)
                    if not 1<=length<=19:raise ValueError('bad RGB vector count')
                    data=handle.read(length*384)
                    if len(data)!=length*384:raise ValueError('truncated input tensor')
                    tensor=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(length,192)
                    image=codec.restore_input(tensor,192,(3,32,32),model,residual=transform=='difference',bpe=True,torch=torch)
                    if sha256(image.tobytes()).hexdigest()!=identity['sha256']:raise ValueError('independent decode differs')
                if handle.read(1):raise ValueError('trailing records')
            checked[key]=1000
    save_json(directory/'independent-decode.json',{'checked':checked,'resource':resource_receipt()})


def summarize(args):
    selection=verify_selection(args);results={}
    for dataset in ('cifar','svhn'):
        directory=args.out/dataset/'e2-test';report=read_json(directory/'report.json')
        checked=read_json(directory/'independent-decode.json')['checked']
        if len(checked)!=4 or set(checked.values())!={1000}:raise ValueError('independent decode incomplete')
        result={}
        for transform in ('raw','difference'):
            modes={}
            for mode in (selection['choices'][dataset][transform],'nd'):
                rows=report['results'][transform+'-'+mode]['records']
                positions=np.array([row['positions'] for row in rows],dtype=np.int64)
                arrival=np.random.Generator(np.random.PCG64(151201)).permutation(1000)
                ordered=positions[arrival]
                modes[mode]={'mean_tokens':float(np.mean([row['tokens'] for row in rows])),
                    'total_positions':int(positions.sum()),'mean_positions':float(positions.mean()),
                    'mean_input_bytes':float(positions.mean()*384),'fewer_vs_fixed_percent':float(100*(16000-positions.sum())/16000),
                    'fifo16_arrival_seed':151201,
                    'fifo16_allocated_positions':sum(len(ordered[i:i+16])*int(ordered[i:i+16].max()) for i in range(0,1000,16))}
            scan=selection['choices'][dataset][transform]
            one=np.array([row['positions'] for row in report['results'][transform+'-'+scan]['records']])
            nd=np.array([row['positions'] for row in report['results'][transform+'-nd']['records']])
            gain=one-nd;rng=np.random.Generator(np.random.PCG64(151302))
            samples=np.concatenate([gain[rng.integers(0,1000,(100,1000))].mean(axis=1) for _ in range(100)])
            result[transform]={'selected_scan':scan,'methods':modes,'nd_mean_gain_positions':float(gain.mean()),
                'nd_gain_ci95':np.quantile(samples,[.025,.975]).tolist(),
                'nd_win_tie_loss':[int((gain>0).sum()),int((gain==0).sum()),int((gain<0).sum())]}
        results[dataset]=result
    save_json(args.out/'summary.json',{'complete':True,'results':results,'resource':resource_receipt()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','fit','evaluate','choose','decode','summarize'))
    for name in ('inputs','prior','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--dataset',choices=('cifar','svhn'),default='cifar')
    p.add_argument('--transform',choices=('raw','difference'),default='raw')
    p.add_argument('--scan',choices=SCANS,default='row')
    p.add_argument('--split',choices=('validation','e2-test'),default='validation')
    args=p.parse_args()
    for name in ('inputs','prior','out'):setattr(args,name,getattr(args,name).resolve())
    process=psutil.Process();process.nice(psutil.IDLE_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[:1])
    if psutil.virtual_memory().available<4*1024**3:raise RuntimeError('insufficient memory')
    globals()[args.stage](args)


if __name__=='__main__':main()
