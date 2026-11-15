"""Frozen CPU comparison of color-coordinate BPE and explicit multiscale templates."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import shutil
import sys
import threading
import time
import numpy as np
import psutil
from rgb_capacity_study import read_json,save_json,digest_file,source_images,select_indices,resource_receipt
from rgb_large_vocab_study import texts,expand
import rgb_input_codec as codec
from color_template_codec import TemplateCodec,fit_templates,color_basis,inverse_color_basis

ROOT=Path(__file__).resolve().parents[1]


def freeze(args):
    args.out.mkdir(exist_ok=False,parents=True);source=args.out/'source';source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,source/p.name)
    shared=read_json(args.shared/'inputs.json');prior=read_json(args.prior/'inputs.json')
    manifest={'seed':151511,'datasets':{},'shared_sha256':digest_file(args.shared/'inputs.json'),
        'prior_sha256':digest_file(args.prior/'inputs.json'),'source_sha256':{p.name:digest_file(p) for p in source.glob('*.py')}}
    for dataset,meta in shared['datasets'].items():
        directory=args.out/dataset;directory.mkdir()
        seen=set()
        for part in prior['datasets'][dataset]['splits'].values():seen.update(r['sha256'] for r in part['samples'])
        for part in meta['splits'].values():seen.update(r['sha256'] for r in part['samples'])
        for name,indices in meta['excluded_indices'].items():
            part=next(p for p in prior['datasets'][dataset]['splits'].values() if p['source']==name)
            images,labels=source_images(dataset,args.backup/name,part['source_sha256'])
            seen.update(sha256(images[i].tobytes()).hexdigest() for i in indices)
            del images,labels
        previous=meta['splits']['e2-test'];name=previous['source']
        images,labels=source_images(dataset,args.backup/name,previous['source_sha256'])
        excluded=set(meta['excluded_indices'][name])
        for part in meta['splits'].values():
            if part['source']==name:excluded.update(r['index'] for r in part['samples'])
        rows=select_indices(labels,images,1000,151511,excluded,seen)
        path=directory/'test.npz'
        with path.open('xb') as handle:np.savez(handle,images=np.stack([images[r['index']] for r in rows]))
        manifest['datasets'][dataset]={'source':name,'source_sha256':previous['source_sha256'],
            'samples':rows,'excluded_indices':sorted(excluded),'array_sha256':digest_file(path)}
        del images,labels
        for budget in (65536,150000):
            old=args.baseline/dataset/f'rules-{budget}.json'
            expected=read_json(args.baseline/dataset/'sealed-before-test.json')['budgets'][str(budget)]['tokenizer_sha256']
            if digest_file(old)!=expected:raise ValueError('old baseline changed')
            shutil.copyfile(old,directory/f'baseline-{budget}.json')
    save_json(args.out/'inputs.json',manifest)
    print('fresh holdouts frozen',flush=True)


def arrays(args,split):
    if split=='learn':
        meta=read_json(args.prior/'inputs.json')['datasets'][args.dataset]['splits']['learn']
        path=args.prior/args.dataset/'learn.npz';rows=meta['samples']
    elif split=='validation':
        meta=read_json(args.shared/'inputs.json')['datasets'][args.dataset]['splits']['validation']
        path=args.shared/args.dataset/'validation.npz';rows=meta['samples']
    else:
        meta=read_json(args.out/'inputs.json')['datasets'][args.dataset]
        path=args.out/args.dataset/'test.npz';rows=meta['samples']
    if digest_file(path)!=meta['array_sha256']:raise ValueError('input arrays changed')
    with np.load(path,allow_pickle=False) as payload:images=payload['images']
    if images.shape!=(1000,3,32,32) or images.dtype!=np.uint8:raise ValueError('bad fixed RGB split')
    if any(sha256(im.tobytes()).hexdigest()!=r['sha256'] for im,r in zip(images,rows,strict=True)):
        raise ValueError('image identity differs')
    return images,rows


def fit_color(args):
    from tokenizers import Tokenizer,models,trainers
    images,_=arrays(args,'learn');directory=args.out/args.dataset
    strings=[s for image in images for s in texts(codec.difference(color_basis(image)))]
    tokenizer=Tokenizer(models.BPE());tokenizer.train_from_iterator(strings,trainer=trainers.BpeTrainer(
        vocab_size=150256,min_frequency=1,initial_alphabet=[chr(256+i) for i in range(256)],show_progress=False))
    data=__import__('json').loads(tokenizer.to_str());full=data['model'];entries={}
    for budget in (65536,150000):
        pairs=full['merges'][:budget];v={chr(256+i):full['vocab'][chr(256+i)] for i in range(256)}
        for a,b in pairs:v[a+b]=full['vocab'][a+b]
        saved=Tokenizer(models.BPE(vocab=v,merges=[tuple(p) for p in pairs]));path=directory/f'color-{budget}.json';saved.save(str(path))
        entries[str(budget)]={'actual_rules':len(pairs),'vocabulary':len(v),'sha256':digest_file(path)}
    save_json(directory/'color-fit.json',{'entries':entries,'resource':resource_receipt()})


def fit_template(args):
    images,_=arrays(args,'learn');directory=args.out/args.dataset
    if args.transform=='difference':images=np.stack([codec.difference(im) for im in images])
    model,stats=fit_templates(images,args.relative,150000)
    key=f'{args.transform}-'+('relative' if args.relative else 'absolute')
    path=directory/(key+'.json');save_json(path,model.dictionary())
    save_json(directory/(key+'-fit.json'),{'stats':stats,'sha256':digest_file(path),'resource':resource_receipt()})
    print(key,stats,flush=True)


def seal(args):
    entries={}
    for dataset in ('cifar','svhn'):
        directory=args.out/dataset
        for path in directory.glob('*.json'):
            if path.name.endswith('-fit.json') or path.name.startswith(('baseline-','color-','raw-','difference-')):
                entries[str(path.relative_to(args.out))]=digest_file(path)
    save_json(args.out/'sealed-models.json',{'inputs_sha256':digest_file(args.out/'inputs.json'),'artifacts':entries})


def verify(args):
    seal=read_json(args.out/'sealed-models.json')
    if digest_file(args.out/'inputs.json')!=seal['inputs_sha256']:raise ValueError('input manifest changed')
    for p,h in seal['artifacts'].items():
        if digest_file(args.out/p)!=h:raise ValueError('frozen dictionary changed')


def evaluation(args):
    from tokenizers import Tokenizer
    verify(args);images,identities=arrays(args,args.split)
    directory=args.out/args.dataset/args.split;directory.mkdir(exist_ok=False);results={}
    methods=[]
    for kind in ('baseline','color'):
        for budget in (65536,150000):methods.append((f'{kind}-{budget}',kind,budget,None))
    for transform in ('raw','difference'):
        for relative in (False,True):
            key=f'{transform}-'+('relative' if relative else 'absolute')
            for budget in (16384,150000):methods.append((f'{key}-{budget}',key,budget,transform))
    for name,kind,budget,transform in methods:
        if transform is None:
            model=Tokenizer.from_file(str(args.out/args.dataset/f'{kind}-{budget}.json'));words={i:w for w,i in model.get_vocab().items()}
        else:model=TemplateCodec.from_dictionary(read_json(args.out/args.dataset/(kind+'.json'))).prefix(budget)
        counts=[];flat=[];totals=[];aggregate=Counter()
        for image,identity in zip(images,identities,strict=True):
            if transform is None:
                prepared=codec.difference(color_basis(image) if kind=='color' else image);ns=[];grids=[];ids=[]
                for text in texts(prepared):
                    channel=model.encode(text,add_special_tokens=False).ids;ids.extend(channel);ns.append(len(channel));grids.append(expand(channel,words))
                restored=codec.restore_difference(np.stack(grids))
                if kind=='color':restored=inverse_color_basis(restored)
                counts.append(ns)
            else:
                prepared=codec.difference(image) if transform=='difference' else image
                ids,stats=model.encode(prepared);restored=model.decode(ids);aggregate.update(stats);counts.append([len(ids)])
                if transform=='difference':restored=codec.restore_difference(restored)
            if sha256(restored.tobytes()).hexdigest()!=identity['sha256']:raise ValueError('pixel roundtrip differs')
            totals.append(len(ids));flat.extend(ids)
        path=directory/(name+'.npz')
        with path.open('xb') as handle:np.savez(handle,ids=np.asarray(flat,dtype=np.uint32),counts=np.asarray(counts,dtype=np.uint32))
        total=sum(totals);results[name]={'tokens':total,'per_image_tokens':totals,'mean_tokens':total/1000,
            'saved_percent':100*(1-total/3072000),'half_images':sum(t<=1536 for t in totals),
            'packet_sha256':digest_file(path),'stats':dict(aggregate),'lossless_images':1000}
        save_json(directory/(name+'.json'),results[name]);print(args.dataset,args.split,name,total/1000,flush=True)
    save_json(directory/'report.json',{'complete':True,'results':results,'resource':resource_receipt()})


def decode(args):
    verify(args);from tokenizers import Tokenizer
    directory=args.out/args.dataset/args.split;report=read_json(directory/'report.json')
    if args.split=='test':identities=read_json(args.out/'inputs.json')['datasets'][args.dataset]['samples']
    else:identities=read_json(args.shared/'inputs.json')['datasets'][args.dataset]['splits']['validation']['samples']
    checked={}
    for name,row in report['results'].items():
        path=directory/(name+'.npz')
        if digest_file(path)!=row['packet_sha256']:raise ValueError('token payload changed')
        kind,budget=name.rsplit('-',1);is_bpe=kind in ('baseline','color')
        if is_bpe:
            data=read_json(args.out/args.dataset/(name+'.json'))['model'];words={i:w for w,i in data['vocab'].items()}
        else:model=TemplateCodec.from_dictionary(read_json(args.out/args.dataset/(kind+'.json'))).prefix(int(budget))
        with np.load(path,allow_pickle=False) as payload:
            values=payload['ids'];counts=payload['counts'];offset=0
            for identity,ns in zip(identities,counts,strict=True):
                if is_bpe:
                    grids=[]
                    for n in ns:grids.append(expand(values[offset:offset+int(n)],words));offset+=int(n)
                    image=codec.restore_difference(np.stack(grids))
                    if kind=='color':image=inverse_color_basis(image)
                else:
                    n=int(ns[0]);image=model.decode(values[offset:offset+n].astype(int).tolist());offset+=n
                    if kind.startswith('difference'):image=codec.restore_difference(image)
                if sha256(image.tobytes()).hexdigest()!=identity['sha256']:raise ValueError('independent decode differs')
            if offset!=len(values):raise ValueError('extra token payload')
        checked[name]=len(identities)
    save_json(directory/'independent-decode.json',{'checked':checked,'resource':resource_receipt()})


def color_control(args):
    """Exploratory validation intervention; keep each channel pool unchanged."""
    from tokenizers import Tokenizer,models,trainers
    verify(args);directory=args.out/'controls'/args.dataset;directory.mkdir(parents=True,exist_ok=False)
    def shuffle(images,seed):
        rng=np.random.Generator(np.random.PCG64(seed));n=len(images);permutations=[]
        for _ in range(2):
            while True:
                order=rng.permutation(n)
                if np.all(order!=np.arange(n)):break
            permutations.append(order)
        result=images.copy();result[:,0]=images[permutations[0],0];result[:,2]=images[permutations[1],2]
        # Same training strings for a channel-wise shared baseline; only RGB coupling changes.
        assert sorted(s for im in images for s in texts(codec.difference(im)))==sorted(s for im in result for s in texts(codec.difference(im)))
        return result,[order.tolist() for order in permutations]
    learn,_=arrays(args,'learn');shuffled,learn_order=shuffle(learn,151512)
    tokenizer=Tokenizer(models.BPE());tokenizer.train_from_iterator(
        (s for im in shuffled for s in texts(codec.difference(color_basis(im)))),
        trainer=trainers.BpeTrainer(vocab_size=150256,min_frequency=1,initial_alphabet=[chr(256+i) for i in range(256)],show_progress=False))
    tokenizer.save(str(directory/'color-tokenizer.json'))
    validation,_=arrays(args,'validation');validation,val_order=shuffle(validation,151513)
    save_json(directory/'permutations.json',{'learn_R_B':learn_order,'validation_R_B':val_order,
        'interpretation':'synthetic channel pairing, validation only; per-channel samples unchanged'})
    results={}
    for kind in ('baseline','color'):
        model=(Tokenizer.from_file(str(args.out/args.dataset/'baseline-150000.json')) if kind=='baseline' else tokenizer)
        words={i:w for w,i in model.get_vocab().items()};totals=[];zero_count=0
        for image in validation:
            prepared=codec.difference(color_basis(image) if kind=='color' else image);zero_count+=int((prepared==0).sum())
            grids=[];total=0
            for text in texts(prepared):
                ids=model.encode(text,add_special_tokens=False).ids;total+=len(ids);grids.append(expand(ids,words))
            restored=codec.restore_difference(np.stack(grids))
            if kind=='color':restored=inverse_color_basis(restored)
            if not np.array_equal(restored,image):raise ValueError('synthetic control inverse differs')
            totals.append(total)
        results[kind]={'mean_tokens':sum(totals)/1000,'per_image_tokens':totals,'zero_fraction':zero_count/3072000,'lossless_images':1000}
    original=read_json(args.out/args.dataset/'validation/report.json')['results']
    if sum(results['baseline']['per_image_tokens'])!=original['baseline-150000']['tokens']:
        raise ValueError('baseline did not preserve channel-pool length')
    save_json(directory/'report.json',{'results':results,'resource':resource_receipt(),
        'original_validation_gain_percent':100*(1-original['color-150000']['tokens']/original['baseline-150000']['tokens']),
        'decorrelated_validation_gain_percent':100*(1-results['color']['mean_tokens']/results['baseline']['mean_tokens']),
        'channel_marginal_preserved':True,'test_split_used':False})


def parsing_control(args):
    """Uniform-cost Viterbi on the exact same BPE vocabulary, validation only."""
    from tokenizers import Tokenizer,models
    verify(args);directory=args.out/'parsing'/args.dataset;directory.mkdir(parents=True,exist_ok=False)
    images,identities=arrays(args,'validation');original=read_json(args.out/args.dataset/'validation/report.json')['results'];results={}
    for kind in ('baseline','color'):
        source=args.out/args.dataset/f'{kind}-150000.json';v=read_json(source)['model']['vocab']
        words={i:w for w,i in v.items()};ordered=sorted(v,key=v.get)
        model=Tokenizer(models.Unigram([(word,-1.0) for word in ordered]))
        if model.get_vocab()!=v:raise ValueError('parser changed vocabulary IDs')
        totals=[];flat=[];counts=[]
        for image,identity,previous in zip(images,identities,original[f'{kind}-150000']['per_image_tokens'],strict=True):
            prepared=codec.difference(color_basis(image) if kind=='color' else image);ns=[];grids=[]
            for text in texts(prepared):
                ids=model.encode(text,add_special_tokens=False).ids;flat.extend(ids);ns.append(len(ids));grids.append(expand(ids,words))
            restored=codec.restore_difference(np.stack(grids))
            if kind=='color':restored=inverse_color_basis(restored)
            if sha256(restored.tobytes()).hexdigest()!=identity['sha256'] or sum(ns)>previous:
                raise ValueError('minimum-parse or inverse contract failed')
            totals.append(sum(ns));counts.append(ns)
        path=directory/(kind+'.npz')
        with path.open('xb') as f:np.savez(f,ids=np.asarray(flat,dtype=np.uint32),counts=np.asarray(counts,dtype=np.uint32))
        results[kind]={'mean_tokens':sum(totals)/1000,'per_image_tokens':totals,'lossless_images':1000,
            'source_vocabulary_sha256':digest_file(source),'packet_sha256':digest_file(path),
            'original_bpe_mean':original[f'{kind}-150000']['mean_tokens'],'same_vocabulary_ids':True}
    save_json(directory/'report.json',{'results':results,'resource':resource_receipt(),'scope':'exploratory validation, no learned probabilities or vocabulary changes'})


def parsing_decode(args):
    verify(args);directory=args.out/'parsing'/args.dataset;report=read_json(directory/'report.json');checked={}
    identities=read_json(args.shared/'inputs.json')['datasets'][args.dataset]['splits']['validation']['samples']
    for kind,row in report['results'].items():
        source=args.out/args.dataset/f'{kind}-150000.json';path=directory/(kind+'.npz')
        if digest_file(source)!=row['source_vocabulary_sha256'] or digest_file(path)!=row['packet_sha256']:
            raise ValueError('parser artifact changed')
        words={i:w for w,i in read_json(source)['model']['vocab'].items()}
        with np.load(path,allow_pickle=False) as payload:
            ids=payload['ids'];counts=payload['counts'];offset=0
            for identity,ns in zip(identities,counts,strict=True):
                grids=[]
                for n in ns:grids.append(expand(ids[offset:offset+int(n)],words));offset+=int(n)
                image=codec.restore_difference(np.stack(grids))
                if kind=='color':image=inverse_color_basis(image)
                if sha256(image.tobytes()).hexdigest()!=identity['sha256']:raise ValueError('independent parse decode failed')
            if offset!=len(ids):raise ValueError('extra parse tokens')
        checked[kind]=len(identities)
    save_json(directory/'independent-decode.json',{'checked':checked,'resource':resource_receipt()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','fit_color','fit_template','seal','evaluation','decode','color_control','parsing_control','parsing_decode'))
    for key in ('out','shared','prior','baseline','backup'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--dataset',choices=('cifar','svhn'),default='cifar');p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--transform',choices=('raw','difference'),default='raw');p.add_argument('--relative',action='store_true');args=p.parse_args()
    for key in ('out','shared','prior','baseline','backup'):setattr(args,key,getattr(args,key).resolve())
    process=psutil.Process();process.nice(psutil.IDLE_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[:1])
    if psutil.virtual_memory().available<4*1024**3:raise RuntimeError('insufficient free memory')
    def monitor():
        while True:
            if process.memory_info().rss>8*1024**3 or psutil.virtual_memory().available<4*1024**3:
                args.out.mkdir(exist_ok=True);(args.out/f'resource-failure-{os.getpid()}.json').write_text(str(resource_receipt()),encoding='utf-8');os._exit(75)
            time.sleep(1)
    threading.Thread(target=monitor,daemon=True).start();globals()[args.stage](args)


if __name__=='__main__':main()
