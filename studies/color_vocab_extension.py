"""CPU budget extension on frozen RGB learning and previously exposed holdout."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[name]='1'
os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false')
import argparse
from hashlib import sha256
from pathlib import Path
import sys
import threading
import time
import numpy as np
import psutil
from rgb_capacity_study import read_json,save_json,digest_file,resource_receipt
from rgb_large_vocab_study import texts,expand
from color_template_codec import color_basis,inverse_color_basis
from rgb_input_codec import difference,restore_difference
from color_template_codec import validate


def prepare_image(image,kind):
    validate(image)
    if kind=='raw':return image
    if kind=='baseline':return difference(image)
    if kind=='color':return difference(color_basis(image))
    raise ValueError('unknown representation')


def restore_image(grids,kind):
    image=np.stack(grids)
    if kind=='raw':
        validate(image);return image
    if kind=='baseline':return restore_difference(image)
    if kind=='color':return inverse_color_basis(restore_difference(image))
    raise ValueError('unknown representation')


def train(args):
    from tokenizers import Tokenizer,models,trainers
    args.out.mkdir(exist_ok=False,parents=True)
    prior=read_json(args.prior/'inputs.json');sealed={}
    for dataset in ('cifar','svhn'):
        directory=args.out/dataset;directory.mkdir()
        source=args.prior/dataset/'learn.npz'
        assert digest_file(source)==prior['datasets'][dataset]['splits']['learn']['array_sha256']
        with np.load(source,allow_pickle=False) as packet:images=packet['images']
        for kind in (('raw',) if args.raw_only else ('baseline','color')):
            strings=[s for im in images for s in texts(prepare_image(im,kind))]
            model=Tokenizer(models.BPE());model.train_from_iterator(strings,trainer=trainers.BpeTrainer(
                vocab_size=200256,min_frequency=1,initial_alphabet=[chr(256+i) for i in range(256)],show_progress=False))
            full=__import__('json').loads(model.to_str())['model']
            previous=None if kind=='raw' else read_json(args.previous/dataset/(kind+'-150000.json'))['model']
            for budget in (150000,200000):
                pairs=full['merges'][:budget];v={chr(256+i):full['vocab'][chr(256+i)] for i in range(256)}
                for a,b in pairs:v[a+b]=full['vocab'][a+b]
                current=Tokenizer(models.BPE(vocab=v,merges=[tuple(p) for p in pairs]))
                path=directory/f'{kind}-{budget}.json';current.save(str(path))
                sealed[str(path.relative_to(args.out))]={'sha256':digest_file(path),'actual_rules':len(pairs),'vocabulary':len(v),
                    'old_150000_prefix_identical':None if previous is None else (full['merges'][:150000]==previous['merges'] and
                        {w:i for w,i in full['vocab'].items() if i<150256}==previous['vocab'])}
            print('trained',dataset,kind,len(full['merges']),flush=True)
    save_json(args.out/'sealed.json',{'models':sealed,'learn_manifest_sha256':digest_file(args.prior/'inputs.json'),
        'holdout_manifest_sha256':digest_file(args.holdout/'inputs.json'),'raw_only':args.raw_only,'test_already_exposed':True,'resource':resource_receipt()})


def evaluate(args):
    from tokenizers import Tokenizer,models
    seal=read_json(args.out/'sealed.json');manifest=read_json(args.holdout/'inputs.json')
    if seal.get('raw_only',False)!=args.raw_only:raise ValueError('representation differs from sealed run')
    assert digest_file(args.holdout/'inputs.json')==seal['holdout_manifest_sha256']
    for dataset in ('cifar','svhn'):
        imagefile=args.holdout/dataset/'test.npz';meta=manifest['datasets'][dataset]
        assert digest_file(imagefile)==meta['array_sha256']
        with np.load(imagefile,allow_pickle=False) as packet:images=packet['images']
        result={}
        for kind in (('raw',) if args.raw_only else ('baseline','color')):
            for budget in (150000,200000):
                dictionary=args.out/dataset/f'{kind}-{budget}.json';entry=seal['models'][str(dictionary.relative_to(args.out))]
                assert digest_file(dictionary)==entry['sha256']
                original=Tokenizer.from_file(str(dictionary));v=original.get_vocab();words={i:w for w,i in v.items()}
                parsers=[('bpe',original)]
                if not args.raw_only:
                    minimum=Tokenizer(models.Unigram([(words[i],-1.0) for i in range(len(words))]));assert minimum.get_vocab()==v
                    parsers.append(('minimum',minimum))
                for parser,model in parsers:
                    lengths=[];flat=[]
                    for image,row in zip(images,meta['samples'],strict=True):
                        grids=[];count=0
                        for string in texts(prepare_image(image,kind)):
                            ids=model.encode(string,add_special_tokens=False).ids
                            flat.extend(ids);count+=len(ids);grids.append(expand(ids,words))
                        recovered=restore_image(grids,kind)
                        assert sha256(recovered.tobytes()).hexdigest()==row['sha256'];lengths.append(count)
                    name=f'{kind}-{budget}-{parser}';path=args.out/dataset/(name+'.npz')
                    with path.open('xb') as handle:np.savez(handle,ids=np.asarray(flat,dtype=np.uint32))
                    result[name]={**entry,'dictionary':dictionary.name,'packet_sha256':digest_file(path),
                        'per_image_tokens':lengths,'mean_tokens':sum(lengths)/len(lengths),'saved_percent':100*(1-sum(lengths)/(len(lengths)*3072)),
                        'half_images':sum(t<=1536 for t in lengths),'lossless_images':len(lengths)}
                    print(dataset,name,result[name]['mean_tokens'],flush=True)
        for kind in ('baseline','color'):
            for budget in (150000,200000):
                if not args.raw_only:
                    assert all(a<=b for a,b in zip(result[f'{kind}-{budget}-minimum']['per_image_tokens'],result[f'{kind}-{budget}-bpe']['per_image_tokens']))
        save_json(args.out/dataset/'report.json',{'results':result,'resource':resource_receipt()})


def decode(args):
    manifest=read_json(args.holdout/'inputs.json')
    for dataset in ('cifar','svhn'):
        report=read_json(args.out/dataset/'report.json');checked={}
        for name,row in report['results'].items():
            path=args.out/dataset/row['dictionary'];assert digest_file(path)==row['sha256']
            words={i:w for w,i in read_json(path)['model']['vocab'].items()};sizes={i:len(w) for i,w in words.items()}
            path=args.out/dataset/(name+'.npz');assert digest_file(path)==row['packet_sha256']
            with np.load(path,allow_pickle=False) as packet:ids=packet['ids'].tolist()
            offset=0
            for identity in manifest['datasets'][dataset]['samples']:
                grids=[]
                for _ in range(3):
                    start=offset;length=0
                    while length<1024:
                        if offset>=len(ids):raise ValueError('truncated channel')
                        length+=sizes[ids[offset]];offset+=1
                    if length!=1024:raise ValueError('invalid boundary')
                    grids.append(expand(ids[start:offset],words))
                image=restore_image(grids,name.split('-')[0])
                assert sha256(image.tobytes()).hexdigest()==identity['sha256']
            assert offset==len(ids);checked[name]=len(manifest['datasets'][dataset]['samples'])
        save_json(args.out/dataset/'independent-decode.json',{'checked':checked,'resource':resource_receipt()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('train','evaluate','decode'))
    p.add_argument('--raw-only',action='store_true',help='raw channel values, ordinary BPE only; no transform or minimum parser')
    for key in ('out','prior','previous','holdout'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();process=psutil.Process();process.nice(psutil.IDLE_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[:1])
    if psutil.virtual_memory().available<4*1024**3:raise RuntimeError('insufficient free memory')
    def watch():
        while True:
            if process.memory_info().rss>8*1024**3 or psutil.virtual_memory().available<4*1024**3:os._exit(75)
            time.sleep(1)
    threading.Thread(target=watch,daemon=True).start();globals()[args.stage](args)


if __name__=='__main__':main()
