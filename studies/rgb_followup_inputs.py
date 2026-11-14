"""New validation and disjoint E4/E2 holdouts, excluding prior RGB inputs."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import psutil

from rgb_capacity_study import SOURCES, digest_file, read_json, save_json, source_images, select_indices


SEEDS = {'validation':151102,'e4-test':151103,'e2-test':151104}


def freeze(backup, old38, prior, out):
    out.mkdir(parents=True,exist_ok=False)
    old = read_json(old38/'inputs.json')
    e1 = read_json(prior/'inputs.json')['datasets']
    entries = {row['path']:row['sha256'] for row in read_json(backup/'manifest.json')['data']['entries']}
    manifest = {'seeds':SEEDS,'images_per_split':1000,'datasets':{},
                'prior_inputs_sha256':digest_file(prior/'inputs.json'),
                'old38_inputs_sha256':digest_file(old38/'inputs.json')}
    for dataset,names in SOURCES.items():
        directory = out/dataset
        directory.mkdir()
        seen,blocked = set(),{}
        for name in names:
            relative = 'data/'+name
            images,labels = source_images(dataset,backup/relative,entries[relative])
            excluded = {row['index'] for part in old[dataset].values() if part['source']==relative for row in part['samples']}
            excluded.update(row['index'] for part in e1[dataset]['splits'].values() if part['source']==relative
                            for row in part['samples'])
            blocked[relative] = sorted(excluded)
            from hashlib import sha256
            seen.update(sha256(images[index].tobytes()).hexdigest() for index in excluded)
            del images,labels
        splits = {}
        for name,split_names in ((names[0],('validation',)),(names[1],('e4-test','e2-test'))):
            relative = 'data/'+name
            images,labels = source_images(dataset,backup/relative,entries[relative])
            for split in split_names:
                rows = select_indices(labels,images,1000,SEEDS[split],set(blocked[relative]),seen)
                path = directory/(split+'.npz')
                with path.open('xb') as handle:
                    np.savez(handle,images=np.stack([images[row['index']] for row in rows]))
                splits[split] = {'source':relative,'source_sha256':entries[relative],
                                 'samples':rows,'array_sha256':digest_file(path)}
                print(json.dumps({'frozen':dataset,'split':split,'images':len(rows)}),flush=True)
            del images,labels
        # Retain only existing shared models, not per-image state.
        for transform in ('raw','difference'):
            source = prior/dataset/'formal'/(transform+'-dictionary.json')
            receipt = read_json(prior/dataset/'formal'/(transform+'-fit.json'))
            if digest_file(source)!=receipt['dictionary_sha256']:
                raise ValueError('E1 shared dictionary differs from learning receipt')
            shutil.copyfile(source,directory/source.name)
        manifest['datasets'][dataset] = {'splits':splits,'excluded_indices':blocked,
            'dictionaries':{t:digest_file(directory/(t+'-dictionary.json')) for t in ('raw','difference')}}
    save_json(out/'inputs.json',manifest)
    return manifest


def load_split(root,dataset,split):
    from hashlib import sha256
    meta = read_json(root/'inputs.json')['datasets'][dataset]['splits'][split]
    path = root/dataset/(split+'.npz')
    if digest_file(path) != meta['array_sha256']:
        raise ValueError('frozen array changed')
    with np.load(path,allow_pickle=False) as data:
        images = data['images']
    if images.shape != (1000,3,32,32) or images.dtype != np.uint8:
        raise ValueError('fixed RGB split differs')
    if any(sha256(image.tobytes()).hexdigest()!=row['sha256'] for image,row in zip(images,meta['samples'],strict=True)):
        raise ValueError('frozen identity changed')
    return images,meta['samples']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('backup','old38','prior','out'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    process=psutil.Process();process.nice(psutil.IDLE_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[:1])
    freeze(args.backup,args.old38,args.prior,args.out)
