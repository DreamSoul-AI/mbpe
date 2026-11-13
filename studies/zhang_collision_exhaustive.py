"""Finish the small unresolved preimage searches; certify with official code."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['NUMBA_NUM_THREADS']='1'
import argparse
import json
from itertools import product
from pathlib import Path
import shutil
import sys
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
if os.environ.get('MBPE_PROBE_DEPS'):sys.path.insert(0,os.environ['MBPE_PROBE_DEPS'])
import numpy as np
if os.environ.get('MBPE_PROBE_NUMBA_SITE'):sys.path.append(os.environ['MBPE_PROBE_NUMBA_SITE'])
from zhang_large_collision_study import load_source
from zhang_direction_trace import expand_tokens,decode_first


def check_small_domains(official):
    reports=[]
    for shape,alphabet in (((2,2),4),((2,3),3),((3,3),2)):
        total=shape[0]*shape[1];groups={}
        for values in product(range(alphabet),repeat=total):
            image=np.asarray(values,dtype=np.int32).reshape(shape)
            sequence=official._apply_merge_2d(image.copy(),0,1,256)
            groups.setdefault(sequence.tobytes(),set()).add(image.tobytes())
        for signature,expected in groups.items():
            sequence=np.frombuffer(signature,dtype=np.int32)
            count=int((sequence==256).sum());actual=set()
            for directions in product((0,1),repeat=count):
                try:image=decode_first(sequence,shape,((0,1),256),directions)
                except ValueError:continue
                result=official._apply_merge_2d(image.copy(),0,1,256)
                if np.array_equal(sequence,result):actual.add(image.tobytes())
            assert actual==expected,(shape,len(actual),len(expected))
        reports.append({'shape':shape,'alphabet':alphabet,'all_inputs':alphabet**total,
                        'output_groups':len(groups),'all_preimage_sets_equal_bruteforce':True})
    return reports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path)
    parser.add_argument('--previous',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--small-domain-check',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    official,_=load_source(Path(__file__).resolve().parents[1])
    if args.small_domain_check:
        reports=check_small_domains(official)
        (args.out/'report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print(json.dumps(reports,indent=2));return
    if args.inputs is None or args.previous is None:raise ValueError('inputs and previous required')
    inputs=json.loads((args.inputs/'input.json').read_text());reports=[]
    previous=json.loads((args.previous/'report.json').read_text())
    for row in previous:
        mode=row['channel'];work=args.out/mode;work.mkdir()
        test=np.load(inputs['channels'][mode]['files']['test1000']['path'])
        rules=json.loads((args.previous/mode/'rules.json').read_text())
        data=np.load(args.previous/mode/'outputs.npz')
        found=np.load(args.previous/mode/'collision_witnesses.npz')
        certified={int(i):(a.copy(),b.copy()) for i,a,b in zip(found['indices'],found['originals'],found['partners'])}
        resolved_unique=[];new=[];assignments=0;enumeration=[]
        for index,count in enumerate(row['direction_counts']):
            if not count or index in certified:continue
            if count>20:raise ValueError('unresolved search too large for this protocol')
            sequence=data['ids'][data['offsets'][index]:data['offsets'][index+1]]
            first=expand_tokens(sequence,rules,keep_first=True)
            preimages={};tried=0
            for directions in product((0,1),repeat=count):
                assignments+=1;tried+=1
                try:candidate=decode_first(first,(32,32),rules[0],directions)
                except ValueError:continue
                result=official._apply_merge_2d(candidate.copy(),*map(int,rules[0][0]),int(rules[0][1]))
                if not np.array_equal(result,first):continue
                preimages[candidate.tobytes()]=candidate
                if not np.array_equal(candidate,test[index]):
                    certified[index]=(test[index].copy(),candidate.copy());new.append(index);break
            if index not in certified:
                assert len(preimages)==1 and test[index].tobytes() in preimages
                resolved_unique.append(index)
            enumeration.append({'index':index,'first_merges':count,'assignments_checked':tried,
                                'ambiguous':index in certified})
        order=sorted(certified)
        np.savez_compressed(work/'collision_witnesses.npz',indices=np.asarray(order),
                            originals=np.stack([certified[i][0] for i in order]),
                            partners=np.stack([certified[i][1] for i in order]))
        shutil.copyfile(args.previous/mode/'rules.json',work/'rules.json')
        shutil.copyfile(args.previous/mode/'outputs.npz',work/'outputs.npz')
        result={'channel':mode,'sample_size':1000,'certified_ambiguous':len(certified),
                'ambiguous_indices':order,'additional_certified':new,'exhausted_unique':resolved_unique,
                'no_first_merge_unique':len(row['first_pass_no_merge']),'unresolved':0,
                'remaining_orientation_assignments_checked':assignments,'enumeration':enumeration,
                'scope':'exact under documented fixed-code first-pass preimage characterization, not population prevalence'}
        assert len(certified)+len(resolved_unique)+len(row['first_pass_no_merge'])==1000
        reports.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ('enumeration','ambiguous_indices','exhausted_unique')},indent=2),flush=True)
    (args.out/'report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')


if __name__=='__main__':main()
