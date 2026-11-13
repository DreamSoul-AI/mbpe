"""Shared CIFAR channels: collisions, certified partners and external repairs."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='8'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['NUMBA_NUM_THREADS']='1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
if os.environ.get('MBPE_PROBE_DEPS'):sys.path.insert(0,os.environ['MBPE_PROBE_DEPS'])
import numpy as np
if os.environ.get('MBPE_PROBE_NUMBA_SITE'):sys.path.append(os.environ['MBPE_PROBE_NUMBA_SITE'])
import numba
numba.set_num_threads(1)
from zhang_direction_trace import (expand_tokens,directions_from_modified,decode_first,
    decode_with_trace,pack_directions,unpack_directions)


def load_source(root):
    path=root/'third_party/Being-VL-0.5/beingvl/model/vbpe.py'
    manifest=json.loads((path.parents[2]/'SOURCE.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files']['beingvl/model/vbpe.py']
    spec=importlib.util.spec_from_file_location('unchanged_large_official_vbpe',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module,manifest


def certify_partner(model,original,sequence,first,directions):
    # A different input is counted only after an actual official re-encode.
    candidates=[('flat_inverse',expand_tokens(sequence,model.vbpe_vocab).reshape(original.shape))]
    for i in range(min(len(directions),8)):
        changed=list(directions);changed[i]=1-changed[i]
        try:candidates.append((f'flip_direction_{i}',decode_first(first,original.shape,model.vbpe_vocab[0],changed)))
        except ValueError:pass
    for name,partner in candidates:
        if not np.array_equal(partner,original):
            encoded=model.merge_token(partner.copy())
            if np.array_equal(encoded,sequence):return name,partner
    return None,None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];official,manifest=load_source(root)
    inputs=json.loads((args.inputs/'input.json').read_text());reports=[]
    for mode,meta in inputs['channels'].items():
        start=time.perf_counter();work=args.out/mode;work.mkdir()
        arrays={}
        for label in ('train','test1000'):
            path=Path(meta['files'][label]['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['files'][label]['sha256']
            arrays[label]=np.load(path)
        train,test=arrays['train'],arrays['test1000']
        model=official.vBPE(256,0,256);model.train(train.copy(),256)
        flat_model=official.vBPE(256,0,256);flat_model.train(train.reshape(len(train),-1).copy(),256)
        rules=model.vbpe_vocab
        (work/'rules.json').write_text(json.dumps(rules),encoding='utf-8')
        (work/'flat_rules.json').write_text(json.dumps(flat_model.vbpe_vocab),encoding='utf-8')
        lengths=[];inverse_bad=[];trace_bad=[];flat_bad=[];flat_lengths=[];counts=[];side_bytes=[]
        witness_indices=[];witness_kind=[];partners=[];originals=[];codes=[];trace_codes=[]
        observed={};collisions={};no_first_merge=[];mutated=[]
        for index,original in enumerate(test):
            working=original.copy();sequence=model.merge_token(working)
            codes.append(sequence.copy());lengths.append(len(sequence))
            if not np.array_equal(working,original):mutated.append(index)
            inverse=expand_tokens(sequence,rules)
            if not np.array_equal(inverse.reshape(original.shape),original):inverse_bad.append(index)
            first=expand_tokens(sequence,rules,keep_first=True)
            directions=directions_from_modified(working,rules[0],first) if rules else []
            packed=pack_directions(directions)
            trace_codes.append(packed.hex());counts.append(len(directions));side_bytes.append(len(packed))
            rebuilt=decode_with_trace(sequence,original.shape,rules,unpack_directions(packed,len(directions)))
            if not np.array_equal(rebuilt,original):trace_bad.append(index)
            flattened=flat_model.merge_token(original.ravel().copy())
            flat_lengths.append(len(flattened))
            if not np.array_equal(expand_tokens(flattened,flat_model.vbpe_vocab).reshape(original.shape),original):flat_bad.append(index)
            signature=sequence.astype('<i4').tobytes()
            input_hash=hashlib.sha256(original.astype('<i4').tobytes()).hexdigest()
            previous=observed.get(signature)
            if previous is not None and previous[1]!=input_hash:
                collisions.setdefault(hashlib.sha256(signature).hexdigest(),[previous[0]]).append(index)
            elif previous is None:observed[signature]=(index,input_hash)
            if not directions:no_first_merge.append(index)
            else:
                kind,partner=certify_partner(model,original,sequence,first,directions)
                if partner is not None:
                    witness_indices.append(index);witness_kind.append(kind)
                    originals.append(original.copy());partners.append(partner)
            if (index+1)%100==0:print(mode,index+1,'certified',len(witness_indices),'inverse_bad',len(inverse_bad),flush=True)
        offsets=np.cumsum([0]+lengths,dtype=np.int64)
        np.savez_compressed(work/'outputs.npz',ids=np.concatenate(codes),offsets=offsets)
        if partners:np.savez_compressed(work/'collision_witnesses.npz',indices=np.asarray(witness_indices),
                                        originals=np.stack(originals),partners=np.stack(partners))
        (work/'direction_payload.json').write_text(json.dumps({'counts':counts,'hex':trace_codes}),encoding='utf-8')
        paired=inputs['paired_positions_in_1000']
        row={'channel':mode,'source_commit':manifest['commit'],'train_images':len(train),'test_images':len(test),
             'new_rules':len(rules),'lengths':lengths,'ratio':sum(lengths)/test.size,
             'paired100_ratio':sum(lengths[i] for i in paired)/(100*1024),
             'paired100_inverse_bad':[i for i in paired if i in inverse_bad],
             'sample_collision_groups':collisions,'sample_collision_distinct_inputs':len(set(i for v in collisions.values() for i in v)),
             'certified_ambiguous_inputs_lower_bound':len(witness_indices),'certified_indices':witness_indices,
             'certification_methods':witness_kind,'simple_inverse_bad':inverse_bad,'first_pass_input_mutated':mutated,
             'first_pass_no_merge':no_first_merge,'direction_trace_bad':trace_bad,'flat_repair_bad':flat_bad,
             'flat_repair_ratio':sum(flat_lengths)/test.size,'flat_lengths':flat_lengths,
             'direction_bits':sum(counts),'direction_payload_bytes':sum(side_bytes),
             'direction_counts':counts,'trace_fixed9bit_sequence_plus_trace_ratio':(sum(lengths)*9+sum(side_bytes)*8)/(test.size*8),
             'cost_excludes_dictionary_and_shape_headers':True,
             'paired_input_pixel_sha256':meta['files']['paired100']['pixel_sha256'],
             'seconds':time.perf_counter()-start,'numba_threads':numba.get_num_threads(),'workers':0}
        reports.append(row)
        (work/'report.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        (args.out/'report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print('COMPLETE',mode,'ratio',row['ratio'],'certified',len(witness_indices),
              'trace_bad',len(trace_bad),'flat_bad',len(flat_bad),flush=True)
    load_source(root)


if __name__=='__main__':main()
