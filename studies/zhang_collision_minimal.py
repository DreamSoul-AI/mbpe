"""Minimal reproduction: load only the unchanged official vbpe.py.

Requires numpy and numba. No project helpers, package shim, images or models.
Run from the repository root with a compatible Python environment.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['NUMBA_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
if os.name=='nt':
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
if os.environ.get('MBPE_PROBE_DEPS'):
    sys.path.insert(0,os.environ['MBPE_PROBE_DEPS'])
import numpy as np
if os.environ.get('MBPE_PROBE_NUMBA_SITE'):
    sys.path.append(os.environ['MBPE_PROBE_NUMBA_SITE'])
import numba
numba.set_num_threads(1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--witnesses',type=Path)
    parser.add_argument('--inputs',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    path=root/'third_party/Being-VL-0.5/beingvl/model/vbpe.py'
    manifest=json.loads((path.parents[2]/'SOURCE.json').read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==manifest['files']['beingvl/model/vbpe.py']
    spec=importlib.util.spec_from_file_location('unchanged_official_vbpe',path)
    official=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=official
    spec.loader.exec_module(official)
    a=np.array([[1,2],[3,4]],dtype=np.int32)
    b=np.array([[1,3],[2,4]],dtype=np.int32)
    assert not np.array_equal(a,b)
    results={}
    for name,fn in [('official_jit',official._apply_merge_2d),
                    ('official_original_python_body',official._apply_merge_2d.py_func)]:
        left=fn(a.copy(),1,2,256);right=fn(b.copy(),1,2,256)
        assert np.array_equal(left,right) and left.tolist()==[256,3,4]
        results[name]=[left.tolist(),right.tolist()]
    model=official.vBPE(init_size=256,vocab_pad=0,vocab_end=256)
    model.train(np.stack([a,b]).copy(),num_merges=1)
    left=model.merge_token(a.copy());right=model.merge_token(b.copy())
    assert np.array_equal(left,right)
    results['official_class_learned_rule']={'rules':model.vbpe_vocab,'outputs':[left.tolist(),right.tolist()]}
    if args.witnesses:
        if args.inputs is None:raise ValueError('--inputs required for real-data replay')
        inputs=json.loads((args.inputs/'input.json').read_text())
        verified={}
        for mode,meta in inputs['channels'].items():
            work=args.witnesses/mode
            rules=json.loads((work/'rules.json').read_text())
            replay=official.vBPE(256,0,256)
            replay.vbpe_vocab=rules  # load the learned dictionary, do not patch methods
            witnesses=np.load(work/'collision_witnesses.npz')
            actual=np.load(meta['files']['test1000']['path'])
            outputs=np.load(work/'outputs.npz')
            for index,original,partner in zip(witnesses['indices'],witnesses['originals'],witnesses['partners']):
                assert np.array_equal(original,actual[int(index)])
                assert not np.array_equal(original,partner)
                first=replay.merge_token(original.copy());second=replay.merge_token(partner.copy())
                assert np.array_equal(first,second)
                assert np.array_equal(first,outputs['ids'][outputs['offsets'][index]:outputs['offsets'][index+1]])
            verified[mode]=len(witnesses['indices'])
        results['independent_real_witness_replay']=verified
    assert digest==hashlib.sha256(path.read_bytes()).hexdigest()
    report={'source_commit':manifest['commit'],'source_sha256':digest,'inputs':[a.tolist(),b.tolist()],
            'same_input_shape':[2,2],'results':results,'official_source_modified':False,
            'project_helper_imports':False,'package_shim':False,'vocab_offset_or_data_transform':False,
            'numpy':np.__version__,'numba':numba.__version__}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
