"""Tiny standalone reproduction; needs compatible NumPy/Numba and psutil."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['NUMBA_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'

import hashlib
import importlib.util
from pathlib import Path
import sys
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
import numpy as np

source=Path(__file__).resolve().parents[1]/'third_party/Being-VL-0.5/beingvl/model/vbpe.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='40766b169ad46a1ebe4adedf9522b12949bd8c27c29bea5951e8261685d2ffe9'
spec=importlib.util.spec_from_file_location('official_collision_repro',source)
official=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=official
spec.loader.exec_module(official)

a=np.array([[1,2],[3,4]],dtype=np.int32)
b=np.array([[1,3],[2,4]],dtype=np.int32)
assert not np.array_equal(a,b)
for function in (official._apply_merge_2d,official._apply_merge_2d.py_func):
    left=function(a.copy(),1,2,256)
    right=function(b.copy(),1,2,256)
    assert np.array_equal(left,right)
    print('Different inputs, identical output:',left.tolist())

model=official.vBPE(init_size=256,vocab_pad=0,vocab_end=256)
model.train(np.stack([a,b]).copy(),num_merges=1)
assert np.array_equal(model.merge_token(a.copy()),model.merge_token(b.copy()))
print('Official class learned rule:',model.vbpe_vocab)
