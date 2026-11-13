"""Run unchanged notebook replacement/export/decoder on shared test grids."""
import os
from pathlib import Path

for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='8'
os.environ.update(CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'tmp' / 'matplotlib'))
import argparse
import hashlib
import json
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
import numpy as np
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
import elsner_svhn_celeba as runner
from elsner_rule_decode import rule_shapes,decode_sequence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    inputs=json.loads((args.inputs/'input.json').read_text())
    digest=hashlib.sha256(runner.NOTEBOOK.read_bytes()).hexdigest()
    reports=[]
    for mode,meta in inputs['channels'].items():
        path=Path(meta['files']['paired100']['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['files']['paired100']['sha256']
        images=np.load(path)
        rules_path=Path(meta['elsner_rules'])
        assert hashlib.sha256(rules_path.read_bytes()).hexdigest()==meta['elsner_rules_sha256']
        work=args.out/mode/'test'
        if work.exists():raise FileExistsError(work)
        print('START',mode,len(images),flush=True)
        runner.PREPARED_GRIDS=[image.astype(np.uint8) for image in images]
        summary=runner._apply_test(work,rules_path.parent,'grids',1,len(images),32,32,256)
        rules=torch.load(rules_path,weights_only=False)
        shapes=rule_shapes(rules);lengths=[];bad=[]
        files=sorted((work/'tokensequences').rglob('*.dat'),key=lambda p:int(p.stem))
        for index,file in enumerate(files):
            sequence=torch.load(file,weights_only=False);lengths.append(len(sequence))
            rebuilt=decode_sequence(sequence,shapes,(32,32))
            if rebuilt!=images[index].tolist():bad.append(index)
        assert len(lengths)==100
        assert hashlib.sha256(runner.NOTEBOOK.read_bytes()).hexdigest()==digest
        row={'channel':mode,'summary':summary,'lengths':lengths,'external_rule_decode_bad':bad,
             'input_pixel_sha256':meta['files']['paired100']['pixel_sha256'],
             'rules_sha256':meta['elsner_rules_sha256'],'notebook_sha256':digest}
        reports.append(row)
        (args.out/'report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print('DONE',mode,summary['compression_percent_of_original'],'native_bad',summary['lossless_bad'],
              'external_bad',bad,flush=True)


if __name__=='__main__':main()
