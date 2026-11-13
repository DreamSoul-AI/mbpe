"""Freeze completed round-two evidence; refuse a partial official comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sources = {
        'inputs': 'inputs/input.json',
        'minimal': 'minimal/report.json',
        'zhang_large': 'zhang-large/report.json',
        'zhang_final_preimages': 'zhang-exhaustive/report.json',
        'independent_replay': 'independent-replay-final/report.json',
        'preimage_small_domain': 'preimage-coverage-check/report.json',
        'elsner_paired100': 'elsner-paired/report.json',
        'elsner_full1000': 'elsner-full/report.json',
        'elsner_shard_validation': 'elsner-full/validation-red/report.json',
        'elsner_resource_snapshot': 'elsner-full/resources-constrained.json',
        'candidate': 'candidate-shape-complete/report.json',
        'candidate_domains': 'candidate-domain-check-known-and-unseen/report.json',
    }
    for channel in ('red', 'green', 'blue'):
        sources[f'zhang_rules_{channel}'] = f'zhang-large/{channel}/rules.json'
        sources[f'zhang_flat_rules_{channel}'] = f'zhang-large/{channel}/flat_rules.json'
        for score in ('frequency', 'realizable'):
            sources[f'candidate_dictionary_{channel}_{score}'] = f'candidate-shape-complete/{channel}-{score}-dictionary.json'
    cases = {}
    for name, relative in sources.items():
        source = args.root/relative
        payload = source.read_bytes()
        cases[name] = dict(source=str(source), sha256=hashlib.sha256(payload).hexdigest(),
                           data=json.loads(payload))
    for row in cases['elsner_full1000']['data']:
        assert row['images'] == len(row['lengths']) == 1000
    assert len(cases['elsner_full1000']['data']) == 3
    report = dict(format='mbpe-research-evidence', version=1,
                  recorded_utc=datetime.now(timezone.utc).isoformat(),
                  scope='CPU raw integer BPE; source snapshots unchanged; no GPU or model encoding',
                  source_qualification='Being-VL-0.5 public snapshot, not verified as the fixed 2410.02155 implementation',
                  cases=cases)
    repo = Path(__file__).resolve().parents[1]
    files = [repo/'third_party/Being-VL-0.5/beingvl/model/vbpe.py',
             repo/'third_party/MDBPE_TF/MDBPE/processor_python.ipynb']
    files += [Path(__file__).with_name(name) for name in (
        'bpe_pair_inputs.py', 'zhang_collision_repro.py', 'zhang_collision_minimal.py',
        'zhang_large_collision_study.py', 'zhang_collision_exhaustive.py',
        'zhang_direction_trace.py', 'elsner_pair_channels.py', 'elsner_full_shards.py',
        'elsner_svhn_celeba.py', 'elsner_rule_decode.py', 'lossless_nd_candidate.py',
        'lossless_nd_candidate_study.py', 'lossless_nd_domain_check.py',
        'nd_bpe_probe.py', 'nd_bpe_graph_probe.py', 'nd_bpe_incremental_probe.py')]
    report['source_sha256'] = {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in files}
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()
