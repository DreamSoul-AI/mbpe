"""Geometry-constrained occurrence overlap, a small research check only."""
from collections import Counter,defaultdict
import argparse
from itertools import product
import json
import os
from pathlib import Path
import random

from nd_bpe_probe import State


def inspect(key,pairs):
    adjacency=defaultdict(set)
    for a,b in pairs:adjacency[a].add(b);adjacency[b].add(a)
    assert max(map(len,adjacency.values())) <= (1 if key[0]!=key[1] else 2)
    unseen=set(adjacency);optimum=0
    while unseen:
        stack=[unseen.pop()];component=set(stack)
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor);component.add(neighbor);stack.append(neighbor)
        edges=sum(len(adjacency[x]) for x in component)//2
        assert edges==len(component)-1  # path, not cycle
        optimum+=len(component)//2
    occupied=set();greedy=0
    for a,b in pairs:
        if a not in occupied and b not in occupied:
            occupied.update((a,b));greedy+=1
    assert greedy==optimum
    return optimum


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if os.name=='nt':
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    rng=random.Random(20261111)
    checks=0
    for shape in ((20,),(5,7),(3,4,4),(2,)*8):
        for seed in range(5):
            cells={p:rng.randrange(3) for p in product(*(range(n) for n in shape)) if rng.random()>.15}
            state=State(shape,cells)
            for step in range(16):
                groups=state.pairs()
                if not groups:break
                for key,pairs in groups.items():
                    inspect(key,pairs);checks+=1
                key=min(groups,key=lambda k:(-len(groups[k]),k))
                if len(groups[key])<2:break
                state.merge(groups[key],256+step)
    # 6 copies of A vs four disjoint BC pairs; raw counts and savings differ.
    cells={(0,x):1 for x in range(6)}
    for start in (8,11,14,17):cells[(0,start)]=2;cells[(0,start+1)]=3
    state=State((1,19),cells)
    groups=state.pairs()
    raw=min(groups,key=lambda k:(-len(groups[k]),k))
    gains={key:inspect(key,pairs) for key,pairs in groups.items()}
    effective=min(gains,key=lambda k:(-gains[k],k))
    report={'checked_candidate_groups':checks,'rank_cases':[1,2,3,8],
            'all_degree_acyclic_and_greedy_checks_passed':True,
            'toy':{'base_cells':len(cells),'raw_frequency_rule':raw,'raw_count':len(groups[raw]),
                   'raw_rule_actual_savings':gains[raw], 'actual_savings_rule':effective,
                   'actual_rule_savings':gains[effective]},
            'scope':'finite sample checks and constructed example; not a general proof or novelty claim'}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
