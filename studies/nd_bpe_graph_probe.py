"""Paired N-D full-cell scan vs boundary-graph mechanism experiment."""
from collections import Counter
import argparse
import json
import os
from pathlib import Path
import time
import random
import statistics

from nd_bpe_probe import State, AxisBPE, add, sub, sample


class GraphState(State):
    def __init__(self, shape, cells, alphabet_size=256):
        super().__init__(shape,cells,alphabet_size)
        self.edges={uid:{} for uid in self.regions}
        for point, uid in sorted(self.cells.items()):
            for axis in range(len(shape)):
                neighbor=list(point);neighbor[axis]+=1
                other=self.cells.get(tuple(neighbor))
                if other is not None:
                    witness=(point,axis)
                    self.edges[uid][other]=witness
                    self.edges[other][uid]=witness

    def pairs(self):
        edges=[]
        for uid,neighbors in self.edges.items():
            for other,witness in neighbors.items():
                if uid<other:
                    edges.append((witness,uid,other))
        matches={}
        for _,left,right in sorted(edges):
            if self.regions[left][1]>self.regions[right][1]:
                left,right=right,left
            a,anchor_a,_=self.regions[left]
            b,anchor_b,_=self.regions[right]
            key=(a,b,sub(anchor_b,anchor_a))
            matches.setdefault(key,[]).append((left,right))
        return matches

    def merge(self,candidates,token):
        count=0
        for left,right in candidates:
            if left not in self.regions or right not in self.regions:
                continue
            neighbors={}
            for old in (left,right):
                for other,witness in self.edges[old].items():
                    if other not in (left,right):
                        neighbors[other]=min(neighbors.get(other,witness),witness)
            super().merge([(left,right)],token)
            uid=self.next_id-1
            self.edges.pop(left);self.edges.pop(right)
            self.edges[uid]=neighbors
            for other,witness in neighbors.items():
                self.edges[other].pop(left,None);self.edges[other].pop(right,None)
                self.edges[other][uid]=witness
            count+=1
        return count


def run(shape,holes,budget=32,noise=False,training=None):
    training=training if training is not None else [sample(shape,phase,holes) for phase in range(3)]
    if noise:
        rng=random.Random(20261111)
        training=[(shape,{p:rng.randrange(256) for p in cells}) for shape,cells in training]
    started=time.perf_counter()
    baseline=[State(*item) for item in training]
    scan_init=time.perf_counter()-started
    started=time.perf_counter()
    graph=[GraphState(*item) for item in training]
    graph_init=time.perf_counter()-started
    scan_time=graph_time=0
    common=0
    merged=[]
    for step in range(budget):
        start=time.perf_counter();left=[s.pairs() for s in baseline];scan_time+=time.perf_counter()-start
        start=time.perf_counter();right=[s.pairs() for s in graph];graph_time+=time.perf_counter()-start
        assert left==right,(shape,step,'candidate dictionary or occurrence order changed')
        started=time.perf_counter()
        counts=Counter()
        for pairs in left:counts.update({key:len(value) for key,value in pairs.items()})
        if not counts:break
        key=min(counts,key=lambda k:(-counts[k],k))
        if counts[key]<2:break
        common+=time.perf_counter()-started
        token=256+step
        start=time.perf_counter()
        l=[s.merge(p.get(key,[]),token) for s,p in zip(baseline,left)]
        scan_time+=time.perf_counter()-start
        start=time.perf_counter()
        r=[s.merge(p.get(key,[]),token) for s,p in zip(graph,right)]
        graph_time+=time.perf_counter()-start
        assert l==r
        for a,b in zip(baseline,graph):assert a.regions==b.regions and a.cells==b.cells
        merged.append({'rule':key,'token':token,'merged':sum(l)})
    return {'shape':shape,'holes':holes,'noise':noise,'rules':len(merged),'scan_seconds':scan_time,
            'graph_seconds':graph_time,'scan_over_graph':scan_time/graph_time,
            'scan_init_seconds':scan_init,'graph_init_seconds':graph_init,'shared_selection_seconds':common,
            'initialization_and_mechanism_ratio':(scan_init+scan_time+common)/(graph_init+graph_time+common),
            'final_lengths':[len(s.regions) for s in baseline],
            'all_step_candidates_and_regions_equal':True,'trace':merged,
            'timing_excludes_initial_state_construction':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if os.name=='nt':
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    args.out.mkdir(parents=True,exist_ok=False)
    reports=[]
    summary=[]
    for shape,holes,noise in [((32,32),False,False),((64,64),False,False),
                             ((32,32),False,True),((8,8,8),True,False)]:
        rows=[run(shape,holes,noise=noise) for _ in range(5)]
        ratios=[r['initialization_and_mechanism_ratio'] for r in rows]
        summary.append({'shape':shape,'holes':holes,'noise':noise,'repetitions':5,
                        'median_ratio':statistics.median(ratios),'min_ratio':min(ratios),'max_ratio':max(ratios),
                        'rules':rows[0]['rules'],'equal_all_steps':True})
        reports.extend(rows)
    (args.out/'report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
