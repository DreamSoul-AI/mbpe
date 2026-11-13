"""Incremental exact pair counts: validate against the full-scan prototype."""
from collections import Counter, defaultdict
import argparse
import heapq
import json
import os
from pathlib import Path
import pickle
import tarfile
import time

import numpy as np
from nd_bpe_probe import State, AxisBPE, sub
from nd_bpe_graph_probe import GraphState


class Index:
    def __init__(self):
        self.counts=Counter()
        self.occurrences=defaultdict(set)
        self.heap=[]

    def change(self,key,occurrence,delta):
        if delta==1:
            assert occurrence not in self.occurrences[key]
            self.occurrences[key].add(occurrence)
        else:
            self.occurrences[key].remove(occurrence)
            if not self.occurrences[key]:self.occurrences.pop(key)
        self.counts[key]+=delta
        if self.counts[key]==0:self.counts.pop(key)
        else:heapq.heappush(self.heap,(-self.counts[key],key))

    def best(self):
        while self.heap:
            negative,key=self.heap[0]
            if self.counts.get(key,0)!=-negative:
                heapq.heappop(self.heap)
            else:return key,-negative
        return None,0


class IndexedState(GraphState):
    def __init__(self,shape,cells,index,image,alphabet_size=256):
        super().__init__(shape,cells,alphabet_size)
        self.index=index;self.image=image
        for uid,neighbors in self.edges.items():
            for other in neighbors:
                if uid<other:self.register(uid,other,1)

    def register(self,left,right,delta):
        occurrence=(self.image,min(left,right),max(left,right))
        if self.regions[left][1]>self.regions[right][1]:left,right=right,left
        a,anchor_a,_=self.regions[left];b,anchor_b,_=self.regions[right]
        self.index.change((a,b,sub(anchor_b,anchor_a)),occurrence,delta)

    def replace(self,left,right,token):
        if left not in self.regions or right not in self.regions:return 0
        old=set()
        for uid in (left,right):
            for other in self.edges[uid]:old.add((min(uid,other),max(uid,other)))
        for a,b in old:self.register(a,b,-1)
        super().merge([(left,right)],token)
        uid=self.next_id-1
        for other in self.edges[uid]:self.register(uid,other,1)
        return 1


def learn_reference(data):
    states=[State(*sample) for sample in data]
    rules=[]
    for step in range(256):
        pairs=[s.pairs() for s in states]
        counts=Counter()
        for row in pairs:counts.update({k:len(v) for k,v in row.items()})
        key=min(counts,key=lambda k:(-counts[k],k))
        token=256+step
        for state,row in zip(states,pairs):state.merge(row.get(key,[]),token)
        rules.append((key,token))
    return rules,states


def learn_incremental(data):
    index=Index()
    states=[IndexedState(*sample,index,i) for i,sample in enumerate(data)]
    rules=[]
    for step in range(256):
        key,_=index.best()
        candidates=[]
        for image,a,b in index.occurrences[key]:
            state=states[image]
            if state.regions[a][1]>state.regions[b][1]:a,b=b,a
            candidates.append((state.edges[a][b],image,a,b))
        token=256+step
        for _,image,a,b in sorted(candidates):states[image].replace(a,b,token)
        rules.append((key,token))
    return rules,states


def benchmark(data):
    reports=[]
    for repeat in range(3):
        results={};seconds={}
        for name in (('reference','incremental') if repeat%2==0 else ('incremental','reference')):
            started=time.perf_counter()
            results[name]=(learn_reference if name=='reference' else learn_incremental)(data)
            seconds[name]=time.perf_counter()-started
        assert results['reference'][0]==results['incremental'][0]
        for a,b in zip(results['reference'][1],results['incremental'][1]):
            assert a.regions==b.regions and a.cells==b.cells
        reports.append({'repeat':repeat,'seconds':seconds,
                        'reference_over_incremental':seconds['reference']/seconds['incremental'],
                        'rules_and_final_regions_equal':True})
        print('BENCHMARK',json.dumps(reports[-1]),flush=True)
    return reports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    if os.name=='nt':
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    root=Path(__file__).resolve().parents[1]
    data=[]
    with tarfile.open(root/'data/cifar-10-python.tar.gz') as archive:
        for member,per_class in [('data_batch_1',2),('test_batch',1)]:
            batch=pickle.load(archive.extractfile('cifar-10-batches-py/'+member),encoding='bytes')
            labels=np.asarray(batch[b'labels'])
            chosen=[int(i) for c in range(10) for i in np.flatnonzero(labels==c)[:per_class]]
            red=batch[b'data'][chosen].reshape(len(chosen),3,32,32)[:,0]
            data.append([((32,32),{(y,x):int(image[y,x]) for y in range(32) for x in range(32)}) for image in red])
    if args.benchmark:
        reports=benchmark(data[0])
        args.out.mkdir(parents=True,exist_ok=False)
        (args.out/'report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        return
    started=time.perf_counter();reference=[State(*sample) for sample in data[0]]
    reference_time=time.perf_counter()-started
    started=time.perf_counter();index=Index()
    indexed=[IndexedState(*sample,index,i) for i,sample in enumerate(data[0])]
    indexed_time=time.perf_counter()-started
    initial_indexed_time=indexed_time
    trace=[]
    max_heap=len(index.heap)
    for step in range(256):
        started=time.perf_counter()
        pairs=[s.pairs() for s in reference]
        counts=Counter()
        for row in pairs:counts.update({key:len(occ) for key,occ in row.items()})
        key=min(counts,key=lambda k:(-counts[k],k))
        reference_time+=time.perf_counter()-started
        assert counts==index.counts,step
        started=time.perf_counter();selected,frequency=index.best()
        candidates=[]
        for image,a,b in index.occurrences[selected]:
            state=indexed[image]
            if state.regions[a][1]>state.regions[b][1]:a,b=b,a
            candidates.append((state.edges[a][b],image,a,b))
        candidates.sort()
        indexed_time+=time.perf_counter()-started
        assert selected==key and frequency==counts[key],step
        token=256+step
        started=time.perf_counter()
        left=[s.merge(row.get(key,[]),token) for s,row in zip(reference,pairs)]
        reference_time+=time.perf_counter()-started
        started=time.perf_counter()
        right=[0]*len(indexed)
        for _,image,a,b in candidates:right[image]+=indexed[image].replace(a,b,token)
        indexed_time+=time.perf_counter()-started
        assert left==right,step
        for a,b in zip(reference,indexed):assert a.regions==b.regions and a.cells==b.cells,step
        trace.append((key,token))
        max_heap=max(max_heap,len(index.heap))
    model=AxisBPE.from_dictionary({'rank':2,'base_alphabet':256,'rules':trace})
    lengths=[]
    for shape,cells in data[1]:
        packet=model.encode(shape,cells)
        assert model.decode(packet)==(shape,cells)
        assert model.decode_without_anchors(model.without_anchors(packet,cells))==(shape,cells)
        lengths.append(len(packet['tokens']))
    report={'reference_seconds':reference_time,'incremental_seconds':indexed_time,
            'reference_over_incremental':reference_time/indexed_time,'incremental_init_seconds':initial_indexed_time,
            'all_256_steps_counts_rules_groups_equal':True,'test_lengths':lengths,
            'test_ratio':sum(lengths)/(1024*len(lengths)),'lossless_bad':0,
            'max_heap_entries':max_heap,'training_images':20,'test_images':10,'rules':trace,
            'timing_scope':'initialization + count/select/replace; excludes equality checks and final test encoding',
            'comparison':'own fullscan vs own incremental prototype, not official benchmark'}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rules'},indent=2))


if __name__=='__main__':main()
