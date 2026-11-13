"""Research candidate: explicit geometric grammar and packet-local literals.

Finite integer-lattice domains, arbitrary positive edge lengths and integer
values. Rank-specific dictionaries; no claim of a finished general codec.
"""
from itertools import product
import json
from types import SimpleNamespace

from nd_bpe_incremental_probe import Index,IndexedState
from nd_bpe_probe import add


def validate_domain(rank,shape,cells):
    if len(shape)!=rank or any(type(n) is not int or n<0 for n in shape):
        raise ValueError('invalid shape or rank')
    for point,value in cells.items():
        if len(point)!=rank or any(type(x) is not int or not 0<=x<n for x,n in zip(point,shape)):
            raise ValueError('invalid domain coordinate')
        if type(value) is not int:raise ValueError('integer atoms required')


class LosslessNDCandidate:
    def __init__(self,rank):
        if type(rank) is not int or rank<0:raise ValueError('nonnegative rank required')
        self.rank=rank;self.base=[];self.rules=[];self.shapes={}

    def _state(self,shape,cells,index,image,alphabet_size):
        if not shape or 0 in shape:
            # Scalar or empty shape has no axis-neighbor pair to index.
            regions={i:(value,point,frozenset([point])) for i,(point,value) in enumerate(sorted(cells.items()))}
            return SimpleNamespace(regions=regions,edges={})
        return IndexedState(shape,cells,index,image,alphabet_size)

    def _realizable(self,key,index,states):
        if key[0]!=key[1]:return index.counts[key]
        matches=[]
        for image,a,b in index.occurrences[key]:
            matches.append((states[image].edges[a][b],image,a,b))
        occupied=set();count=0
        for _,image,a,b in sorted(matches):
            if (image,a) not in occupied and (image,b) not in occupied:
                occupied.update(((image,a),(image,b)));count+=1
        return count

    def learn(self,samples,budget=64,score='frequency'):
        if score not in ('frequency','realizable'):raise ValueError('unknown score')
        for shape,cells in samples:validate_domain(self.rank,shape,cells)
        self.base=sorted({v for _,cells in samples for v in cells.values()})
        ids={v:i for i,v in enumerate(self.base)}
        index=Index();states=[]
        for image,(shape,cells) in enumerate(samples):
            states.append(self._state(shape,{p:ids[v] for p,v in cells.items()},index,image,len(self.base)))
        zero=(0,)*self.rank
        self.shapes={i:{zero:v} for i,v in enumerate(self.base)};self.rules=[]
        self.trace=[]
        for step in range(budget):
            if not index.counts:break
            if score=='frequency':key,gain=index.best()
            else:
                gains={key:self._realizable(key,index,states) for key in index.counts}
                key=min(gains,key=lambda k:(-gains[k],k));gain=gains[key]
            if gain<2:break
            a,b,delta=key;token=len(self.base)+step
            shape_atoms=dict(self.shapes[a])
            for point,value in self.shapes[b].items():
                shifted=add(point,delta)
                if shifted in shape_atoms:raise ValueError('overlapping shape rule')
                shape_atoms[shifted]=value
            if min(shape_atoms)!=zero:raise ValueError('noncanonical anchor')
            self.shapes[token]=shape_atoms;self.rules.append((key,token))
            candidates=[]
            for image,left,right in index.occurrences[key]:
                state=states[image]
                if state.regions[left][1]>state.regions[right][1]:left,right=right,left
                candidates.append((state.edges[left][right],image,left,right))
            saved=0
            for _,image,left,right in sorted(candidates):saved+=states[image].replace(left,right,token)
            if score=='realizable' and saved!=gain:raise ValueError('gain contract violated')
            self.trace.append({'chosen_score':gain,'actual_saved':saved})
        return [len(state.regions) for state in states]

    def dictionary(self):
        return {'format':'nd-bpe-candidate-dictionary','version':1,'rank':self.rank,
                'base':self.base,'rules':self.rules}

    @classmethod
    def from_dictionary(cls,data):
        if data.get('format')!='nd-bpe-candidate-dictionary' or data.get('version')!=1:
            raise ValueError('unsupported dictionary')
        model=cls(data['rank']);model.base=list(data['base'])
        if any(type(v) is not int for v in model.base) or len(set(model.base))!=len(model.base):
            raise ValueError('invalid base alphabet')
        zero=(0,)*model.rank
        model.shapes={i:{zero:v} for i,v in enumerate(model.base)}
        for (a,b,delta),token in data['rules']:
            if token!=len(model.base)+len(model.rules) or a not in model.shapes or b not in model.shapes:
                raise ValueError('noncausal rule')
            if len(delta)!=model.rank or any(type(x) is not int for x in delta):raise ValueError('invalid displacement')
            atoms=dict(model.shapes[a])
            for point,value in model.shapes[b].items():
                shifted=add(point,delta)
                if shifted in atoms:raise ValueError('overlapping rule')
                atoms[shifted]=value
            if min(atoms)!=zero:raise ValueError('invalid anchor')
            model.shapes[token]=atoms;model.rules.append(((a,b,tuple(delta)),token))
        return model

    def encode(self,shape,cells):
        validate_domain(self.rank,shape,cells)
        base={v:i for i,v in enumerate(self.base)}
        unknown=sorted(set(cells.values())-set(base))
        literals={len(self.shapes)+i:v for i,v in enumerate(unknown)}
        ids={**base,**{v:i for i,v in literals.items()}}
        index=Index()
        state=self._state(shape,{p:ids[v] for p,v in cells.items()},index,0,len(self.shapes)+len(unknown))
        for key,token in self.rules:
            candidates=[]
            for _,a,b in index.occurrences.get(key,()):
                if state.regions[a][1]>state.regions[b][1]:a,b=b,a
                candidates.append((state.edges[a][b],a,b))
            for _,a,b in sorted(candidates):state.replace(a,b,token)
        tokens=[token for token,_,_ in sorted(state.regions.values(),key=lambda r:r[1])]
        dense_size=1
        for n in shape:dense_size*=n
        domain=None if len(cells)==dense_size else [list(p) for p in sorted(cells)]
        return {'format':'nd-bpe-candidate-packet','version':1,'shape':list(shape),'domain':domain,
                'literals':[[i,v] for i,v in literals.items()],'tokens':tokens}

    def decode(self,packet):
        if packet.get('format')!='nd-bpe-candidate-packet' or packet.get('version')!=1:
            raise ValueError('unsupported packet')
        shape=tuple(packet['shape']);validate_domain(self.rank,shape,{})
        points=list(product(*(range(n) for n in shape))) if packet['domain'] is None else [tuple(p) for p in packet['domain']]
        validate_domain(self.rank,shape,{p:0 for p in points})
        if len(set(points))!=len(points) or points!=sorted(points):raise ValueError('invalid domain')
        shapes=dict(self.shapes);zero=(0,)*self.rank
        for token,value in packet['literals']:
            if type(token) is not int or token<len(self.shapes) or token in shapes or type(value) is not int:
                raise ValueError('invalid literal')
            shapes[token]={zero:value}
        domain=set(points);cells={};cursor=0
        for token in packet['tokens']:
            if type(token) is not int or token not in shapes:raise ValueError('unknown symbol')
            while cursor<len(points) and points[cursor] in cells:cursor+=1
            if cursor==len(points):raise ValueError('extra token')
            for offset,value in shapes[token].items():
                point=add(points[cursor],offset)
                if point not in domain or point in cells:raise ValueError('invalid coverage')
                cells[point]=value
        if set(cells)!=domain:raise ValueError('incomplete coverage')
        return shape,cells
