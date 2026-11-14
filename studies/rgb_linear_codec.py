"""Strong fixed-scan 1D grammars using the existing exact BPE loop."""

from functools import lru_cache

import numpy as np

from batch30_mask_baselines import learn_apply, make_orders, merge_one


SCANS=('row','column','snake','hilbert')


@lru_cache(maxsize=4)
def scan_points(mode):
    if mode not in SCANS:raise ValueError('unknown fixed scan')
    return tuple(make_orders(32)[mode])


class LinearRGBGrammar:
    rank=2

    def __init__(self,mode,rules=()):
        if mode not in SCANS:raise ValueError('unknown fixed scan')
        self.mode=mode;self.base=list(range(256));self.rules=[]
        self.shapes={value:(value,) for value in self.base}
        for left,right in rules:
            token=len(self.shapes)
            if (type(left) is not int or type(right) is not int
                    or not 0<=left<token or not 0<=right<token or token>=512):
                raise ValueError('noncausal or excessive linear rule')
            phrase=self.shapes[left]+self.shapes[right]
            if len(phrase)>1024:raise ValueError('linear phrase exceeds a channel')
            self.rules.append((left,right));self.shapes[token]=phrase

    def dictionary(self):
        return {'format':'rgb-linear-grammar','version':1,'mode':self.mode,'rules':self.rules}

    @classmethod
    def from_dictionary(cls,data):
        if data.get('format')!='rgb-linear-grammar' or data.get('version')!=1:
            raise ValueError('unsupported linear dictionary')
        return cls(data['mode'],data['rules'])

    def encode(self,shape,cells):
        if tuple(shape)!=(32,32) or len(cells)!=1024:
            raise ValueError('fixed 32x32 dense channel required')
        points=scan_points(self.mode)
        if set(cells)!=set(points) or any(type(v) is not int or not 0<=v<256 for v in cells.values()):
            raise ValueError('invalid byte grid')
        values=np.fromiter((cells[p] for p in points),dtype=np.int32)
        for step,(left,right) in enumerate(self.rules):
            values=merge_one(values,left,right,256+step)
        return {'format':'nd-bpe-candidate-packet','version':1,'shape':[32,32],
                'domain':None,'literals':[],'tokens':values.astype(int).tolist()}

    def decode(self,packet):
        if (packet.get('format')!='nd-bpe-candidate-packet' or packet.get('version')!=1
                or packet.get('shape')!=[32,32] or packet.get('domain') is not None or packet.get('literals')):
            raise ValueError('fixed dense byte packet required')
        flat=[]
        for token in packet['tokens']:
            if type(token) is not int or token not in self.shapes:raise ValueError('unknown linear token')
            flat.extend(self.shapes[token])
            if len(flat)>1024:raise ValueError('overfilled linear grid')
        if len(flat)!=1024:raise ValueError('incomplete linear grid')
        points=scan_points(self.mode)
        return (32,32),dict(zip(points,flat,strict=True))


def learn_rgb(images,mode,budget=256):
    if images.dtype!=np.uint8 or images.ndim!=4 or images.shape[1:]!=(3,32,32):
        raise ValueError('RGB learning arrays required')
    points=scan_points(mode)
    ys,xs=zip(*points)
    sequences=images[:,:,ys,xs].reshape(-1,1024).astype(np.int32)
    rules,_=learn_apply(list(sequences),{'audit':[]},256,budget,set())
    return LinearRGBGrammar(mode,[(int(left),int(right)) for left,right,_ in rules])
