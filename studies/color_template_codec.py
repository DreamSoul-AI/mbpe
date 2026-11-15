"""Fixed-grid CPU prototypes: reversible color basis and explicit template parameters."""
from collections import Counter
import numpy as np


def validate(image):
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.shape!=(3,32,32):
        raise ValueError('uint8 [3,32,32] required')


def color_basis(image):
    validate(image)
    red,green,blue=image
    return np.stack((green,np.subtract(red,green,dtype=np.uint8),np.subtract(blue,green,dtype=np.uint8)))


def inverse_color_basis(image):
    validate(image)
    green,red_delta,blue_delta=image
    return np.stack((np.add(red_delta,green,dtype=np.uint8),green,np.add(blue_delta,green,dtype=np.uint8)))


def patch_key(block,relative):
    if relative:
        return np.subtract(block,block[0,0],dtype=np.uint8).tobytes()
    return block.tobytes()


def fit_templates(images,relative,budget=150000):
    counts=Counter()
    for image in images:
        validate(image)
        for side in (2,4,8):
            for channel in image:
                for y in range(0,32,side):
                    for x in range(0,32,side):
                        counts[(side,patch_key(channel[y:y+side,x:x+side],relative))]+=1
    order=sorted(counts,key=lambda key:(-counts[key],-key[0],key[1]))[:budget]
    return TemplateCodec(relative,order),{'unique_patterns':len(counts),'observations':sum(counts.values()),
        'selected':len(order),'selected_repeated':sum(counts[k]>1 for k in order)}


class TemplateCodec:
    def __init__(self,relative,entries):
        if type(relative) is not bool:raise ValueError('relative mode must be boolean')
        self.relative=relative;self.entries=[];self.lookup={}
        for side,pattern in entries:
            if side not in (2,4,8) or not isinstance(pattern,bytes) or len(pattern)!=side*side:
                raise ValueError('invalid square template')
            if relative and pattern[0]!=0:raise ValueError('relative template must start at zero')
            key=(side,pattern)
            if key in self.lookup:raise ValueError('duplicate template')
            self.lookup[key]=256+len(self.entries);self.entries.append(key)

    def dictionary(self):
        return {'format':'rgb-multiscale-template','version':1,'relative':self.relative,
            'shape':[3,32,32],'entries':[[s,b.hex()] for s,b in self.entries]}

    @classmethod
    def from_dictionary(cls,data):
        if data.get('format')!='rgb-multiscale-template' or data.get('version')!=1 or data.get('shape')!=[3,32,32]:
            raise ValueError('unsupported template dictionary')
        return cls(data['relative'],[(s,bytes.fromhex(b)) for s,b in data['entries']])

    def prefix(self,budget):
        return TemplateCodec(self.relative,self.entries[:budget])

    def encode(self,image):
        validate(image);tokens=[];stats=Counter()
        for channel in image:
            pieces=[]
            def visit(y,x,side):
                if side==1:
                    pieces.append((y*32+x,[int(channel[y,x])],1));return
                block=channel[y:y+side,x:x+side]
                token=self.lookup.get((side,patch_key(block,self.relative)))
                if token is not None:
                    values=[token,int(block[0,0])] if self.relative else [token]
                    pieces.append((y*32+x,values,side));return
                half=side//2
                for dy,dx in ((0,0),(0,half),(half,0),(half,half)):
                    visit(y+dy,x+dx,half)
            for y in range(0,32,8):
                for x in range(0,32,8):visit(y,x,8)
            # Anchor ordering is part of the implicit-position contract.
            for _,values,side in sorted(pieces):
                tokens.extend(values);stats['literal_cells' if side==1 else f'templates_{side}']+=1
                if side>1:stats['template_cells']+=side*side
        if len(tokens)>3072:raise ValueError('fallback length contract violated')
        return tokens,dict(stats)

    def decode(self,tokens):
        if any(type(t) is not int or t<0 for t in tokens):raise ValueError('nonnegative token IDs required')
        image=np.zeros((3,32,32),dtype=np.uint8);position=0
        for channel in image:
            filled=bytearray(1024);cursor=0;covered=0
            while covered<1024:
                cursor=filled.find(0,cursor)
                if cursor<0 or position>=len(tokens):raise ValueError('incomplete grid')
                token=tokens[position];position+=1;y,x=divmod(cursor,32)
                if token<256:side=1;pattern=bytes([token]);anchor=0
                else:
                    if token-256>=len(self.entries):raise ValueError('unknown template ID')
                    side,pattern=self.entries[token-256];anchor=0
                    if self.relative:
                        if position>=len(tokens) or tokens[position]>=256:raise ValueError('missing or invalid color parameter')
                        anchor=tokens[position];position+=1
                if y+side>32 or x+side>32:raise ValueError('template exceeds grid')
                for dy in range(side):
                    for dx in range(side):
                        index=(y+dy)*32+x+dx
                        if filled[index]:raise ValueError('overlapping coverage')
                        channel[y+dy,x+dx]=(pattern[dy*side+dx]+anchor)%256;filled[index]=1
                covered+=side*side
        if position!=len(tokens):raise ValueError('trailing tokens')
        return image
