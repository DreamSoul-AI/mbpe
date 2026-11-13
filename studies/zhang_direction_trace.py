"""External reversible envelope for the unchanged public first 2-D pass.

Only the first pass is 2-D in that implementation. Its orientations must
be retained before subsequent 1-D pair expansions can reconstruct a grid.
"""
import numpy as np


def expand_tokens(sequence,rules,keep_first=False):
    chosen=rules[1:] if keep_first else rules
    vocabulary={int(token):tuple(map(int,pair)) for pair,token in chosen}
    output=[];stack=list(reversed(list(map(int,sequence))))
    while stack:
        token=stack.pop()
        if token in vocabulary:stack.extend(reversed(vocabulary[token]))
        else:output.append(token)
    return np.asarray(output,dtype=np.int32)


def directions_from_modified(modified,first_rule,first_sequence):
    """Read the official pass's visible new anchors, retaining consumed sides.

    No truth pixels are passed to decoding. Validate the inferred removal
    mask against the actual official first-pass output before using it.
    """
    (a,b),new=first_rule
    if a==b or new<=255:raise ValueError('unsupported first rule')
    height,width=modified.shape
    flat=modified.ravel();removed=np.zeros(flat.size,dtype=bool);directions=[]
    for index in np.flatnonzero(flat==new):
        if removed[index]:raise ValueError('anchor was consumed')
        row,col=divmod(int(index),width)
        if col+1<width and flat[index+1]==b and not removed[index+1]:
            directions.append(0);removed[index+1]=True
        elif row+1<height and flat[index+width]==b and not removed[index+width]:
            directions.append(1);removed[index+width]=True
        else:raise ValueError('cannot determine original official consumed side')
    if not np.array_equal(flat[~removed],first_sequence):
        raise ValueError('trace does not reproduce official first-pass sequence')
    return directions


def decode_first(first_sequence,shape,first_rule,directions):
    height,width=shape
    grid=np.full(height*width,-1,dtype=np.int32)
    pair,new=first_rule
    cursor=0;used=0
    for token in first_sequence:
        while cursor<len(grid) and grid[cursor]>=0:cursor+=1
        if cursor>=len(grid):raise ValueError('extra token')
        if int(token)==int(new):
            if used>=len(directions):raise ValueError('missing direction')
            direction=int(directions[used]);used+=1
            row,col=divmod(cursor,width)
            if direction==0 and col+1<width:other=cursor+1
            elif direction==1 and row+1<height:other=cursor+width
            else:raise ValueError('invalid direction or outside image')
            if grid[other]>=0:raise ValueError('overlapping pair')
            grid[cursor]=int(pair[0]);grid[other]=int(pair[1])
        elif 0<=int(token)<=255:grid[cursor]=int(token)
        else:raise ValueError('unknown base token')
        cursor+=1
    if used!=len(directions) or np.any(grid<0):raise ValueError('incomplete trace/grid')
    return grid.reshape(shape)


def decode_with_trace(sequence,shape,rules,directions):
    if not rules:
        return np.asarray(sequence,dtype=np.int32).reshape(shape)
    first=expand_tokens(sequence,rules,keep_first=True)
    return decode_first(first,shape,rules[0],directions)


def pack_directions(directions):
    return np.packbits(np.asarray(directions,dtype=np.uint8),bitorder='little').tobytes()


def unpack_directions(data,count):
    if len(data)!=(count+7)//8:raise ValueError('invalid direction payload length')
    return np.unpackbits(np.frombuffer(data,dtype=np.uint8),bitorder='little')[:count].tolist()
