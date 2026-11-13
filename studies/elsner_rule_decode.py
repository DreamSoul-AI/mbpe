"""External geometric decoder from official learned rules, no source edits.

Unlike the notebook's cropped-shape export, relative coordinates can be
negative in a later axis. Ground-truth pixels are not inputs to decoding.
"""


def rule_shapes(rules, alphabet=256):
    shapes={i:{(0,0):i} for i in range(alphabet)}
    for index,rule in enumerate(rules):
        a,b,dx,dy=map(int,rule)
        joined=dict(shapes[a])
        for (x,y),value in shapes[b].items():
            point=(x-dx,y-dy)
            if point in joined:
                raise ValueError('overlapping rule definition')
            joined[point]=value
        anchor=min(joined)
        shapes[alphabet+index]={(x-anchor[0],y-anchor[1]):value for (x,y),value in joined.items()}
    return shapes


def decode_sequence(sequence,shapes,shape):
    height,width=shape
    grid=[[None]*width for _ in range(height)]
    for token,x,y in sequence:
        for (dx,dy),value in shapes[int(token)].items():
            row,col=int(x)+dx,int(y)+dy
            if not (0<=row<height and 0<=col<width):
                raise ValueError('decoded region outside domain')
            if grid[row][col] is not None:
                raise ValueError('decoded regions overlap')
            grid[row][col]=value
    if any(value is None for row in grid for value in row):
        raise ValueError('incomplete coverage')
    return grid
