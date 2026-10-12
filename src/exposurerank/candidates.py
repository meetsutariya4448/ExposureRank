import numpy as np

def sample_candidates(valid,item_weights,rng,q,sampler):
    out=np.zeros_like(valid)
    for u in range(len(valid)):
        indices=np.flatnonzero(valid[u]);size=min(q,len(indices))
        if size==len(indices): chosen=indices
        elif sampler=='uniform': chosen=rng.choice(indices,size=size,replace=False)
        elif sampler=='popularity':
            w=item_weights[u,indices];chosen=rng.choice(indices,size=size,replace=False,p=w/w.sum())
        else: raise ValueError(sampler)
        out[u,chosen]=True
    return out
