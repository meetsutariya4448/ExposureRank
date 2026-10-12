"""Padded observed catalogs. Padding is never a candidate/negative."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Catalog:
    u: np.ndarray
    i: np.ndarray
    indices: np.ndarray
    valid: np.ndarray
    sizes: np.ndarray
    ties: np.ndarray

    @classmethod
    def build(cls, frame, tie_keys, n_users):
        counts=np.bincount(frame['u'],minlength=n_users)
        if np.any(counts<1): raise ValueError('Missing catalog user')
        m=int(counts.max());indices=np.zeros((n_users,m),dtype=np.int32)
        valid=np.arange(m)[None,:]<counts[:,None]
        for u in range(n_users):
            ix=np.flatnonzero(frame['u']==u);indices[u,:len(ix)]=ix
        items=frame['i'][indices]
        return cls(np.broadcast_to(np.arange(n_users)[:,None],indices.shape),items,indices,valid,counts,tie_keys[items])

    def order(self, flat_scores):
        scores=np.where(self.valid,flat_scores[self.indices],-np.inf)
        if not np.isfinite(scores[self.valid]).all(): raise ValueError('Nonfinite scores')
        # score first, then frozen item hash; raw mapped item provides fallback.
        return np.lexsort((self.i,self.ties,-scores),axis=1)

    def outcomes(self,flat): return np.where(self.valid,flat[self.indices],0)

def restrict_order(order, selected):
    ranks=np.argsort(order,axis=1)
    return np.argsort(np.where(selected,ranks,order.shape[1]+ranks),axis=1)
