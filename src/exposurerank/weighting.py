"""Observed-only interfaces: no complete outcome or complete pi vector."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ObservedView:
    indices: np.ndarray
    values: np.ndarray
    probabilities: np.ndarray
    size: int

    def __post_init__(self):
        if len(self.indices)!=len(self.values) or len(self.values)!=len(self.probabilities): raise ValueError('Observed arrays disagree')
        if np.any(self.probabilities<=0) or np.any(self.probabilities>1): raise ValueError('Observed positivity violated')
        if np.any(np.diff(self.indices)<=0): raise ValueError('Observed indices must be distinct and sorted')

    def lookup(self,index):
        found=np.flatnonzero(self.indices==index)
        if not len(found): raise KeyError('Outcome was not observed')
        return self.values[found[0]]

def ht(coefficients,observed,clip=None):
    a=np.asarray(coefficients,dtype=float).ravel()
    if len(a)!=observed.size: raise ValueError('Wrong declared frame size')
    aa=a[observed.indices];pi=observed.probabilities;weights=1/pi
    if clip is not None: weights=np.minimum(weights,clip)
    contributions=aa*observed.values
    value=float(np.sum(contributions*weights))
    variance=float(np.sum(contributions**2*(1-pi)*weights**2))
    contributing=aa!=0;w=weights[contributing]
    ess=float(w.sum()**2/np.sum(w*w)) if len(w) else 0
    return {'estimate':value,'variance':variance,'ess':ess,'max_weight':float(w.max()) if len(w) else 0,'clipped_fraction':float(np.mean(1/pi[contributing]>clip)) if clip is not None and np.any(contributing) else 0}
