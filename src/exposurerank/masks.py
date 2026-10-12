import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from .weighting import ObservedView

def offset(z,alpha,epsilon=.02,beta=2):
    if not epsilon<alpha<1-epsilon: raise ValueError('Density must lie inside attainable floor')
    return float(brentq(lambda c:np.mean(epsilon+(1-2*epsilon)*expit(c+beta*z))-alpha,-80,80,xtol=1e-12))

def probability(mechanism,alpha,pop_z,y,constant=None,epsilon=.02):
    if alpha==1: return np.ones_like(y,dtype=float)
    if mechanism=='uniform': return np.full_like(y,alpha,dtype=float)
    z=pop_z if mechanism=='popularity' else 2*y-1
    if mechanism not in ('popularity','outcome'): raise ValueError(mechanism)
    if constant is None: raise ValueError('Validation-frozen offset is required')
    return epsilon+(1-2*epsilon)*expit(constant+2*z)

def observe(values,probabilities,valid,rng):
    # Only this simulator owns the entire probability/outcome array.
    chosen=valid & (rng.random(valid.shape)<probabilities)
    ix=np.flatnonzero(chosen.ravel())
    return ObservedView(ix,values.ravel()[ix].copy(),probabilities.ravel()[ix].copy(),values.size)

def naive_zero(view,shape):
    out=np.zeros(view.size);out[view.indices]=view.values
    return out.reshape(shape)
