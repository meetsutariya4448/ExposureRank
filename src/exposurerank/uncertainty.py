import numpy as np
from scipy.stats import norm

def paired_bootstrap(a,b,rng,draws=5000,level=.975):
    d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    if d.ndim!=1 or not len(d): raise ValueError('One paired value per user required')
    out=[]
    for start in range(0,draws,100):
        ix=rng.integers(0,len(d),(min(100,draws-start),len(d)));out.extend(d[ix].mean(axis=1))
    lo,hi=np.quantile(out,[(1-level)/2,1-(1-level)/2])
    return {'gap':float(d.mean()),'lo':float(lo),'hi':float(hi),'users':len(d),'draws':draws,'level':level}

def wilson(k,n,level=.95):
    if n<=0: raise ValueError('Positive repetitions required')
    z=norm.ppf((1+level)/2);p=k/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom;radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return float(0 if k==0 else max(0,center-radius)),float(1 if k==n else min(1,center+radius))

def mc_summary(values,reference):
    v=np.asarray(values,dtype=float);error=v-reference
    return {'mean':float(v.mean()),'bias':float(error.mean()),'rmse':float(np.sqrt(np.mean(error**2))),'mcse':float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0,'repetitions':len(v)}

def reversal(values,reference,tolerance=.001):
    if abs(reference)<=tolerance: return {'state':'reference near tie','reversals':0,'rate':float('nan'),'lo':float('nan'),'hi':float('nan')}
    count=int(np.count_nonzero(np.asarray(values)*reference<0));lo,hi=wilson(count,len(values))
    return {'state':'reference separated','reversals':count,'rate':count/len(values),'lo':lo,'hi':hi}
