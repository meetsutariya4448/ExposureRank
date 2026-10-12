import time
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

def priors(u,i,y,n_users,n_items,smoothing=20):
    p0=(y.sum(dtype=np.float64)+1)/(len(y)+2)
    n=np.bincount(i,minlength=n_items);pos=np.bincount(i,weights=y,minlength=n_items)
    return {'p0':float(p0),'n':n,'pos':pos,'item':(pos+smoothing*p0)/(n+smoothing)}

def fit_bias(u,i,y,n_users,n_items,l2=.001):
    start=time.perf_counter();mu=np.log((y.sum()+1)/(len(y)-y.sum()+1))
    def objective(x):
        z=x[0]+x[1:1+n_users][u]+x[1+n_users:][i]
        error=(expit(z)-y)/len(y)
        loss=np.mean(np.logaddexp(0,z)-y*z)+l2/2*np.dot(x[1:],x[1:])
        gradient=np.r_[error.sum(),np.bincount(u,weights=error,minlength=n_users),np.bincount(i,weights=error,minlength=n_items)]
        gradient[1:]+=l2*x[1:]
        return loss,gradient
    x=np.zeros(1+n_users+n_items);x[0]=mu
    fit=minimize(objective,x,jac=True,method='L-BFGS-B',options={'maxiter':250,'ftol':1e-12,'gtol':1e-7})
    # Centering preserves every fitted logit; intercept absorbs both means.
    b=fit.x[1:1+n_users];c=fit.x[1+n_users:];mu=float(fit.x[0]+b.mean()+c.mean())
    if not fit.success: raise RuntimeError(f'Bias convergence failed: {fit.message}')
    return {'mu':mu,'b':b-b.mean(),'c':c-c.mean(),'seconds':time.perf_counter()-start,'iterations':fit.nit,'gradient_max':float(np.max(np.abs(fit.jac))),'converged':bool(fit.success)}

def baseline_score(name,params,frame):
    if name=='global': return np.full(len(frame['u']),params['p0'])
    if name=='item': return params['item'][frame['i']]
    if name=='bias': return expit(params['mu']+params['b'][frame['u']]+params['c'][frame['i']])
    raise ValueError(name)
