import numpy as np

DISCOUNT=1/np.log2(np.arange(2,22,dtype=np.float64))
IDEAL=np.r_[0,np.cumsum(DISCOUNT)]

def metrics(y,r,order,valid):
    sizes=valid.sum(axis=1)
    yy=np.take_along_axis(y,order,axis=1)
    rr=np.take_along_axis(r,order,axis=1)
    vv=np.take_along_axis(valid,order,axis=1)
    positives=(y*valid).sum(axis=1)
    k10=np.minimum(sizes,10);k20=np.minimum(sizes,20)
    rank10=np.arange(10)[None,:]<k10[:,None]
    rank20=np.arange(20)[None,:]<k20[:,None]
    # Pad short synthetic catalogs, so K can exceed candidate count.
    if yy.shape[1]<20:
        pad=20-yy.shape[1];yy=np.pad(yy,((0,0),(0,pad)));rr=np.pad(rr,((0,0),(0,pad)))
    dcg=(yy[:,:10]*rank10*DISCOUNT[:10]).sum(axis=1)
    ideal=IDEAL[np.minimum(positives,10).astype(int)]
    return {
      'ndcg10':np.divide(dcg,ideal,out=np.zeros_like(dcg),where=ideal>0),
      'recall10':np.divide((yy[:,:10]*rank10).sum(axis=1),positives,out=np.zeros_like(dcg),where=positives>0),
      'recall20':np.divide((yy[:,:20]*rank20).sum(axis=1),positives,out=np.zeros_like(dcg),where=positives>0),
      'dcg10':dcg,
      'reward10':np.divide((rr[:,:10]*rank10).sum(axis=1),k10,out=np.zeros_like(dcg),where=k10>0),
      'zero_positive':(positives==0).astype(float)
    }

def linear_coefficients(order,valid,k=10,reward=False):
    u,m=order.shape;a=np.zeros((u,m));sizes=valid.sum(axis=1)
    for row in range(u):
        kk=min(k,int(sizes[row]))
        if kk: a[row,order[row,:kk]]=1/(u*kk) if reward else DISCOUNT[:kk]/u
    return a
