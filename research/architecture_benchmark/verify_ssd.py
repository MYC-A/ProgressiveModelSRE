"""Independent recurrence control for the dense selective-state reference."""
import torch
import torch.nn.functional as F
from networks import SSD
from common import ROOT,dump

torch.manual_seed(77)
model=SSD(d=16,h=2,n=3).eval()
x=torch.randn(2,17,16)
captured={}
def save_projection(m,a,o): captured['projection']=o.detach()
def save_convolution(m,a,o): captured['convolution']=o.detach()
handles=[model.proj.register_forward_hook(save_projection),model.conv.register_forward_hook(save_convolution)]
with torch.no_grad():
    actual=model(x)
    for handle in handles: handle.remove()
    b,t,d=x.shape; h,n,p=model.h,model.n,model.p
    _,gate,B,C,dt=captured['projection'].split([d,d,h*n,h*n,h],-1)
    values=F.silu(captured['convolution'][:,:,:t].transpose(1,2)).reshape(b,t,h,p)
    B=B.reshape(b,t,h,n); C=C.reshape(b,t,h,n); dt=F.softplus(dt-3)
    state=torch.zeros(b,h,n,p); outputs=[]
    for i in range(t):
        decay=torch.exp(-model.logA.exp()[None,:]*dt[:,i])
        state=decay[:,:,None,None]*state+dt[:,i,:,None,None]*B[:,i,:,:,None]*values[:,i,:,None,:]
        y=(C[:,i,:,:,None]*state).sum(-2)+model.D[None,:,None]*values[:,i]
        outputs.append(y.reshape(b,d)*F.silu(gate[:,i]))
    reference=x+model.out(torch.stack(outputs,1))
    error=float((actual-reference).abs().max())
    causality=float((model(x[:,:8])-actual[:,:8]).abs().max())
assert error<1e-5,error
assert causality<1e-5,causality
model(x.requires_grad_()).square().mean().backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
dump(ROOT/'ssd_verification.json',dict(max_recurrence_error=error,max_prefix_error=causality,
    gradients_finite=True,scope='CPU reference SSD prototype; not official optimized Mamba-2 kernels'))
print('VERIFIED_SSD',error,causality)
