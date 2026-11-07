"""Small trained prototypes. SSD is a CPU reference, not an optimized Mamba checkpoint."""
import hashlib
import re
import torch
from torch import nn
import torch.nn.functional as F

def tokens(text):
    return re.findall(r'[a-z_]+|\d|[^\w\s]',text.lower())

def encode(text):
    result=[1+int.from_bytes(hashlib.blake2s(t.encode(),digest_size=4).digest(),'little')%4095 for t in tokens(text)]
    assert len(result)<=256,len(result)
    return result

def batch_encode(rows):
    seqs=[encode(r['prompt']) for r in rows]
    out=torch.zeros(len(seqs),max(map(len,seqs)),dtype=torch.long)
    for i,s in enumerate(seqs): out[i,:len(s)]=torch.tensor(s)
    return out

class SSD(nn.Module):
    """Input-dependent state-space recurrence, evaluated in parallel by its exact kernel.

    S_t=exp(A*dt_t)S_{t-1}+dt_t B_t x_t, y_t=C_t S_t+D x_t.
    Per-head A, input-dependent B/C/dt; causal depthwise conv and gated output.
    Dense O(T^2) CPU reference is used here: do not claim production Mamba speed.
    """
    def __init__(self,d=48,h=4,n=8):
        super().__init__(); self.h=h; self.n=n; self.p=d//h
        self.norm=nn.LayerNorm(d)
        self.proj=nn.Linear(d,2*d+2*h*n+h)
        self.conv=nn.Conv1d(d,d,3,groups=d,padding=2)
        self.logA=nn.Parameter(torch.zeros(h))
        self.D=nn.Parameter(torch.ones(h))
        self.out=nn.Linear(d,d)
    def forward(self,x):
        b,t,d=x.shape; h,n,p=self.h,self.n,self.p
        a,z,B,C,dt=self.proj(self.norm(x)).split([d,d,h*n,h*n,h],-1)
        a=F.silu(self.conv(a.transpose(1,2))[:,:,:t].transpose(1,2)).reshape(b,t,h,p).transpose(1,2)
        B=B.reshape(b,t,h,n).transpose(1,2); C=C.reshape(b,t,h,n).transpose(1,2)
        dt=F.softplus(dt.transpose(1,2)-3)
        cumulative=(-self.logA.exp()[None,:,None]*dt).cumsum(-1)
        causal=torch.ones(t,t,dtype=torch.bool,device=x.device).tril()
        log_kernel=cumulative[:,:,:,None]-cumulative[:,:,None,:]
        decay=log_kernel.masked_fill(~causal,-torch.inf).exp()
        coupling=torch.matmul(C,B.transpose(-1,-2))
        y=torch.matmul(decay*coupling*dt[:,:,None,:],a)+self.D[None,:,None,None]*a
        y=y.transpose(1,2).reshape(b,t,d)*F.silu(z)
        return x+self.out(y)

class TCN(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.layers=nn.ModuleList([nn.Conv1d(d,d,3,dilation=2**i,padding=2*2**i) for i in range(7)])
    def forward(self,x):
        x=x.transpose(1,2); t=x.shape[-1]
        for layer in self.layers: x=x+F.gelu(layer(x)[:,:,:t])
        return x.transpose(1,2)

class EventGNN(nn.Module):
    """Graph over actual chronological successors, not inferred service causality."""
    def __init__(self,d):
        super().__init__()
        self.self_layers=nn.ModuleList([nn.Linear(d,d) for _ in range(3)])
        self.neighbor_layers=nn.ModuleList([nn.Linear(d,d) for _ in range(3)])
    def forward(self,x):
        for own,other in zip(self.self_layers,self.neighbor_layers):
            previous=F.pad(x[:,:-1],(0,0,1,0))
            following=F.pad(x[:,1:],(0,0,0,1))
            x=F.gelu(own(x)+other((previous+following)/2))
        return x

class Classifier(nn.Module):
    def __init__(self,kind,d=48,input_dim=None,classes=3):
        super().__init__(); self.kind=kind
        self.embedding=nn.Embedding(4096,d,padding_idx=0) if input_dim is None else nn.Linear(input_dim,d)
        if kind in ('gru','lstm'):
            self.body={'gru':nn.GRU,'lstm':nn.LSTM}[kind](d,d,batch_first=True)
        elif kind=='tcn': self.body=TCN(d)
        elif kind=='ssd': self.body=nn.Sequential(SSD(d),SSD(d))
        elif kind=='gnn': self.body=EventGNN(d)
        else: self.body=nn.Identity()
        self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,classes))
    def forward(self,ids):
        mask=ids.ne(0) if ids.ndim==2 else torch.ones(ids.shape[:2],dtype=torch.bool,device=ids.device)
        x=self.embedding(ids)
        x=self.body(x)
        if isinstance(x,tuple): x=x[0]
        mean=(x*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        last=x[torch.arange(len(x)),mask.sum(1)-1]
        return self.head(torch.cat([mean,last],-1))
