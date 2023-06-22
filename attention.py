import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, Shard, Replicate, distribute_tensor

import math

import os
import numpy as np

import torch
import torch.distributed as dist
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PairwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from torch.nn import functional as F

import copy

world_size = torch.cuda.device_count()

def print0(msg):
  if int(os.getenv('RANK')) == 0:
    print(msg)

class Config:
  block_size: int = 1024
  vocab_size: 50304
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768
  dropout: float = 0.0
  bias: bool = True

class Attention(nn.Module):
  def __init__(self, mesh, config):
    super().__init__()
    self.k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) 
    self.q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) 
    self.v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) 
    self.o = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) 

    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.dropout = config.dropout
    self.block_size = config.block_size

    self.mesh = mesh

    self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    self.attn_dropout = nn.Dropout(config.dropout)
    self.resid_dropout = nn.Dropout(config.dropout)

  def forward(self, x, sharded=True):
    B,T,C = x.size() # Batch, Sequence length, Embeding size

    if not sharded:
      q = self.q(x).split(self.n_embd, dim=2)[0].view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
      k = self.k(x).split(self.n_embd, dim=2)[0].view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
      v = self.v(x).split(self.n_embd, dim=2)[0].view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
      att = F.softmax(att, dim=-1)
      att = self.attn_dropout(att)
      y = att @ v
      y = y.transpose(1,2).contiguous().view(B,T,C)
      y = self.resid_dropout(self.o(y))
      return y

    print0(f'X size: {x.size()}')

    channel_head_size = C // self.n_head
    tp_size = world_size 
    _rank = int(os.getenv('RANK'))

    #print(f'Q weight at rank {_rank} of shape {self.q.weight.shape} and local shape {self.q.weight.to_local().shape}') # = {self.q.weight}')
    print(f'[Rank{_rank}] X (12, 1024, 768) x WQ (768, 768) = {self.q(x).shape}, local = {self.q(x).to_local().shape}')
    print(f'[Rank{_rank}] Q(x) = {self.q(x)}')

    #print0(self.q(x).split(self.n_embd // tp_size, dim=2)) # tuple of duplicates 
                                                            # however we do not want duplicates at this point

    # shard in colwise: [Shard(2)] due to the batch in dim=0
    q = self.q(x).redistribute(self.mesh, [Shard(2)]).view(B, T, self.n_head, C // self.n_head).transpose(1,2) # no need to divide by tp_size as we 'redistributed' the tensor 
    k = self.k(x).redistribute(self.mesh, [Shard(2)]).view(B, T, self.n_head, C // self.n_head).transpose(1,2) # no need to divide by tp_size as we 'redistributed' the tensor 
    v = self.v(x).redistribute(self.mesh, [Shard(2)]).view(B, T, self.n_head, C // self.n_head).transpose(1,2) # no need to divide by tp_size as we 'redistributed' the tensor 
    print(f'[Rank{_rank}] redistributed q shape: {q.to_local().shape}')
    #print(f'[Rank{_rank}] redistributed q value: {q}')

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    print0(f'att size: {att.shape}, att local size: {att.to_local().shape}') # again, here they seem to all-reduce the result again.. 
                                                                             # this is actually not I expected, but it makes easy to apply softmax 
    print0(f'bias size: {self.bias.shape}')

    """
    dbias = distribute_tensor(self.bias.view(self.block_size, self.block_size), device_mesh = mesh, placements=colwise).view(1,1,self.block_size, self.block_size)
    print0(f'Distributed Bias Shape: {dbias.shape}')
    print0(f'Attention Shape: {att.shape}')
    att = att.masked_fill(dbias[:,:,:T,:T] == 0, float('-inf'))
    """
    # TODO Need to apply softmax and dropout to 'Sharded tensor' not to 'Replicated tensor'
    replicated_bias = distribute_tensor(self.bias, device_mesh=mesh, placements=[Replicate()]).view(1,1,self.block_size, self.block_size)
    att = att.masked_fill(replicated_bias[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)

    # at this moment, att is Replicate()
    att = att.redistribute(self.mesh, [Shard(1)])  
    print0(f'Attn shape {att.shape}, local shape {att.to_local().shape}, att: {att}') 
    print0(f'v shape {v.shape}, v: {v}')
    y = att @ v
    y = y.transpose(1,2).contiguous().view(B,T,C)
    y = y.redistribute(self.mesh, [Shard(2)])
    print0(f'y shape {y.shape}, local shape: {y.to_local().shape}, y: {y}')
    print0(f'o shape {self.o.weight.shape}, local shape: {self.o.weight.to_local().shape}, o: {self.o.weight}')

    y = self.resid_dropout(self.o(y))
    # How do they handle multiplication of Replicate() (y) and Rowwise() (o)?

    return y


_rank = int(os.getenv('RANK'))
device = f'cuda:{_rank}'
device_type = 'cuda'
torch.cuda.set_device(device)
torch.backends.cuda.enable_mem_efficient_sdp(enabled=False)

dist.init_process_group(backend='nccl', rank=_rank, world_size=world_size)

mesh = DeviceMesh('cuda', list(range(world_size)))
config = Config()
model = Attention(mesh, config)

iter_max = 100

dataset = 'shakespeare_char'
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

torch.manual_seed(1337)
torch.random.manual_seed(1337)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

sequence_size = 1024
batch_size = 12
n_embd = 768

X = torch.rand(batch_size, sequence_size, n_embd)


sd = model.state_dict()

model_b = Attention(mesh, config)
model_b.load_state_dict(sd)

not_sharded = model_b(X, sharded=False).to(device) 
print (f'Not sharded: {not_sharded}')

dist.barrier()

# tp
parallelize_module(model, mesh, {'q': ColwiseParallel(),
                                 'k': ColwiseParallel(),
                                 'v': ColwiseParallel(),
                                 'o': ColwiseParallel()})

#for name, module in model.named_modules():
#  if name in ['k', 'q', 'v', 'o']:
#    print(name, module.weight)

sharded = model(X)
print (f'Sharded: {sharded}')

