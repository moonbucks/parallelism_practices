import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed.tensor.parallel as tpp
from torch.distributed._tensor import DTensor, DeviceMesh, Shard, Replicate, distribute_tensor
from torch.optim import AdamW

#import sys
#sys.path.append('/data/home/yro/practices/parallelism/transformers')
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers import T5Config, T5Model
from transformers import set_seed

import time
import os
import argparse

from mingpt.model import GPT
from addition_dataset import AdditionDataset

train_dataset = AdditionDataset(ndigit=2, split="train")
test_dataset = AdditionDataset(ndigit=2, split="test")

def print0(msg, rank):
  if rank == 0:
    print(msg)

def pp(model):
  return

def tp(model, args):
  from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel, ColwiseParallel, RowwiseParallel

  def parallelize_MLP_block(model, module_path, twod_mesh):
    block = model.get_submodule(module_path)
    parallelized_block = parallelize_module(
        module=block,
        device_mesh=twod_mesh,
        parallelize_plan={
            "wi": ColwiseParallel(),
            "wo": RowwiseParallel(),
        },
        tp_mesh_dim=0,
    )
    return parallelized_block

  def parallelize_Attn_block(model, module_path, twod_mesh):
      block = model.get_submodule(module_path)
      parallelized_block = parallelize_module(
          module=block,
          device_mesh=twod_mesh,
          parallelize_plan={
              "q": ColwiseParallel(),
              "k": ColwiseParallel(),
              "v": ColwiseParallel(),
              "o": RowwiseParallel(),
          },
          tp_mesh_dim=0,
      )
      return parallelized_block

  for name, module in model.named_modules():
    if type(module) == transformers.models.t5.modeling_t5.T5Attention:
      tpp.parallelize_module(module, mesh, {'q': ColwiseParallel(), 
                                            'k': ColwiseParallel(),
                                            'v': ColwiseParallel(),
                                            'o': RowwiseParallel()}) 

  model = model.to(args.device)
  mesh = DeviceMesh(args.device.type, list(range(args.world_size)))

  for name, module in model.named_modules():
    print0(name, args.rank) #, type(module))

  for i in range(6):
      block = parallelize_MLP_block(model, f"encoder.block.{i}.layer.{1}.DenseReluDense", mesh)
      block = parallelize_MLP_block(model, f"decoder.block.{i}.layer.{2}.DenseReluDense", mesh)
      block = parallelize_Attn_block(model, f"encoder.block.{i}.layer.{0}.SelfAttention", mesh)
      block = parallelize_Attn_block(model, f"decoder.block.{i}.layer.{0}.SelfAttention", mesh)
      block = parallelize_Attn_block(model, f"decoder.block.{i}.layer.{1}.EncDecAttention", mesh)

  return model

def pp_and_tp(model):
  return

def get_model(args):
  if args.name == 't5-small':
    config = AutoConfig.from_pretrained('t5-small')
    model = T5Model(config).to(args.device) 
    args.seq_length = config.n_positions # max sequence length
  elif args.name == 'gpt2':
    config = GPT.get_default_config()
    config.model_type = 'gpt2'
    config.vocab_size = 50257 # openai's model vocabulary
    config.block_size = 1024 
    args.seq_length = config.block_size
    model = GPT(config)

  return model, config

def run(args):
  torch.manual_seed(args.seed)

  model, config = get_model(args)

  # 1. tp
  model = tp(model, args)
  train(args, model, config)

  # 2. pp 
  #model = pp(model, args)

  pass

def train(args, model, config):
  if args.name == 't5-small':
    t5_train(args, model, config)
  elif args.name == 'gpt2':
    gpt2_train(args, model, config)
  else:
    assert False, 'no model given'

def gpt2_train(args, model, config):
  optimizer = AdamW(params=model.parameters(), lr=args.lr)
  for ep in range(0, args.epochs):
    optimizer.zero_grad()
    inp = torch.tensor([1, 2, 3, 4], dtype=torch.long).repeat(args.batch, 1).to(args.device)
    outputs = model(inp)
    print0(f'FWD epoch {ep}', args.rank)
    outputs.sum().backward()
    print0(f'BWD epoch {ep}', args.rank)
    optimizer.step()
    print0(f'OPT epoch {ep}', args.rank)
    print(f'epoch-{ep} rank-{args.rank} loss-{outputs.loss.item}')

def t5_train(args, model, config):
  optimizer = AdamW(params=model.parameters(), lr=args.lr)
  for ep in range(0, args.epochs):
    optimizer.zero_grad()
    x = torch.empty(args.batch, args.seq_length, dtype=torch.long, device=args.device).random_(config.vocab_size-1)
    y = torch.empty(args.batch, args.seq_length, dtype=torch.long, device=args.device).random_(config.vocab_size-1)
    d = { 'input_ids': x, 'decoder_input_ids': x} #, 'labels': y} 
    outputs = model(**d)
    outputs.loss.backward()
    optimizer.step()
    print(f'epoch-{ep} rank-{args.rank} loss-{outputs.loss.item}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', type=float, default=2e-5)
  parser.add_argument('--batch', type=int, default=16)
  parser.add_argument('--world_size', type=int, default=int(os.getenv('WORLD_SIZE', 8)))
  parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', '127.0.0.1'))
  parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
  parser.add_argument('--epochs', type=int, default=5)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--rank', type=int, default=int(os.getenv('RANK', -1)))
  parser.add_argument('--name', type=str, default='gpt2')

  args = parser.parse_args()

  if torch.cuda.is_available():
    dev_id = args.rank % torch.cuda.device_count()
    args.device = torch.device(f'cuda:{dev_id}')
    torch.cuda.set_device(args.device)
    backend = 'nccl'
    print(f'rank={args.rank}, device={dev_id}, world_size={args.world_size}, master_addr={args.master_addr}, master_port={args.master_port}')
  else:
    args.device = torch.device('cpu')
    backend = 'gloo'

  dist.init_process_group(backend, rank=args.rank, world_size=args.world_size)

  run(args)
