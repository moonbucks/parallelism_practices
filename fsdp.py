def fsdp_accelerator(model, args):
  # to run, use accelerator CLI
  # example: 
  #   accelerator config
  #   accelerator launch main.py
  # for more information: https://github.com/huggingface/accelerate

  from accelerate import Accelerator
  from datasets import load_dataset

  import evaluate

  set_seed(args.seed)
  def get_dataloaders(accelerator, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    dataset = load_dataset('glue', 'mrpc')

    def tokenize_function(examples):
      outputs = tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, max_length=None)
      return outputs

    with accelerator.main_process_first():
      tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=['idx', 'sentence1', 'sentence2'],
      )

    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

    def collate_fn(examples):
      return

    return train_dataloader, eval_dataloader

  metric = evaluate.glue('glue', 'mrpc')


  accelertor = Accelerator()
  device = accelerator.device
  model.to(device)
  model = accelerator.prepare(model)

  optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
  optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )


  data = torch.utils.data.DataLoader(dataset, shuffle=True)
  data = accelerator.prepare(data)

  model.train()
  time_taken = time.time()
  for epoch in range(args.epochs):
    for source, targets in data:
      source = source.to(device)
      targets = targets.to(device)

      optimizer.zero_grad()

      output = model(source)
      loss = F.cross_entropy(output, targets)

      accelerator.backward(loss)

      optimizer.step()

  time_taken = time_taken - time.time()

  # TODO save state_dict

  return time_taken

def fsdp_native(model, args):
  # to run, use torch launcher
  # example:
  #   python -m torchrun --nproc_per_node 2 --use_env main.py
  # example - multi node:
  #   python -m torchrun --nproc_per_node 2 --use_env --node_rank 0 --master_addr localhost main.py -- master node
  #   python -m torchrun --nproc_per_node 2 --use_env --node_rank 1 --master_addr localhost main.py -- second node

  from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

  return

def fsdp(model, args):
  t = fsdp_accelerator(model, args)
  #fsdp_native()
  return t


