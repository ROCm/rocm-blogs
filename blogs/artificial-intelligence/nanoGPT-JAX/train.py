# Originally written by Andrej Karpathy in PyTorch; adapted by Douglas Jia to JAX.
"""
This script can be run on systems with GPUs.
You can refer to this blog on how to run pre-training, fine-tuning
and text generation with this script: https://rocm.blogs.amd.com/artificial-intelligence/nanoGPT-JAX/README.
"""
import os
import jax
import optax
import pickle
import time
import numpy as np
from pathlib import Path
from flax.training import train_state
from functools import partial
import jax.numpy as jnp
import orbax.checkpoint as ocp
# Define the GPT model and configuration
from model import GPTConfig, GPT
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt-fine-tune'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 5
eval_iters = 100
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# data
dataset = 'openwebtext'
batch_size = 24
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
bias = False # do we use bias inside LayerNorm and Linear layers?
use_einsum=False # Use einsum in attention calculation?
# adamw optimizer
learning_rate = 1e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# -----------------------------------------------------------------------------
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k:v for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))}
# -----------------------------------------------------------------------------
# data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))
    x = jnp.stack([data[i:i+block_size].astype(jnp.int32) for i in ix], axis=0)
    y = jnp.stack([data[i+1:i+1+block_size].astype(jnp.int32) for i in ix], axis=0)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

#checkpointing
check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
check_path = Path(os.getcwd(), out_dir, 'checkpoint')
checkpoint_manager = ocp.CheckpointManager(check_path, options=check_options, item_names=('state', 'metadata'))

# Define function to initialize the training state.
def init_train_state(model, params, learning_rate, weight_decay=None, beta1=None, beta2=None, decay_lr=True, warmup_iters=None, 
                     lr_decay_iters=None, min_lr=None) -> train_state.TrainState:
    # learning rate decay scheduler (cosine with warmup)
    if decay_lr:
        assert warmup_iters is not None, "warmup_iters must be provided"
        assert lr_decay_iters is not None, "lr_decay_iters must be provided"
        assert min_lr is not None, "min_lr must be provided"
        assert lr_decay_iters >= warmup_iters, "lr_decay_iters must be greater than or equal to warmup_iters"
        lr_schedule = optax.warmup_cosine_decay_schedule(init_value=1e-9, peak_value=learning_rate,
                                                         warmup_steps=warmup_iters, decay_steps=lr_decay_iters, end_value=min_lr)
    else:
        lr_schedule = learning_rate
    # Create the optimizer
    naive_optimizer = model.configure_optimizers(params, weight_decay=weight_decay, learning_rate=lr_schedule, betas=(beta1, beta2))
    # Add gradient clipping
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), naive_optimizer)
    # Create a State
    return train_state.TrainState.create(
        apply_fn = model.apply,
        tx=optimizer,
        params=params
    ), lr_schedule

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, use_einsum=use_einsum) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    idx = jnp.ones((3, gptconf.block_size), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(1), idx)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    latest_step = checkpoint_manager.latest_step()
    assert latest_step is not None, "No checkpoint found"
    # restore checkpoint from the latest step in the checkpoint folder.
    # this restored_ckpt only contains restored metadata
    
    with ocp.CheckpointManager(check_path, options=check_options, item_names=('state', 'metadata')) as mngr:
        restored_ckpt = mngr.restore(mngr.latest_step(),args=ocp.args.Composite(metadata=ocp.args.JsonRestore()))
    model_args_ckpt, iter_num_ckpt, best_val_loss_ckpt, _, config_ckpt = restored_ckpt['metadata']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = model_args_ckpt[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    idx = jnp.ones((3, gptconf.block_size), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(1), idx)
    iter_num = iter_num_ckpt+1
    best_val_loss = best_val_loss_ckpt
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout, use_einsum=use_einsum)
    model, params = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    raise RuntimeError(f"init_from={init_from} is not supported.")

state, lr_schedule = init_train_state(model, params['params'], learning_rate, weight_decay, beta1, beta2, decay_lr, warmup_iters, 
                     lr_decay_iters, min_lr)
if init_from == "resume":
    with ocp.CheckpointManager(check_path, options=check_options, item_names=('state', 'metadata')) as mngr:
        state = mngr.restore(mngr.latest_step(), args=ocp.args.Composite(state=ocp.args.StandardRestore(state)))['state']

@partial(jax.jit, static_argnames=('train'))
def loss_fn(params, x, targets=None, train=True, rng=jax.random.key(0)):
    _, loss = state.apply_fn({'params': params}, x, targets=targets, train=train, rng=rng)
    return loss


@partial(jax.jit, donate_argnums=(0,))
def train_step(state: train_state.TrainState, batch: jnp.ndarray, rng: jnp.ndarray=jax.random.key(0)):
    x, y = batch
    key0, key1 = jax.random.split(rng)
    gradient_fn = jax.value_and_grad(loss_fn)
    loss, grads = gradient_fn(state.params, x, targets=y, rng=key0, train=True)
    state = state.apply_gradients(grads=grads)
    return state, loss, key1

def estimate_loss(state):
    result = {}
    for split in ['train', 'val']:
        loss = np.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            loss[i] = loss_fn(state.params, x, targets=y, train=False)
        result[split] = loss.mean()
    return result

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
print(f"Data is on device: {X.devices()}") #check the device is cpu or gpu.
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
# running_mfu = -1
rng_key = jax.random.key(1)

while True:
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        print(f'Evaluating at iter_num == {iter_num}...')
        losses = estimate_loss(state)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}; best val loss to now: {min(losses['val'].item(), best_val_loss):.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": float(lr_schedule(state.step)) if callable(lr_schedule) else lr_schedule,
            })
        if losses['val'].item() < best_val_loss or always_save_checkpoint:
            best_val_loss = min(losses['val'].item(), best_val_loss)
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}")
                with ocp.CheckpointManager(check_path, options=check_options, item_names=('state', 'metadata')) as mngr:
                    mngr.save(
                        step=iter_num,
                        args=ocp.args.Composite(
                            state=ocp.args.StandardSave(state),
                            metadata=ocp.args.JsonSave((model_args, iter_num, best_val_loss, losses['val'].item(), config))))
            
    if iter_num == 0 and eval_only:
        break

    state, loss, rng_key = train_step(state, get_batch('train'), rng_key)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() 
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break