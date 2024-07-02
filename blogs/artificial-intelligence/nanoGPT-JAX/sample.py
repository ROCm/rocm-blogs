# Originally written by Andrej Karpathy in PyTorch; adapted by Douglas Jia to JAX.
"""
Sample from a trained model
"""
import os
import pickle
import jax
from pathlib import Path
import jax.numpy as jnp
import tiktoken
import orbax.checkpoint as ocp
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-shakespeare-char' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 3 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
override_args={}
exec(open('configurator.py').read()) # overrides from command line or config file

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
    check_path = Path(os.getcwd(), out_dir, 'checkpoint')
    checkpoint_manager = ocp.CheckpointManager(check_path, options=check_options, item_names=('state', 'metadata'))
    with ocp.CheckpointManager(check_path, options=check_options, item_names=('state', 'metadata')) as mngr:
        restored_ckpt = mngr.restore(mngr.latest_step(),args=ocp.args.Composite(state=ocp.args.StandardRestore(), metadata=ocp.args.JsonRestore()))
    model_args_ckpt, iter_num_ckpt, best_val_loss_ckpt, _, config_ckpt = restored_ckpt['metadata']
    state = restored_ckpt['state']
    # create the model
    gptconf = GPTConfig(**model_args_ckpt)
    model = GPT(gptconf)
    params = {'params': state['params']}

elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model, params = GPT.from_pretrained(init_from, override_args=override_args)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'dataset' in config_ckpt:
    meta_path = os.path.join('data', config_ckpt['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

@jax.jit
def generate(idx, seed):
    """
    Take a conditioning batch of sequences idx (numpy.ndarry of shape (b,) where b is the batch size 
    and each element in the array correspond to a sequence of tokens) and complete
    each sequence max_new_tokens times, feeding the predictions back into the model each time.
    Please note that this implementation is different from Andrej Karpathy's original implementation
    which requires the sequences to be of the same length.
    """
    sequence_len = list(map(len, idx))
    max_context_len = max(sequence_len)
    # Pad the input sequence with a special padding token (e.g. 0, or 50256 which is the end of sequence token) so that they have the same length
    idx = jnp.array([jnp.pad(jnp.array(sub_arr), (max_context_len + max_new_tokens- len(sub_arr), 0), mode='constant', constant_values=0)
                    for sub_arr in idx])
    
    def body_fn(i, idx):
        logits, _ = model.apply(params, idx[:,-model.config.block_size:], train=False)
        logits = logits[:, -1, :] / temperature #shape (b, 1)
        if top_k is not None:
            logits_, _ = jax.lax.top_k(logits, min(top_k, logits.shape[-1])) #0: value, 1: index; descending order
            logits = jnp.where(logits < jnp.expand_dims(logits_[:, -1], -1), -jnp.inf, logits)
        idx_next = jax.random.categorical(jax.random.PRNGKey(i+seed), logits=logits)
        idx = jnp.concatenate([idx[:, 1:], jnp.expand_dims(idx_next, -1)], axis=1)
        return idx
    idx = jax.lax.fori_loop(0, max_new_tokens, body_fn, idx)
    idx = [ind[(max_context_len-l):] for l, ind in zip(list(sequence_len), list(idx))]
    return idx

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)

for i in range(num_samples):
    output = generate([jnp.array(start_ids)], seed+i)
    print(f'\nGenerated output __{i}__: \n__________________________________\n{decode(output[0].tolist())}\n__________________________________')
