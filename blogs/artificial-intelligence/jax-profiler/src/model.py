import jax
import optax
import flax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_einsum: bool = False # Whether to ues Einstein summation or jnp.matmul for matrix multiplication in self attention calculation

class CausalSelfAttention(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, train=False, rng1=None, rng2=None):
        assert self.config.n_embd % self.config.n_head == 0
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = jnp.split(nn.Dense(self.config.n_embd * 3, name="c_attn")(x), 3, axis=-1)
        k = k.reshape(B, T, self.config.n_head, C // self.config.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        q = q.reshape(B, T, self.config.n_head, C // self.config.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        v = v.reshape(B, T, self.config.n_head, C // self.config.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        with jax.named_scope("attn_q_k"):
            att = (jnp.einsum('bhts,bhqs->bhtq', q, k, optimize=True) if self.config.use_einsum else jnp.matmul(q, k.swapaxes(-2, -1))) * (1.0 / jnp.sqrt(k.shape[-1]))
        mask = jnp.tril(jnp.ones((T, T))).reshape((1, 1, T, T))
        att = jnp.where(mask == 0, float('-inf'), att)
        att = nn.softmax(att, axis=-1)
        att = nn.Dropout(self.config.dropout, name='attn_dropout', deterministic=not train)(att, rng=rng1)
        with jax.named_scope("attn_att_v"):
            y = jnp.einsum('bhts,bhsq->bhtq', att, v, optimize=True) if self.config.use_einsum else jnp.matmul(att, v)   # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.swapaxes(1, 2).reshape(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = nn.Dense(self.config.n_embd, name='c_proj')(y)
        y = nn.Dropout(self.config.dropout, name='resid_dropout', deterministic=not train)(y, rng=rng2)

        return y
    
class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, train=False, rng=None):
        x = nn.Dense(4 * self.config.n_embd, use_bias=self.config.bias)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.config.n_embd, use_bias=self.config.bias)(x)
        x = nn.Dropout(self.config.dropout, deterministic=not train)(x, rng=rng)
        return x
    
class Block(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, train=False, rng1=None, rng2=None, rng3=None):        
        x = x + CausalSelfAttention(self.config, name='attn')(nn.LayerNorm(use_bias=self.config.bias, name='ln_1')(x), train=train, rng1=rng1, rng2=rng2)
        x = x + MLP(self.config, name='mlp')(nn.LayerNorm(use_bias=self.config.bias, name='ln_2')(x), train=train, rng=rng3)
        return x

class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        assert self.config.vocab_size is not None
        assert self.config.block_size is not None
        
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embed(self.config.block_size, self.config.n_embd)
        self.drop = nn.Dropout(self.config.dropout)
        self.h = [Block(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm(use_bias=self.config.bias)

    def __call__(self, idx, targets=None, train=False, rng=jax.random.key(0)):
        _, t = idx.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.arange(t, dtype=jnp.int32)

        # forward the GPT model itself
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (t, n_embd)

        rng0, rng1, rng2, rng3 = jax.random.split(rng, 4)
        x = self.drop(tok_emb + pos_emb, deterministic=False, rng=rng0) 
        for block in self.h:
            x = block(x, train=train, rng1=rng1, rng2=rng2, rng3=rng3)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.wte.attend(x) # (b, t, vocab_size); implements weight tying (https://github.com/google/flax/discussions/2186)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        else:
            logits = self.wte.attend(x[:, -1:, :])
            loss = None

        return logits, loss
    
    @staticmethod
    def get_num_params(params, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.size for p in jax.tree.leaves(params))
        if non_embedding:
            wpe_params = params['wpe']['embedding']
            n_params -= wpe_params.size
        return n_params
    
    # TODO: imlement model sugery method so that it can fit smaller block sizes.

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k in ('dropout', 'use_einsum') for k in override_args)
        from transformers import FlaxGPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        if 'use_einsum' in override_args:
            print(f"overriding use_einsum to {override_args['use_einsum']}")
            config_args['use_einsum'] = override_args['use_einsum']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        idx = jnp.ones((3, config.block_size), dtype=jnp.int32)
        params = model.init(jax.random.PRNGKey(1), idx, train=False)

        # init a huggingface/transformers model
        model_hf = FlaxGPT2LMHeadModel.from_pretrained(model_type)

        #Update parameters of the Flax model.
        params['params']['wpe'] = model_hf.params['transformer']['wpe']
        params['params']['wte'] = model_hf.params['transformer']['wte']
        params['params']['ln_f'] = model_hf.params['transformer']['ln_f']

        for i in jnp.arange(config.n_layer):
            params['params']['h_'+str(i)]['ln_1'] = model_hf.params['transformer']['h'][str(i)]['ln_1']
            params['params']['h_'+str(i)]['ln_2'] = model_hf.params['transformer']['h'][str(i)]['ln_2']
            params['params']['h_'+str(i)]['mlp']['Dense_0']['bias'] = model_hf.params['transformer']['h'][str(i)]['mlp']['c_fc']['bias']
            params['params']['h_'+str(i)]['mlp']['Dense_1']['bias'] = model_hf.params['transformer']['h'][str(i)]['mlp']['c_proj']['bias']
            params['params']['h_'+str(i)]['mlp']['Dense_0']['kernel'] = model_hf.params['transformer']['h'][str(i)]['mlp']['c_fc']['kernel'].T
            params['params']['h_'+str(i)]['mlp']['Dense_1']['kernel'] = model_hf.params['transformer']['h'][str(i)]['mlp']['c_proj']['kernel'].T
            params['params']['h_'+str(i)]['attn']['c_attn']['bias'] = model_hf.params['transformer']['h'][str(i)]['attn']['c_attn']['bias']
            params['params']['h_'+str(i)]['attn']['c_proj']['bias'] = model_hf.params['transformer']['h'][str(i)]['attn']['c_proj']['bias']
            params['params']['h_'+str(i)]['attn']['c_attn']['kernel'] = model_hf.params['transformer']['h'][str(i)]['attn']['c_attn']['kernel'].T
            params['params']['h_'+str(i)]['attn']['c_proj']['kernel'] = model_hf.params['transformer']['h'][str(i)]['attn']['c_proj']['kernel'].T
        return model, params

    def configure_optimizers(self, params, weight_decay, learning_rate, betas):

        # Only implement weight decay on weights that are involved in matmul.
        # Reference: https://stats.stackexchange.com/questions/576463/why-not-perform-weight-decay-on-layernorm-embedding
        label_fn = lambda path, value: "no_decay" if (value.ndim < 2) or ('embedding' in path) else "decay"

        # Create optimization groups
        decay_opt = optax.adamw(learning_rate, weight_decay=weight_decay, b1=betas[0], b2=betas[1])
        nodecay_opt = optax.adam(learning_rate, b1=betas[0], b2=betas[1])

        tx = optax.multi_transform({
            'decay': decay_opt,
            'no_decay': nodecay_opt
        }, flax.traverse_util.path_aware_map(label_fn, params))

        return tx
    
    def estimate_mfu(self, params, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params(params)
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu