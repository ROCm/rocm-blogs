from dataclasses import dataclass
import jmp
from jax import numpy as jnp
from jmp._src.policy import _cast_floating_to

@dataclass(frozen=True)
class Policy(jmp.Policy):
    reduce_ops_dtype: jnp.dtype

    def cast_to_reduce_ops(self, x):
        return _cast_floating_to(x, self.reduce_ops_dtype)