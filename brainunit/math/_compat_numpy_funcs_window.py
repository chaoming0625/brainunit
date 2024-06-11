# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from functools import wraps

import jax.numpy as jnp
from jax import Array

__all__ = [

  # window funcs
  'bartlett', 'blackman', 'hamming', 'hanning', 'kaiser',
]


# window funcs
# ------------

def wrap_window_funcs(func):
  @wraps(func)
  def f(*args, **kwargs):
    return func(*args, **kwargs)

  f.__module__ = 'brainunit.math'
  return f


@wrap_window_funcs
def bartlett(M: int) -> Array:
  return jnp.bartlett(M)


@wrap_window_funcs
def blackman(M: int) -> Array:
  return jnp.blackman(M)


@wrap_window_funcs
def hamming(M: int) -> Array:
  return jnp.hamming(M)


@wrap_window_funcs
def hanning(M: int) -> Array:
  return jnp.hanning(M)


@wrap_window_funcs
def kaiser(M: int, beta: float) -> Array:
  return jnp.kaiser(M, beta)


# docs for functions above
bartlett.__doc__ = jnp.bartlett.__doc__
blackman.__doc__ = jnp.blackman.__doc__
hamming.__doc__ = jnp.hamming.__doc__
hanning.__doc__ = jnp.hanning.__doc__
kaiser.__doc__ = jnp.kaiser.__doc__
