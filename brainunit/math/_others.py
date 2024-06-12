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

import jax.numpy as jnp

from _misc import set_module_as
from ._compat_numpy_funcs_accept_unitless import funcs_only_accept_unitless_unary

__all__ = [
  'exprel',
]


def _exprel(x):
  # following the implementation of exprel from scipy.special
  x = jnp.asarray(x)
  dtype = x.dtype

  # Adjust the tolerance based on the dtype of x
  if dtype == jnp.float64:
    small_threshold = 1e-16
    big_threshold = 717
  elif dtype == jnp.float32:
    small_threshold = 1e-8
    big_threshold = 100
  elif dtype == jnp.float16:
    small_threshold = 1e-4
    big_threshold = 10
  else:
    small_threshold = 1e-4
    big_threshold = 10

  small = jnp.abs(x) < small_threshold
  big = x > big_threshold
  origin = jnp.expm1(x) / x
  return jnp.where(small, 1.0, jnp.where(big, jnp.inf, origin))


@set_module_as('brainunit.math')
def exprel(x):
  """
  Relative error exponential, ``(exp(x) - 1)/x``.

  When ``x`` is near zero, ``exp(x)`` is near 1, so the numerical calculation of ``exp(x) - 1`` can
  suffer from catastrophic loss of precision. ``exprel(x)`` is implemented to avoid the loss of
  precision that occurs when ``x`` is near zero.

  Args:
    x: ndarray. Input array. ``x`` must contain real numbers.

  Returns:
    ``(exp(x) - 1)/x``, computed element-wise.
  """
  return funcs_only_accept_unitless_unary(_exprel, x)
