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
from typing import (Union, Optional)

import jax.numpy as jnp
from brainstate._utils import set_module_as
from jax import Array

from .._base import (Quantity,
                     )

__all__ = [

  # math funcs remove unit (unary)
  'signbit', 'sign', 'histogram', 'bincount',

  # math funcs remove unit (binary)
  'corrcoef', 'correlate', 'cov', 'digitize',
]


# math funcs remove unit (unary)
# ------------------------------

def funcs_remove_unit_unary(func, x, *args, **kwargs):
  if isinstance(x, Quantity):
    return func(x.value, *args, **kwargs)
  else:
    return func(x, *args, **kwargs)


@set_module_as('brainunit.math')
def signbit(x: Union[Array, Quantity]) -> Array:
  '''
  Returns element-wise True where signbit is set (less than zero).

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  '''
  return funcs_remove_unit_unary(jnp.signbit, x)


@set_module_as('brainunit.math')
def sign(x: Union[Array, Quantity]) -> Array:
  '''
  Returns the sign of each element in the input array.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  '''
  return funcs_remove_unit_unary(jnp.sign, x)


@set_module_as('brainunit.math')
def histogram(x: Union[Array, Quantity]) -> tuple[Array, Array]:
  '''
  Compute the histogram of a set of data.

  Args:
    x: array_like, Quantity

  Returns:
    tuple[jax.Array]: Tuple of arrays (hist, bin_edges)
  '''
  return funcs_remove_unit_unary(jnp.histogram, x)


@set_module_as('brainunit.math')
def bincount(x: Union[Array, Quantity]) -> Array:
  '''
  Count number of occurrences of each value in array of non-negative integers.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  '''
  return funcs_remove_unit_unary(jnp.bincount, x)


# math funcs remove unit (binary)
# -------------------------------
def funcs_remove_unit_binary(func, x, y, *args, **kwargs):
  if isinstance(x, Quantity):
    x_value = x.value
  if isinstance(y, Quantity):
    y_value = y.value
  if isinstance(x, Quantity) or isinstance(y, Quantity):
    return func(jnp.array(x_value), jnp.array(y_value), *args, **kwargs)
  else:
    return func(x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def corrcoef(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
  '''
  Return Pearson product-moment correlation coefficients.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    jax.Array: an array
  '''
  return funcs_remove_unit_binary(jnp.corrcoef, x, y)


@set_module_as('brainunit.math')
def correlate(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
  '''
  Cross-correlation of two sequences.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    jax.Array: an array
  '''
  return funcs_remove_unit_binary(jnp.correlate, x, y)


@set_module_as('brainunit.math')
def cov(x: Union[Array, Quantity], y: Optional[Union[Array, Quantity]] = None) -> Array:
  '''
  Covariance matrix.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity (optional, if not provided, x is assumed to be a 2D array)

  Returns:
    jax.Array: an array
  '''
  return funcs_remove_unit_binary(jnp.cov, x, y)


@set_module_as('brainunit.math')
def digitize(x: Union[Array, Quantity], bins: Union[Array, Quantity]) -> Array:
  '''
  Return the indices of the bins to which each value in input array belongs.

  Args:
    x: array_like, Quantity
    bins: array_like, Quantity

  Returns:
    jax.Array: an array
  '''
  return funcs_remove_unit_binary(jnp.digitize, x, bins)
