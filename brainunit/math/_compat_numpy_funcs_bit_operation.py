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
from typing import (Union)

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from brainunit._misc import set_module_as
from .._base import Quantity

__all__ = [

  # Elementwise bit operations (unary)
  'bitwise_not', 'invert',

  # Elementwise bit operations (binary)
  'bitwise_and', 'bitwise_or', 'bitwise_xor', 'left_shift', 'right_shift',
]


# Elementwise bit operations (unary)
# ----------------------------------

def elementwise_bit_operation_unary(func, x, *args, **kwargs):
  if isinstance(x, Quantity):
    raise ValueError(f'Expected integers, got {x}')
  elif isinstance(x, (jax.Array, np.ndarray)):
    return func(x, *args, **kwargs)
  else:
    raise ValueError(f'Unsupported types {type(x)} for {func.__name__}')


@set_module_as('brainunit.math')
def bitwise_not(x: Union[Quantity, jax.typing.ArrayLike]) -> Array:
  """
  Compute the bit-wise NOT of an array, element-wise.

  Args:
    x: array_like

  Returns:
    jax.Array: an array
  """
  return elementwise_bit_operation_unary(jnp.bitwise_not, x)


@set_module_as('brainunit.math')
def invert(x: Union[Quantity, jax.typing.ArrayLike]) -> Array:
  """
  Compute bit-wise inversion, or bit-wise NOT, element-wise.

  Args:
    x: array_like

  Returns:
    jax.Array: an array
  """
  return elementwise_bit_operation_unary(jnp.invert, x)


# Elementwise bit operations (binary)
# -----------------------------------


def elementwise_bit_operation_binary(func, x, y, *args, **kwargs):
  if isinstance(x, Quantity) or isinstance(y, Quantity):
    raise ValueError(f'Expected integers, got {x} and {y}')
  elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray, int, float)):
    return func(x, y, *args, **kwargs)
  else:
    raise ValueError(f'Unsupported types {type(x)} and {type(y)} for {func.__name__}')


@set_module_as('brainunit.math')
def bitwise_and(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Compute the bit-wise AND of two arrays element-wise.

  Args:
    x: array_like
    y: array_like

  Returns:
    jax.Array: an array
  """
  return elementwise_bit_operation_binary(jnp.bitwise_and, x, y)


@set_module_as('brainunit.math')
def bitwise_or(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Compute the bit-wise OR of two arrays element-wise.

  Args:
    x: array_like
    y: array_like

  Returns:
    jax.Array: an array
  """
  return elementwise_bit_operation_binary(jnp.bitwise_or, x, y)


@set_module_as('brainunit.math')
def bitwise_xor(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Compute the bit-wise XOR of two arrays element-wise.

  Args:
    x: array_like
    y: array_like

  Returns:
    jax.Array: an array
  """
  return elementwise_bit_operation_binary(jnp.bitwise_xor, x, y)


@set_module_as('brainunit.math')
def left_shift(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Shift the bits of an integer to the left.

  Args:
    x: array_like
    y: array_like

  Returns:
    jax.Array: an array
  """
  return elementwise_bit_operation_binary(jnp.left_shift, x, y)


@set_module_as('brainunit.math')
def right_shift(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Shift the bits of an integer to the right.

  Args:
    x: array_like
    y: array_like

  Returns:
    jax.Array: an array
  """
  return elementwise_bit_operation_binary(jnp.right_shift, x, y)
