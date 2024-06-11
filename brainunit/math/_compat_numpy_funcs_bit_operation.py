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
from typing import (Union)

import brainstate as bst
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._base import (Quantity,
                     )

__all__ = [

  # Elementwise bit operations (unary)
  'bitwise_not', 'invert',

  # Elementwise bit operations (binary)
  'bitwise_and', 'bitwise_or', 'bitwise_xor', 'left_shift', 'right_shift',
]


# Elementwise bit operations (unary)
# ----------------------------------

def wrap_elementwise_bit_operation_unary(func):
  @wraps(func)
  def f(x, *args, **kwargs):
    if isinstance(x, Quantity):
      raise ValueError(f'Expected integers, got {x}')
    elif isinstance(x, (jax.Array, np.ndarray)):
      return func(x, *args, **kwargs)
    else:
      raise ValueError(f'Unsupported types {type(x)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


@wrap_elementwise_bit_operation_unary
def bitwise_not(x: Union[Quantity, bst.typing.ArrayLike]) -> Array:
  return jnp.bitwise_not(x)


@wrap_elementwise_bit_operation_unary
def invert(x: Union[Quantity, bst.typing.ArrayLike]) -> Array:
  return jnp.invert(x)


# docs for functions above
bitwise_not.__doc__ = '''
  Compute the bit-wise NOT of an array, element-wise.

  Args:
    x: array_like

  Returns:
    jax.Array: an array
'''

invert.__doc__ = '''
  Compute bit-wise inversion, or bit-wise NOT, element-wise.

  Args:
    x: array_like

  Returns:
    jax.Array: an array
'''


# Elementwise bit operations (binary)
# -----------------------------------

def wrap_elementwise_bit_operation_binary(func):
  @wraps(func)
  def f(x, y, *args, **kwargs):
    if isinstance(x, Quantity) or isinstance(y, Quantity):
      raise ValueError(f'Expected integers, got {x} and {y}')
    elif isinstance(x, bst.typing.ArrayLike) and isinstance(y, bst.typing.ArrayLike):
      return func(x, y, *args, **kwargs)
    else:
      raise ValueError(f'Unsupported types {type(x)} and {type(y)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


@wrap_elementwise_bit_operation_binary
def bitwise_and(x: Union[Quantity, bst.typing.ArrayLike], y: Union[Quantity, bst.typing.ArrayLike]) -> Array:
  return jnp.bitwise_and(x, y)


@wrap_elementwise_bit_operation_binary
def bitwise_or(x: Union[Quantity, bst.typing.ArrayLike], y: Union[Quantity, bst.typing.ArrayLike]) -> Array:
  return jnp.bitwise_or(x, y)


@wrap_elementwise_bit_operation_binary
def bitwise_xor(x: Union[Quantity, bst.typing.ArrayLike], y: Union[Quantity, bst.typing.ArrayLike]) -> Array:
  return jnp.bitwise_xor(x, y)


@wrap_elementwise_bit_operation_binary
def left_shift(x: Union[Quantity, bst.typing.ArrayLike], y: Union[Quantity, bst.typing.ArrayLike]) -> Array:
  return jnp.left_shift(x, y)


@wrap_elementwise_bit_operation_binary
def right_shift(x: Union[Quantity, bst.typing.ArrayLike], y: Union[Quantity, bst.typing.ArrayLike]) -> Array:
  return jnp.right_shift(x, y)


# docs for functions above
bitwise_and.__doc__ = '''
  Compute the bit-wise AND of two arrays element-wise.

  Args:
    x: array_like
    y: array_like

  Returns:
    jax.Array: an array
'''

bitwise_or.__doc__ = '''
  Compute the bit-wise OR of two arrays element-wise.

  Args:
    x: array_like
    y: array_like

  Returns:
    jax.Array: an array
'''

bitwise_xor.__doc__ = '''
  Compute the bit-wise XOR of two arrays element-wise.

  Args:
    x: array_like
    y: array_like

  Returns:
    jax.Array: an array
'''

left_shift.__doc__ = '''
  Shift the bits of an integer to the left.

  Args:
    x: array_like
    y: array_like

  Returns:
    jax.Array: an array
'''

right_shift.__doc__ = '''
  Shift the bits of an integer to the right.

  Args:
    x: array_like
    y: array_like

  Returns:
    jax.Array: an array
'''
