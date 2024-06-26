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
from __future__ import annotations

from typing import (Union)

import jax
import jax.numpy as jnp
from jax import Array

from .._base import Quantity
from .._misc import set_module_as

__all__ = [
  # Elementwise bit operations (unary)
  'bitwise_not', 'invert',

  # Elementwise bit operations (binary)
  'bitwise_and', 'bitwise_or', 'bitwise_xor', 'left_shift', 'right_shift',
]


# Elementwise bit operations (unary)
# ----------------------------------

def _fun_unitless_unary(func, x, *args, **kwargs):
  if isinstance(x, Quantity):
    assert x.is_unitless, f'Expected unitless array, got {x}'
    x = x.value
  return func(x, *args, **kwargs)


@set_module_as('brainunit.math')
def bitwise_not(x: Union[Quantity, jax.typing.ArrayLike]) -> Array:
  """
  Compute the bit-wise NOT of an array, element-wise.

  Parameters
  ----------
  x: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_unary(jnp.bitwise_not, x)


@set_module_as('brainunit.math')
def invert(x: Union[Quantity, jax.typing.ArrayLike]) -> Array:
  """
  Compute bit-wise inversion, or bit-wise NOT, element-wise.

  Parameters
  ----------
  x: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_unary(jnp.invert, x)


# Elementwise bit operations (binary)
# -----------------------------------


def _fun_unitless_binary(func, x, y, *args, **kwargs):
  if isinstance(x, Quantity):
    assert x.is_unitless, f'Expected unitless array, got {x}'
    x = x.value
  if isinstance(y, Quantity):
    assert y.is_unitless, f'Expected unitless array, got {y}'
    y = y.value
  return func(x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def bitwise_and(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Compute the bit-wise AND of two arrays element-wise.

  Parameters
  ----------
  x: array_like, quantity
    Input array.
  y: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_binary(jnp.bitwise_and, x, y)


@set_module_as('brainunit.math')
def bitwise_or(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Compute the bit-wise OR of two arrays element-wise.

  Parameters
  ----------
  x: array_like, quantity
    Input array.
  y: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_binary(jnp.bitwise_or, x, y)


@set_module_as('brainunit.math')
def bitwise_xor(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Compute the bit-wise XOR of two arrays element-wise.

  Parameters
  ----------
  x: array_like, quantity
    Input array.
  y: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_binary(jnp.bitwise_xor, x, y)


@set_module_as('brainunit.math')
def left_shift(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Shift the bits of an integer to the left.

  Parameters
  ----------
  x: array_like, quantity
    Input array.
  y: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_binary(jnp.left_shift, x, y)


@set_module_as('brainunit.math')
def right_shift(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Shift the bits of an integer to the right.

  Parameters
  ----------
  x: array_like, quantity
    Input array.
  y: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_binary(jnp.right_shift, x, y)
