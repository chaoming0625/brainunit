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

import jax
import jax.numpy as jnp
import numpy as np
from brainstate._utils import set_module_as
from jax import Array

from .._base import (Quantity,
                     fail_for_dimension_mismatch,
                     )

__all__ = [
  # logic funcs (unary)
  'all', 'any', 'logical_not',

  # logic funcs (binary)
  'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
  'array_equal', 'isclose', 'allclose', 'logical_and',
  'logical_or', 'logical_xor', "alltrue", 'sometrue',
]


# logic funcs (unary)
# -------------------


def logic_func_unary(func, x, *args, **kwargs):
  if isinstance(x, Quantity):
    raise ValueError(f'Expected booleans, got {x}')
  elif isinstance(x, (jax.Array, np.ndarray)):
    return func(x, *args, **kwargs)
  else:
    raise ValueError(f'Unsupported types {type(x)} for {func.__name__}')


@set_module_as('brainunit.math')
def all(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    out: Optional[Array] = None,
    keepdims: bool = False,
    where: Optional[Array] = None
) -> Union[bool, Array]:
  '''
  Test whether all array elements along a given axis evaluate to True.

  Args:
    a: array_like
    axis: int, optional
    out: array, optional
    keepdims: bool, optional
    where: array_like of bool, optional

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_unary(jnp.all, x, axis=axis, out=out, keepdims=keepdims, where=where)


@set_module_as('brainunit.math')
def any(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    out: Optional[Array] = None,
    keepdims: bool = False,
    where: Optional[Array] = None
) -> Union[bool, Array]:
  '''
  Test whether any array element along a given axis evaluates to True.

  Args:
    a: array_like
    axis: int, optional
    out: array, optional
    keepdims: bool, optional
    where: array_like of bool, optional

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_unary(jnp.any, x, axis=axis, out=out, keepdims=keepdims, where=where)


@set_module_as('brainunit.math')
def logical_not(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[bool, Array]:
  '''
  Compute the truth value of NOT x element-wise.

  Args:
    x: array_like
    out: array, optional

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_unary(jnp.logical_not, x)


alltrue = all
sometrue = any


# logic funcs (binary)
# --------------------


def logic_func_binary(func, x, y, *args, **kwargs):
  if isinstance(x, Quantity) and isinstance(y, Quantity):
    fail_for_dimension_mismatch(x, y)
    return func(x.value, y.value, *args, **kwargs)
  elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
    return func(x, y, *args, **kwargs)
  else:
    raise ValueError(f'Unsupported types {type(x)} and {type(y)} for {func.__name__}')


@set_module_as('brainunit.math')
def equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[bool, Array]:
  '''
  Return (x == y) element-wise and have the same unit if x and y are Quantity.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_binary(jnp.equal, x, y)


@set_module_as('brainunit.math')
def not_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[bool, Array]:
  '''
  Return (x != y) element-wise and have the same unit if x and y are Quantity.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_binary(jnp.not_equal, x, y)


@set_module_as('brainunit.math')
def greater(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[bool, Array]:
  '''
  Return (x > y) element-wise and have the same unit if x and y are Quantity.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_binary(jnp.greater, x, y)


@set_module_as('brainunit.math')
def greater_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[
  bool, Array]:
  '''
  Return (x >= y) element-wise and have the same unit if x and y are Quantity.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_binary(jnp.greater_equal, x, y)


@set_module_as('brainunit.math')
def less(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[bool, Array]:
  '''
  Return (x < y) element-wise and have the same unit if x and y are Quantity.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_binary(jnp.less, x, y)


@set_module_as('brainunit.math')
def less_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[
  bool, Array]:
  '''
  Return (x <= y) element-wise and have the same unit if x and y are Quantity.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_binary(jnp.less_equal, x, y)


@set_module_as('brainunit.math')
def array_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[
  bool, Array]:
  '''
  Return True if two arrays have the same shape, elements, and units (if they are Quantity), False otherwise.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_binary(jnp.array_equal, x, y)


@set_module_as('brainunit.math')
def isclose(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False
) -> Union[bool, Array]:
  '''
  Returns a boolean array where two arrays are element-wise equal within a tolerance and have the same unit if they are Quantity.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity
    rtol: float, optional
    atol: float, optional
    equal_nan: bool, optional

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_binary(jnp.isclose, x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)


@set_module_as('brainunit.math')
def allclose(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False
) -> Union[bool, Array]:
  '''
  Returns True if the two arrays are equal within the given tolerance and have the same unit if they are Quantity; False otherwise.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity
    rtol: float, optional
    atol: float, optional
    equal_nan: bool, optional

  Returns:
    bool: boolean result
  '''
  return logic_func_binary(jnp.allclose, x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)


@set_module_as('brainunit.math')
def logical_and(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[
  bool, Array]:
  '''
  Compute the truth value of x AND y element-wise and have the same unit if x and y are Quantity.

  Args:
    x: array_like
    y: array_like
    out: array, optional

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_binary(jnp.logical_and, x, y)


@set_module_as('brainunit.math')
def logical_or(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[
  bool, Array]:
  '''
  Compute the truth value of x OR y element-wise and have the same unit if x and y are Quantity.

  Args:
    x: array_like
    y: array_like
    out: array, optional

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_binary(jnp.logical_or, x, y)


@set_module_as('brainunit.math')
def logical_xor(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[
  bool, Array]:
  '''
  Compute the truth value of x XOR y element-wise and have the same unit if x and y are Quantity.

  Args:
    x: array_like
    y: array_like
    out: array, optional

  Returns:
    Union[bool, jax.Array]: bool or array
  '''
  return logic_func_binary(jnp.logical_xor, x, y)
