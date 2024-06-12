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

import jax.numpy as jnp
from jax import Array

from brainunit._misc import set_module_as
from ._compat_numpy_funcs_change_unit import funcs_change_unit_binary
from ._compat_numpy_funcs_keep_unit import funcs_keep_unit_unary
from .._base import Quantity

__all__ = [

  # linear algebra
  'dot', 'vdot', 'inner', 'outer', 'kron', 'matmul', 'trace',

]


# linear algebra
# --------------

@set_module_as('brainunit.math')
def dot(a: Union[Array, Quantity], b: Union[Array, Quantity]) -> Union[Array, Quantity]:
  """
  Dot product of two arrays or quantities.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `x` and the unit of `y`, else an array.
  """
  return funcs_change_unit_binary(jnp.dot,
                                  lambda x, y: x * y,
                                  a, b)


@set_module_as('brainunit.math')
def vdot(a: Union[Array, Quantity], b: Union[Array, Quantity]) -> Union[Array, Quantity]:
  """
  Return the dot product of two vectors or quantities.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
  """
  return funcs_change_unit_binary(jnp.vdot,
                                  lambda x, y: x * y,
                                  a, b)


@set_module_as('brainunit.math')
def inner(a: Union[Array, Quantity], b: Union[Array, Quantity]) -> Union[Array, Quantity]:
  """
  Inner product of two arrays or quantities.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
  """
  return funcs_change_unit_binary(jnp.inner,
                                  lambda x, y: x * y,
                                  a, b)


@set_module_as('brainunit.math')
def outer(a: Union[Array, Quantity], b: Union[Array, Quantity]) -> Union[Array, Quantity]:
  """
  Compute the outer product of two vectors or quantities.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
  """
  return funcs_change_unit_binary(jnp.outer,
                                  lambda x, y: x * y,
                                  a, b)


@set_module_as('brainunit.math')
def kron(a: Union[Array, Quantity], b: Union[Array, Quantity]) -> Union[Array, Quantity]:
  """
  Compute the Kronecker product of two arrays or quantities.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
  """
  return funcs_change_unit_binary(jnp.kron,
                                  lambda x, y: x * y,
                                  a, b)


@set_module_as('brainunit.math')
def matmul(a: Union[Array, Quantity], b: Union[Array, Quantity]) -> Union[Array, Quantity]:
  """
  Matrix product of two arrays or quantities.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
  """
  return funcs_change_unit_binary(jnp.matmul,
                                  lambda x, y: x * y,
                                  a, b)


@set_module_as('brainunit.math')
def trace(a: Union[Array, Quantity]) -> Union[Array, Quantity]:
  """
  Return the sum of the diagonal elements of a matrix or quantity.

  Args:
    a: array_like, Quantity
    offset: int, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if the input is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.trace, a)
