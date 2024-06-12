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

from ._compat_numpy_funcs_change_unit import wrap_math_funcs_change_unit_binary
from ._compat_numpy_funcs_keep_unit import wrap_math_funcs_keep_unit_unary
from .._base import (Quantity,
                     )

__all__ = [

  # linear algebra
  'dot', 'vdot', 'inner', 'outer', 'kron', 'matmul', 'trace',

]




# linear algebra
# --------------

@wrap_math_funcs_change_unit_binary(lambda x, y: x * y)
def dot(a: Union[Array, Quantity], b: Union[Array, Quantity]) -> Union[Array, Quantity]:
  return jnp.dot(a, b)


@wrap_math_funcs_change_unit_binary(lambda x, y: x * y)
def vdot(a: Union[Array, Quantity], b: Union[Array, Quantity]) -> Union[Array, Quantity]:
  return jnp.vdot(a, b)


@wrap_math_funcs_change_unit_binary(lambda x, y: x * y)
def inner(a: Union[Array, Quantity], b: Union[Array, Quantity]) -> Union[Array, Quantity]:
  return jnp.inner(a, b)


@wrap_math_funcs_change_unit_binary(lambda x, y: x * y)
def outer(a: Union[Array, Quantity], b: Union[Array, Quantity]) -> Union[Array, Quantity]:
  return jnp.outer(a, b)


@wrap_math_funcs_change_unit_binary(lambda x, y: x * y)
def kron(a: Union[Array, Quantity], b: Union[Array, Quantity]) -> Union[Array, Quantity]:
  return jnp.kron(a, b)


@wrap_math_funcs_change_unit_binary(lambda x, y: x * y)
def matmul(a: Union[Array, Quantity], b: Union[Array, Quantity]) -> Union[Array, Quantity]:
  return jnp.matmul(a, b)


@wrap_math_funcs_keep_unit_unary
def trace(a: Union[Array, Quantity]) -> Union[Array, Quantity]:
  return jnp.trace(a)


# docs for functions above
dot.__doc__ = '''
  Dot product of two arrays or quantities.
  
  Args:
    a: array_like, Quantity
    b: array_like, Quantity
    
  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `x` and the unit of `y`, else an array.
'''

vdot.__doc__ = '''
  Return the dot product of two vectors or quantities.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
'''

inner.__doc__ = '''
  Inner product of two arrays or quantities.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
'''

outer.__doc__ = '''
  Compute the outer product of two vectors or quantities.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
'''

kron.__doc__ = '''
  Compute the Kronecker product of two arrays or quantities.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
'''

matmul.__doc__ = '''
  Matrix product of two arrays or quantities.

  Args:
    a: array_like, Quantity
    b: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
'''

trace.__doc__ = '''
  Return the sum of the diagonal elements of a matrix or quantity.

  Args:
    a: array_like, Quantity
    offset: int, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if the input is a Quantity, else an array.
'''
