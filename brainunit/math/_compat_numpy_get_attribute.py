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
import numpy as np

from .._base import Quantity
from .._misc import set_module_as

__all__ = [
  # getting attribute funcs
  'ndim', 'isreal', 'isscalar', 'isfinite', 'isinf',
  'isnan', 'shape', 'size',
]


@set_module_as('brainunit.math')
def ndim(a: Union[Quantity, jax.typing.ArrayLike]) -> int:
  """
  Return the number of dimensions of an array.

  Args:
    a: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: int
  """
  if isinstance(a, Quantity):
    return a.ndim
  else:
    return jnp.ndim(a)


@set_module_as('brainunit.math')
def isreal(a: Union[Quantity, jax.typing.ArrayLike]) -> jax.Array:
  """
  Return True if the input array is real.

  Args:
    a: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: boolean array
  """
  if isinstance(a, Quantity):
    return a.isreal
  else:
    return jnp.isreal(a)


@set_module_as('brainunit.math')
def isscalar(a: Union[Quantity, jax.typing.ArrayLike]) -> bool:
  """
  Return True if the input is a scalar.

  Args:
    a: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: boolean array
  """
  if isinstance(a, Quantity):
    return a.isscalar
  else:
    return jnp.isscalar(a)


@set_module_as('brainunit.math')
def isfinite(a: Union[Quantity, jax.typing.ArrayLike]) -> jax.Array:
  """
  Return each element of the array is finite or not.

  Args:
    a: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: boolean array
  """
  if isinstance(a, Quantity):
    return a.isfinite
  else:
    return jnp.isfinite(a)


@set_module_as('brainunit.math')
def isinf(a: Union[Quantity, jax.typing.ArrayLike]) -> jax.Array:
  """
  Return each element of the array is infinite or not.

  Args:
    a: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: boolean array
  """
  if isinstance(a, Quantity):
    return a.isinf
  else:
    return jnp.isinf(a)


@set_module_as('brainunit.math')
def isnan(a: Union[Quantity, jax.typing.ArrayLike]) -> jax.Array:
  """
  Return each element of the array is NaN or not.

  Args:
    a: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: boolean array
  """
  if isinstance(a, Quantity):
    return a.isnan
  else:
    return jnp.isnan(a)


@set_module_as('brainunit.math')
def shape(a: Union[Quantity, jax.typing.ArrayLike]) -> tuple[int, ...]:
  """
  Return the shape of an array.

  Parameters
  ----------
  a : array_like
      Input array.

  Returns
  -------
  shape : tuple of ints
      The elements of the shape tuple give the lengths of the
      corresponding array dimensions.

  See Also
  --------
  len : ``len(a)`` is equivalent to ``np.shape(a)[0]`` for N-D arrays with
        ``N>=1``.
  ndarray.shape : Equivalent array method.

  Examples
  --------
  >>> brainunit.math.shape(brainunit.math.eye(3))
  (3, 3)
  >>> brainunit.math.shape([[1, 3]])
  (1, 2)
  >>> brainunit.math.shape([0])
  (1,)
  >>> brainunit.math.shape(0)
  ()

  """
  if isinstance(a, (Quantity, jax.Array, np.ndarray)):
    return a.shape
  else:
    return np.shape(a)


@set_module_as('brainunit.math')
def size(a: Union[Quantity, jax.typing.ArrayLike], axis: int = None) -> int:
  """
  Return the number of elements along a given axis.

  Parameters
  ----------
  a : array_like
      Input data.
  axis : int, optional
      Axis along which the elements are counted.  By default, give
      the total number of elements.

  Returns
  -------
  element_count : int
      Number of elements along the specified axis.

  See Also
  --------
  shape : dimensions of array
  Array.shape : dimensions of array
  Array.size : number of elements in array

  Examples
  --------
  >>> a = Quantity([[1,2,3], [4,5,6]])
  >>> brainunit.math.size(a)
  6
  >>> brainunit.math.size(a, 1)
  3
  >>> brainunit.math.size(a, 0)
  2
  """
  if isinstance(a, (Quantity, jax.Array, np.ndarray)):
    if axis is None:
      return a.size
    else:
      return a.shape[axis]
  else:
    return np.size(a, axis=axis)
