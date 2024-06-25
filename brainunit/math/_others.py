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

from typing import Sequence, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ._numpy_accept_unitless import _func_accept_unitless_unary
from ._numpy_keep_unit import _fun_keep_unit_unary
from .._base import Quantity, Unit
from .._misc import set_module_as

__all__ = [
  'exprel',
  'flatten',
  'unflatten',
  'remove_diag',
  'from_numpy',
  'as_numpy',
  'tree_ones_like',
  'tree_zeros_like',
  'get_dtype',
  'is_float',
  'is_int',
]

environ = None  # type: ignore[assignment]


def _exprel_v1(x):  # This approximation has problems of the gradient vanishing at x=0
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


def _exprel_v2(x, *, level: int = 2):
  x = jnp.asarray(x)
  dtype = x.dtype
  assert jnp.issubdtype(dtype, jnp.floating), f'The input array must contain real numbers. Got {x}'

  # Adjust the tolerance based on the dtype of x
  if dtype == jnp.float64:
    threshold = 1e-8
  elif dtype == jnp.float32:
    threshold = 1e-5
  elif dtype in [jnp.float16, jnp.bfloat16]:
    threshold = 1e-3
  else:
    threshold = 1e-3

  assert level in [0, 1, 2, 3], 'The approximation level should be 0, 1, 2, or 3.'
  if level == 0:
    return jax.numpy.where(jnp.abs(x) <= threshold, 1., jnp.expm1(x) / x)
  elif level == 1:
    return jax.numpy.where(jnp.abs(x) <= threshold, 1. + x / 2., jnp.expm1(x) / x)
  elif level == 2:
    return jax.numpy.where(jnp.abs(x) <= threshold, 1. + x / 2. + x * x / 6., jnp.expm1(x) / x)
  elif level == 3:
    x2 = x * x
    return jax.numpy.where(jnp.abs(x) <= threshold, 1. + x / 2. + x2 / 6. + x2 * x / 24., jnp.expm1(x) / x)
  else:
    raise ValueError(f'Unsupported approximation level {level}.')


@set_module_as('brainunit.math')
def exprel(x, *, level: int = 2):
  """
  Relative error exponential, ``(exp(x) - 1)/x``.

  When ``x`` is near zero, ``exp(x)`` is near 1, so the numerical calculation of ``exp(x) - 1`` can
  suffer from catastrophic loss of precision. ``exprel(x)`` is implemented to avoid the loss of
  precision that occurs when ``x`` is near zero.

  Args:
    x: ndarray. Input array. ``x`` must contain real numbers.
    level: int. The approximation level of the function. The higher the level, the more accurate the result.

  Returns:
    ``(exp(x) - 1)/x``, computed element-wise.
  """
  return _func_accept_unitless_unary(_exprel_v2, x, level=level)


@set_module_as('brainunit.math')
def flatten(
    x: jax.typing.ArrayLike | Quantity,
    start_axis: Optional[int] = None,
    end_axis: Optional[int] = None
) -> jax.Array | Quantity:
  """Flattens input by reshaping it into a one-dimensional tensor.
  If ``start_dim`` or ``end_dim`` are passed, only dimensions starting
  with ``start_dim`` and ending with ``end_dim`` are flattened.
  The order of elements in input is unchanged.

  .. note::
     Flattening a zero-dimensional tensor will return a one-dimensional view.

  Parameters
  ----------
  x: Array, Quantity
    The input array.
  start_axis: int
    the first dim to flatten
  end_axis: int
    the last dim to flatten

  Returns
  -------
  out: Array, Quantity
  """
  shape = x.shape
  ndim = x.ndim
  if ndim == 0:
    ndim = 1
  if start_axis is None:
    start_axis = 0
  elif start_axis < 0:
    start_axis = ndim + start_axis
  if end_axis is None:
    end_axis = ndim - 1
  elif end_axis < 0:
    end_axis = ndim + end_axis
  end_axis += 1
  if start_axis < 0 or start_axis > ndim:
    raise ValueError(f'start_axis {start_axis} is out of size.')
  if end_axis < 0 or end_axis > ndim:
    raise ValueError(f'end_axis {end_axis} is out of size.')
  new_shape = shape[:start_axis] + (np.prod(shape[start_axis: end_axis], dtype=int),) + shape[end_axis:]
  return _fun_keep_unit_unary(jnp.reshape, x, shape=new_shape)


@set_module_as('brainunit.math')
def unflatten(
    x: jax.typing.ArrayLike | Quantity,
    axis: int,
    sizes: Sequence[int]
) -> jax.Array | Quantity:
  """
  Expands a dimension of the input tensor over multiple dimensions.

  Args:
    x: input tensor.
    axis: Dimension to be unflattened, specified as an index into ``x.shape``.
    sizes: New shape of the unflattened dimension. One of its elements can be -1
        in which case the corresponding output dimension is inferred.
        Otherwise, the product of ``sizes`` must equal ``input.shape[dim]``.

  Returns:
    A tensor with the same data as ``input``, but with ``dim`` split into multiple dimensions.
    The returned tensor has one more dimension than the input tensor.
    The returned tensor shares the same underlying data with this tensor.
  """
  assert x.ndim > axis, ('The dimension to be unflattened should '
                         'be less than the tensor dimension. '
                         f'Got {axis} and {x.ndim}.')
  shape = x.shape
  new_shape = shape[:axis] + tuple(sizes) + shape[axis + 1:]
  return _fun_keep_unit_unary(jnp.reshape, x, shape=new_shape)


@set_module_as('brainunit.math')
def remove_diag(x: jax.typing.ArrayLike | Quantity) -> jax.Array | Quantity:
  """Remove the diagonal of the matrix.

  Parameters
  ----------
  x: Array, Quantity
    The matrix with the shape of `(M, N)`.

  Returns
  -------
  arr: Array, Quantity
    The matrix without diagonal which has the shape of `(M, N-1)`.
  """
  dim = None
  if isinstance(x, Quantity):
    x = x.value
    dim = x.dim

  if x.ndim != 2:
    raise ValueError(f'Only support 2D matrix, while we got a {x.ndim}D array.')
  eyes = jnp.fill_diagonal(jnp.ones(x.shape, dtype=bool), False)
  x = jnp.reshape(x[eyes], (x.shape[0], x.shape[1] - 1))
  if dim is not None:
    return Quantity(x, dim=dim)
  return x


@set_module_as('brainunit.math')
def from_numpy(
    x: np.ndarray,
    unit: Unit = None
) -> jax.Array | Quantity:
  """
  Convert the numpy array to jax array.

  Args:
    x: The numpy array.
    unit: The unit of the array.

  Returns:
    The jax array.
  """
  if unit is not None:
    return jnp.array(x) * unit
  return jnp.array(x)


@set_module_as('brainunit.math')
def as_numpy(x):
  """
  Convert the array to numpy array.

  Args:
    x: The array.

  Returns:
    The numpy array.
  """
  return np.array(x)


@set_module_as('brainunit.math')
def tree_zeros_like(tree):
  """
  Create a tree with the same structure as the input tree, but with zeros in each leaf.

  Args:
    tree: The input tree.

  Returns:
    The tree with zeros in each leaf.
  """
  return jax.tree_map(jnp.zeros_like, tree)


@set_module_as('brainunit.math')
def tree_ones_like(tree):
  """
  Create a tree with the same structure as the input tree, but with ones in each leaf.

  Args:
    tree: The input tree.

  Returns:
    The tree with ones in each leaf.

  """
  return jax.tree_map(jnp.ones_like, tree)


@set_module_as('brainstate.math')
def get_dtype(a):
  """
  Get the dtype of a.
  """
  if hasattr(a, 'dtype'):
    return a.dtype
  else:
    global environ
    if isinstance(a, bool):
      return bool
    elif isinstance(a, int):
      if environ is None:
        from brainstate import environ
      return environ.ditype()
    elif isinstance(a, float):
      if environ is None:
        from brainstate import environ
      return environ.dftype()
    elif isinstance(a, complex):
      if environ is None:
        from brainstate import environ
      return environ.dctype()
    else:
      raise ValueError(f'Can not get dtype of {a}.')


@set_module_as('brainstate.math')
def is_float(array):
  """
  Check if the array is a floating point array.

  Args:
    array: The input array.

  Returns:
    A boolean value indicating if the array is a floating point array.
  """
  return jnp.issubdtype(get_dtype(array), jnp.floating)


@set_module_as('brainstate.math')
def is_int(array):
  """
  Check if the array is an integer array.

  Args:
    array: The input array.

  Returns:
    A boolean value indicating if the array is an integer array.
  """
  return jnp.issubdtype(get_dtype(array), jnp.integer)
