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

from collections.abc import Sequence
from typing import (Union, Optional, List, Any)

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._base import (DIMENSIONLESS,
                     Quantity,
                     Unit,
                     fail_for_dimension_mismatch,
                     is_unitless, DimensionMismatchError, )
from .._misc import set_module_as

Shape = Union[int, Sequence[int]]

__all__ = [
  # array creation(given shape)
  'full', 'eye', 'identity', 'tri',
  'empty', 'ones', 'zeros',

  # array creation(given array)
  'full_like', 'diag', 'tril', 'triu',
  'empty_like', 'ones_like', 'zeros_like', 'fill_diagonal',

  # array creation(misc)
  'array', 'asarray', 'arange', 'linspace', 'logspace',
  'meshgrid', 'vander',

  # indexing funcs
  'tril_indices', 'tril_indices_from', 'triu_indices',
  'triu_indices_from',

  # others
  'from_numpy',
  'as_numpy',
  'tree_ones_like',
  'tree_zeros_like',
]


@set_module_as('brainunit.math')
def full(
    shape: Shape,
    fill_value: Union[Quantity, int, float],
    dtype: Optional[jax.typing.DTypeLike] = None,
) -> Union[Array, Quantity]:
  """
  Returns a quantity of `shape`, filled with `fill_value` if `fill_value` is a Quantity.
  else return an array of `shape` filled with `fill_value`.

  Parameters
  ----------
  shape : int or sequence of ints
    Shape of the new array, e.g., ``(2, 3)`` or ``2``.
  fill_value : scalar, array_like or Quantity
      Fill value.
  dtype : data-type, optional
    The desired data-type for the array  The default, None, means ``np.array(fill_value).dtype`

  Returns
  -------
  out : quantity or ndarray
    Quantity with the given shape if `fill_value` is a Quantity, else an array.
    Array of `fill_value` with the given shape, dtype, and order.
  """
  if isinstance(fill_value, Quantity):
    return Quantity(jnp.full(shape, fill_value.value, dtype=dtype), dim=fill_value.dim)
  return jnp.full(shape, fill_value, dtype=dtype)


@set_module_as('brainunit.math')
def eye(
    N: int,
    M: Optional[int] = None,
    k: int = 0,
    dtype: Optional[jax.typing.DTypeLike] = None,
    unit: Optional[Unit] = None,
) -> Union[Array, Quantity]:
  """
  Returns a 2-D quantity or array of `shape` and `unit` with ones on the diagonal and zeros elsewhere.

  Parameters
  ----------
  N : int
    Number of rows in the output.
  M : int, optional
    Number of columns in the output. If None, defaults to `N`.
  k : int, optional
    Index of the diagonal: 0 (the default) refers to the main diagonal,
    a positive value refers to an upper diagonal, and a negative value
    to a lower diagonal.
  dtype : data-type, optional
    Data-type of the returned array.
  unit : Unit, optional
    Unit of the returned Quantity.

  Returns
  -------
  I : quantity or ndarray of shape (N,M)
    An array where all elements are equal to zero, except for the `k`-th
    diagonal, whose values are equal to one.
  """
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
    return jnp.eye(N, M, k, dtype=dtype) * unit
  else:
    return jnp.eye(N, M, k, dtype=dtype)


@set_module_as('brainunit.math')
def identity(
    n: int,
    dtype: Optional[jax.typing.DTypeLike] = None,
    unit: Optional[Unit] = None
) -> Union[Array, Quantity]:
  """
  Return the identity Quantity or array.

  The identity array is a square array with ones on
  the main diagonal.

  Parameters
  ----------
  n : int
    Number of rows (and columns) in `n` x `n` output.
  dtype : data-type, optional
    Data-type of the output.  Defaults to ``float``.
  unit : Unit, optional
    Unit of the returned Quantity.

  Returns
  -------
  out : quantity or ndarray
    `n` x `n` quantity or array with its main diagonal set to one,
    and all other elements 0.
  """
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
    return jnp.identity(n, dtype=dtype) * unit
  else:
    return jnp.identity(n, dtype=dtype)


@set_module_as('brainunit.math')
def tri(
    N: int,
    M: Optional[int] = None,
    k: int = 0,
    dtype: Optional[jax.typing.DTypeLike] = None,
    unit: Optional[Unit] = None
) -> Union[Array, Quantity]:
  """
  A quantity or an array with ones at and below the given diagonal and zeros elsewhere.

  Parameters
  ----------
  N : int
    Number of rows in the array.
  M : int, optional
    Number of columns in the array.
    By default, `M` is taken equal to `N`.
  k : int, optional
    The sub-diagonal at and below which the array is filled.
    `k` = 0 is the main diagonal, while `k` < 0 is below it,
    and `k` > 0 is above.  The default is 0.
  dtype : dtype, optional
    Data type of the returned array.  The default is float.
  unit : Unit, optional
    Unit of the returned Quantity.

  Returns
  -------
  tri : quantity or ndarray of shape (N, M)
    quantity or array with its lower triangle filled with ones and zero elsewhere;
    in other words ``T[i,j] == 1`` for ``j <= i + k``, 0 otherwise.
  """
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
    return jnp.tri(N, M, k, dtype=dtype) * unit
  else:
    return jnp.tri(N, M, k, dtype=dtype)


@set_module_as('brainunit.math')
def empty(
    shape: Shape,
    dtype: Optional[jax.typing.DTypeLike] = None,
    unit: Optional[Unit] = None
) -> Union[Array, Quantity]:
  """
  Return a new quantity or array of given shape and type, without initializing entries.

  Parameters
  ----------
  shape : sequence of int
    Shape of the empty quantity or array.
  dtype : data-type, optional
    Data-type of the output.  Defaults to ``float``.
  unit : Unit, optional
    Unit of the returned Quantity.

  Returns
  -------
  out : quantity or ndarray
    quantity or array of uninitialized (arbitrary) data of the given shape, dtype, and order.
  """
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
    return jnp.empty(shape, dtype=dtype) * unit
  else:
    return jnp.empty(shape, dtype=dtype)


@set_module_as('brainunit.math')
def ones(
    shape: Shape,
    dtype: Optional[jax.typing.DTypeLike] = None,
    unit: Optional[Unit] = None
) -> Union[Array, Quantity]:
  """
  Returns a new quantity or array of given shape and type, filled with ones.

  Parameters
  ----------
  shape : sequence of int
    Shape of the new quantity or array.
  dtype : data-type, optional
    The desired data-type for the array.  Default is `float`.
  unit : Unit, optional
    Unit of the returned Quantity.

  Returns
  -------
  out : quantity or ndarray
    Array of ones with the given shape, dtype, and order.
  """
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
    return jnp.ones(shape, dtype=dtype) * unit
  else:
    return jnp.ones(shape, dtype=dtype)


@set_module_as('brainunit.math')
def zeros(
    shape: Shape,
    dtype: Optional[jax.typing.DTypeLike] = None,
    unit: Optional[Unit] = None
) -> Union[Array, Quantity]:
  """
  Returns a new quantity or array of given shape and type, filled with zeros.

  Parameters
  ----------
  shape : sequence of int
    Shape of the new quantity or array.
  dtype : data-type, optional
    The desired data-type for the array.  Default is `float`.
  unit : Unit, optional
    Unit of the returned Quantity.

  Returns
  -------
  out : quantity or ndarray
    Array of zeros with the given shape, dtype, and order.
  """
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
    return jnp.zeros(shape, dtype=dtype) * unit
  else:
    return jnp.zeros(shape, dtype=dtype)


@set_module_as('brainunit.math')
def full_like(
    a: Union[Quantity, jax.typing.ArrayLike],
    fill_value: Union[Quantity, jax.typing.ArrayLike],
    dtype: Optional[jax.typing.DTypeLike] = None,
    shape: Shape = None
) -> Union[Quantity, jax.Array]:
  """
  Return a new quantity or array with the same shape and type as a given array or quantity, filled with `fill_value`.

  Parameters
  ----------
  a : quantity or ndarray
    The shape and data-type of `a` define these same attributes of the returned quantity or array.
  fill_value : quantity or ndarray
    Value to fill the new quantity or array with.
  dtype : data-type, optional
    Overrides the data type of the result.
  shape : sequence of int, optional
    Overrides the shape of the result. If `shape` is not given, the shape of `a` is used.

  Returns
  -------
  out : quantity or ndarray
    New quantity or array with the same shape and type as `a`, filled with `fill_value`.
  """
  if isinstance(fill_value, Quantity):
    if isinstance(a, Quantity):
      fail_for_dimension_mismatch(a, fill_value, error_message="a and fill_value have to have the same units.")
      return Quantity(jnp.full_like(a.value, fill_value.value, dtype=dtype, shape=shape),
                      dim=a.dim)
    else:
      assert fill_value.is_unitless, 'fill_value must be unitless when a is not a Quantity.'
      return Quantity(jnp.full_like(a, fill_value.value, dtype=dtype, shape=shape),
                      dim=fill_value.dim)
  else:
    if isinstance(a, Quantity):
      assert a.is_unitless, 'a must be unitless when fill_value is not a Quantity.'
      return jnp.full_like(a.value, fill_value, dtype=dtype, shape=shape)
    else:
      return jnp.full_like(a, fill_value, dtype=dtype, shape=shape)


@set_module_as('brainunit.math')
def diag(
    v: Union[Quantity, jax.typing.ArrayLike],
    k: int = 0,
    unit: Optional[Unit] = None
) -> Union[Quantity, jax.Array]:
  """
  Extract a diagonal or construct a diagonal array.

  Parameters
  ----------
  v : quantity or ndarray
    If `a` is a 1-D array, `diag` constructs a 2-D array with `v` on the `k`-th diagonal.
    If `a` is a 2-D array, `diag` extracts the `k`-th diagonal and returns a 1-D array.
  k : int, optional
    Diagonal in question. The default is 0. Use `k>0` for diagonals above the main diagonal, and `k<0` for diagonals
    below the main diagonal.
  unit : Unit, optional
    Unit of the returned Quantity.

  Returns
  -------
  out : quantity or ndarray
    The extracted diagonal or constructed diagonal array.
  """
  if isinstance(v, Quantity):
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      fail_for_dimension_mismatch(v, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.diag(v.value, k=k), dim=v.dim)
  else:
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      return jnp.diag(v, k=k) * unit
    else:
      return jnp.diag(v, k=k)


@set_module_as('brainunit.math')
def tril(
    m: Union[Quantity, jax.typing.ArrayLike],
    k: int = 0,
    unit: Optional[Unit] = None
) -> Union[Quantity, jax.Array]:
  """
  Lower triangle of an array.

  Return a copy of a matrix with the elements above the `k`-th diagonal zeroed.
  For quantities or arrays with ``ndim`` exceeding 2, `tril` will apply to the final two axes.

  Parameters
  ----------
  m : quantity or ndarray
    Input array.
  k : int, optional
    Diagonal above which to zero elements. `k = 0` is the main diagonal, `k < 0` is below it, and `k > 0` is above.
  unit : Unit, optional
    Unit of the returned Quantity.

  Returns
  -------
  out : quantity or ndarray
    Lower triangle of `m`, of the same shape and data-type as `m`.
  """
  if isinstance(m, Quantity):
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      fail_for_dimension_mismatch(m, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.tril(m.value, k=k), dim=m.dim)
  else:
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      return jnp.tril(m, k=k) * unit
    else:
      return jnp.tril(m, k=k)


@set_module_as('brainunit.math')
def triu(
    m: Union[Quantity, jax.typing.ArrayLike],
    k: int = 0,
    unit: Optional[Unit] = None
) -> Union[Quantity, jax.Array]:
  """
  Upper triangle of a quantity or an array.

  Return a copy of an array with the elements below the `k`-th diagonal
  zeroed. For arrays with ``ndim`` exceeding 2, `triu` will apply to the
  final two axes.

  Please refer to the documentation for `tril` for further details.

  See Also
  --------
  tril : lower triangle of an array
  """
  if isinstance(m, Quantity):
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      fail_for_dimension_mismatch(m, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.triu(m.value, k=k), dim=m.dim)
  else:
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      return jnp.triu(m, k=k) * unit
    else:
      return jnp.triu(m, k=k)


@set_module_as('brainunit.math')
def empty_like(
    prototype: Union[Quantity, jax.typing.ArrayLike],
    dtype: Optional[jax.typing.DTypeLike] = None,
    shape: Shape = None,
    unit: Optional[Unit] = None
) -> Union[Quantity, jax.Array]:
  """
  Return a new quantity or array with the same shape and type as a given array.

  Parameters
  ----------
  prototype : quantity or ndarray
    The shape and data-type of `prototype` define these same attributes of the returned array.
  dtype : data-type, optional
    Overrides the data type of the result.
  shape : int or tuple of ints, optional
    Overrides the shape of the result. If not given, `prototype.shape` is used.
  unit : Unit, optional
    Unit of the returned Quantity.

  Returns
  -------
  out : quantity or ndarray
    Array of uninitialized (arbitrary) data with the same shape and type as `prototype`.
  """
  if isinstance(prototype, Quantity):
    if unit is not None:
      assert isinstance(unit, Unit), 'unit must be an instance of Unit.'
      fail_for_dimension_mismatch(prototype, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.empty_like(prototype.value, dtype=dtype), dim=prototype.dim)
  else:
    if unit is not None:
      assert isinstance(unit, Unit), 'unit must be an instance of Unit.'
      return jnp.empty_like(prototype, dtype=dtype, shape=shape) * unit
    else:
      return jnp.empty_like(prototype, dtype=dtype, shape=shape)


@set_module_as('brainunit.math')
def ones_like(
    a: Union[Quantity, jax.typing.ArrayLike],
    dtype: Optional[jax.typing.DTypeLike] = None,
    shape: Shape = None,
    unit: Optional[Unit] = None
) -> Union[Quantity, jax.Array]:
  """
  Return a quantity or an array of ones with the same shape and type as a given array.

  Parameters
  ----------
  a : quantity or ndarray
    The shape and data-type of `a` define these same attributes of the returned array.
  dtype : data-type, optional
    Overrides the data type of the result.
  shape : int or tuple of ints, optional
    Overrides the shape of the result. If not given, `a.shape` is used.
  unit : Unit, optional
    Unit of the returned Quantity.

  Returns
  -------
  out : quantity or ndarray
    Array of ones with the same shape and type as `a`.
  """
  if isinstance(a, Quantity):
    if unit is not None:
      assert isinstance(unit, Unit), 'unit must be an instance of Unit.'
      fail_for_dimension_mismatch(a, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.ones_like(a.value, dtype=dtype, shape=shape), dim=a.dim)
  else:
    if unit is not None:
      assert isinstance(unit, Unit), 'unit must be an instance of Unit.'
      return jnp.ones_like(a, dtype=dtype, shape=shape) * unit
    else:
      return jnp.ones_like(a, dtype=dtype, shape=shape)


@set_module_as('brainunit.math')
def zeros_like(
    a: Union[Quantity, jax.typing.ArrayLike],
    dtype: Optional[jax.typing.DTypeLike] = None,
    shape: Shape = None,
    unit: Optional[Unit] = None
) -> Union[Quantity, jax.Array]:
  """
  Return a quantity or an array of zeros with the same shape and type as a given array.

  Parameters
  ----------
  a : quantity or ndarray
    The shape and data-type of `a` define these same attributes of the returned array.
  dtype : data-type, optional
    Overrides the data type of the result.
  shape : int or tuple of ints, optional
    Overrides the shape of the result. If not given, `a.shape` is used.
  unit : Unit, optional
    Unit of the returned Quantity.

  Returns
  -------
  out : quantity or ndarray
    Array of zeros with the same shape and type as `a`.
  """
  if isinstance(a, Quantity):
    if unit is not None:
      assert isinstance(unit, Unit), 'unit must be an instance of Unit.'
      fail_for_dimension_mismatch(a, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.zeros_like(a.value, dtype=dtype, shape=shape), dim=a.dim)
  else:
    if unit is not None:
      assert isinstance(unit, Unit), 'unit must be an instance of Unit.'
      return jnp.zeros_like(a, dtype=dtype, shape=shape) * unit
    else:
      return jnp.zeros_like(a, dtype=dtype, shape=shape)


@set_module_as('brainunit.math')
def asarray(
    a: Any,
    dtype: Optional[jax.typing.DTypeLike] = None,
    order: Optional[str] = None,
    unit: Optional[Unit] = None,
) -> Union[Quantity, jax.Array]:
  """
  Convert the input to a quantity or array.

  If unit is provided, the input will be checked whether it has the same unit as the provided unit.
  (If they have same dimension but different magnitude, the input will be converted to the provided unit.)
  If unit is not provided, the input will be converted to an array.

  Parameters
  ----------
  a : quantity, ndarray, list[Quantity], list[ndarray]
    Input data, in any form that can be converted to an array.
  dtype : data-type, optional
    By default, the data-type is inferred from the input data.
  order : {'C', 'F', 'A', 'K'}, optional
    Whether to use row-major (C-style) or column-major (Fortran-style) memory representation.
    Defaults to 'K', which means that the memory layout is used in the order the array elements are stored in memory.
  unit : Unit, optional
    Unit of the returned Quantity.

  Returns
  -------
  out : quantity or array
    Array interpretation of `a`. No copy is made if the input is already an array.
  """
  # get leaves
  leaves, treedef = jax.tree.flatten(a, is_leaf=lambda x: isinstance(x, Quantity))
  a = treedef.unflatten([leaf.value if isinstance(leaf, Quantity) else leaf for leaf in leaves])

  # get unit
  dims = [leaf.dim if isinstance(leaf, Quantity) else DIMENSIONLESS for leaf in leaves]
  if any(dims[0] != d for d in dims):
    raise DimensionMismatchError(f'Units do not match for asarray operation. Got {dims}')
  dim = dims[0]
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
    if dim != DIMENSIONLESS:
      fail_for_dimension_mismatch(Unit(1., dim, register=False), unit,
                                  error_message="a and unit have to have the same units.")

  # compute
  r = jnp.asarray(a, dtype=dtype, order=order)

  # returns
  if dim != DIMENSIONLESS:
    return Quantity(r, dim=dim)
  if unit is not None:
    return r * unit
  return r


array = asarray


@set_module_as('brainunit.math')
def arange(
    start: Union[Quantity, jax.typing.ArrayLike] = None,
    stop: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
    step: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[Quantity, jax.Array]:
  """
  Return evenly spaced values within a given interval.

  Parameters
  ----------
  start : Quantity or array, optional
      Start of the interval. The interval includes this value. The default start value is 0.
  stop : Quantity or array
      End of the interval. The interval does not include this value, except in some cases where `step` is not an integer
      and floating point round-off affects the length of `out`.
  step : Quantity or array, optional
      Spacing between values. For any output `out`, this is the distance between two adjacent values, `out[i+1] - out[i]`.
      The default step size is 1.
  dtype : data-type, optional
      The type of the output array. If `dtype` is not given, infer the data type from the other input arguments.

  Returns
  -------
  out : quantity or array
      Array of evenly spaced values.
  """

  arg_len = len([x for x in [start, stop, step] if x is not None])

  if arg_len == 1:
    if stop is not None:
      raise TypeError("Duplicate definition of 'stop'")
    stop = start
    start = 0
  elif arg_len == 2:
    if start is not None and stop is None:
      stop = start
      start = 0

  elif arg_len > 3:
    raise TypeError("Need between 1 and 3 non-keyword arguments")

  # default values
  if start is None:
    start = 0
  if step is None:
    step = 1

  if stop is None:
    raise TypeError("Missing stop argument.")
  if stop is not None and not is_unitless(stop):
    start = Quantity(start, dim=stop.dim)

  fail_for_dimension_mismatch(
    start,
    stop,
    error_message="Start value {start} and stop value {stop} have to have the same units.",
    start=start,
    stop=stop,
  )
  fail_for_dimension_mismatch(
    stop,
    step,
    error_message="Stop value {stop} and step value {step} have to have the same units.",
    stop=stop,
    step=step,
  )

  unit = getattr(stop, "dim", DIMENSIONLESS)

  if start == 0:
    return Quantity(
      jnp.arange(
        start=start.value if isinstance(start, Quantity) else jnp.asarray(start),
        stop=stop.value if isinstance(stop, Quantity) else jnp.asarray(stop),
        step=step.value if isinstance(step, Quantity) else jnp.asarray(step),
        dtype=dtype,
      ),
      dim=unit,
    ) if unit != DIMENSIONLESS else jnp.arange(
      start=start.value if isinstance(start, Quantity) else jnp.asarray(start),
      stop=stop.value if isinstance(stop, Quantity) else jnp.asarray(stop),
      step=step.value if isinstance(step, Quantity) else jnp.asarray(step),
      dtype=dtype,
    )
  else:
    return Quantity(
      jnp.arange(
        start.value if isinstance(start, Quantity) else jnp.asarray(start),
        stop=stop.value if isinstance(stop, Quantity) else jnp.asarray(stop),
        step=step.value if isinstance(step, Quantity) else jnp.asarray(step),
        dtype=dtype,
      ),
      dim=unit,
    ) if unit != DIMENSIONLESS else jnp.arange(
      start.value if isinstance(start, Quantity) else jnp.asarray(start),
      stop=stop.value if isinstance(stop, Quantity) else jnp.asarray(stop),
      step=step.value if isinstance(step, Quantity) else jnp.asarray(step),
      dtype=dtype,
    )



@set_module_as('brainunit.math')
def linspace(
    start: Union[Quantity, jax.typing.ArrayLike],
    stop: Union[Quantity, jax.typing.ArrayLike],
    num: int = 50,
    endpoint: Optional[bool] = True,
    retstep: Optional[bool] = False,
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[Quantity, jax.Array]:
  """
  Return evenly spaced numbers over a specified interval.

  Returns `num` evenly spaced samples, calculated over the interval [`start`, `stop`].
  The endpoint of the interval can optionally be excluded.

  Parameters
  ----------
  start : Quantity or array
    The starting value of the sequence.
  stop : Quantity or array
    The end value of the sequence.
  num : int, optional
    Number of samples to generate. Default is 50.
  endpoint : bool, optional
    If True, `stop` is the last sample. Otherwise, it is not included. Default is True.
  retstep : bool, optional
    If True, return (`samples`, `step`), where `step` is the spacing between samples.
  dtype : data-type, optional
    The type of the output array. If `dtype` is not given, infer the data type from the other input arguments.

  Returns
  -------
  samples : quantity or array
    There are `num` equally spaced samples in the closed interval [`start`, `stop`] or the half-open interval [`start`, `stop`).
  """
  fail_for_dimension_mismatch(
    start,
    stop,
    error_message="Start value {start} and stop value {stop} have to have the same units.",
    start=start,
    stop=stop,
  )
  unit = getattr(start, "dim", DIMENSIONLESS)
  start = start.value if isinstance(start, Quantity) else start
  stop = stop.value if isinstance(stop, Quantity) else stop

  result = jnp.linspace(start, stop, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype)
  return Quantity(result, dim=unit) if unit != DIMENSIONLESS else result


@set_module_as('brainunit.math')
def logspace(
    start: Union[Quantity, jax.typing.ArrayLike],
    stop: Union[Quantity, jax.typing.ArrayLike],
    num: Optional[int] = 50,
    endpoint: Optional[bool] = True,
    base: Optional[float] = 10.0,
    dtype: Optional[jax.typing.DTypeLike] = None
):
  """
  Return numbers spaced evenly on a log scale.

  In linear space, the sequence starts at `base ** start` (`base` to the power of `start`) and ends with `base ** stop` in `num` steps.

  Parameters
  ----------
  start : Quantity or array
    The starting value of the sequence.
  stop : Quantity or array
    The end value of the sequence.
  num : int, optional
    Number of samples to generate. Default is 50.
  endpoint : bool, optional
    If True, `stop` is the last sample. Otherwise, it is not included. Default is True.
  base : float, optional
    The base of the log space. The step size between the elements in `ln(samples)` is `base`.
  dtype : data-type, optional
    The type of the output array. If `dtype` is not given, infer the data type from the other input arguments.

  Returns
  -------
  samples : quantity or array
    There are `num` equally spaced samples in the closed interval [`start`, `stop`] or the half-open interval [`start`, `stop`).
  """
  fail_for_dimension_mismatch(
    start,
    stop,
    error_message="Start value {start} and stop value {stop} have to have the same units.",
    start=start,
    stop=stop,
  )
  unit = getattr(start, "dim", DIMENSIONLESS)
  start = start.value if isinstance(start, Quantity) else start
  stop = stop.value if isinstance(stop, Quantity) else stop

  result = jnp.logspace(start, stop, num=num, endpoint=endpoint, base=base, dtype=dtype)
  return Quantity(result, dim=unit) if unit != DIMENSIONLESS else result


@set_module_as('brainunit.math')
def fill_diagonal(
    a: Union[Quantity, jax.typing.ArrayLike],
    val: Union[Quantity, jax.typing.ArrayLike],
    wrap: Optional[bool] = False,
    inplace: Optional[bool] = False
) -> Union[Quantity, jax.Array]:
  """
  Fill the main diagonal of the given array of any dimensionality.

  For an array `a` with `a.ndim >= 2`, the diagonal is the list of locations with indices `a[i, i, ..., i]`
  all identical.

  Parameters
  ----------
  a : Quantity or array
    Array in which to fill the diagonal.
  val : Quantity or array
    Value to be written on the diagonal. Its type must be compatible with that of the array a.
  wrap : bool, optional
    For tall matrices in NumPy version 1.6.2 and earlier, the matrix is considered "tall" if `a.shape[0] > a.shape[1]`.
    If `wrap` is True, the diagonal is "wrapped" after `a.shape[1]` and continues in the first column.
  inplace : bool, optional
    If True, the diagonal is filled in-place. Default is False.

  Returns
  -------
  out : Quantity or array
    The input array with the diagonal filled.
  """
  if isinstance(val, Quantity):
    if isinstance(a, Quantity):
      fail_for_dimension_mismatch(a, val, error_message="Array and value have to have the same units.")
      return Quantity(jnp.fill_diagonal(a.value, val.value, wrap, inplace=inplace), dim=a.dim)
    else:
      return Quantity(jnp.fill_diagonal(a, val.value, wrap, inplace=inplace), dim=val.dim)
  else:
    if isinstance(a, Quantity):
      return jnp.fill_diagonal(a.value, val, wrap, inplace=inplace)
    else:
      return jnp.fill_diagonal(a, val, wrap, inplace=inplace)


@set_module_as('brainunit.math')
def meshgrid(
    *xi: Union[Quantity, jax.typing.ArrayLike],
    copy: Optional[bool] = True,
    sparse: Optional[bool] = False,
    indexing: Optional[str] = 'xy'
) -> List[Union[Quantity, jax.Array]]:
  """
  Return coordinate matrices from coordinate vectors.

  Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields over N-D grids,
  given one-dimensional coordinate arrays x1, x2,..., xn.

  Parameters
  ----------
  xi : Quantity or array
    1-D arrays representing the coordinates of a grid.
  copy : bool, optional
    If True (default), the returned arrays are copies. If False, the view is returned.
  sparse : bool, optional
    If True, return a sparse grid (meshgrid) instead of a dense grid.
  indexing : {'xy', 'ij'}, optional
    Cartesian ('xy', default) or matrix ('ij') indexing of output.

  Returns
  -------
  X1, X2,..., XN : Quantity or array
    For vectors x1, x2,..., 'xn' with lengths Ni=len(xi), return (N1, N2, N3,..., Nn) shaped arrays if indexing='ij'
    or (N2, N1, N3,..., Nn) shaped arrays if indexing='xy' with the elements of xi repeated to fill the matrix along
    the first dimension for x1, the second for x2 and so on.
  """

  args = [asarray(x) for x in xi]
  if not copy:
    raise ValueError("jax.numpy.meshgrid only supports copy=True")
  if indexing not in ["xy", "ij"]:
    raise ValueError(f"Valid values for indexing are 'xy' and 'ij', got {indexing}")
  if any(a.ndim != 1 for a in args):
    raise ValueError("Arguments to jax.numpy.meshgrid must be 1D, got shapes "
                     f"{[a.shape for a in args]}")
  if indexing == "xy" and len(args) >= 2:
    args[0], args[1] = args[1], args[0]
  shape = [1 if sparse else a.shape[0] for a in args]
  f_shape = lambda i, a: [*shape[:i], a.shape[0], *shape[i + 1:]] if sparse else shape
  # use jax.tree.map to compatible with Quantity
  output = [
    jax.tree.map(lambda x: jax.lax.broadcast_in_dim(x, f_shape(i, x), (i,)), a)
    for i, a, in enumerate(args)
  ]
  if indexing == "xy" and len(args) >= 2:
    output[0], output[1] = output[1], output[0]
  return output


@set_module_as('brainunit.math')
def vander(
    x: Union[Quantity, jax.typing.ArrayLike],
    N: Optional[bool] = None,
    increasing: Optional[bool] = False,
    unit: Optional[Unit] = None,
) -> Union[Quantity, jax.Array]:
  """
  Generate a Vandermonde matrix.

  The Vandermonde matrix is a matrix with the terms of a geometric progression in each row.
  The geometric progression is defined by the vector `x` and the number of columns `N`.

  Parameters
  ----------
  x : Quantity or array
    1-D input array.
  N : int, optional
    Number of columns in the output. If `N` is not specified, a square array is returned (N = len(x)).
  increasing : bool, optional
    Order of the powers of the columns. If True, the powers increase from left to right, if False (the default),
    they are reversed.

  Returns
  -------
  out : Quantity or array
    Vandermonde matrix. If `increasing` is False, the first column is `x^(N-1)`, the second `x^(N-2)` and so forth.
  """
  if isinstance(x, Quantity):
    assert x.is_unitless, f'x must be unitless for function {vander.__name__}.'
    x = x.value
  r = jnp.vander(x, N=N, increasing=increasing)
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
    return Quantity(r, unit=unit)
  else:
    return r


# indexing funcs
# --------------

tril_indices = jnp.tril_indices


@set_module_as('brainunit.math')
def tril_indices_from(
    arr: Union[Quantity, jax.typing.ArrayLike],
    k: Optional[int] = 0
) -> tuple[jax.Array, jax.Array]:
  """
  Return the indices for the lower-triangle of an (n, m) array.

  Parameters
  ----------
  arr : array_like, Quantity
    The arrays for which the returned indices will be valid.
  k : int, optional
    Diagonal above which to zero elements. k = 0 is the main diagonal, k < 0 subdiagonal and k > 0 superdiagonal.

  Returns
  -------
  out : tuple[jax.Array]
    tuple of arrays
  """
  if isinstance(arr, Quantity):
    return jnp.tril_indices_from(arr.value, k=k)
  else:
    return jnp.tril_indices_from(arr, k=k)


triu_indices = jnp.triu_indices


@set_module_as('brainunit.math')
def triu_indices_from(
    arr: Union[Quantity, jax.typing.ArrayLike],
    k: Optional[int] = 0
) -> tuple[jax.Array, jax.Array]:
  """
  Return the indices for the upper-triangle of an (n, m) array.

  Parameters
  ----------
  arr : array_like, Quantity
    The arrays for which the returned indices will be valid.
  k : int, optional
    Diagonal above which to zero elements. k = 0 is the main diagonal, k < 0 subdiagonal and k > 0 superdiagonal.

  Returns
  -------
  out : tuple[jax.Array]
    tuple of arrays
  """
  if isinstance(arr, Quantity):
    return jnp.triu_indices_from(arr.value, k=k)
  else:
    return jnp.triu_indices_from(arr, k=k)


# --- others ---


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
