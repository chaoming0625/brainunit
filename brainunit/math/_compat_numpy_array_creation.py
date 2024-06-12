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

from collections.abc import Sequence
from typing import (Union, Optional, Any)

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from brainunit._misc import set_module_as
from .._base import (
  DIMENSIONLESS,
  Quantity,
  Unit,
  fail_for_dimension_mismatch,
  is_unitless,
)

__all__ = [
  # array creation
  'full', 'full_like', 'eye', 'identity', 'diag', 'tri', 'tril', 'triu',
  'empty', 'empty_like', 'ones', 'ones_like', 'zeros', 'zeros_like',
  'array', 'asarray', 'arange', 'linspace', 'logspace', 'fill_diagonal',
  'array_split', 'meshgrid', 'vander',
]


@set_module_as('brainunit.math')
def full(
    shape: Sequence[int],
    fill_value: Union[Quantity, int, float],
    dtype: Optional[Any] = None,
) -> Union[Array, Quantity]:
  """
  Returns a Quantity of `shape`, filled with `fill_value` if `fill_value` is a Quantity.
  else return an array of `shape` filled with `fill_value`.

  Args:
    shape: sequence of integers, describing the shape of the output array.
    fill_value: the value to fill the new array with.
    dtype: the type of the output array, or `None`. If not `None`, `fill_value`
      will be cast to `dtype`.

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if isinstance(fill_value, Quantity):
    return Quantity(jnp.full(shape, fill_value.value, dtype=dtype), dim=fill_value.dim)
  return jnp.full(shape, fill_value, dtype=dtype)


@set_module_as('brainunit.math')
def eye(
    N: int,
    M: Optional[int] = None,
    k: int = 0,
    dtype: Optional[Any] = None,
    unit: Optional[Unit] = None
) -> Union[Array, Quantity]:
  """
  Returns a Quantity of `shape` and `unit`, representing an identity matrix if `unit` is provided.
  else return an identity matrix of `shape`.

  Args:
    n: the number of rows (and columns) in the output array.
    k: the index of the diagonal: 0 (the default) refers to the main diagonal,
       a positive value refers to an upper diagonal, and a negative value to a
       lower diagonal.
    dtype: the type of the output array, or `None`. If not `None`, elements
      will be cast to `dtype`.
    unit: the unit of the output array, or `None`.

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
    return jnp.eye(N, M, k, dtype=dtype) * unit
  else:
    return jnp.eye(N, M, k, dtype=dtype)


@set_module_as('brainunit.math')
def identity(
    n: int,
    dtype: Optional[Any] = None,
    unit: Optional[Unit] = None
) -> Union[Array, Quantity]:
  """
  Returns a Quantity of `shape` and `unit`, representing an identity matrix if `unit` is provided.
  else return an identity matrix of `shape`.

  Args:
    n: the number of rows (and columns) in the output array.
    dtype: the type of the output array, or `None`. If not `None`, elements
      will be cast to `dtype`.
    unit: the unit of the output array, or `None`.

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
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
    dtype: Optional[Any] = None,
    unit: Optional[Unit] = None
) -> Union[Array, Quantity]:
  """
  Returns a Quantity of `shape` and `unit`, representing a triangular matrix if `unit` is provided.
  else return a triangular matrix of `shape`.

  Args:
    n: the number of rows in the output array.
    m: the number of columns with default being `n`.
    k: the index of the diagonal: 0 (the default) refers to the main diagonal,
       a positive value refers to an upper diagonal, and a negative value to a
       lower diagonal.
    dtype: the type of the output array, or `None`. If not `None`, elements
      will be cast to `dtype`.
    unit: the unit of the output array, or `None`.

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
    return jnp.tri(N, M, k, dtype=dtype) * unit
  else:
    return jnp.tri(N, M, k, dtype=dtype)


@set_module_as('brainunit.math')
def empty(
    shape: Sequence[int],
    dtype: Optional[Any] = None,
    unit: Optional[Unit] = None
) -> Union[Array, Quantity]:
  """
  Returns a Quantity of `shape` and `unit`, with uninitialized values if `unit` is provided.
  else return an array of `shape` with uninitialized values.

  Args:
    shape: sequence of integers, describing the shape of the output array.
    dtype: the type of the output array, or `None`. If not `None`, elements
      will be of type `dtype`.
    unit: the unit of the output array, or `None`.

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
    return jnp.empty(shape, dtype=dtype) * unit
  else:
    return jnp.empty(shape, dtype=dtype)


@set_module_as('brainunit.math')
def ones(
    shape: Sequence[int],
    dtype: Optional[Any] = None,
    unit: Optional[Unit] = None
) -> Union[Array, Quantity]:
  """
  Returns a Quantity of `shape` and `unit`, filled with 1 if `unit` is provided.
  else return an array of `shape` filled with 1.

  Args:
    shape: sequence of integers, describing the shape of the output array.
    dtype: the type of the output array, or `None`. If not `None`, elements
      will be cast to `dtype`.
    unit: the unit of the output array, or `None`.

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
    return jnp.ones(shape, dtype=dtype) * unit
  else:
    return jnp.ones(shape, dtype=dtype)


@set_module_as('brainunit.math')
def zeros(
    shape: Sequence[int],
    dtype: Optional[Any] = None,
    unit: Optional[Unit] = None
) -> Union[Array, Quantity]:
  """
  Returns a Quantity of `shape` and `unit`, filled with 0 if `unit` is provided.
  else return an array of `shape` filled with 0.

  Args:
    shape: sequence of integers, describing the shape of the output array.
    dtype: the type of the output array, or `None`. If not `None`, elements
      will be cast to `dtype`.
    unit: the unit of the output array, or `None`.

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
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
    shape: Any = None
) -> Union[Quantity, jax.Array]:
  """
  Return a Quantity if `a` and `fill_value` are Quantities that have the same unit or only `fill_value` is a Quantity.
  else return an array of `a` filled with `fill_value`.

  Args:
    a: array_like, Quantity, shape, or dtype
    fill_value: scalar or array_like
    dtype: data-type, optional
    shape: sequence of ints, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if isinstance(fill_value, Quantity):
    if isinstance(a, Quantity):
      fail_for_dimension_mismatch(a, fill_value, error_message="a and fill_value have to have the same units.")
      return Quantity(jnp.full_like(a.value, fill_value.value, dtype=dtype, shape=shape), dim=a.dim)
    else:
      return Quantity(jnp.full_like(a, fill_value.value, dtype=dtype, shape=shape), dim=fill_value.dim)
  else:
    if isinstance(a, Quantity):
      return jnp.full_like(a.value, fill_value, dtype=dtype, shape=shape)
    else:
      return jnp.full_like(a, fill_value, dtype=dtype, shape=shape)


@set_module_as('brainunit.math')
def diag(a: Union[Quantity, jax.typing.ArrayLike],
         k: int = 0,
         unit: Optional[Unit] = None) -> Union[Quantity, jax.Array]:
  """
  Extract a diagonal or construct a diagonal array.

  Args:
    a: array_like, Quantity
    k: int, optional
    unit: Unit, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if isinstance(a, Quantity):
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      fail_for_dimension_mismatch(a, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.diag(a.value, k=k), dim=a.dim)
  elif isinstance(a, (jax.Array, np.ndarray)):
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      return jnp.diag(a, k=k) * unit
    else:
      return jnp.diag(a, k=k)
  else:
    return jnp.diag(a, k=k)


@set_module_as('brainunit.math')
def tril(a: Union[Quantity, jax.typing.ArrayLike],
         k: int = 0,
         unit: Optional[Unit] = None) -> Union[Quantity, jax.Array]:
  """
  Lower triangle of an array.

  Args:
    a: array_like, Quantity
    k: int, optional
    unit: Unit, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if isinstance(a, Quantity):
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      fail_for_dimension_mismatch(a, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.tril(a.value, k=k), dim=a.dim)
  elif isinstance(a, (jax.Array, np.ndarray)):
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      return jnp.tril(a, k=k) * unit
    else:
      return jnp.tril(a, k=k)
  else:
    return jnp.tril(a, k=k)


@set_module_as('brainunit.math')
def triu(a: Union[Quantity, jax.typing.ArrayLike],
         k: int = 0,
         unit: Optional[Unit] = None) -> Union[Quantity, jax.Array]:
  """
  Upper triangle of an array.

  Args:
    a: array_like, Quantity
    k: int, optional
    unit: Unit, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if isinstance(a, Quantity):
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      fail_for_dimension_mismatch(a, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.triu(a.value, k=k), dim=a.dim)
  elif isinstance(a, (jax.Array, np.ndarray)):
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      return jnp.triu(a, k=k) * unit
    else:
      return jnp.triu(a, k=k)
  else:
    return jnp.triu(a, k=k)


@set_module_as('brainunit.math')
def empty_like(a: Union[Quantity, jax.typing.ArrayLike],
               dtype: Optional[jax.typing.DTypeLike] = None,
               shape: Any = None,
               unit: Optional[Unit] = None) -> Union[Quantity, jax.Array]:
  """
  Return a Quantity of `a` and `unit`, with uninitialized values if `unit` is provided.
  else return an array of `a` with uninitialized values.

  Args:
    a: array_like, Quantity, shape, or dtype
    dtype: data-type, optional
    shape: sequence of ints, optional
    unit: Unit, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if isinstance(a, Quantity):
    if unit is not None:
      assert isinstance(unit, Unit)
      fail_for_dimension_mismatch(a, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.empty_like(a.value, dtype=dtype, shape=shape), dim=a.dim)
  elif isinstance(a, (jax.Array, np.ndarray)):
    if unit is not None:
      assert isinstance(unit, Unit)
      return jnp.empty_like(a, dtype=dtype, shape=shape) * unit
    else:
      return jnp.empty_like(a, dtype=dtype, shape=shape)
  else:
    return jnp.empty_like(a, dtype=dtype, shape=shape)


@set_module_as('brainunit.math')
def ones_like(a: Union[Quantity, jax.typing.ArrayLike],
              dtype: Optional[jax.typing.DTypeLike] = None,
              shape: Any = None,
              unit: Optional[Unit] = None) -> Union[Quantity, jax.Array]:
  """
  Return a Quantity of `a` and `unit`, filled with 1 if `unit` is provided.
  else return an array of `a` filled with 1.

  Args:
    a: array_like, Quantity, shape, or dtype
    dtype: data-type, optional
    shape: sequence of ints, optional
    unit: Unit, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if isinstance(a, Quantity):
    if unit is not None:
      assert isinstance(unit, Unit)
      fail_for_dimension_mismatch(a, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.ones_like(a.value, dtype=dtype, shape=shape), dim=a.dim)
  elif isinstance(a, (jax.Array, np.ndarray)):
    if unit is not None:
      assert isinstance(unit, Unit)
      return jnp.ones_like(a, dtype=dtype, shape=shape) * unit
    else:
      return jnp.ones_like(a, dtype=dtype, shape=shape)
  else:
    return jnp.ones_like(a, dtype=dtype, shape=shape)


@set_module_as('brainunit.math')
def zeros_like(a: Union[Quantity, jax.typing.ArrayLike],
               dtype: Optional[jax.typing.DTypeLike] = None,
               shape: Any = None,
               unit: Optional[Unit] = None) -> Union[Quantity, jax.Array]:
  """
  Return a Quantity of `a` and `unit`, filled with 0 if `unit` is provided.
  else return an array of `a` filled with 0.

  Args:
    a: array_like, Quantity, shape, or dtype
    dtype: data-type, optional
    shape: sequence of ints, optional
    unit: Unit, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if isinstance(a, Quantity):
    if unit is not None:
      assert isinstance(unit, Unit)
      fail_for_dimension_mismatch(a, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.zeros_like(a.value, dtype=dtype, shape=shape), dim=a.dim)
  elif isinstance(a, (jax.Array, np.ndarray)):
    if unit is not None:
      assert isinstance(unit, Unit)
      return jnp.zeros_like(a, dtype=dtype, shape=shape) * unit
    else:
      return jnp.zeros_like(a, dtype=dtype, shape=shape)
  else:
    return jnp.zeros_like(a, dtype=dtype, shape=shape)


@set_module_as('brainunit.math')
def asarray(
    a: Union[Quantity, jax.typing.ArrayLike, Sequence[Quantity], Sequence[jax.typing.ArrayLike]],
    dtype: Optional[jax.typing.DTypeLike] = None,
    order: Optional[str] = None,
    unit: Optional[Unit] = None,
) -> Union[Quantity, jax.Array]:
  """
  Convert the input to a quantity or array.

  If unit is provided, the input will be checked whether it has the same unit as the provided unit.
  (If they have same dimension but different magnitude, the input will be converted to the provided unit.)
  If unit is not provided, the input will be converted to an array.

  Args:
    a: array_like, Quantity, or Sequence[Quantity]
    dtype: data-type, optional
    order: {'C', 'F', 'A', 'K'}, optional
    unit: Unit, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `unit` is provided, else an array.
  """
  if unit is not None:
    assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
  if isinstance(a, Quantity):
    if unit is not None:
      fail_for_dimension_mismatch(a, unit, error_message="a and unit have to have the same units.")
    return Quantity(jnp.asarray(a.value, dtype=dtype, order=order), dim=a.dim)
  elif isinstance(a, (jax.Array, np.ndarray)):
    if unit is not None:
      assert isinstance(unit, Unit)
      return jnp.asarray(a, dtype=dtype, order=order) * unit
    else:
      return jnp.asarray(a, dtype=dtype, order=order)
    # list[Quantity]
  elif isinstance(a, Sequence):
    leaves, tree = jax.tree.flatten(a, is_leaf=lambda x: isinstance(x, Quantity))
    if all([isinstance(leaf, Quantity) for leaf in leaves]):
      # check all elements have the same unit
      if any(x.dim != leaves[0].dim for x in leaves):
        raise ValueError('Units do not match for asarray operation.')
      values = jax.tree.unflatten(tree, [x.value for x in a])
      if unit is not None:
        fail_for_dimension_mismatch(a[0], unit, error_message="a and unit have to have the same units.")
      unit = a[0].dim
      # Convert the values to a jnp.ndarray and create a Quantity object
      return Quantity(jnp.asarray(values, dtype=dtype, order=order), dim=unit)
    else:
      values = jax.tree.unflatten(tree, leaves)
      val = jnp.asarray(values, dtype=dtype, order=order)
      if unit is not None:
        return val * unit
      else:
        return val
  else:
    raise TypeError('Invalid input type for asarray.')


array = asarray


@set_module_as('brainunit.math')
def arange(*args, **kwargs):
  """
  Return a Quantity of `arange` and `unit`, with uninitialized values if `unit` is provided.

  Args:
    start: number, Quantity, optional
    stop: number, Quantity, optional
    step: number, optional
    dtype: dtype, optional
    unit: Unit, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if start and stop are Quantities that have the same unit, else an array.
  """
  # arange has a bit of a complicated argument structure unfortunately
  # we leave the actual checking of the number of arguments to numpy, though

  # default values
  start = kwargs.pop("start", 0)
  step = kwargs.pop("step", 1)
  stop = kwargs.pop("stop", None)
  if len(args) == 1:
    if stop is not None:
      raise TypeError("Duplicate definition of 'stop'")
    stop = args[0]
  elif len(args) == 2:
    if start != 0:
      raise TypeError("Duplicate definition of 'start'")
    if stop is not None:
      raise TypeError("Duplicate definition of 'stop'")
    start, stop = args
  elif len(args) == 3:
    if start != 0:
      raise TypeError("Duplicate definition of 'start'")
    if stop is not None:
      raise TypeError("Duplicate definition of 'stop'")
    if step != 1:
      raise TypeError("Duplicate definition of 'step'")
    start, stop, step = args
  elif len(args) > 3:
    raise TypeError("Need between 1 and 3 non-keyword arguments")

  if stop is None:
    raise TypeError("Missing stop argument.")
  if stop is not None and not is_unitless(stop):
    start = Quantity(start, dim=stop.dim)

  fail_for_dimension_mismatch(
    start,
    stop,
    error_message=(
      "Start value {start} and stop value {stop} have to have the same units."
    ),
    start=start,
    stop=stop,
  )
  fail_for_dimension_mismatch(
    stop,
    step,
    error_message=(
      "Stop value {stop} and step value {step} have to have the same units."
    ),
    stop=stop,
    step=step,
  )
  unit = getattr(stop, "dim", DIMENSIONLESS)
  # start is a position-only argument in numpy 2.0
  # https://numpy.org/devdocs/release/2.0.0-notes.html#arange-s-start-argument-is-positional-only
  # TODO: check whether this is still the case in the final release
  if start == 0:
    return Quantity(
      jnp.arange(
        start=start.value if isinstance(start, Quantity) else jnp.asarray(start),
        stop=stop.value if isinstance(stop, Quantity) else jnp.asarray(stop),
        step=step.value if isinstance(step, Quantity) else jnp.asarray(step),
        **kwargs,
      ),
      dim=unit,
    )
  else:
    return Quantity(
      jnp.arange(
        start.value if isinstance(start, Quantity) else jnp.asarray(start),
        stop=stop.value if isinstance(stop, Quantity) else jnp.asarray(stop),
        step=step.value if isinstance(step, Quantity) else jnp.asarray(step),
        **kwargs,
      ),
      dim=unit,
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
  Return a Quantity of `linspace` and `unit`, with uninitialized values if `unit` is provided.

  Args:
    start: number, Quantity
    stop: number, Quantity
    num: int, optional
    endpoint: bool, optional
    retstep: bool, optional
    dtype: dtype, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if start and stop are Quantities that have the same unit, else an array.
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
  return Quantity(result, dim=unit)


@set_module_as('brainunit.math')
def logspace(start: Union[Quantity, jax.typing.ArrayLike],
             stop: Union[Quantity, jax.typing.ArrayLike],
             num: Optional[int] = 50,
             endpoint: Optional[bool] = True,
             base: Optional[float] = 10.0,
             dtype: Optional[jax.typing.DTypeLike] = None):
  """
  Return a Quantity of `logspace` and `unit`, with uninitialized values if `unit` is provided.

  Args:
    start: number, Quantity
    stop: number, Quantity
    num: int, optional
    endpoint: bool, optional
    base: float, optional
    dtype: dtype, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if start and stop are Quantities that have the same unit, else an array.
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
  return Quantity(result, dim=unit)


@set_module_as('brainunit.math')
def fill_diagonal(a: Union[Quantity, jax.typing.ArrayLike],
                  val: Union[Quantity, jax.typing.ArrayLike],
                  wrap: Optional[bool] = False,
                  inplace: Optional[bool] = False) -> Union[Quantity, jax.Array]:
  """
  Fill the main diagonal of the given array of `a` with `val`.

  Args:
    a: array_like, Quantity
    val: scalar, Quantity
    wrap: bool, optional
    unit: Unit, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `a` and `val` are Quantities that have the same unit, else an array.
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
def array_split(ary: Union[Quantity, jax.typing.ArrayLike],
                indices_or_sections: Union[int, jax.typing.ArrayLike],
                axis: Optional[int] = 0) -> Union[list[Quantity], list[Array]]:
  """
  Split an array into multiple sub-arrays.

  Args:
    ary: array_like, Quantity
    indices_or_sections: int, array_like
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `ary` is a Quantity, else an array.
  """
  if isinstance(ary, Quantity):
    return [Quantity(x, dim=ary.dim) for x in jnp.array_split(ary.value, indices_or_sections, axis)]
  elif isinstance(ary, (jax.Array, np.ndarray)):
    return jnp.array_split(ary, indices_or_sections, axis)
  else:
    raise ValueError(f'Unsupported type: {type(ary)} for array_split')


@set_module_as('brainunit.math')
def meshgrid(*xi: Union[Quantity, jax.typing.ArrayLike],
             copy: Optional[bool] = True,
             sparse: Optional[bool] = False,
             indexing: Optional[str] = 'xy'):
  """
  Return coordinate matrices from coordinate vectors.

  Args:
    xi: array_like, Quantity
    copy: bool, optional
    sparse: bool, optional
    indexing: str, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `xi` are Quantities that have the same unit, else an array.
  """
  from builtins import all as origin_all
  if origin_all(isinstance(x, Quantity) for x in xi):
    fail_for_dimension_mismatch(*xi)
    return Quantity(jnp.meshgrid(*[x.value for x in xi], copy=copy, sparse=sparse, indexing=indexing), dim=xi[0].dim)
  elif origin_all(isinstance(x, (jax.Array, np.ndarray)) for x in xi):
    return jnp.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)
  else:
    raise ValueError(f'Unsupported types : {type(xi)} for meshgrid')


@set_module_as('brainunit.math')
def vander(x: Union[Quantity, jax.typing.ArrayLike],
           N: Optional[bool] = None,
           increasing: Optional[bool] = False) -> Union[Quantity, jax.Array]:
  """
  Generate a Vandermonde matrix.

  Args:
    x: array_like, Quantity
    N: int, optional
    increasing: bool, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  """
  if isinstance(x, Quantity):
    return Quantity(jnp.vander(x.value, N=N, increasing=increasing), dim=x.dim)
  elif isinstance(x, (jax.Array, np.ndarray)):
    return jnp.vander(x, N=N, increasing=increasing)
  else:
    raise ValueError(f'Unsupported type: {type(x)} for vander')
