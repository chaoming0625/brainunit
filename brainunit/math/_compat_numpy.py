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
import functools
from collections.abc import Sequence
from functools import wraps
from typing import (Callable, Union, Optional, Any, List)

import brainstate as bst
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum
from brainstate._utils import set_module_as
from jax import Array
from jax._src.numpy.lax_numpy import _einsum

from ._utils import _compatible_with_quantity
from .. import Quantity
from .._base import (DIMENSIONLESS,
                     Quantity,
                     Unit,
                     fail_for_dimension_mismatch,
                     is_unitless,
                     get_unit, )
from .._base import _return_check_unitless

__all__ = [
  # array creation
  'full', 'full_like', 'eye', 'identity', 'diag', 'tri', 'tril', 'triu',
  'empty', 'empty_like', 'ones', 'ones_like', 'zeros', 'zeros_like',
  'array', 'asarray', 'arange', 'linspace', 'logspace', 'fill_diagonal',
  'array_split', 'meshgrid', 'vander',

  # getting attribute funcs
  'ndim', 'isreal', 'isscalar', 'isfinite', 'isinf',
  'isnan', 'shape', 'size',

  # math funcs keep unit (unary)
  'real', 'imag', 'conj', 'conjugate', 'negative', 'positive',
  'abs', 'round', 'around', 'round_', 'rint',
  'floor', 'ceil', 'trunc', 'fix', 'sum', 'nancumsum', 'nansum',
  'cumsum', 'ediff1d', 'absolute', 'fabs', 'median',
  'nanmin', 'nanmax', 'ptp', 'average', 'mean', 'std',
  'nanmedian', 'nanmean', 'nanstd', 'diff', 'modf',

  # math funcs keep unit (binary)
  'fmod', 'mod', 'copysign', 'heaviside',
  'maximum', 'minimum', 'fmax', 'fmin', 'lcm', 'gcd',

  # math funcs keep unit (n-ary)
  'interp', 'clip',

  # math funcs match unit (binary)
  'add', 'subtract', 'nextafter',

  # math funcs change unit (unary)
  'reciprocal', 'prod', 'product', 'nancumprod', 'nanprod', 'cumprod',
  'cumproduct', 'var', 'nanvar', 'cbrt', 'square', 'frexp', 'sqrt',

  # math funcs change unit (binary)
  'multiply', 'divide', 'power', 'cross', 'ldexp',
  'true_divide', 'floor_divide', 'float_power',
  'divmod', 'remainder', 'convolve',

  # math funcs only accept unitless (unary)
  'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2',
  'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
  'arctanh', 'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan',
  'tanh', 'deg2rad', 'rad2deg', 'degrees', 'radians', 'angle',
  'percentile', 'nanpercentile', 'quantile', 'nanquantile',

  # math funcs only accept unitless (binary)
  'hypot', 'arctan2', 'logaddexp', 'logaddexp2',

  # math funcs remove unit (unary)
  'signbit', 'sign', 'histogram', 'bincount',

  # math funcs remove unit (binary)
  'corrcoef', 'correlate', 'cov', 'digitize',

  # array manipulation
  'reshape', 'moveaxis', 'transpose', 'swapaxes', 'row_stack',
  'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',
  'split', 'dsplit', 'hsplit', 'vsplit', 'tile', 'repeat', 'unique',
  'append', 'flip', 'fliplr', 'flipud', 'roll', 'atleast_1d', 'atleast_2d',
  'atleast_3d', 'expand_dims', 'squeeze', 'sort', 'argsort', 'argmax', 'argmin',
  'argwhere', 'nonzero', 'flatnonzero', 'searchsorted', 'extract',
  'count_nonzero', 'max', 'min', 'amax', 'amin', 'block', 'compress',
  'diagflat', 'diagonal', 'choose', 'ravel',

  # Elementwise bit operations (unary)
  'bitwise_not', 'invert', 'left_shift', 'right_shift',

  # Elementwise bit operations (binary)
  'bitwise_and', 'bitwise_or', 'bitwise_xor',

  # logic funcs (unary)
  'all', 'any', 'logical_not',

  # logic funcs (binary)
  'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
  'array_equal', 'isclose', 'allclose', 'logical_and',
  'logical_or', 'logical_xor', "alltrue", 'sometrue',

  # indexing funcs
  'nonzero', 'where', 'tril_indices', 'tril_indices_from', 'triu_indices',
  'triu_indices_from', 'take', 'select',

  # window funcs
  'bartlett', 'blackman', 'hamming', 'hanning', 'kaiser',

  # constants
  'e', 'pi', 'inf',

  # linear algebra
  'dot', 'vdot', 'inner', 'outer', 'kron', 'matmul', 'trace',

  # data types
  'dtype', 'finfo', 'iinfo',

  # more
  'broadcast_arrays', 'broadcast_shapes',
  'einsum', 'gradient', 'intersect1d', 'nan_to_num', 'nanargmax', 'nanargmin',
  'rot90', 'tensordot',

]


# array creation
# --------------

def wrap_array_creation_function(func):
  def f(*args, unit: Unit = None, **kwargs):
    if unit is not None:
      assert isinstance(unit, Unit), f'unit must be an instance of Unit, got {type(unit)}'
      return func(*args, **kwargs) * unit
    else:
      return func(*args, **kwargs)

  f.__module__ = 'brainunit.math'
  return f


# array creation
# --------------

full = wrap_array_creation_function(jnp.full)
eye = wrap_array_creation_function(jnp.eye)
identity = wrap_array_creation_function(jnp.identity)
tri = wrap_array_creation_function(jnp.tri)
empty = wrap_array_creation_function(jnp.empty)
ones = wrap_array_creation_function(jnp.ones)
zeros = wrap_array_creation_function(jnp.zeros)

# docs for full, eye, identity, tri, empty, ones, zeros

full.__doc__ = """
  Returns a Quantity of `shape` and `unit`, filled with `fill_value` if `unit` is provided.
  else return an array of `shape` filled with `fill_value`.

  Args:
    shape: sequence of integers, describing the shape of the output array.
    fill_value: the value to fill the new array with.
    dtype: the type of the output array, or `None`. If not `None`, `fill_value`
      will be cast to `dtype`.
    sharding: an optional sharding specification for the resulting array,
      note, sharding will currently be ignored in jitted mode, this might change
      in the future.
    unit: the unit of the output array, or `None`.
  
  Returns:
    out: Quantity if `unit` is provided, else an array.
"""

eye.__doc__ = """
  Returns a Quantity of `shape` and `unit`, representing an identity matrix if `unit` is provided.
  else return an identity matrix of `shape`.

  Args:
    n: the number of rows (and columns) in the output array.
    k: the index of the diagonal: 0 (the default) refers to the main diagonal,
       a positive value refers to an upper diagonal, and a negative value to a
       lower diagonal.
    dtype: the type of the output array, or `None`. If not `None`, elements
      will be cast to `dtype`.
    sharding: an optional sharding specification for the resulting array,
      note, sharding will currently be ignored in jitted mode, this might change
      in the future.
    unit: the unit of the output array, or `None`.
    
  Returns:
    out: Quantity if `unit` is provided, else an array.
"""

identity.__doc__ = """
  Returns a Quantity of `shape` and `unit`, representing an identity matrix if `unit` is provided.
  else return an identity matrix of `shape`.

  Args:
    n: the number of rows (and columns) in the output array.
    dtype: the type of the output array, or `None`. If not `None`, elements
      will be cast to `dtype`.
    sharding: an optional sharding specification for the resulting array,
      note, sharding will currently be ignored in jitted mode, this might change
      in the future.
    unit: the unit of the output array, or `None`.
    
  Returns:
    out: Quantity if `unit` is provided, else an array.
"""

tri.__doc__ = """
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
    sharding: an optional sharding specification for the resulting array,
      note, sharding will currently be ignored in jitted mode, this might change
      in the future.
    unit: the unit of the output array, or `None`.
    
  Returns:
    out: Quantity if `unit` is provided, else an array.
"""

# empty
empty.__doc__ = """
  Returns a Quantity of `shape` and `unit`, with uninitialized values if `unit` is provided.
  else return an array of `shape` with uninitialized values.

  Args:
    shape: sequence of integers, describing the shape of the output array.
    dtype: the type of the output array, or `None`. If not `None`, elements
      will be of type `dtype`.
    sharding: an optional sharding specification for the resulting array,
      note, sharding will currently be ignored in jitted mode, this might change
      in the future.
    unit: the unit of the output array, or `None`.
    
  Returns:
    out: Quantity if `unit` is provided, else an array.
"""

# ones
ones.__doc__ = """
  Returns a Quantity of `shape` and `unit`, filled with 1 if `unit` is provided.
  else return an array of `shape` filled with 1.

  Args:
    shape: sequence of integers, describing the shape of the output array.
    dtype: the type of the output array, or `None`. If not `None`, elements
      will be cast to `dtype`.
    sharding: an optional sharding specification for the resulting array,
      note, sharding will currently be ignored in jitted mode, this might change
      in the future.
    unit: the unit of the output array, or `None`.
    
  Returns:
    out: Quantity if `unit` is provided, else an array.
"""

# zeros
zeros.__doc__ = """
  Returns a Quantity of `shape` and `unit`, filled with 0 if `unit` is provided.
  else return an array of `shape` filled with 0.

  Args:
    shape: sequence of integers, describing the shape of the output array.
    dtype: the type of the output array, or `None`. If not `None`, elements
      will be cast to `dtype`.
    sharding: an optional sharding specification for the resulting array,
      note, sharding will currently be ignored in jitted mode, this might change
      in the future.
    unit: the unit of the output array, or `None`.
    
  Returns:
    out: Quantity if `unit` is provided, else an array.
"""


@set_module_as('brainunit.math')
def full_like(a: Union[Quantity, bst.typing.ArrayLike],
              fill_value: Union[bst.typing.ArrayLike],
              unit: Unit = None,
              dtype: Optional[bst.typing.DTypeLike] = None,
              shape: Any = None) -> Union[Quantity, jax.Array]:
  '''
  Return a Quantity of `a` and `unit`, filled with `fill_value` if `unit` is provided.
  else return an array of `a` filled with `fill_value`.

  Args:
    a: array_like, Quantity, shape, or dtype
    fill_value: scalar or array_like
    unit: Unit, optional
    dtype: data-type, optional
    shape: sequence of ints, optional

  Returns:
    out: Quantity if `unit` is provided, else an array.
  '''
  if unit is not None:
    assert isinstance(unit, Unit)
    if isinstance(a, Quantity):
      return jnp.full_like(a.value, fill_value, dtype=dtype, shape=shape) * unit
    else:
      return jnp.full_like(a, fill_value, dtype=dtype, shape=shape) * unit
  else:
    return jnp.full_like(a, fill_value, dtype=dtype, shape=shape)


@set_module_as('brainunit.math')
def diag(a: Union[Quantity, bst.typing.ArrayLike],
         k: int = 0,
         unit: Unit = None) -> Union[Quantity, jax.Array]:
  '''
  Extract a diagonal or construct a diagonal array.

  Args:
    a: array_like, Quantity
    k: int, optional
    unit: Unit, optional

  Returns:
    out: Quantity if `unit` is provided, else an array.
  '''
  if unit is not None:
    assert isinstance(unit, Unit)
    if isinstance(a, Quantity):
      return jnp.diag(a.value, k=k) * unit
    else:
      return jnp.diag(a, k=k) * unit
  else:
    return jnp.diag(a, k=k)


@set_module_as('brainunit.math')
def tril(a: Union[Quantity, bst.typing.ArrayLike],
         k: int = 0,
         unit: Unit = None) -> Union[Quantity, jax.Array]:
  '''
  Lower triangle of an array.

  Args:
    a: array_like, Quantity
    k: int, optional
    unit: Unit, optional

  Returns:
    out: Quantity if `unit` is provided, else an array.
  '''
  if unit is not None:
    assert isinstance(unit, Unit)
    if isinstance(a, Quantity):
      return jnp.tril(a.value, k=k) * unit
    else:
      return jnp.tril(a, k=k) * unit
  else:
    return jnp.tril(a, k=k)


@set_module_as('brainunit.math')
def triu(a: Union[Quantity, bst.typing.ArrayLike],
         k: int = 0,
         unit: Unit = None) -> Union[Quantity, jax.Array]:
  '''
  Upper triangle of an array.

  Args:
    a: array_like, Quantity
    k: int, optional
    unit: Unit, optional

  Returns:
    out: Quantity if `unit` is provided, else an array.
  '''
  if unit is not None:
    assert isinstance(unit, Unit)
    if isinstance(a, Quantity):
      return jnp.triu(a.value, k=k) * unit
    else:
      return jnp.triu(a, k=k) * unit
  else:
    return jnp.triu(a, k=k)


@set_module_as('brainunit.math')
def empty_like(a: Union[Quantity, bst.typing.ArrayLike],
               dtype: Optional[bst.typing.DTypeLike] = None,
               shape: Any = None,
               unit: Unit = None) -> Union[Quantity, jax.Array]:
  '''
  Return a Quantity of `a` and `unit`, with uninitialized values if `unit` is provided.
  else return an array of `a` with uninitialized values.

  Args:
    a: array_like, Quantity, shape, or dtype
    dtype: data-type, optional
    shape: sequence of ints, optional
    unit: Unit, optional

  Returns:
    out: Quantity if `unit` is provided, else an array.
  '''
  if unit is not None:
    assert isinstance(unit, Unit)
    if isinstance(a, Quantity):
      return jnp.empty_like(a.value, dtype=dtype, shape=shape) * unit
    else:
      return jnp.empty_like(a, dtype=dtype, shape=shape) * unit
  else:
    return jnp.empty_like(a, dtype=dtype, shape=shape)


@set_module_as('brainunit.math')
def ones_like(a: Union[Quantity, bst.typing.ArrayLike],
              dtype: Optional[bst.typing.DTypeLike] = None,
              shape: Any = None,
              unit: Unit = None) -> Union[Quantity, jax.Array]:
  '''
  Return a Quantity of `a` and `unit`, filled with 1 if `unit` is provided.
  else return an array of `a` filled with 1.

  Args:
    a: array_like, Quantity, shape, or dtype
    dtype: data-type, optional
    shape: sequence of ints, optional
    unit: Unit, optional

  Returns:
    out: Quantity if `unit` is provided, else an array.
  '''
  if unit is not None:
    assert isinstance(unit, Unit)
    if isinstance(a, Quantity):
      return jnp.ones_like(a.value, dtype=dtype, shape=shape) * unit
    else:
      return jnp.ones_like(a, dtype=dtype, shape=shape) * unit
  else:
    return jnp.ones_like(a, dtype=dtype, shape=shape)


@set_module_as('brainunit.math')
def zeros_like(a: Union[Quantity, bst.typing.ArrayLike],
               dtype: Optional[bst.typing.DTypeLike] = None,
               shape: Any = None,
               unit: Unit = None) -> Union[Quantity, jax.Array]:
  '''
  Return a Quantity of `a` and `unit`, filled with 0 if `unit` is provided.
  else return an array of `a` filled with 0.

  Args:
    a: array_like, Quantity, shape, or dtype
    dtype: data-type, optional
    shape: sequence of ints, optional
    unit: Unit, optional

  Returns:
    out: Quantity if `unit` is provided, else an array.
  '''
  if unit is not None:
    assert isinstance(unit, Unit)
    if isinstance(a, Quantity):
      return jnp.zeros_like(a.value, dtype=dtype, shape=shape) * unit
    else:
      return jnp.zeros_like(a, dtype=dtype, shape=shape) * unit
  else:
    return jnp.zeros_like(a, dtype=dtype, shape=shape)


@set_module_as('brainunit.math')
def asarray(
    a: Union[Quantity, bst.typing.ArrayLike, Sequence[Quantity]],
    dtype: Optional[bst.typing.DTypeLike] = None,
    order: Optional[str] = None,
    unit: Optional[Unit] = None,
) -> Union[Quantity, jax.Array]:
  from builtins import all as origin_all
  from builtins import any as origin_any
  if isinstance(a, Quantity):
    return Quantity(jnp.asarray(a.value, dtype=dtype, order=order), unit=a.unit)
  elif isinstance(a, (jax.Array, np.ndarray)):
    return jnp.asarray(a, dtype=dtype, order=order)
  # list[Quantity]
  elif isinstance(a, Sequence) and origin_all(isinstance(x, Quantity) for x in a):
    # check all elements have the same unit
    if origin_any(x.unit != a[0].unit for x in a):
      raise ValueError('Units do not match for asarray operation.')
    values = [x.value for x in a]
    unit = a[0].unit
    # Convert the values to a jnp.ndarray and create a Quantity object
    return Quantity(jnp.asarray(values, dtype=dtype, order=order), unit=unit)
  else:
    return jnp.asarray(a, dtype=dtype, order=order)


array = asarray


@set_module_as('brainunit.math')
def arange(*args, **kwargs):
  '''
  Return a Quantity of `arange` and `unit`, with uninitialized values if `unit` is provided.

  Args:
    start: number, Quantity, optional
    stop: number, Quantity, optional
    step: number, optional
    dtype: dtype, optional
    unit: Unit, optional

  Returns:
    out: Quantity if start and stop are Quantities that have the same unit, else an array.
  '''
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
    start = Quantity(start, unit=stop.unit)

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
  unit = getattr(stop, "unit", DIMENSIONLESS)
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
      unit=unit,
    )
  else:
    return Quantity(
      jnp.arange(
        start.value if isinstance(start, Quantity) else jnp.asarray(start),
        stop=stop.value if isinstance(stop, Quantity) else jnp.asarray(stop),
        step=step.value if isinstance(step, Quantity) else jnp.asarray(step),
        **kwargs,
      ),
      unit=unit,
    )


@set_module_as('brainunit.math')
def linspace(start: Union[Quantity, bst.typing.ArrayLike],
             stop: Union[Quantity, bst.typing.ArrayLike],
             num: int = 50,
             endpoint: Optional[bool] = True,
             retstep: Optional[bool] = False,
             dtype: Optional[bst.typing.DTypeLike] = None) -> Union[Quantity, jax.Array]:
  '''
  Return a Quantity of `linspace` and `unit`, with uninitialized values if `unit` is provided.

  Args:
    start: number, Quantity
    stop: number, Quantity
    num: int, optional
    endpoint: bool, optional
    retstep: bool, optional
    dtype: dtype, optional

  Returns:
    out: Quantity if start and stop are Quantities that have the same unit, else an array.
  '''
  fail_for_dimension_mismatch(
    start,
    stop,
    error_message="Start value {start} and stop value {stop} have to have the same units.",
    start=start,
    stop=stop,
  )
  unit = getattr(start, "unit", DIMENSIONLESS)
  start = start.value if isinstance(start, Quantity) else start
  stop = stop.value if isinstance(stop, Quantity) else stop

  result = jnp.linspace(start, stop, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype)
  return Quantity(result, unit=unit)


@set_module_as('brainunit.math')
def logspace(start: Union[Quantity, bst.typing.ArrayLike],
             stop: Union[Quantity, bst.typing.ArrayLike],
             num: Optional[int] = 50,
             endpoint: Optional[bool] = True,
             base: Optional[float] = 10.0,
             dtype: Optional[bst.typing.DTypeLike] = None):
  '''
  Return a Quantity of `logspace` and `unit`, with uninitialized values if `unit` is provided.

  Args:
    start: number, Quantity
    stop: number, Quantity
    num: int, optional
    endpoint: bool, optional
    base: float, optional
    dtype: dtype, optional

  Returns:
    out: Quantity if start and stop are Quantities that have the same unit, else an array.
  '''
  fail_for_dimension_mismatch(
    start,
    stop,
    error_message="Start value {start} and stop value {stop} have to have the same units.",
    start=start,
    stop=stop,
  )
  unit = getattr(start, "unit", DIMENSIONLESS)
  start = start.value if isinstance(start, Quantity) else start
  stop = stop.value if isinstance(stop, Quantity) else stop

  result = jnp.logspace(start, stop, num=num, endpoint=endpoint, base=base, dtype=dtype)
  return Quantity(result, unit=unit)


@set_module_as('brainunit.math')
def fill_diagonal(a: Union[Quantity, bst.typing.ArrayLike],
                  val: Union[Quantity, bst.typing.ArrayLike],
                  wrap: Optional[bool] = False,
                  inplace: Optional[bool] = True) -> Union[Quantity, jax.Array]:
  '''
  Fill the main diagonal of the given array of `a` with `val`.

  Args:
    a: array_like, Quantity
    val: scalar, Quantity
    wrap: bool, optional
    inplace: bool, optional

  Returns:
    out: Quantity if `a` and `val` are Quantities that have the same unit, else an array.
  '''
  if isinstance(a, Quantity) and isinstance(val, Quantity):
    fail_for_dimension_mismatch(a, val)
    return Quantity(jnp.fill_diagonal(a.value, val.value, wrap=wrap, inplace=inplace), unit=a.unit)
  elif isinstance(a, (jax.Array, np.ndarray)) and isinstance(val, (jax.Array, np.ndarray)):
    return jnp.fill_diagonal(a, val, wrap=wrap, inplace=inplace)
  elif is_unitless(a) or is_unitless(val):
    return jnp.fill_diagonal(a, val, wrap=wrap, inplace=inplace)
  else:
    raise ValueError(f'Unsupported types : {type(a)} abd {type(val)} for fill_diagonal')


@set_module_as('brainunit.math')
def array_split(ary: Union[Quantity, bst.typing.ArrayLike],
                indices_or_sections: Union[int, bst.typing.ArrayLike],
                axis: Optional[int] = 0) -> list[Quantity] | list[Array]:
  '''
  Split an array into multiple sub-arrays.

  Args:
    ary: array_like, Quantity
    indices_or_sections: int, array_like
    axis: int, optional

  Returns:
    out: Quantity if `ary` is a Quantity, else an array.
  '''
  if isinstance(ary, Quantity):
    return [Quantity(x, unit=ary.unit) for x in jnp.array_split(ary.value, indices_or_sections, axis)]
  elif isinstance(ary, (bst.typing.ArrayLike)):
    return jnp.array_split(ary, indices_or_sections, axis)
  else:
    raise ValueError(f'Unsupported type: {type(ary)} for array_split')


@set_module_as('brainunit.math')
def meshgrid(*xi: Union[Quantity, bst.typing.ArrayLike],
             copy: Optional[bool] = True,
             sparse: Optional[bool] = False,
             indexing: Optional[str] = 'xy'):
  '''
  Return coordinate matrices from coordinate vectors.

  Args:
    xi: array_like, Quantity
    copy: bool, optional
    sparse: bool, optional
    indexing: str, optional

  Returns:
    out: Quantity if `xi` are Quantities that have the same unit, else an array.
  '''
  from builtins import all as origin_all
  if origin_all(isinstance(x, Quantity) for x in xi):
    fail_for_dimension_mismatch(*xi)
    return Quantity(jnp.meshgrid(*[x.value for x in xi], copy=copy, sparse=sparse, indexing=indexing), unit=xi[0].unit)
  elif origin_all(isinstance(x, (jax.Array, np.ndarray)) for x in xi):
    return jnp.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)
  else:
    raise ValueError(f'Unsupported types : {type(xi)} for meshgrid')


@set_module_as('brainunit.math')
def vander(x: Union[Quantity, bst.typing.ArrayLike],
           N: Optional[bool] = None,
           increasing: Optional[bool] = False) -> Union[Quantity, jax.Array]:
  '''
  Generate a Vandermonde matrix.

  Args:
    x: array_like, Quantity
    N: int, optional
    increasing: bool, optional

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
  '''
  if isinstance(x, Quantity):
    return Quantity(jnp.vander(x.value, N=N, increasing=increasing), unit=x.unit)
  elif isinstance(x, (jax.Array, np.ndarray)):
    return jnp.vander(x, N=N, increasing=increasing)
  else:
    raise ValueError(f'Unsupported type: {type(x)} for vander')


# getting attribute funcs
# -----------------------

@set_module_as('brainunit.math')
def ndim(a: Union[Quantity, bst.typing.ArrayLike]) -> int:
  '''
  Return the number of dimensions of an array.

  Args:
    a: array_like, Quantity

  Returns:
    out: int
  '''
  if isinstance(a, Quantity):
    return a.ndim
  else:
    return jnp.ndim(a)


@set_module_as('brainunit.math')
def isreal(a: Union[Quantity, bst.typing.ArrayLike]) -> jax.Array:
  '''
  Return True if the input array is real.

  Args:
    a: array_like, Quantity

  Returns:
    out: boolean array
  '''
  if isinstance(a, Quantity):
    return a.isreal
  else:
    return jnp.isreal(a)


@set_module_as('brainunit.math')
def isscalar(a: Union[Quantity, bst.typing.ArrayLike]) -> bool:
  '''
  Return True if the input is a scalar.

  Args:
    a: array_like, Quantity

  Returns:
    out: boolean array
  '''
  if isinstance(a, Quantity):
    return a.isscalar
  else:
    return jnp.isscalar(a)


@set_module_as('brainunit.math')
def isfinite(a: Union[Quantity, bst.typing.ArrayLike]) -> jax.Array:
  '''
  Return each element of the array is finite or not.

  Args:
    a: array_like, Quantity

  Returns:
    out: boolean array
  '''
  if isinstance(a, Quantity):
    return a.isfinite
  else:
    return jnp.isfinite(a)


@set_module_as('brainunit.math')
def isinf(a: Union[Quantity, bst.typing.ArrayLike]) -> jax.Array:
  '''
  Return each element of the array is infinite or not.

  Args:
    a: array_like, Quantity

  Returns:
    out: boolean array
  '''
  if isinstance(a, Quantity):
    return a.isinf
  else:
    return jnp.isinf(a)


@set_module_as('brainunit.math')
def isnan(a: Union[Quantity, bst.typing.ArrayLike]) -> jax.Array:
  '''
  Return each element of the array is NaN or not.

  Args:
    a: array_like, Quantity

  Returns:
    out: boolean array
  '''
  if isinstance(a, Quantity):
    return a.isnan
  else:
    return jnp.isnan(a)


@set_module_as('brainunit.math')
def shape(a: Union[Quantity, bst.typing.ArrayLike]) -> tuple[int, ...]:
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
def size(a: Union[Quantity, bst.typing.ArrayLike], axis: int = None) -> int:
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


# math funcs keep unit (unary)
# ----------------------------

def wrap_math_funcs_keep_unit_unary(func):
  def f(x, *args, **kwargs):
    if isinstance(x, Quantity):
      return Quantity(func(x.value, *args, **kwargs), unit=x.unit)
    elif isinstance(x, (jax.Array, np.ndarray)):
      return func(x, *args, **kwargs)
    else:
      raise ValueError(f'Unsupported type: {type(x)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


real = wrap_math_funcs_keep_unit_unary(jnp.real)
imag = wrap_math_funcs_keep_unit_unary(jnp.imag)
conj = wrap_math_funcs_keep_unit_unary(jnp.conj)
conjugate = wrap_math_funcs_keep_unit_unary(jnp.conjugate)
negative = wrap_math_funcs_keep_unit_unary(jnp.negative)
positive = wrap_math_funcs_keep_unit_unary(jnp.positive)
abs = wrap_math_funcs_keep_unit_unary(jnp.abs)
round_ = wrap_math_funcs_keep_unit_unary(jnp.round)
around = wrap_math_funcs_keep_unit_unary(jnp.around)
round = wrap_math_funcs_keep_unit_unary(jnp.round)
rint = wrap_math_funcs_keep_unit_unary(jnp.rint)
floor = wrap_math_funcs_keep_unit_unary(jnp.floor)
ceil = wrap_math_funcs_keep_unit_unary(jnp.ceil)
trunc = wrap_math_funcs_keep_unit_unary(jnp.trunc)
fix = wrap_math_funcs_keep_unit_unary(jnp.fix)
sum = wrap_math_funcs_keep_unit_unary(jnp.sum)
nancumsum = wrap_math_funcs_keep_unit_unary(jnp.nancumsum)
nansum = wrap_math_funcs_keep_unit_unary(jnp.nansum)
cumsum = wrap_math_funcs_keep_unit_unary(jnp.cumsum)
ediff1d = wrap_math_funcs_keep_unit_unary(jnp.ediff1d)
absolute = wrap_math_funcs_keep_unit_unary(jnp.absolute)
fabs = wrap_math_funcs_keep_unit_unary(jnp.fabs)
median = wrap_math_funcs_keep_unit_unary(jnp.median)
nanmin = wrap_math_funcs_keep_unit_unary(jnp.nanmin)
nanmax = wrap_math_funcs_keep_unit_unary(jnp.nanmax)
ptp = wrap_math_funcs_keep_unit_unary(jnp.ptp)
average = wrap_math_funcs_keep_unit_unary(jnp.average)
mean = wrap_math_funcs_keep_unit_unary(jnp.mean)
std = wrap_math_funcs_keep_unit_unary(jnp.std)
nanmedian = wrap_math_funcs_keep_unit_unary(jnp.nanmedian)
nanmean = wrap_math_funcs_keep_unit_unary(jnp.nanmean)
nanstd = wrap_math_funcs_keep_unit_unary(jnp.nanstd)
diff = wrap_math_funcs_keep_unit_unary(jnp.diff)
modf = wrap_math_funcs_keep_unit_unary(jnp.modf)

# docs for the functions above
real.__doc__ = '''
  Return the real part of the complex argument.
  
  Args:
    x: array_like, Quantity
    
  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

imag.__doc__ = '''
  Return the imaginary part of the complex argument.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

conj.__doc__ = '''
  Return the complex conjugate of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

conjugate.__doc__ = '''
  Return the complex conjugate of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

negative.__doc__ = '''
  Return the negative of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

positive.__doc__ = '''
  Return the positive of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

abs.__doc__ = '''
  Return the absolute value of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

round_.__doc__ = '''
  Round an array to the nearest integer.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

around.__doc__ = '''
  Round an array to the nearest integer.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

round.__doc__ = '''
  Round an array to the nearest integer.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

rint.__doc__ = '''
  Round an array to the nearest integer.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

floor.__doc__ = '''
  Return the floor of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

ceil.__doc__ = '''
  Return the ceiling of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

trunc.__doc__ = '''
  Return the truncated value of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

fix.__doc__ = '''
  Return the nearest integer towards zero.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

sum.__doc__ = '''
  Return the sum of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

nancumsum.__doc__ = '''
  Return the cumulative sum of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

nansum.__doc__ = '''
  Return the sum of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

cumsum.__doc__ = '''
  Return the cumulative sum of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

ediff1d.__doc__ = '''
  Return the differences between consecutive elements of the array.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

absolute.__doc__ = '''
  Return the absolute value of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

fabs.__doc__ = '''
  Return the absolute value of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

median.__doc__ = '''
  Return the median of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

nanmin.__doc__ = '''
  Return the minimum of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

nanmax.__doc__ = '''
  Return the maximum of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

ptp.__doc__ = '''
  Return the range of the array elements (maximum - minimum).

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

average.__doc__ = '''
  Return the weighted average of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

mean.__doc__ = '''
  Return the mean of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

std.__doc__ = '''
  Return the standard deviation of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

nanmedian.__doc__ = '''
  Return the median of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

nanmean.__doc__ = '''
  Return the mean of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

nanstd.__doc__ = '''
  Return the standard deviation of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

diff.__doc__ = '''
  Return the differences between consecutive elements of the array.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''

modf.__doc__ = '''
  Return the fractional and integer parts of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity tuple if `x` is a Quantity, else an array tuple.
'''


# math funcs keep unit (binary)
# -----------------------------

def wrap_math_funcs_keep_unit_binary(func):
  def f(x1, x2, *args, **kwargs):
    if isinstance(x1, Quantity) and isinstance(x2, Quantity):
      return Quantity(func(x1.value, x2.value, *args, **kwargs), unit=x1.unit)
    elif isinstance(x1, (jax.Array, np.ndarray)) and isinstance(x2, (jax.Array, np.ndarray)):
      return func(x1, x2, *args, **kwargs)
    else:
      raise ValueError(f'Unsupported type: {type(x1)} and {type(x2)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


fmod = wrap_math_funcs_keep_unit_binary(jnp.fmod)
mod = wrap_math_funcs_keep_unit_binary(jnp.mod)
copysign = wrap_math_funcs_keep_unit_binary(jnp.copysign)
heaviside = wrap_math_funcs_keep_unit_binary(jnp.heaviside)
maximum = wrap_math_funcs_keep_unit_binary(jnp.maximum)
minimum = wrap_math_funcs_keep_unit_binary(jnp.minimum)
fmax = wrap_math_funcs_keep_unit_binary(jnp.fmax)
fmin = wrap_math_funcs_keep_unit_binary(jnp.fmin)
lcm = wrap_math_funcs_keep_unit_binary(jnp.lcm)
gcd = wrap_math_funcs_keep_unit_binary(jnp.gcd)

# docs for the functions above
fmod.__doc__ = '''
  Return the element-wise remainder of division.
  
  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity
    
  Returns:
    out: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

mod.__doc__ = '''
  Return the element-wise modulus of division.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

copysign.__doc__ = '''
  Return a copy of the first array elements with the sign of the second array.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

heaviside.__doc__ = '''
  Compute the Heaviside step function.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

maximum.__doc__ = '''
  Element-wise maximum of array elements.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

minimum.__doc__ = '''
  Element-wise minimum of array elements.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

fmax.__doc__ = '''
  Element-wise maximum of array elements ignoring NaNs.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

fmin.__doc__ = '''
  Element-wise minimum of array elements ignoring NaNs.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

lcm.__doc__ = '''
  Return the least common multiple of `x1` and `x2`.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

gcd.__doc__ = '''
  Return the greatest common divisor of `x1` and `x2`.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''


# math funcs keep unit (n-ary)
# ----------------------------
@set_module_as('brainunit.math')
def interp(x: Union[Quantity, bst.typing.ArrayLike],
           xp: Union[Quantity, bst.typing.ArrayLike],
           fp: Union[Quantity, bst.typing.ArrayLike],
           left: Union[Quantity, bst.typing.ArrayLike] = None,
           right: Union[Quantity, bst.typing.ArrayLike] = None,
           period: Union[Quantity, bst.typing.ArrayLike] = None) -> Union[Quantity, jax.Array]:
  '''
  One-dimensional linear interpolation.

  Args:
    x: array_like, Quantity
    xp: array_like, Quantity
    fp: array_like, Quantity
    left: array_like, Quantity, optional
    right: array_like, Quantity, optional
    period: array_like, Quantity, optional

  Returns:
    out: Quantity if `x`, `xp`, and `fp` are Quantities that have the same unit, else an array.
  '''
  unit = None
  if isinstance(x, Quantity) or isinstance(xp, Quantity) or isinstance(fp, Quantity):
    unit = x.unit if isinstance(x, Quantity) else xp.unit if isinstance(xp, Quantity) else fp.unit
  if isinstance(x, Quantity):
    x_value = x.value
  else:
    x_value = x
  if isinstance(xp, Quantity):
    xp_value = xp.value
  else:
    xp_value = xp
  if isinstance(fp, Quantity):
    fp_value = fp.value
  else:
    fp_value = fp
  result = jnp.interp(x_value, xp_value, fp_value, left=left, right=right, period=period)
  if unit is not None:
    return Quantity(result, unit=unit)
  else:
    return result


@set_module_as('brainunit.math')
def clip(a: Union[Quantity, bst.typing.ArrayLike],
         a_min: Union[Quantity, bst.typing.ArrayLike],
         a_max: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Clip (limit) the values in an array.

  Args:
    a: array_like, Quantity
    a_min: array_like, Quantity
    a_max: array_like, Quantity

  Returns:
    out: Quantity if `a`, `a_min`, and `a_max` are Quantities that have the same unit, else an array.
  '''
  unit = None
  if isinstance(a, Quantity) or isinstance(a_min, Quantity) or isinstance(a_max, Quantity):
    unit = a.unit if isinstance(a, Quantity) else a_min.unit if isinstance(a_min, Quantity) else a_max.unit
  if isinstance(a, Quantity):
    a_value = a.value
  else:
    a_value = a
  if isinstance(a_min, Quantity):
    a_min_value = a_min.value
  else:
    a_min_value = a_min
  if isinstance(a_max, Quantity):
    a_max_value = a_max.value
  else:
    a_max_value = a_max
  result = jnp.clip(a_value, a_min_value, a_max_value)
  if unit is not None:
    return Quantity(result, unit=unit)
  else:
    return result


# math funcs match unit (binary)
# ------------------------------

def wrap_math_funcs_match_unit_binary(func):
  def f(x, y, *args, **kwargs):
    if isinstance(x, Quantity) and isinstance(y, Quantity):
      fail_for_dimension_mismatch(x, y)
      return Quantity(func(x.value, y.value, *args, **kwargs), unit=x.unit)
    elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
      return func(x, y, *args, **kwargs)
    elif isinstance(x, Quantity):
      if x.is_unitless:
        return Quantity(func(x.value, y, *args, **kwargs), unit=x.unit)
      else:
        raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {func.__name__}')
    elif isinstance(y, Quantity):
      if y.is_unitless:
        return Quantity(func(x, y.value, *args, **kwargs), unit=y.unit)
      else:
        raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {func.__name__}')
    else:
      raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


add = wrap_math_funcs_match_unit_binary(jnp.add)
subtract = wrap_math_funcs_match_unit_binary(jnp.subtract)
nextafter = wrap_math_funcs_match_unit_binary(jnp.nextafter)

# docs for the functions above
add.__doc__ = '''
  Add arguments element-wise.
  
  Args:
    x: array_like, Quantity
    y: array_like, Quantity
    
  Returns:
    out: Quantity if `x` and `y` are Quantities that have the same unit, else an array.
'''

subtract.__doc__ = '''
  Subtract arguments element-wise.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    out: Quantity if `x` and `y` are Quantities that have the same unit, else an array.
'''

nextafter.__doc__ = '''
  Return the next floating-point value after `x1` towards `x2`.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''


# math funcs change unit (unary)
# ------------------------------

def wrap_math_funcs_change_unit_unary(func, change_unit_func):
  def f(x, *args, **kwargs):
    if isinstance(x, Quantity):
      return _return_check_unitless(Quantity(func(x.value, *args, **kwargs), unit=change_unit_func(x.unit)))
    elif isinstance(x, (jax.Array, np.ndarray)):
      return func(x, *args, **kwargs)
    else:
      raise ValueError(f'Unsupported type: {type(x)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


reciprocal = wrap_math_funcs_change_unit_unary(jnp.reciprocal, lambda x: x ** -1)
reciprocal.__doc__ = '''
  Return the reciprocal of the argument.
  
  Args:
    x: array_like, Quantity
    
  Returns:
    out: Quantity if `x` is a Quantity, else an array.
'''


@set_module_as('brainunit.math')
def prod(x: Union[Quantity, bst.typing.ArrayLike],
         axis: Optional[int] = None,
         dtype: Optional[bst.typing.DTypeLike] = None,
         out: Optional[...] = None,
         keepdims: Optional[bool] = False,
         initial: Union[Quantity, bst.typing.ArrayLike] = None,
         where: Union[Quantity, bst.typing.ArrayLike] = None,
         promote_integers: bool = True) -> Union[Quantity, jax.Array]:
  '''
  Return the product of array elements over a given axis.

  Args:
    x: array_like, Quantity
    axis: int, optional
    dtype: dtype, optional
    out: array, optional
    keepdims: bool, optional
    initial: array_like, Quantity, optional
    where: array_like, Quantity, optional
    promote_integers: bool, optional

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
  '''
  if isinstance(x, Quantity):
    return x.prod(axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where,
                  promote_integers=promote_integers)
  else:
    return jnp.prod(x, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where,
                    promote_integers=promote_integers)


@set_module_as('brainunit.math')
def nanprod(x: Union[Quantity, bst.typing.ArrayLike],
            axis: Optional[int] = None,
            dtype: Optional[bst.typing.DTypeLike] = None,
            out: Optional[...] = None,
            keepdims: Optional[...] = False,
            initial: Union[Quantity, bst.typing.ArrayLike] = None,
            where: Union[Quantity, bst.typing.ArrayLike] = None):
  '''
  Return the product of array elements over a given axis treating Not a Numbers (NaNs) as one.

  Args:
    x: array_like, Quantity
    axis: int, optional
    dtype: dtype, optional
    out: array, optional
    keepdims: bool, optional
    initial: array_like, Quantity, optional
    where: array_like, Quantity, optional

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
  '''
  if isinstance(x, Quantity):
    return x.nanprod(axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)
  else:
    return jnp.nanprod(x, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)


product = prod


@set_module_as('brainunit.math')
def cumprod(x: Union[Quantity, bst.typing.ArrayLike],
            axis: Optional[int] = None,
            dtype: Optional[bst.typing.DTypeLike] = None,
            out: Optional[...] = None) -> Union[Quantity, bst.typing.ArrayLike]:
  '''
  Return the cumulative product of elements along a given axis.

  Args:
    x: array_like, Quantity
    axis: int, optional
    dtype: dtype, optional
    out: array, optional

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
  '''
  if isinstance(x, Quantity):
    return x.cumprod(axis=axis, dtype=dtype, out=out)
  else:
    return jnp.cumprod(x, axis=axis, dtype=dtype, out=out)


@set_module_as('brainunit.math')
def nancumprod(x: Union[Quantity, bst.typing.ArrayLike],
               axis: Optional[int] = None,
               dtype: Optional[bst.typing.DTypeLike] = None,
               out: Optional[...] = None) -> Union[Quantity, bst.typing.ArrayLike]:
  '''
  Return the cumulative product of elements along a given axis treating Not a Numbers (NaNs) as one.

  Args:
    x: array_like, Quantity
    axis: int, optional
    dtype: dtype, optional
    out: array, optional

  Returns:
    out: Quantity if `x` is a Quantity, else an array.
  '''
  if isinstance(x, Quantity):
    return x.nancumprod(axis=axis, dtype=dtype, out=out)
  else:
    return jnp.nancumprod(x, axis=axis, dtype=dtype, out=out)


cumproduct = cumprod

var = wrap_math_funcs_change_unit_unary(jnp.var, lambda x: x ** 2)
nanvar = wrap_math_funcs_change_unit_unary(jnp.nanvar, lambda x: x ** 2)
frexp = wrap_math_funcs_change_unit_unary(jnp.frexp, lambda x, y: x * 2 ** y)
sqrt = wrap_math_funcs_change_unit_unary(jnp.sqrt, lambda x: x ** 0.5)
cbrt = wrap_math_funcs_change_unit_unary(jnp.cbrt, lambda x: x ** (1 / 3))
square = wrap_math_funcs_change_unit_unary(jnp.square, lambda x: x ** 2)

# docs for the functions above
var.__doc__ = '''
  Compute the variance along the specified axis.
  
  Args:
    x: array_like, Quantity
    
  Returns:
    out: Quantity if the final unit is the square of the unit of `x`, else an array.
'''

nanvar.__doc__ = '''
  Compute the variance along the specified axis, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if the final unit is the square of the unit of `x`, else an array.
'''

frexp.__doc__ = '''
  Decompose a floating-point number into its mantissa and exponent.

  Args:
    x: array_like, Quantity

  Returns:
    out: Tuple of Quantity if the final unit is the product of the unit of `x` and 2 raised to the power of the exponent, else a tuple of arrays.
'''

sqrt.__doc__ = '''
  Compute the square root of each element.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if the final unit is the square root of the unit of `x`, else an array.
'''

cbrt.__doc__ = '''
  Compute the cube root of each element.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if the final unit is the cube root of the unit of `x`, else an array.
'''

square.__doc__ = '''
  Compute the square of each element.

  Args:
    x: array_like, Quantity

  Returns:
    out: Quantity if the final unit is the square of the unit of `x`, else an array.
'''


# math funcs change unit (binary)
# -------------------------------

def wrap_math_funcs_change_unit_binary(func, change_unit_func):
  def f(x, y, *args, **kwargs):
    if isinstance(x, Quantity) and isinstance(y, Quantity):
      return _return_check_unitless(
        Quantity(func(x.value, y.value, *args, **kwargs), unit=change_unit_func(x.unit, y.unit))
      )
    elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
      return func(x, y, *args, **kwargs)
    elif isinstance(x, Quantity):
      return _return_check_unitless(
        Quantity(func(x.value, y, *args, **kwargs), unit=change_unit_func(x.unit, DIMENSIONLESS)))
    elif isinstance(y, Quantity):
      return _return_check_unitless(
        Quantity(func(x, y.value, *args, **kwargs), unit=change_unit_func(DIMENSIONLESS, y.unit)))
    else:
      raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


multiply = wrap_math_funcs_change_unit_binary(jnp.multiply, lambda x, y: x * y)
divide = wrap_math_funcs_change_unit_binary(jnp.divide, lambda x, y: x / y)
cross = wrap_math_funcs_change_unit_binary(jnp.cross, lambda x, y: x * y)
ldexp = wrap_math_funcs_change_unit_binary(jnp.ldexp, lambda x, y: x * 2 ** y)
true_divide = wrap_math_funcs_change_unit_binary(jnp.true_divide, lambda x, y: x / y)
divmod = wrap_math_funcs_change_unit_binary(jnp.divmod, lambda x, y: x / y)
convolve = wrap_math_funcs_change_unit_binary(jnp.convolve, lambda x, y: x * y)

# docs for the functions above
multiply.__doc__ = '''
  Multiply arguments element-wise.
  
  Args:
    x: array_like, Quantity
    y: array_like, Quantity
    
  Returns:
    out: Quantity if the final unit is the product of the unit of `x` and the unit of `y`, else an array.
'''

divide.__doc__ = '''
  Divide arguments element-wise.
  
  Args:
    x: array_like, Quantity
    
  Returns:
    out: Quantity if the final unit is the quotient of the unit of `x` and the unit of `y`, else an array.
'''

cross.__doc__ = '''
  Return the cross product of two (arrays of) vectors.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    out: Quantity if the final unit is the product of the unit of `x` and the unit of `y`, else an array.
'''

ldexp.__doc__ = '''
  Return x1 * 2**x2, element-wise.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    out: Quantity if the final unit is the product of the unit of `x` and 2 raised to the power of the unit of `y`, else an array. 
'''

true_divide.__doc__ = '''
  Returns a true division of the inputs, element-wise.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    out: Quantity if the final unit is the quotient of the unit of `x` and the unit of `y`, else an array.
'''

divmod.__doc__ = '''
  Return element-wise quotient and remainder simultaneously.
  
  Args:
    x: array_like, Quantity
    y: array_like, Quantity
    
  Returns:
    out: Quantity if the final unit is the quotient of the unit of `x` and the unit of `y`, else an array.
'''

convolve.__doc__ = '''
  Returns the discrete, linear convolution of two one-dimensional sequences.
  
  Args:
    x: array_like, Quantity
    y: array_like, Quantity
    
  Returns:
    out: Quantity if the final unit is the product of the unit of `x` and the unit of `y`, else an array.
'''


@set_module_as('brainunit.math')
def power(x: Union[Quantity, bst.typing.ArrayLike],
          y: Union[Quantity, bst.typing.ArrayLike], ) -> Union[Quantity, jax.Array]:
  '''
  First array elements raised to powers from second array, element-wise.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    out: Quantity if the final unit is the product of the unit of `x` and the unit of `y`, else an array.
  '''
  if isinstance(x, Quantity) and isinstance(y, Quantity):
    return _return_check_unitless(Quantity(jnp.power(x.value, y.value), unit=x.unit ** y.unit))
  elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
    return jnp.power(x, y)
  elif isinstance(x, Quantity):
    return _return_check_unitless(Quantity(jnp.power(x.value, y), unit=x.unit ** y))
  elif isinstance(y, Quantity):
    return _return_check_unitless(Quantity(jnp.power(x, y.value), unit=x ** y.unit))
  else:
    raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {jnp.power.__name__}')


@set_module_as('brainunit.math')
def floor_divide(x: Union[Quantity, bst.typing.ArrayLike],
                 y: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the largest integer smaller or equal to the division of the inputs.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    out: Quantity if the final unit is the quotient of the unit of `x` and the unit of `y`, else an array.
  '''
  if isinstance(x, Quantity) and isinstance(y, Quantity):
    return _return_check_unitless(Quantity(jnp.floor_divide(x.value, y.value), unit=x.unit / y.unit))
  elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
    return jnp.floor_divide(x, y)
  elif isinstance(x, Quantity):
    return _return_check_unitless(Quantity(jnp.floor_divide(x.value, y), unit=x.unit / y))
  elif isinstance(y, Quantity):
    return _return_check_unitless(Quantity(jnp.floor_divide(x, y.value), unit=x / y.unit))
  else:
    raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {jnp.floor_divide.__name__}')


@set_module_as('brainunit.math')
def float_power(x: Union[Quantity, bst.typing.ArrayLike],
                y: bst.typing.ArrayLike) -> Union[Quantity, jax.Array]:
  '''
  First array elements raised to powers from second array, element-wise.

  Args:
    x: array_like, Quantity
    y: array_like

  Returns:
    out: Quantity if the final unit is the product of the unit of `x` and the unit of `y`, else an array.
  '''
  assert isscalar(y), f'{jnp.float_power.__name__} only supports scalar exponent'
  if isinstance(x, Quantity):
    return _return_check_unitless(Quantity(jnp.float_power(x.value, y), unit=x.unit ** y.unit))
  elif isinstance(x, (jax.Array, np.ndarray)):
    return jnp.float_power(x, y)
  else:
    raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {jnp.float_power.__name__}')


@set_module_as('brainunit.math')
def remainder(x: Union[Quantity, bst.typing.ArrayLike],
              y: Union[Quantity, bst.typing.ArrayLike]):
  if isinstance(x, Quantity) and isinstance(y, Quantity):
    return _return_check_unitless(Quantity(jnp.remainder(x.value, y.value), unit=x.unit / y.unit))
  elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
    return jnp.remainder(x, y)
  elif isinstance(x, Quantity):
    return _return_check_unitless(Quantity(jnp.remainder(x.value, y), unit=x.unit % y))
  elif isinstance(y, Quantity):
    return _return_check_unitless(Quantity(jnp.remainder(x, y.value), unit=x % y.unit))
  else:
    raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {jnp.remainder.__name__}')


# math funcs only accept unitless (unary)
# ---------------------------------------

def wrap_math_funcs_only_accept_unitless_unary(func):
  def f(x, *args, **kwargs):
    if isinstance(x, Quantity):
      fail_for_dimension_mismatch(
        x,
        error_message="%s expects a dimensionless argument but got {value}" % func.__name__,
        value=x,
      )
      return func(jnp.array(x.value), *args, **kwargs)
    else:
      return func(x, *args, **kwargs)

  f.__module__ = 'brainunit.math'
  return f


exp = wrap_math_funcs_only_accept_unitless_unary(jnp.exp)
exp2 = wrap_math_funcs_only_accept_unitless_unary(jnp.exp2)
expm1 = wrap_math_funcs_only_accept_unitless_unary(jnp.expm1)
log = wrap_math_funcs_only_accept_unitless_unary(jnp.log)
log10 = wrap_math_funcs_only_accept_unitless_unary(jnp.log10)
log1p = wrap_math_funcs_only_accept_unitless_unary(jnp.log1p)
log2 = wrap_math_funcs_only_accept_unitless_unary(jnp.log2)
arccos = wrap_math_funcs_only_accept_unitless_unary(jnp.arccos)
arccosh = wrap_math_funcs_only_accept_unitless_unary(jnp.arccosh)
arcsin = wrap_math_funcs_only_accept_unitless_unary(jnp.arcsin)
arcsinh = wrap_math_funcs_only_accept_unitless_unary(jnp.arcsinh)
arctan = wrap_math_funcs_only_accept_unitless_unary(jnp.arctan)
arctanh = wrap_math_funcs_only_accept_unitless_unary(jnp.arctanh)
cos = wrap_math_funcs_only_accept_unitless_unary(jnp.cos)
cosh = wrap_math_funcs_only_accept_unitless_unary(jnp.cosh)
sin = wrap_math_funcs_only_accept_unitless_unary(jnp.sin)
sinc = wrap_math_funcs_only_accept_unitless_unary(jnp.sinc)
sinh = wrap_math_funcs_only_accept_unitless_unary(jnp.sinh)
tan = wrap_math_funcs_only_accept_unitless_unary(jnp.tan)
tanh = wrap_math_funcs_only_accept_unitless_unary(jnp.tanh)
deg2rad = wrap_math_funcs_only_accept_unitless_unary(jnp.deg2rad)
rad2deg = wrap_math_funcs_only_accept_unitless_unary(jnp.rad2deg)
degrees = wrap_math_funcs_only_accept_unitless_unary(jnp.degrees)
radians = wrap_math_funcs_only_accept_unitless_unary(jnp.radians)
angle = wrap_math_funcs_only_accept_unitless_unary(jnp.angle)
percentile = wrap_math_funcs_only_accept_unitless_unary(jnp.percentile)
nanpercentile = wrap_math_funcs_only_accept_unitless_unary(jnp.nanpercentile)
quantile = wrap_math_funcs_only_accept_unitless_unary(jnp.quantile)
nanquantile = wrap_math_funcs_only_accept_unitless_unary(jnp.nanquantile)

# docs for the functions above
exp.__doc__ = '''
  Calculate the exponential of all elements in the input array.
  
  Args:
    x: array_like, Quantity
    
  Returns:
    out: an array
'''

exp2.__doc__ = '''
  Calculate 2 raised to the power of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

expm1.__doc__ = '''
  Calculate the exponential of the input elements minus 1.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

log.__doc__ = '''
  Natural logarithm, element-wise.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

log10.__doc__ = '''
  Base-10 logarithm of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

log1p.__doc__ = '''
  Natural logarithm of 1 + the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

log2.__doc__ = '''
  Base-2 logarithm of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

arccos.__doc__ = '''
  Compute the arccosine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

arccosh.__doc__ = '''
  Compute the hyperbolic arccosine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

arcsin.__doc__ = '''
  Compute the arcsine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

arcsinh.__doc__ = '''
  Compute the hyperbolic arcsine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

arctan.__doc__ = '''
  Compute the arctangent of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

arctanh.__doc__ = '''
  Compute the hyperbolic arctangent of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

cos.__doc__ = '''
  Compute the cosine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

cosh.__doc__ = '''
  Compute the hyperbolic cosine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

sin.__doc__ = '''
  Compute the sine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

sinc.__doc__ = '''
  Compute the sinc function of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

sinh.__doc__ = '''
  Compute the hyperbolic sine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

tan.__doc__ = '''
  Compute the tangent of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

tanh.__doc__ = '''
  Compute the hyperbolic tangent of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

deg2rad.__doc__ = '''
  Convert angles from degrees to radians.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

rad2deg.__doc__ = '''
  Convert angles from radians to degrees.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

degrees.__doc__ = '''
  Convert angles from radians to degrees.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

radians.__doc__ = '''
  Convert angles from degrees to radians.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

angle.__doc__ = '''
  Return the angle of the complex argument.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

percentile.__doc__ = '''
  Compute the nth percentile of the input array along the specified axis.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

nanpercentile.__doc__ = '''
  Compute the nth percentile of the input array along the specified axis, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

quantile.__doc__ = '''
  Compute the qth quantile of the input array along the specified axis.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

nanquantile.__doc__ = '''
  Compute the qth quantile of the input array along the specified axis, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''


# math funcs only accept unitless (binary)
# ----------------------------------------

def wrap_math_funcs_only_accept_unitless_binary(func):
  def f(x, y, *args, **kwargs):
    if isinstance(x, Quantity):
      x_value = x.value
    if isinstance(y, Quantity):
      y_value = y.value
    if isinstance(x, Quantity) or isinstance(y, Quantity):
      fail_for_dimension_mismatch(
        x,
        error_message="%s expects a dimensionless argument but got {value}" % func.__name__,
        value=x,
      )
      fail_for_dimension_mismatch(
        y,
        error_message="%s expects a dimensionless argument but got {value}" % func.__name__,
        value=y,
      )
      return func(jnp.array(x_value), jnp.array(y_value), *args, **kwargs)
    else:
      return func(x, y, *args, **kwargs)

  f.__module__ = 'brainunit.math'
  return f


hypot = wrap_math_funcs_only_accept_unitless_binary(jnp.hypot)
arctan2 = wrap_math_funcs_only_accept_unitless_binary(jnp.arctan2)
logaddexp = wrap_math_funcs_only_accept_unitless_binary(jnp.logaddexp)
logaddexp2 = wrap_math_funcs_only_accept_unitless_binary(jnp.logaddexp2)

# docs for the functions above
hypot.__doc__ = '''
  Given the “legs” of a right triangle, return its hypotenuse.
  
  Args:
    x: array_like, Quantity
    y: array_like, Quantity
    
  Returns:
    out: an array
'''

arctan2.__doc__ = '''
  Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: an array
'''

logaddexp.__doc__ = '''
  Logarithm of the sum of exponentiations of the inputs.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: an array
'''

logaddexp2.__doc__ = '''
  Logarithm of the sum of exponentiations of the inputs in base-2.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    out: an array
'''


# math funcs remove unit (unary)
# ------------------------------
def wrap_math_funcs_remove_unit_unary(func):
  def f(x, *args, **kwargs):
    if isinstance(x, Quantity):
      return func(x.value, *args, **kwargs)
    else:
      return func(x, *args, **kwargs)

  f.__module__ = 'brainunit.math'
  return f


signbit = wrap_math_funcs_remove_unit_unary(jnp.signbit)
sign = wrap_math_funcs_remove_unit_unary(jnp.sign)
histogram = wrap_math_funcs_remove_unit_unary(jnp.histogram)
bincount = wrap_math_funcs_remove_unit_unary(jnp.bincount)

# docs for the functions above
signbit.__doc__ = '''
  Returns element-wise True where signbit is set (less than zero).
  
  Args:
    x: array_like, Quantity
    
  Returns:
    out: an array
'''

sign.__doc__ = '''
  Returns the sign of each element in the input array.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''

histogram.__doc__ = '''
  Compute the histogram of a set of data.

  Args:
    x: array_like, Quantity

  Returns:
    out: Tuple of arrays (hist, bin_edges)
'''

bincount.__doc__ = '''
  Count number of occurrences of each value in array of non-negative integers.

  Args:
    x: array_like, Quantity

  Returns:
    out: an array
'''


# math funcs remove unit (binary)
# -------------------------------
def wrap_math_funcs_remove_unit_binary(func):
  def f(x, y, *args, **kwargs):
    if isinstance(x, Quantity):
      x_value = x.value
    if isinstance(y, Quantity):
      y_value = y.value
    if isinstance(x, Quantity) or isinstance(y, Quantity):
      return func(jnp.array(x_value), jnp.array(y_value), *args, **kwargs)
    else:
      return func(x, y, *args, **kwargs)

  f.__module__ = 'brainunit.math'
  return f


corrcoef = wrap_math_funcs_remove_unit_binary(jnp.corrcoef)
correlate = wrap_math_funcs_remove_unit_binary(jnp.correlate)
cov = wrap_math_funcs_remove_unit_binary(jnp.cov)
digitize = wrap_math_funcs_remove_unit_binary(jnp.digitize)

# docs for the functions above
corrcoef.__doc__ = '''
  Return Pearson product-moment correlation coefficients.
  
  Args:
    x: array_like, Quantity
    y: array_like, Quantity
    
  Returns:
    out: an array
'''

correlate.__doc__ = '''
  Cross-correlation of two sequences.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    out: an array
'''

cov.__doc__ = '''
  Covariance matrix.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity (optional, if not provided, x is assumed to be a 2D array)

  Returns:
    out: an array
'''

digitize.__doc__ = '''
  Return the indices of the bins to which each value in input array belongs.

  Args:
    x: array_like, Quantity
    bins: array_like, Quantity

  Returns:
    out: an array
'''

# array manipulation
# ------------------

reshape = _compatible_with_quantity(jnp.reshape)
moveaxis = _compatible_with_quantity(jnp.moveaxis)
transpose = _compatible_with_quantity(jnp.transpose)
swapaxes = _compatible_with_quantity(jnp.swapaxes)
concatenate = _compatible_with_quantity(jnp.concatenate)
stack = _compatible_with_quantity(jnp.stack)
vstack = _compatible_with_quantity(jnp.vstack)
row_stack = vstack
hstack = _compatible_with_quantity(jnp.hstack)
dstack = _compatible_with_quantity(jnp.dstack)
column_stack = _compatible_with_quantity(jnp.column_stack)
split = _compatible_with_quantity(jnp.split)
dsplit = _compatible_with_quantity(jnp.dsplit)
hsplit = _compatible_with_quantity(jnp.hsplit)
vsplit = _compatible_with_quantity(jnp.vsplit)
tile = _compatible_with_quantity(jnp.tile)
repeat = _compatible_with_quantity(jnp.repeat)
unique = _compatible_with_quantity(jnp.unique)
append = _compatible_with_quantity(jnp.append)
flip = _compatible_with_quantity(jnp.flip)
fliplr = _compatible_with_quantity(jnp.fliplr)
flipud = _compatible_with_quantity(jnp.flipud)
roll = _compatible_with_quantity(jnp.roll)
atleast_1d = _compatible_with_quantity(jnp.atleast_1d)
atleast_2d = _compatible_with_quantity(jnp.atleast_2d)
atleast_3d = _compatible_with_quantity(jnp.atleast_3d)
expand_dims = _compatible_with_quantity(jnp.expand_dims)
squeeze = _compatible_with_quantity(jnp.squeeze)
sort = _compatible_with_quantity(jnp.sort)

max = _compatible_with_quantity(jnp.max)
min = _compatible_with_quantity(jnp.min)

amax = max
amin = min

choose = _compatible_with_quantity(jnp.choose)
block = _compatible_with_quantity(jnp.block)
compress = _compatible_with_quantity(jnp.compress)
diagflat = _compatible_with_quantity(jnp.diagflat)

# return jax.numpy.Array, not Quantity
argsort = _compatible_with_quantity(jnp.argsort, return_quantity=False)
argmax = _compatible_with_quantity(jnp.argmax, return_quantity=False)
argmin = _compatible_with_quantity(jnp.argmin, return_quantity=False)
argwhere = _compatible_with_quantity(jnp.argwhere, return_quantity=False)
nonzero = _compatible_with_quantity(jnp.nonzero, return_quantity=False)
flatnonzero = _compatible_with_quantity(jnp.flatnonzero, return_quantity=False)
searchsorted = _compatible_with_quantity(jnp.searchsorted, return_quantity=False)
extract = _compatible_with_quantity(jnp.extract, return_quantity=False)
count_nonzero = _compatible_with_quantity(jnp.count_nonzero, return_quantity=False)


def wrap_function_to_method(func):
  @wraps(func)
  def f(x, *args, **kwargs):
    if isinstance(x, Quantity):
      return Quantity(func(x.value, *args, **kwargs), unit=x.unit)
    else:
      return func(x, *args, **kwargs)

  f.__module__ = 'brainunit.math'
  return f


diagonal = wrap_function_to_method(jnp.diagonal)
ravel = wrap_function_to_method(jnp.ravel)


# Elementwise bit operations (unary)
# ----------------------------------

def wrap_elementwise_bit_operation_unary(func):
  def f(x, *args, **kwargs):
    if isinstance(x, Quantity):
      raise ValueError(f'Expected integers, got {x}')
    elif isinstance(x, (jax.Array, np.ndarray)):
      return func(x, *args, **kwargs)
    else:
      raise ValueError(f'Unsupported types {type(x)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


bitwise_not = wrap_elementwise_bit_operation_unary(jnp.bitwise_not)
invert = wrap_elementwise_bit_operation_unary(jnp.invert)
left_shift = wrap_elementwise_bit_operation_unary(jnp.left_shift)
right_shift = wrap_elementwise_bit_operation_unary(jnp.right_shift)


# Elementwise bit operations (binary)
# -----------------------------------

def wrap_elementwise_bit_operation_binary(func):
  def f(x, y, *args, **kwargs):
    if isinstance(x, Quantity) or isinstance(y, Quantity):
      raise ValueError(f'Expected integers, got {x} and {y}')
    elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
      return func(x, y, *args, **kwargs)
    else:
      raise ValueError(f'Unsupported types {type(x)} and {type(y)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


bitwise_and = wrap_elementwise_bit_operation_binary(jnp.bitwise_and)
bitwise_or = wrap_elementwise_bit_operation_binary(jnp.bitwise_or)
bitwise_xor = wrap_elementwise_bit_operation_binary(jnp.bitwise_xor)


# logic funcs (unary)
# -------------------

def wrap_logic_func_unary(func):
  def f(x, *args, **kwargs):
    if isinstance(x, Quantity):
      raise ValueError(f'Expected booleans, got {x}')
    elif isinstance(x, (jax.Array, np.ndarray)):
      return func(x, *args, **kwargs)
    else:
      raise ValueError(f'Unsupported types {type(x)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


all = wrap_logic_func_unary(jnp.all)
any = wrap_logic_func_unary(jnp.any)
alltrue = all
sometrue = any
logical_not = wrap_logic_func_unary(jnp.logical_not)


# logic funcs (binary)
# --------------------

def wrap_logic_func_binary(func):
  def f(x, y, *args, **kwargs):
    if isinstance(x, Quantity) and isinstance(y, Quantity):
      fail_for_dimension_mismatch(x, y)
      return func(x.value, y.value, *args, **kwargs)
    elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
      return func(x, y, *args, **kwargs)
    else:
      raise ValueError(f'Unsupported types {type(x)} and {type(y)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


equal = wrap_logic_func_binary(jnp.equal)
not_equal = wrap_logic_func_binary(jnp.not_equal)
greater = wrap_logic_func_binary(jnp.greater)
greater_equal = wrap_logic_func_binary(jnp.greater_equal)
less = wrap_logic_func_binary(jnp.less)
less_equal = wrap_logic_func_binary(jnp.less_equal)
array_equal = wrap_logic_func_binary(jnp.array_equal)
isclose = wrap_logic_func_binary(jnp.isclose)
allclose = wrap_logic_func_binary(jnp.allclose)
logical_and = wrap_logic_func_binary(jnp.logical_and)

logical_or = wrap_logic_func_binary(jnp.logical_or)
logical_xor = wrap_logic_func_binary(jnp.logical_xor)


# indexing funcs
# --------------
@set_module_as('brainunit.math')
def where(condition, *args, **kwds):  # pylint: disable=C0111
  condition = jnp.asarray(condition)
  if len(args) == 0:
    # nothing to do
    return jnp.where(condition, *args, **kwds)
  elif len(args) == 2:
    # check that x and y have the same dimensions
    fail_for_dimension_mismatch(
      args[0], args[1], "x and y need to have the same dimensions"
    )
    new_args = []
    for arg in args:
      if isinstance(arg, Quantity):
        new_args.append(arg.value)
    if is_unitless(args[0]):
      if len(new_args) == 2:
        return jnp.where(condition, *new_args, **kwds)
      else:
        return jnp.where(condition, *args, **kwds)
    else:
      # as both arguments have the same unit, just use the first one's
      dimensionless_args = [jnp.asarray(arg.value) if isinstance(arg, Quantity) else jnp.asarray(arg) for arg in args]
      return Quantity.with_units(
        jnp.where(condition, *dimensionless_args), args[0].unit
      )
  else:
    # illegal number of arguments
    if len(args) == 1:
      raise ValueError("where() takes 2 or 3 positional arguments but 1 was given")
    elif len(args) > 2:
      raise TypeError("where() takes 2 or 3 positional arguments but {} were given".format(len(args)))


tril_indices = jnp.tril_indices


@set_module_as('brainunit.math')
def tril_indices_from(arr, k=0):
  if isinstance(arr, Quantity):
    return jnp.tril_indices_from(arr.value, k=k)
  else:
    return jnp.tril_indices_from(arr, k=k)


triu_indices = jnp.triu_indices


@set_module_as('brainunit.math')
def triu_indices_from(arr, k=0):
  if isinstance(arr, Quantity):
    return jnp.triu_indices_from(arr.value, k=k)
  else:
    return jnp.triu_indices_from(arr, k=k)


@set_module_as('brainunit.math')
def take(a, indices, axis=None, mode=None):
  if isinstance(a, Quantity):
    return a.take(indices, axis=axis, mode=mode)
  else:
    return jnp.take(a, indices, axis=axis, mode=mode)


@set_module_as('brainunit.math')
def select(condlist: list[Union[jnp.array, np.ndarray]], choicelist: Union[Quantity, jax.Array, np.ndarray], default=0):
  from builtins import all as origin_all
  from builtins import any as origin_any
  if origin_all(isinstance(choice, Quantity) for choice in choicelist):
    if origin_any(choice.unit != choicelist[0].unit for choice in choicelist):
      raise ValueError("All choices must have the same unit")
    else:
      return Quantity(jnp.select(condlist, [choice.value for choice in choicelist], default=default),
                      unit=choicelist[0].unit)
  elif origin_all(isinstance(choice, (jax.Array, np.ndarray)) for choice in choicelist):
    return jnp.select(condlist, choicelist, default=default)
  else:
    raise ValueError(f"Unsupported types : {type(condlist)} and {type(choicelist)} for select")


# window funcs
# ------------

def wrap_window_funcs(func):
  def f(*args, **kwargs):
    return Quantity(func(*args, **kwargs))

  f.__module__ = 'brainunit.math'
  return f


bartlett = wrap_window_funcs(jnp.bartlett)
blackman = wrap_window_funcs(jnp.blackman)
hamming = wrap_window_funcs(jnp.hamming)
hanning = wrap_window_funcs(jnp.hanning)
kaiser = wrap_window_funcs(jnp.kaiser)

# constants
# ---------
e = jnp.e
pi = jnp.pi
inf = jnp.inf

# linear algebra
# --------------
dot = wrap_math_funcs_change_unit_binary(jnp.dot, lambda x, y: x * y)
vdot = wrap_math_funcs_change_unit_binary(jnp.vdot, lambda x, y: x * y)
inner = wrap_math_funcs_change_unit_binary(jnp.inner, lambda x, y: x * y)
outer = wrap_math_funcs_change_unit_binary(jnp.outer, lambda x, y: x * y)
kron = wrap_math_funcs_change_unit_binary(jnp.kron, lambda x, y: x * y)
matmul = wrap_math_funcs_change_unit_binary(jnp.matmul, lambda x, y: x * y)
trace = wrap_math_funcs_keep_unit_unary(jnp.trace)

# data types
# ----------
dtype = jnp.dtype


@set_module_as('brainunit.math')
def finfo(a):
  if isinstance(a, Quantity):
    return jnp.finfo(a.value)
  else:
    return jnp.finfo(a)


@set_module_as('brainunit.math')
def iinfo(a):
  if isinstance(a, Quantity):
    return jnp.iinfo(a.value)
  else:
    return jnp.iinfo(a)


# more
# ----
@set_module_as('brainunit.math')
def broadcast_arrays(*args):
  from builtins import all as origin_all
  from builtins import any as origin_any
  if origin_all(isinstance(arg, Quantity) for arg in args):
    if origin_any(arg.unit != args[0].unit for arg in args):
      raise ValueError("All arguments must have the same unit")
    return Quantity(jnp.broadcast_arrays(*[arg.value for arg in args]), unit=args[0].unit)
  elif origin_all(isinstance(arg, (jax.Array, np.ndarray)) for arg in args):
    return jnp.broadcast_arrays(*args)
  else:
    raise ValueError(f"Unsupported types : {type(args)} for broadcast_arrays")


broadcast_shapes = jnp.broadcast_shapes


@set_module_as('brainunit.math')
def einsum(
    subscripts, /,
    *operands,
    out: None = None,
    optimize: Union[str, bool] = "optimal",
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: Union[jax.typing.DTypeLike, None] = None,
    _dot_general: Callable[..., jax.Array] = jax.lax.dot_general,
) -> Union[jax.Array, Quantity]:
  operands = (subscripts, *operands)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.einsum is not supported.")
  spec = operands[0] if isinstance(operands[0], str) else None
  optimize = 'optimal' if optimize is True else optimize

  # Allow handling of shape polymorphism
  non_constant_dim_types = {
    type(d) for op in operands if not isinstance(op, str)
    for d in np.shape(op) if not jax.core.is_constant_dim(d)
  }
  if not non_constant_dim_types:
    contract_path = opt_einsum.contract_path
  else:
    from jax._src.numpy.lax_numpy import _default_poly_einsum_handler
    contract_path = _default_poly_einsum_handler

  operands, contractions = contract_path(
    *operands, einsum_call=True, use_blas=True, optimize=optimize)

  unit = None
  for i in range(len(contractions) - 1):
    if contractions[i][4] == 'False':

      fail_for_dimension_mismatch(
        Quantity([], unit=unit), operands[i + 1], 'einsum'
      )
    elif contractions[i][4] == 'DOT' or \
        contractions[i][4] == 'TDOT' or \
        contractions[i][4] == 'GEMM' or \
        contractions[i][4] == 'OUTER/EINSUM':
      if i == 0:
        if isinstance(operands[i], Quantity) and isinstance(operands[i + 1], Quantity):
          unit = operands[i].unit * operands[i + 1].unit
        elif isinstance(operands[i], Quantity):
          unit = operands[i].unit
        elif isinstance(operands[i + 1], Quantity):
          unit = operands[i + 1].unit
      else:
        if isinstance(operands[i + 1], Quantity):
          unit = unit * operands[i + 1].unit

  contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)

  einsum = jax.jit(_einsum, static_argnums=(1, 2, 3, 4), inline=True)
  if spec is not None:
    einsum = jax.named_call(einsum, name=spec)
  operands = [op.value if isinstance(op, Quantity) else op for op in operands]
  r = einsum(operands, contractions, precision,  # type: ignore[operator]
             preferred_element_type, _dot_general)
  if unit is not None:
    return Quantity(r, unit=unit)
  else:
    return r


@set_module_as('brainunit.math')
def gradient(
    f: Union[bst.typing.ArrayLike, Quantity],
    *varargs: Union[bst.typing.ArrayLike, Quantity],
    axis: Union[int, Sequence[int], None] = None,
    edge_order: Union[int, None] = None,
) -> Union[jax.Array, list[jax.Array], Quantity, list[Quantity]]:
  if edge_order is not None:
    raise NotImplementedError("The 'edge_order' argument to jnp.gradient is not supported.")

  if len(varargs) == 0:
    if isinstance(f, Quantity) and not is_unitless(f):
      return Quantity(jnp.gradient(f.value, axis=axis), unit=f.unit)
    else:
      return jnp.gradient(f)
  elif len(varargs) == 1:
    unit = get_unit(f) / get_unit(varargs[0])
    if unit is None or unit == DIMENSIONLESS:
      return jnp.gradient(f, varargs[0], axis=axis)
    else:
      return [Quantity(r, unit=unit) for r in jnp.gradient(f.value, varargs[0].value, axis=axis)]
  else:
    unit_list = [get_unit(f) / get_unit(v) for v in varargs]
    f = f.value if isinstance(f, Quantity) else f
    varargs = [v.value if isinstance(v, Quantity) else v for v in varargs]
    result_list = jnp.gradient(f, *varargs, axis=axis)
    return [Quantity(r, unit=unit) if unit is not None else r for r, unit in zip(result_list, unit_list)]


@set_module_as('brainunit.math')
def intersect1d(
    ar1: Union[bst.typing.ArrayLike],
    ar2: Union[bst.typing.ArrayLike],
    assume_unique: bool = False,
    return_indices: bool = False
) -> Union[jax.Array, Quantity, tuple[Union[jax.Array, Quantity], jax.Array, jax.Array]]:
  fail_for_dimension_mismatch(ar1, ar2, 'intersect1d')
  unit = None
  if isinstance(ar1, Quantity):
    unit = ar1.unit
  ar1 = ar1.value if isinstance(ar1, Quantity) else ar1
  ar2 = ar2.value if isinstance(ar2, Quantity) else ar2
  result = jnp.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
  if return_indices:
    if unit is not None:
      return (Quantity(result[0], unit=unit), result[1], result[2])
    else:
      return result
  else:
    if unit is not None:
      return Quantity(result, unit=unit)
    else:
      return result


nan_to_num = wrap_math_funcs_keep_unit_unary(jnp.nan_to_num)
nanargmax = _compatible_with_quantity(jnp.nanargmax, return_quantity=False)
nanargmin = _compatible_with_quantity(jnp.nanargmin, return_quantity=False)

rot90 = wrap_math_funcs_keep_unit_unary(jnp.rot90)
tensordot = wrap_math_funcs_change_unit_binary(jnp.tensordot, lambda x, y: x * y)
