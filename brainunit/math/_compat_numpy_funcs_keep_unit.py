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
from typing import (Union, Callable)

import jax
import jax.numpy as jnp
import numpy as np
from brainstate._utils import set_module_as

from .._base import (Quantity,
                     )

__all__ = [
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
]


# math funcs keep unit (unary)
# ----------------------------


def funcs_keep_unit_unary(func, x, *args, **kwargs):
  if isinstance(x, Quantity):
    return Quantity(func(x.value, *args, **kwargs), dim=x.dim)
  elif isinstance(x, (jax.Array, np.ndarray)):
    return func(x, *args, **kwargs)
  else:
    raise ValueError(f'Unsupported type: {type(x)} for {func.__name__}')


@set_module_as('brainunit.math')
def real(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the real part of the complex argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.real, x)


@set_module_as('brainunit.math')
def imag(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the imaginary part of the complex argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.imag, x)


@set_module_as('brainunit.math')
def conj(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the complex conjugate of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.conj, x)


@set_module_as('brainunit.math')
def conjugate(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the complex conjugate of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.conjugate, x)


@set_module_as('brainunit.math')
def negative(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the negative of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.negative, x)


@set_module_as('brainunit.math')
def positive(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the positive of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.positive, x)


@set_module_as('brainunit.math')
def abs(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the absolute value of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.abs, x)


@set_module_as('brainunit.math')
def round_(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Round an array to the nearest integer.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.round_, x)


@set_module_as('brainunit.math')
def around(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Round an array to the nearest integer.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.around, x)


@set_module_as('brainunit.math')
def round(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Round an array to the nearest integer.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.round, x)


@set_module_as('brainunit.math')
def rint(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Round an array to the nearest integer.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.rint, x)


@set_module_as('brainunit.math')
def floor(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the floor of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.floor, x)


@set_module_as('brainunit.math')
def ceil(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the ceiling of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.ceil, x)


@set_module_as('brainunit.math')
def trunc(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the truncated value of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.trunc, x)


@set_module_as('brainunit.math')
def fix(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the nearest integer towards zero.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.fix, x)


@set_module_as('brainunit.math')
def sum(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the sum of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.sum, x)


@set_module_as('brainunit.math')
def nancumsum(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the cumulative sum of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.nancumsum, x)


@set_module_as('brainunit.math')
def nansum(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the sum of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.nansum, x)


@set_module_as('brainunit.math')
def cumsum(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the cumulative sum of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.cumsum, x)


@set_module_as('brainunit.math')
def ediff1d(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the differences between consecutive elements of the array.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.ediff1d, x)


@set_module_as('brainunit.math')
def absolute(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the absolute value of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.absolute, x)


@set_module_as('brainunit.math')
def fabs(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the absolute value of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.fabs, x)


@set_module_as('brainunit.math')
def median(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the median of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.median, x)


@set_module_as('brainunit.math')
def nanmin(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the minimum of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.nanmin, x)


@set_module_as('brainunit.math')
def nanmax(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the maximum of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.nanmax, x)


@set_module_as('brainunit.math')
def ptp(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the range of the array elements (maximum - minimum).

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.ptp, x)


@set_module_as('brainunit.math')
def average(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the weighted average of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.average, x)


@set_module_as('brainunit.math')
def mean(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the mean of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.mean, x)


@set_module_as('brainunit.math')
def std(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the standard deviation of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.std, x)


@set_module_as('brainunit.math')
def nanmedian(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the median of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.nanmedian, x)


@set_module_as('brainunit.math')
def nanmean(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the mean of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.nanmean, x)


@set_module_as('brainunit.math')
def nanstd(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the standard deviation of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.nanstd, x)


@set_module_as('brainunit.math')
def diff(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the differences between consecutive elements of the array.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  return funcs_keep_unit_unary(jnp.diff, x)


@set_module_as('brainunit.math')
def modf(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the fractional and integer parts of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity tuple if `x` is a Quantity, else an array tuple.
  '''
  return funcs_keep_unit_unary(jnp.modf, x)


# math funcs keep unit (binary)
# -----------------------------

def funcs_keep_unit_binary(func, x1, x2, *args, **kwargs):
  if isinstance(x1, Quantity) and isinstance(x2, Quantity):
    return Quantity(func(x1.value, x2.value, *args, **kwargs), dim=x1.dim)
  elif isinstance(x1, (jax.Array, np.ndarray)) and isinstance(x2, (jax.Array, np.ndarray)):
    return func(x1, x2, *args, **kwargs)
  else:
    raise ValueError(f'Unsupported type: {type(x1)} and {type(x2)} for {func.__name__}')

@set_module_as('brainunit.math')
def fmod(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  '''
  Return the element-wise remainder of division.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  '''
  return funcs_keep_unit_binary(jnp.fmod, x1, x2)


@set_module_as('brainunit.math')
def mod(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  '''
  Return the element-wise modulus of division.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  '''
  return funcs_keep_unit_binary(jnp.mod, x1, x2)


@set_module_as('brainunit.math')
def copysign(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  '''
  Return a copy of the first array elements with the sign of the second array.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  '''
  return funcs_keep_unit_binary(jnp.copysign, x1, x2)


@set_module_as('brainunit.math')
def heaviside(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  '''
  Compute the Heaviside step function.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  '''
  return funcs_keep_unit_binary(jnp.heaviside, x1, x2)


@set_module_as('brainunit.math')
def maximum(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  '''
  Element-wise maximum of array elements.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  '''
  return funcs_keep_unit_binary(jnp.maximum, x1, x2)


@set_module_as('brainunit.math')
def minimum(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  '''
  Element-wise minimum of array elements.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  '''
  return funcs_keep_unit_binary(jnp.minimum, x1, x2)


@set_module_as('brainunit.math')
def fmax(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  '''
  Element-wise maximum of array elements ignoring NaNs.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  '''
  return funcs_keep_unit_binary(jnp.fmax, x1, x2)


@set_module_as('brainunit.math')
def fmin(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  '''
  Element-wise minimum of array elements ignoring NaNs.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  '''
  return funcs_keep_unit_binary(jnp.fmin, x1, x2)


@set_module_as('brainunit.math')
def lcm(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  '''
  Return the least common multiple of `x1` and `x2`.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  '''
  return funcs_keep_unit_binary(jnp.lcm, x1, x2)


@set_module_as('brainunit.math')
def gcd(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  '''
  Return the greatest common divisor of `x1` and `x2`.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  '''
  return funcs_keep_unit_binary(jnp.gcd, x1, x2)


# math funcs keep unit (n-ary)
# ----------------------------
@set_module_as('brainunit.math')
def interp(x: Union[Quantity, jax.typing.ArrayLike],
           xp: Union[Quantity, jax.typing.ArrayLike],
           fp: Union[Quantity, jax.typing.ArrayLike],
           left: Union[Quantity, jax.typing.ArrayLike] = None,
           right: Union[Quantity, jax.typing.ArrayLike] = None,
           period: Union[Quantity, jax.typing.ArrayLike] = None) -> Union[Quantity, jax.Array]:
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
    Union[jax.Array, Quantity]: Quantity if `x`, `xp`, and `fp` are Quantities that have the same unit, else an array.
  '''
  unit = None
  if isinstance(x, Quantity) or isinstance(xp, Quantity) or isinstance(fp, Quantity):
    unit = x.dim if isinstance(x, Quantity) else xp.dim if isinstance(xp, Quantity) else fp.dim
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
    return Quantity(result, dim=unit)
  else:
    return result


@set_module_as('brainunit.math')
def clip(a: Union[Quantity, jax.typing.ArrayLike],
         a_min: Union[Quantity, jax.typing.ArrayLike],
         a_max: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Clip (limit) the values in an array.

  Args:
    a: array_like, Quantity
    a_min: array_like, Quantity
    a_max: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `a`, `a_min`, and `a_max` are Quantities that have the same unit, else an array.
  '''
  unit = None
  if isinstance(a, Quantity) or isinstance(a_min, Quantity) or isinstance(a_max, Quantity):
    unit = a.dim if isinstance(a, Quantity) else a_min.dim if isinstance(a_min, Quantity) else a_max.dim
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
    return Quantity(result, dim=unit)
  else:
    return result
