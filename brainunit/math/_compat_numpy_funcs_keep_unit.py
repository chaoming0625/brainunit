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
from typing import (Union)

import brainstate as bst
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

def wrap_math_funcs_keep_unit_unary(func):
  @wraps(func)
  def f(x, *args, **kwargs):
    if isinstance(x, Quantity):
      return Quantity(func(x.value, *args, **kwargs), dim=x.dim)
    elif isinstance(x, (jax.Array, np.ndarray)):
      return func(x, *args, **kwargs)
    else:
      raise ValueError(f'Unsupported type: {type(x)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


@wrap_math_funcs_keep_unit_unary
def real(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.real(x)


@wrap_math_funcs_keep_unit_unary
def imag(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.imag(x)


@wrap_math_funcs_keep_unit_unary
def conj(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.conj(x)


@wrap_math_funcs_keep_unit_unary
def conjugate(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.conjugate(x)


@wrap_math_funcs_keep_unit_unary
def negative(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.negative(x)


@wrap_math_funcs_keep_unit_unary
def positive(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.positive(x)


@wrap_math_funcs_keep_unit_unary
def abs(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.abs(x)


@wrap_math_funcs_keep_unit_unary
def round_(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.round(x)


@wrap_math_funcs_keep_unit_unary
def around(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.around(x)


@wrap_math_funcs_keep_unit_unary
def round(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.round(x)


@wrap_math_funcs_keep_unit_unary
def rint(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.rint(x)


@wrap_math_funcs_keep_unit_unary
def floor(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.floor(x)


@wrap_math_funcs_keep_unit_unary
def ceil(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.ceil(x)


@wrap_math_funcs_keep_unit_unary
def trunc(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.trunc(x)


@wrap_math_funcs_keep_unit_unary
def fix(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.fix(x)


@wrap_math_funcs_keep_unit_unary
def sum(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.sum(x)


@wrap_math_funcs_keep_unit_unary
def nancumsum(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.nancumsum(x)


@wrap_math_funcs_keep_unit_unary
def nansum(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.nansum(x)


@wrap_math_funcs_keep_unit_unary
def cumsum(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.cumsum(x)


@wrap_math_funcs_keep_unit_unary
def ediff1d(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.ediff1d(x)


@wrap_math_funcs_keep_unit_unary
def absolute(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.absolute(x)


@wrap_math_funcs_keep_unit_unary
def fabs(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.fabs(x)


@wrap_math_funcs_keep_unit_unary
def median(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.median(x)


@wrap_math_funcs_keep_unit_unary
def nanmin(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.nanmin(x)


@wrap_math_funcs_keep_unit_unary
def nanmax(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.nanmax(x)


@wrap_math_funcs_keep_unit_unary
def ptp(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.ptp(x)


@wrap_math_funcs_keep_unit_unary
def average(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.average(x)


@wrap_math_funcs_keep_unit_unary
def mean(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.mean(x)


@wrap_math_funcs_keep_unit_unary
def std(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.std(x)


@wrap_math_funcs_keep_unit_unary
def nanmedian(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.nanmedian(x)


@wrap_math_funcs_keep_unit_unary
def nanmean(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.nanmean(x)


@wrap_math_funcs_keep_unit_unary
def nanstd(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.nanstd(x)


@wrap_math_funcs_keep_unit_unary
def diff(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.diff(x)


@wrap_math_funcs_keep_unit_unary
def modf(x: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.modf(x)


# docs for the functions above
real.__doc__ = '''
  Return the real part of the complex argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

imag.__doc__ = '''
  Return the imaginary part of the complex argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

conj.__doc__ = '''
  Return the complex conjugate of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

conjugate.__doc__ = '''
  Return the complex conjugate of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

negative.__doc__ = '''
  Return the negative of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

positive.__doc__ = '''
  Return the positive of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

abs.__doc__ = '''
  Return the absolute value of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

round_.__doc__ = '''
  Round an array to the nearest integer.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

around.__doc__ = '''
  Round an array to the nearest integer.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

round.__doc__ = '''
  Round an array to the nearest integer.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

rint.__doc__ = '''
  Round an array to the nearest integer.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

floor.__doc__ = '''
  Return the floor of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

ceil.__doc__ = '''
  Return the ceiling of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

trunc.__doc__ = '''
  Return the truncated value of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

fix.__doc__ = '''
  Return the nearest integer towards zero.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

sum.__doc__ = '''
  Return the sum of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

nancumsum.__doc__ = '''
  Return the cumulative sum of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

nansum.__doc__ = '''
  Return the sum of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

cumsum.__doc__ = '''
  Return the cumulative sum of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

ediff1d.__doc__ = '''
  Return the differences between consecutive elements of the array.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

absolute.__doc__ = '''
  Return the absolute value of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

fabs.__doc__ = '''
  Return the absolute value of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

median.__doc__ = '''
  Return the median of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

nanmin.__doc__ = '''
  Return the minimum of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

nanmax.__doc__ = '''
  Return the maximum of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

ptp.__doc__ = '''
  Return the range of the array elements (maximum - minimum).

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

average.__doc__ = '''
  Return the weighted average of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

mean.__doc__ = '''
  Return the mean of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

std.__doc__ = '''
  Return the standard deviation of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

nanmedian.__doc__ = '''
  Return the median of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

nanmean.__doc__ = '''
  Return the mean of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

nanstd.__doc__ = '''
  Return the standard deviation of the array elements, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

diff.__doc__ = '''
  Return the differences between consecutive elements of the array.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

modf.__doc__ = '''
  Return the fractional and integer parts of the array elements.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity tuple if `x` is a Quantity, else an array tuple.
'''


# math funcs keep unit (binary)
# -----------------------------

def wrap_math_funcs_keep_unit_binary(func):
  @wraps(func)
  def f(x1, x2, *args, **kwargs):
    if isinstance(x1, Quantity) and isinstance(x2, Quantity):
      return Quantity(func(x1.value, x2.value, *args, **kwargs), dim=x1.dim)
    elif isinstance(x1, (jax.Array, np.ndarray)) and isinstance(x2, (jax.Array, np.ndarray)):
      return func(x1, x2, *args, **kwargs)
    else:
      raise ValueError(f'Unsupported type: {type(x1)} and {type(x2)} for {func.__name__}')

  f.__module__ = 'brainunit.math'
  return f


@wrap_math_funcs_keep_unit_binary
def fmod(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  return jnp.fmod(x1, x2)


@wrap_math_funcs_keep_unit_binary
def mod(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  return jnp.mod(x1, x2)


@wrap_math_funcs_keep_unit_binary
def copysign(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  return jnp.copysign(x1, x2)


@wrap_math_funcs_keep_unit_binary
def heaviside(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  return jnp.heaviside(x1, x2)


@wrap_math_funcs_keep_unit_binary
def maximum(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  return jnp.maximum(x1, x2)


@wrap_math_funcs_keep_unit_binary
def minimum(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  return jnp.minimum(x1, x2)


@wrap_math_funcs_keep_unit_binary
def fmax(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  return jnp.fmax(x1, x2)


@wrap_math_funcs_keep_unit_binary
def fmin(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  return jnp.fmin(x1, x2)


@wrap_math_funcs_keep_unit_binary
def lcm(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  return jnp.lcm(x1, x2)


@wrap_math_funcs_keep_unit_binary
def gcd(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  return jnp.gcd(x1, x2)


# docs for the functions above
fmod.__doc__ = '''
  Return the element-wise remainder of division.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

mod.__doc__ = '''
  Return the element-wise modulus of division.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

copysign.__doc__ = '''
  Return a copy of the first array elements with the sign of the second array.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

heaviside.__doc__ = '''
  Compute the Heaviside step function.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

maximum.__doc__ = '''
  Element-wise maximum of array elements.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

minimum.__doc__ = '''
  Element-wise minimum of array elements.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

fmax.__doc__ = '''
  Element-wise maximum of array elements ignoring NaNs.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

fmin.__doc__ = '''
  Element-wise minimum of array elements ignoring NaNs.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

lcm.__doc__ = '''
  Return the least common multiple of `x1` and `x2`.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''

gcd.__doc__ = '''
  Return the greatest common divisor of `x1` and `x2`.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
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
