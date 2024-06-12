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

import jax
import jax.numpy as jnp
from jax import Array

from .._base import (Quantity,
                     fail_for_dimension_mismatch,
                     )

__all__ = [
  # math funcs only accept unitless (unary)
  'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2',
  'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
  'arctanh', 'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan',
  'tanh', 'deg2rad', 'rad2deg', 'degrees', 'radians', 'angle',
  'percentile', 'nanpercentile', 'quantile', 'nanquantile',

  # math funcs only accept unitless (binary)
  'hypot', 'arctan2', 'logaddexp', 'logaddexp2',
]


# math funcs only accept unitless (unary)
# ---------------------------------------

def wrap_math_funcs_only_accept_unitless_unary(func):
  @wraps(func)
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


@wrap_math_funcs_only_accept_unitless_unary
def exp(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Array, Quantity]:
  return jnp.exp(x)


@wrap_math_funcs_only_accept_unitless_unary
def exp2(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Array, Quantity]:
  return jnp.exp2(x)


@wrap_math_funcs_only_accept_unitless_unary
def expm1(x: Union[Array, Quantity]) -> Array:
  return jnp.expm1(x)


@wrap_math_funcs_only_accept_unitless_unary
def log(x: Union[Array, Quantity]) -> Array:
  return jnp.log(x)


@wrap_math_funcs_only_accept_unitless_unary
def log10(x: Union[Array, Quantity]) -> Array:
  return jnp.log10(x)


@wrap_math_funcs_only_accept_unitless_unary
def log1p(x: Union[Array, Quantity]) -> Array:
  return jnp.log1p(x)


@wrap_math_funcs_only_accept_unitless_unary
def log2(x: Union[Array, Quantity]) -> Array:
  return jnp.log2(x)


@wrap_math_funcs_only_accept_unitless_unary
def arccos(x: Union[Array, Quantity]) -> Array:
  return jnp.arccos(x)


@wrap_math_funcs_only_accept_unitless_unary
def arccosh(x: Union[Array, Quantity]) -> Array:
  return jnp.arccosh(x)


@wrap_math_funcs_only_accept_unitless_unary
def arcsin(x: Union[Array, Quantity]) -> Array:
  return jnp.arcsin(x)


@wrap_math_funcs_only_accept_unitless_unary
def arcsinh(x: Union[Array, Quantity]) -> Array:
  return jnp.arcsinh(x)


@wrap_math_funcs_only_accept_unitless_unary
def arctan(x: Union[Array, Quantity]) -> Array:
  return jnp.arctan(x)


@wrap_math_funcs_only_accept_unitless_unary
def arctanh(x: Union[Array, Quantity]) -> Array:
  return jnp.arctanh(x)


@wrap_math_funcs_only_accept_unitless_unary
def cos(x: Union[Array, Quantity]) -> Array:
  return jnp.cos(x)


@wrap_math_funcs_only_accept_unitless_unary
def cosh(x: Union[Array, Quantity]) -> Array:
  return jnp.cosh(x)


@wrap_math_funcs_only_accept_unitless_unary
def sin(x: Union[Array, Quantity]) -> Array:
  return jnp.sin(x)


@wrap_math_funcs_only_accept_unitless_unary
def sinc(x: Union[Array, Quantity]) -> Array:
  return jnp.sinc(x)


@wrap_math_funcs_only_accept_unitless_unary
def sinh(x: Union[Array, Quantity]) -> Array:
  return jnp.sinh(x)


@wrap_math_funcs_only_accept_unitless_unary
def tan(x: Union[Array, Quantity]) -> Array:
  return jnp.tan(x)


@wrap_math_funcs_only_accept_unitless_unary
def tanh(x: Union[Array, Quantity]) -> Array:
  return jnp.tanh(x)


@wrap_math_funcs_only_accept_unitless_unary
def deg2rad(x: Union[Array, Quantity]) -> Array:
  return jnp.deg2rad(x)


@wrap_math_funcs_only_accept_unitless_unary
def rad2deg(x: Union[Array, Quantity]) -> Array:
  return jnp.rad2deg(x)


@wrap_math_funcs_only_accept_unitless_unary
def degrees(x: Union[Array, Quantity]) -> Array:
  return jnp.degrees(x)


@wrap_math_funcs_only_accept_unitless_unary
def radians(x: Union[Array, Quantity]) -> Array:
  return jnp.radians(x)


@wrap_math_funcs_only_accept_unitless_unary
def angle(x: Union[Array, Quantity]) -> Array:
  return jnp.angle(x)


# docs for the functions above
exp.__doc__ = '''
  Calculate the exponential of all elements in the input array.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

exp2.__doc__ = '''
  Calculate 2 raised to the power of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

expm1.__doc__ = '''
  Calculate the exponential of the input elements minus 1.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

log.__doc__ = '''
  Natural logarithm, element-wise.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

log10.__doc__ = '''
  Base-10 logarithm of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

log1p.__doc__ = '''
  Natural logarithm of 1 + the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

log2.__doc__ = '''
  Base-2 logarithm of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

arccos.__doc__ = '''
  Compute the arccosine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

arccosh.__doc__ = '''
  Compute the hyperbolic arccosine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

arcsin.__doc__ = '''
  Compute the arcsine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

arcsinh.__doc__ = '''
  Compute the hyperbolic arcsine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

arctan.__doc__ = '''
  Compute the arctangent of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

arctanh.__doc__ = '''
  Compute the hyperbolic arctangent of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

cos.__doc__ = '''
  Compute the cosine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

cosh.__doc__ = '''
  Compute the hyperbolic cosine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

sin.__doc__ = '''
  Compute the sine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

sinc.__doc__ = '''
  Compute the sinc function of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

sinh.__doc__ = '''
  Compute the hyperbolic sine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

tan.__doc__ = '''
  Compute the tangent of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

tanh.__doc__ = '''
  Compute the hyperbolic tangent of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

deg2rad.__doc__ = '''
  Convert angles from degrees to radians.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

rad2deg.__doc__ = '''
  Convert angles from radians to degrees.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

degrees.__doc__ = '''
  Convert angles from radians to degrees.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

radians.__doc__ = '''
  Convert angles from degrees to radians.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

angle.__doc__ = '''
  Return the angle of the complex argument.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''


# math funcs only accept unitless (binary)
# ----------------------------------------

def wrap_math_funcs_only_accept_unitless_binary(func):
  @wraps(func)
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


@wrap_math_funcs_only_accept_unitless_binary
def hypot(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
  return jnp.hypot(x, y)


@wrap_math_funcs_only_accept_unitless_binary
def arctan2(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
  return jnp.arctan2(x, y)


@wrap_math_funcs_only_accept_unitless_binary
def logaddexp(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
  return jnp.logaddexp(x, y)


@wrap_math_funcs_only_accept_unitless_binary
def logaddexp2(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
  return jnp.logaddexp2(x, y)


@wrap_math_funcs_only_accept_unitless_binary
def percentile(a: Union[Array, Quantity], q: Union[Array, Quantity], *args, **kwargs) -> Array:
  return jnp.percentile(a, q, *args, **kwargs)


@wrap_math_funcs_only_accept_unitless_binary
def nanpercentile(a: Union[Array, Quantity], q: Union[Array, Quantity], *args, **kwargs) -> Array:
  return jnp.nanpercentile(a, q, *args, **kwargs)


@wrap_math_funcs_only_accept_unitless_binary
def quantile(a: Union[Array, Quantity], q: Union[Array, Quantity], *args, **kwargs) -> Array:
  return jnp.quantile(a, q, *args, **kwargs)


@wrap_math_funcs_only_accept_unitless_binary
def nanquantile(a: Union[Array, Quantity], q: Union[Array, Quantity], *args, **kwargs) -> Array:
  return jnp.nanquantile(a, q, *args, **kwargs)


# docs for the functions above
hypot.__doc__ = '''
  Given the “legs” of a right triangle, return its hypotenuse.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    jax.Array: an array
'''

arctan2.__doc__ = '''
  Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    jax.Array: an array
'''

logaddexp.__doc__ = '''
  Logarithm of the sum of exponentiations of the inputs.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    jax.Array: an array
'''

logaddexp2.__doc__ = '''
  Logarithm of the sum of exponentiations of the inputs in base-2.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    jax.Array: an array
'''

percentile.__doc__ = '''
  Compute the nth percentile of the input array along the specified axis.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

nanpercentile.__doc__ = '''
  Compute the nth percentile of the input array along the specified axis, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

quantile.__doc__ = '''
  Compute the qth quantile of the input array along the specified axis.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''

nanquantile.__doc__ = '''
  Compute the qth quantile of the input array along the specified axis, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
'''
