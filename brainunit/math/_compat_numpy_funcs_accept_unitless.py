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

import jax
import jax.numpy as jnp
from jax import Array

from _misc import set_module_as
from .._base import (Quantity, fail_for_dimension_mismatch, )

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

def funcs_only_accept_unitless_unary(func, x, *args, **kwargs):
  if isinstance(x, Quantity):
    fail_for_dimension_mismatch(
      x,
      error_message="%s expects a dimensionless argument but got {value}" % func.__name__,
      value=x,
    )
    return func(jnp.array(x.value), *args, **kwargs)
  else:
    return func(x, *args, **kwargs)


@set_module_as('brainunit.math')
def exp(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Array, Quantity]:
  """
  Calculate the exponential of all elements in the input array.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.exp, x)


@set_module_as('brainunit.math')
def exp2(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Array, Quantity]:
  """
  Calculate 2 raised to the power of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.exp2, x)


@set_module_as('brainunit.math')
def expm1(x: Union[Array, Quantity]) -> Array:
  """
  Calculate the exponential of the input elements minus 1.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.expm1, x)


@set_module_as('brainunit.math')
def log(x: Union[Array, Quantity]) -> Array:
  """
  Natural logarithm, element-wise.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.log, x)


@set_module_as('brainunit.math')
def log10(x: Union[Array, Quantity]) -> Array:
  """
  Base-10 logarithm of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.log10, x)


@set_module_as('brainunit.math')
def log1p(x: Union[Array, Quantity]) -> Array:
  """
  Natural logarithm of 1 + the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.log1p, x)


@set_module_as('brainunit.math')
def log2(x: Union[Array, Quantity]) -> Array:
  """
  Base-2 logarithm of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.log2, x)


@set_module_as('brainunit.math')
def arccos(x: Union[Array, Quantity]) -> Array:
  """
  Compute the arccosine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.arccos, x)


@set_module_as('brainunit.math')
def arccosh(x: Union[Array, Quantity]) -> Array:
  """
  Compute the hyperbolic arccosine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.arccosh, x)


@set_module_as('brainunit.math')
def arcsin(x: Union[Array, Quantity]) -> Array:
  """
  Compute the arcsine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.arcsin, x)


@set_module_as('brainunit.math')
def arcsinh(x: Union[Array, Quantity]) -> Array:
  """
  Compute the hyperbolic arcsine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.arcsinh, x)


@set_module_as('brainunit.math')
def arctan(x: Union[Array, Quantity]) -> Array:
  """
  Compute the arctangent of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.arctan, x)


@set_module_as('brainunit.math')
def arctanh(x: Union[Array, Quantity]) -> Array:
  """
  Compute the hyperbolic arctangent of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.arctanh, x)


@set_module_as('brainunit.math')
def cos(x: Union[Array, Quantity]) -> Array:
  """
  Compute the cosine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.cos, x)


@set_module_as('brainunit.math')
def cosh(x: Union[Array, Quantity]) -> Array:
  """
  Compute the hyperbolic cosine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.cosh, x)


@set_module_as('brainunit.math')
def sin(x: Union[Array, Quantity]) -> Array:
  """
  Compute the sine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.sin, x)


@set_module_as('brainunit.math')
def sinc(x: Union[Array, Quantity]) -> Array:
  """
  Compute the sinc function of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.sinc, x)


@set_module_as('brainunit.math')
def sinh(x: Union[Array, Quantity]) -> Array:
  """
  Compute the hyperbolic sine of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.sinh, x)


@set_module_as('brainunit.math')
def tan(x: Union[Array, Quantity]) -> Array:
  """
  Compute the tangent of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.tan, x)


@set_module_as('brainunit.math')
def tanh(x: Union[Array, Quantity]) -> Array:
  """
  Compute the hyperbolic tangent of the input elements.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.tanh, x)


@set_module_as('brainunit.math')
def deg2rad(x: Union[Array, Quantity]) -> Array:
  """
  Convert angles from degrees to radians.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.deg2rad, x)


@set_module_as('brainunit.math')
def rad2deg(x: Union[Array, Quantity]) -> Array:
  """
  Convert angles from radians to degrees.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.rad2deg, x)


@set_module_as('brainunit.math')
def degrees(x: Union[Array, Quantity]) -> Array:
  """
  Convert angles from radians to degrees.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.degrees, x)


@set_module_as('brainunit.math')
def radians(x: Union[Array, Quantity]) -> Array:
  """
  Convert angles from degrees to radians.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.radians, x)


@set_module_as('brainunit.math')
def angle(x: Union[Array, Quantity]) -> Array:
  """
  Return the angle of the complex argument.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_unary(jnp.angle, x)


# math funcs only accept unitless (binary)
# ----------------------------------------


def funcs_only_accept_unitless_binary(func, x, y, *args, **kwargs):
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


@set_module_as('brainunit.math')
def hypot(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
  """
  Given the “legs” of a right triangle, return its hypotenuse.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_binary(jnp.hypot, x, y)


@set_module_as('brainunit.math')
def arctan2(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
  """
  Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_binary(jnp.arctan2, x, y)


@set_module_as('brainunit.math')
def logaddexp(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
  """
  Logarithm of the sum of exponentiations of the inputs.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_binary(jnp.logaddexp, x, y)


@set_module_as('brainunit.math')
def logaddexp2(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
  """
  Logarithm of the sum of exponentiations of the inputs in base-2.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_binary(jnp.logaddexp2, x, y)


@set_module_as('brainunit.math')
def percentile(a: Union[Array, Quantity], q: Union[Array, Quantity], *args, **kwargs) -> Array:
  """
  Compute the nth percentile of the input array along the specified axis.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_binary(jnp.percentile, a, q, *args, **kwargs)


@set_module_as('brainunit.math')
def nanpercentile(a: Union[Array, Quantity], q: Union[Array, Quantity], *args, **kwargs) -> Array:
  """
  Compute the nth percentile of the input array along the specified axis, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_binary(jnp.nanpercentile, a, q, *args, **kwargs)


@set_module_as('brainunit.math')
def quantile(a: Union[Array, Quantity], q: Union[Array, Quantity], *args, **kwargs) -> Array:
  """
  Compute the qth quantile of the input array along the specified axis.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_binary(jnp.quantile, a, q, *args, **kwargs)


@set_module_as('brainunit.math')
def nanquantile(a: Union[Array, Quantity], q: Union[Array, Quantity], *args, **kwargs) -> Array:
  """
  Compute the qth quantile of the input array along the specified axis, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    jax.Array: an array
  """
  return funcs_only_accept_unitless_binary(jnp.nanquantile, a, q, *args, **kwargs)
