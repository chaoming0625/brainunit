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

from typing import (Union, Optional, Tuple, Any, Callable)

import jax
import jax.numpy as jnp
from jax import Array

from .._base import (Quantity, Unit, fail_for_dimension_mismatch)
from .._misc import set_module_as

__all__ = [
  # math funcs only accept unitless (unary)
  'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2',
  'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
  'arctanh', 'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan',
  'tanh', 'deg2rad', 'rad2deg', 'degrees', 'radians', 'angle',
  'percentile', 'nanpercentile', 'quantile', 'nanquantile',
  'round', 'around', 'round_', 'rint',
  'floor', 'ceil', 'trunc', 'fix', 'modf', 'frexp',
  'corrcoef', 'correlate', 'cov',

  # math funcs only accept unitless (binary)
  'hypot', 'arctan2', 'logaddexp', 'logaddexp2',

  # Elementwise bit operations (unary)
  'bitwise_not', 'invert',

  # Elementwise bit operations (binary)
  'bitwise_and', 'bitwise_or', 'bitwise_xor', 'left_shift', 'right_shift',
]


# math funcs only accept unitless (unary)
# ---------------------------------------

def _func_accept_unitless_unary(
    func: Callable,
    x: jax.typing.ArrayLike | Quantity,
    *args,
    unit_to_scale: Optional[Unit] = None,
    **kwargs
):
  if isinstance(x, Quantity):
    if unit_to_scale is None:
      assert x.is_unitless, (f'Input should be unitless for the function "{func}" '
                             f'when scaling "unit_to_scale" is not provided.')
      x = x.value
      return func(x, *args, **kwargs)
    else:
      fail_for_dimension_mismatch(
        x,
        unit_to_scale,
        error_message="Unit mismatch: {value} != {unit_to_scale}",
        value=x,
        unit_to_scale=unit_to_scale
      )
      return Quantity(func(x / unit_to_scale, *args, **kwargs), unit=unit_to_scale)
  else:
    assert unit_to_scale is None, f'Unit should be None for the function "{func}" when "x" is not a Quantity.'
    return func(x, *args, **kwargs)


@set_module_as('brainunit.math')
def exp(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Calculate the exponential of all elements in the input quantity or array.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.exp, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def exp2(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Calculate 2**p for all p in the input quantity or array.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.exp2, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def expm1(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Calculate the exponential of the input elements minus 1.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.expm1, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def log(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Natural logarithm, element-wise.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.log, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def log10(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Base-10 logarithm of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.log10, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def log1p(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Natural logarithm of 1 + the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.log1p, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def log2(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Base-2 logarithm of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.log2, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arccos(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the arccosine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.arccos, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arccosh(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the hyperbolic arccosine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.arccosh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arcsin(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the arcsine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.arcsin, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arcsinh(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the hyperbolic arcsine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.arcsinh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arctan(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the arctangent of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.arctan, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arctanh(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the hyperbolic arctangent of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.arctanh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def cos(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the cosine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.cos, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def cosh(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the hyperbolic cosine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.cosh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def sin(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the sine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.sin, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def sinc(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the sinc function of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.sinc, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def sinh(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the hyperbolic sine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.sinh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def tan(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the tangent of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.tan, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def tanh(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Compute the hyperbolic tangent of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.tanh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def deg2rad(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Convert angles from degrees to radians.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.deg2rad, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def rad2deg(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Convert angles from radians to degrees.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.rad2deg, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def degrees(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Convert angles from radians to degrees.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.degrees, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def radians(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Convert angles from degrees to radians.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.radians, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def angle(
    x: Union[Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> Array | Quantity:
  """
  Return the angle of the complex argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _func_accept_unitless_unary(jnp.angle, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def round_(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Round an array to the nearest integer.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _func_accept_unitless_unary(jnp.round_, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def around(
    x: Union[Quantity, jax.typing.ArrayLike],
    decimals: int = 0,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Round an array to the nearest integer.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  decimals : int, optional
    Number of decimal places to round to (default is 0).
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _func_accept_unitless_unary(jnp.around, x, unit_to_scale=unit_to_scale, decimals=decimals)


@set_module_as('brainunit.math')
def round(
    x: Union[Quantity, jax.typing.ArrayLike],
    decimals: int = 0,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array | Quantity:
  """
  Round an array to the nearest integer.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  decimals : int, optional
    Number of decimal places to round to (default is 0).
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _func_accept_unitless_unary(jnp.round, x, unit_to_scale=unit_to_scale, decimals=decimals)


@set_module_as('brainunit.math')
def rint(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> Union[Quantity, jax.Array]:
  """
  Round an array to the nearest integer.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _func_accept_unitless_unary(jnp.rint, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def floor(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Return the floor of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _func_accept_unitless_unary(jnp.floor, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def ceil(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Return the ceiling of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _func_accept_unitless_unary(jnp.ceil, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def trunc(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Return the truncated value of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _func_accept_unitless_unary(jnp.trunc, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def fix(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Return the nearest integer towards zero.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _func_accept_unitless_unary(jnp.fix, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def modf(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> Tuple[jax.Array, jax.Array]:
  """
  Return the fractional and integer parts of the array elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  The fractional and integral parts of the input, both with the same dimension.
  """
  return _func_accept_unitless_unary(jnp.modf, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def frexp(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> Tuple[jax.Array, jax.Array]:
  """
  Decompose the elements of x into mantissa and twos exponent.

  Returns (`mantissa`, `exponent`), where ``x = mantissa * 2**exponent``.
  The mantissa lies in the open interval(-1, 1), while the twos
  exponent is a signed integer.

  Parameters
  ----------
  x : array_like, Quantity
    Array of numbers to be decomposed.
  unit_to_scale : Unit, optional
    The unit_to_scale to scale the ``x``.

  Returns
  -------
  mantissa : ndarray
    Floating values between -1 and 1.
    This is a scalar if `x` is a scalar.
  exponent : ndarray
    Integer exponents of 2.
    This is a scalar if `x` is a scalar.
  """
  return _func_accept_unitless_unary(jnp.frexp, x, unit_to_scale=unit_to_scale)


# math funcs only accept unitless (binary)
# ----------------------------------------


def _fun_accept_unitless_binary(func, x, y, *args, **kwargs):
  x_value = x.value if isinstance(x, Quantity) else x
  y_value = y.value if isinstance(y, Quantity) else y
  if isinstance(x, Quantity) or isinstance(y, Quantity):
    fail_for_dimension_mismatch(
      x,
      y,
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
def hypot(
    x: Union[Array, Quantity],
    y: Union[Array, Quantity],
) -> Array | Quantity:
  """
  Given the “legs” of a right triangle, return its hypotenuse.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  y : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_binary(jnp.hypot, x, y)


@set_module_as('brainunit.math')
def arctan2(
    x: Union[Array, Quantity],
    y: Union[Array, Quantity],
) -> Array | Quantity:
  """
  Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  y : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_binary(jnp.arctan2, x, y)


@set_module_as('brainunit.math')
def logaddexp(
    x: Union[Array, Quantity],
    y: Union[Array, Quantity],
) -> Array | Quantity:
  """
  Logarithm of the sum of exponentiations of the inputs.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  y : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_binary(jnp.logaddexp, x, y)


@set_module_as('brainunit.math')
def logaddexp2(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array | Quantity:
  """
  Logarithm of the sum of exponentiations of the inputs in base-2.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  y : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_binary(jnp.logaddexp2, x, y)


@set_module_as('brainunit.math')
def percentile(
    a: Union[Array, Quantity],
    q: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int]]] = None,
    overwrite_input: Optional[bool] = None,
    method: str = 'linear',
    keepdims: Optional[bool] = False,
) -> Array:
  """
  Compute the q-th percentile of the data along the specified axis.

  Returns the q-th percentile(s) of the array elements.

  Parameters
  ----------
  a : array_like, Quantity
    Input array or Quantity.
  q : array_like, Quantity
    Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
  out : array_like, Quantity, optional
    Alternative output array in which to place the result.
    It must have the same shape and buffer length as the expected output but the type will be cast if necessary.
  overwrite_input : bool, optional
    If True, then allow the input array a to be modified by intermediate calculations, to save memory.
  method : str, optional
    This parameter specifies the method to use for estimating the
    percentile.  There are many different methods, some unique to NumPy.
    See the notes for explanation.  The options sorted by their R type
    as summarized in the H&F paper [1]_ are:

    1. 'inverted_cdf'
    2. 'averaged_inverted_cdf'
    3. 'closest_observation'
    4. 'interpolated_inverted_cdf'
    5. 'hazen'
    6. 'weibull'
    7. 'linear'  (default)
    8. 'median_unbiased'
    9. 'normal_unbiased'

    The first three methods are discontinuous.  NumPy further defines the
    following discontinuous variations of the default 'linear' (7.) option:

    * 'lower'
    * 'higher',
    * 'midpoint'
    * 'nearest'
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_binary(
    jnp.percentile, a, q, axis=axis, overwrite_input=overwrite_input,
    method=method, keepdims=keepdims
  )


@set_module_as('brainunit.math')
def nanpercentile(
    a: Union[Array, Quantity],
    q: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int]]] = None,
    overwrite_input: Optional[bool] = None,
    method: str = 'linear',
    keepdims: Optional[bool] = False,
) -> Array:
  """
  Compute the q-th percentile of the data along the specified axis, while ignoring nan values.

  Returns the q-th percentile(s) of the array elements, while ignoring nan values.

  Parameters
  ----------
  a : array_like, Quantity
    Input array or Quantity.
  q : array_like, Quantity
    Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
  out : array_like, Quantity, optional
    Alternative output array in which to place the result.
    It must have the same shape and buffer length as the expected output but the type will be cast if necessary.
  overwrite_input : bool, optional
    If True, then allow the input array a to be modified by intermediate calculations, to save memory.
  method : str, optional
    This parameter specifies the method to use for estimating the
    percentile.  There are many different methods, some unique to NumPy.
    See the notes for explanation.  The options sorted by their R type
    as summarized in the H&F paper [1]_ are:

    1. 'inverted_cdf'
    2. 'averaged_inverted_cdf'
    3. 'closest_observation'
    4. 'interpolated_inverted_cdf'
    5. 'hazen'
    6. 'weibull'
    7. 'linear'  (default)
    8. 'median_unbiased'
    9. 'normal_unbiased'

    The first three methods are discontinuous.  NumPy further defines the
    following discontinuous variations of the default 'linear' (7.) option:

    * 'lower'
    * 'higher',
    * 'midpoint'
    * 'nearest'
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_binary(
    jnp.nanpercentile, a, q,
    axis=axis, ooverwrite_input=overwrite_input,
    method=method, keepdims=keepdims
  )


@set_module_as('brainunit.math')
def quantile(
    a: Union[Array, Quantity],
    q: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int]]] = None,
    overwrite_input: Optional[bool] = None,
    method: str = 'linear',
    keepdims: Optional[bool] = False,
) -> Array:
  """
  Compute the q-th percentile of the data along the specified axis.

  Returns the q-th percentile(s) of the array elements.

  Parameters
  ----------
  a : array_like, Quantity
    Input array or Quantity.
  q : array_like, Quantity
    Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
  overwrite_input : bool, optional
    If True, then allow the input array a to be modified by intermediate calculations, to save memory.
  method : str, optional
    This parameter specifies the method to use for estimating the
    percentile.  There are many different methods, some unique to NumPy.
    See the notes for explanation.  The options sorted by their R type
    as summarized in the H&F paper [1]_ are:

    1. 'inverted_cdf'
    2. 'averaged_inverted_cdf'
    3. 'closest_observation'
    4. 'interpolated_inverted_cdf'
    5. 'hazen'
    6. 'weibull'
    7. 'linear'  (default)
    8. 'median_unbiased'
    9. 'normal_unbiased'

    The first three methods are discontinuous.  NumPy further defines the
    following discontinuous variations of the default 'linear' (7.) option:

    * 'lower'
    * 'higher',
    * 'midpoint'
    * 'nearest'
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_binary(
    jnp.quantile, a, q,
    axis=axis, overwrite_input=overwrite_input,
    method=method, keepdims=keepdims
  )


@set_module_as('brainunit.math')
def nanquantile(
    a: Union[Array, Quantity],
    q: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int]]] = None,
    overwrite_input: Optional[bool] = None,
    method: str = 'linear',
    keepdims: Optional[bool] = False,
) -> Array:
  """
  Compute the q-th percentile of the data along the specified axis, while ignoring nan values.

  Returns the q-th percentile(s) of the array elements, while ignoring nan values.

  Parameters
  ----------
  a : array_like, Quantity
    Input array or Quantity.
  q : array_like, Quantity
    Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
  overwrite_input : bool, optional
    If True, then allow the input array a to be modified by intermediate calculations, to save memory.
  method : str, optional
    This parameter specifies the method to use for estimating the
    percentile.  There are many different methods, some unique to NumPy.
    See the notes for explanation.  The options sorted by their R type
    as summarized in the H&F paper [1]_ are:

    1. 'inverted_cdf'
    2. 'averaged_inverted_cdf'
    3. 'closest_observation'
    4. 'interpolated_inverted_cdf'
    5. 'hazen'
    6. 'weibull'
    7. 'linear'  (default)
    8. 'median_unbiased'
    9. 'normal_unbiased'

    The first three methods are discontinuous.  NumPy further defines the
    following discontinuous variations of the default 'linear' (7.) option:

    * 'lower'
    * 'higher',
    * 'midpoint'
    * 'nearest'
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_binary(
    jnp.nanquantile, a, q,
    axis=axis, overwrite_input=overwrite_input,
    method=method, keepdims=keepdims
  )


@set_module_as('brainunit.math')
def corrcoef(
    x: Union[Array, Quantity],
    y: Union[Array, Quantity],
    rowvar: bool = True,
    unit_to_scale: Optional[Unit] = None,
) -> Array:
  """
  Return Pearson product-moment correlation coefficients.

  Please refer to the documentation for `cov` for more detail.  The
  relationship between the correlation coefficient matrix, `R`, and the
  covariance matrix, `C`, is

  .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} C_{jj} } }

  The values of `R` are between -1 and 1, inclusive.

  Parameters
  ----------
  x : array_like, Quantity
    A 1-D or 2-D array containing multiple variables and observations.
    Each row of `x` represents a variable, and each column a single
    observation of all those variables. Also see `rowvar` below.
  y : array_like, Quantity, optional
    An additional set of variables and observations. `y` has the same
    shape as `x`.
  rowvar : bool, optional
    If `rowvar` is True (default), then each row represents a
    variable, with observations in the columns. Otherwise, the relationship
    is transposed: each column represents a variable, while the rows
    contain observations.

  Returns
  -------
  R : ndarray
    The correlation coefficient matrix of the variables.
  """
  if isinstance(x, Quantity):
    assert x.is_unitless, 'Input should be unitless for the function "corrcoef".'
    x = x.value
  if isinstance(y, Quantity):
    assert y.is_unitless, 'Input should be unitless for the function "corrcoef".'
    y = y.value
  return jnp.corrcoef(x, y, rowvar=rowvar)


@set_module_as('brainunit.math')
def correlate(
    a: Union[Array, Quantity],
    v: Union[Array, Quantity],
    mode: str = 'valid',
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None,
    unit_to_scale: Optional[Unit] = None,
) -> Array:
  """
  Cross-correlation of two 1-dimensional sequences.

  This function computes the correlation as generally defined in signal
  processing texts:

  .. math:: c_k = \sum_n a_{n+k} \cdot \overline{v}_n

  with a and v sequences being zero-padded where necessary and
  :math:`\overline x` denoting complex conjugation.

  Parameters
  ----------
  a, v : array_like, Quantity
    Input sequences.
  mode : {'valid', 'same', 'full'}, optional
    Refer to the `convolve` docstring.  Note that the default
    is 'valid', unlike `convolve`, which uses 'full'.
  precision : Optional. Either ``None``, which means the default precision for
    the backend, a :class:`~jax.lax.Precision` enum value
    (``Precision.DEFAULT``, ``Precision.HIGH`` or ``Precision.HIGHEST``), a
    string (e.g. 'highest' or 'fastest', see the
    ``jax.default_matmul_precision`` context manager), or a tuple of two
    :class:`~jax.lax.Precision` enums or strings indicating precision of
    ``lhs`` and ``rhs``.
  preferred_element_type : Optional. Either ``None``, which means the default
    accumulation type for the input types, or a datatype, indicating to
    accumulate results to and return a result with that datatype.

  Returns
  -------
  out : ndarray
    Discrete cross-correlation of `a` and `v`.
  """
  if isinstance(a, Quantity):
    assert a.is_unitless, 'Input should be unitless for the function "correlate".'
    a = a.value
  if isinstance(v, Quantity):
    assert v.is_unitless, 'Input should be unitless for the function "correlate".'
    v = v.value
  return jnp.correlate(a, v, mode=mode, precision=precision,
                       preferred_element_type=preferred_element_type)


@set_module_as('brainunit.math')
def cov(
    m: Union[Array, Quantity],
    y: Optional[Union[Array, Quantity]] = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[jax.typing.ArrayLike] = None,
    aweights: Optional[jax.typing.ArrayLike] = None,
    unit_to_scale: Optional[Unit] = None,
) -> Array:
  """
  Estimate a covariance matrix, given data and weights.

  Covariance indicates the level to which two variables vary together.
  If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
  then the covariance matrix element :math:`C_{ij}` is the covariance of
  :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
  of :math:`x_i`.

  See the notes for an outline of the algorithm.

  Parameters
  ----------
  m : array_like, Quantity
    A 1-D or 2-D array containing multiple variables and observations.
    Each row of `m` represents a variable, and each column a single
    observation of all those variables. Also see `rowvar` below.
  y : array_like, Quantity or optional
    An additional set of variables and observations. `y` has the same form
    as that of `m`.
  rowvar : bool, optional
    If `rowvar` is True (default), then each row represents a
    variable, with observations in the columns. Otherwise, the relationship
    is transposed: each column represents a variable, while the rows
    contain observations.
  bias : bool, optional
    Default normalization (False) is by ``(N - 1)``, where ``N`` is the
    number of observations given (unbiased estimate). If `bias` is True,
    then normalization is by ``N``. These values can be overridden by using
    the keyword ``ddof`` in numpy versions >= 1.5.
  ddof : int, optional
    If not ``None`` the default value implied by `bias` is overridden.
    Note that ``ddof=1`` will return the unbiased estimate, even if both
    `fweights` and `aweights` are specified, and ``ddof=0`` will return
    the simple average. See the notes for the details. The default value
    is ``None``.
  fweights : array_like, int, optional
    1-D array of integer frequency weights; the number of times each
    observation vector should be repeated.
  aweights : array_like, optional
    1-D array of observation vector weights. These relative weights are
    typically large for observations considered "important" and smaller for
    observations considered less "important". If ``ddof=0`` the array of
    weights can be used to assign probabilities to observation vectors.

  Returns
  -------
  out : ndarray
    The covariance matrix of the variables.
  """
  if isinstance(m, Quantity):
    assert m.is_unitless, 'Input should be unitless for the function "cov".'
    m = m.value
  if isinstance(y, Quantity):
    assert y.is_unitless, 'Input should be unitless for the function "cov".'
    y = y.value
  return jnp.cov(m, y,
                 rowvar=rowvar, bias=bias, ddof=ddof, fweights=fweights,
                 aweights=aweights)


# Elementwise bit operations (unary)
# ----------------------------------

def _fun_unitless_unary(func, x, *args, **kwargs):
  if isinstance(x, Quantity):
    assert x.is_unitless, f'Expected unitless array, got {x}'
    x = x.value
  return func(x, *args, **kwargs)


@set_module_as('brainunit.math')
def bitwise_not(x: Union[Quantity, jax.typing.ArrayLike]) -> Array:
  """
  Compute the bit-wise NOT of an array, element-wise.

  Parameters
  ----------
  x: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_unary(jnp.bitwise_not, x)


@set_module_as('brainunit.math')
def invert(x: Union[Quantity, jax.typing.ArrayLike]) -> Array:
  """
  Compute bit-wise inversion, or bit-wise NOT, element-wise.

  Parameters
  ----------
  x: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_unary(jnp.invert, x)


# Elementwise bit operations (binary)
# -----------------------------------


def _fun_unitless_binary(func, x, y, *args, **kwargs):
  if isinstance(x, Quantity):
    assert x.is_unitless, f'Expected unitless array, got {x}'
    x = x.value
  if isinstance(y, Quantity):
    assert y.is_unitless, f'Expected unitless array, got {y}'
    y = y.value
  return func(x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def bitwise_and(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Compute the bit-wise AND of two arrays element-wise.

  Parameters
  ----------
  x: array_like, quantity
    Input array.
  y: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_binary(jnp.bitwise_and, x, y)


@set_module_as('brainunit.math')
def bitwise_or(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Compute the bit-wise OR of two arrays element-wise.

  Parameters
  ----------
  x: array_like, quantity
    Input array.
  y: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_binary(jnp.bitwise_or, x, y)


@set_module_as('brainunit.math')
def bitwise_xor(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Compute the bit-wise XOR of two arrays element-wise.

  Parameters
  ----------
  x: array_like, quantity
    Input array.
  y: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_binary(jnp.bitwise_xor, x, y)


@set_module_as('brainunit.math')
def left_shift(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Shift the bits of an integer to the left.

  Parameters
  ----------
  x: array_like, quantity
    Input array.
  y: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_binary(jnp.left_shift, x, y)


@set_module_as('brainunit.math')
def right_shift(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Array:
  """
  Shift the bits of an integer to the right.

  Parameters
  ----------
  x: array_like, quantity
    Input array.
  y: array_like, quantity
    Input array.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_unitless_binary(jnp.right_shift, x, y)
