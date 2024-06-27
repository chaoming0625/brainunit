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

from .._base import (Quantity, Unit, fail_for_dimension_mismatch)
from .._misc import set_module_as

__all__ = [
  # math funcs only accept unitless (unary)
  'exprel', 'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2',
  'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
  'arctanh', 'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan',
  'tanh', 'deg2rad', 'rad2deg', 'degrees', 'radians', 'angle',
  'percentile', 'nanpercentile', 'quantile', 'nanquantile',
  'corrcoef', 'correlate', 'cov',

  # math funcs only accept unitless (unary) can return Quantity
  'round', 'around', 'round_', 'rint',
  'floor', 'ceil', 'trunc', 'fix', 'modf', 'frexp',


  # math funcs only accept unitless (binary)
  'hypot', 'arctan2', 'logaddexp', 'logaddexp2',

  # Elementwise bit operations (unary)
  'bitwise_not', 'invert',

  # Elementwise bit operations (binary)
  'bitwise_and', 'bitwise_or', 'bitwise_xor', 'left_shift', 'right_shift',
]


# math funcs only accept unitless (unary)
# ---------------------------------------

def _fun_accept_unitless_unary(
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
      return func(x.to_value(unit_to_scale), *args, **kwargs)
  else:
    assert unit_to_scale is None, f'Unit should be None for the function "{func}" when "x" is not a Quantity.'
    return func(x, *args, **kwargs)


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
  return _fun_accept_unitless_unary(_exprel_v2, x, level=level)


@set_module_as('brainunit.math')
def exp(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Calculate the exponential of all elements in the input quantity or array.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.exp, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def exp2(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Calculate 2**p for all p in the input quantity or array.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.exp2, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def expm1(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Calculate the exponential of the input elements minus 1.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.expm1, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def log(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Natural logarithm, element-wise.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.log, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def log10(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Base-10 logarithm of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.log10, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def log1p(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Natural logarithm of 1 + the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.log1p, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def log2(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Base-2 logarithm of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.log2, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arccos(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the arccosine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.arccos, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arccosh(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the hyperbolic arccosine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.arccosh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arcsin(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the arcsine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.arcsin, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arcsinh(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the hyperbolic arcsine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.arcsinh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arctan(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the arctangent of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.arctan, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arctanh(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the hyperbolic arctangent of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.arctanh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def cos(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the cosine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.cos, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def cosh(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the hyperbolic cosine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.cosh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def sin(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the sine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.sin, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def sinc(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the sinc function of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.sinc, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def sinh(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the hyperbolic sine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.sinh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def tan(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the tangent of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.tan, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def tanh(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the hyperbolic tangent of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.tanh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def deg2rad(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Convert angles from degrees to radians.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.deg2rad, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def rad2deg(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Convert angles from radians to degrees.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.rad2deg, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def degrees(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Convert angles from radians to degrees.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.degrees, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def radians(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Convert angles from degrees to radians.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.radians, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def angle(
    x: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Return the angle of the complex argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return _fun_accept_unitless_unary(jnp.angle, x, unit_to_scale=unit_to_scale)


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
    The unit to scale the ``x``.

  Returns
  -------
  mantissa : ndarray
    Floating values between -1 and 1.
    This is a scalar if `x` is a scalar.
  exponent : ndarray
    Integer exponents of 2.
    This is a scalar if `x` is a scalar.
  """
  return _fun_accept_unitless_unary(jnp.frexp, x, unit_to_scale=unit_to_scale)


def _fun_accept_unitless_return_keep_unit(
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
      r = func(x.to_value(unit_to_scale), *args, **kwargs)
      return jax.tree.map(lambda a: a * unit_to_scale, r)
  else:
    assert unit_to_scale is None, f'Unit should be None for the function "{func}" when "x" is not a Quantity.'
    return func(x, *args, **kwargs)


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
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _fun_accept_unitless_return_keep_unit(jnp.round_, x, unit_to_scale=unit_to_scale)


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
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _fun_accept_unitless_return_keep_unit(jnp.around, x, unit_to_scale=unit_to_scale, decimals=decimals)


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
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _fun_accept_unitless_return_keep_unit(jnp.round, x, unit_to_scale=unit_to_scale, decimals=decimals)


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
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _fun_accept_unitless_return_keep_unit(jnp.rint, x, unit_to_scale=unit_to_scale)


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
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _fun_accept_unitless_return_keep_unit(jnp.floor, x, unit_to_scale=unit_to_scale)


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
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _fun_accept_unitless_return_keep_unit(jnp.ceil, x, unit_to_scale=unit_to_scale)


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
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _fun_accept_unitless_return_keep_unit(jnp.trunc, x, unit_to_scale=unit_to_scale)


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
    The unit to scale the ``x``.

  Returns
  -------
  out : jax.Array
  """
  return _fun_accept_unitless_return_keep_unit(jnp.fix, x, unit_to_scale=unit_to_scale)


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
    The unit to scale the ``x``.

  Returns
  -------
  The fractional and integral parts of the input, both with the same dimension.
  """
  return _fun_accept_unitless_return_keep_unit(jnp.modf, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def percentile(
    a: Union[jax.Array, Quantity],
    q: jax.typing.ArrayLike,
    axis: Optional[Union[int, Tuple[int]]] = None,
    method: str = 'linear',
    keepdims: Optional[bool] = False,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the q-th percentile of the data along the specified axis.

  Returns the q-th percentile(s) of the array elements.

  Parameters
  ----------
  a : array_like, Quantity
    Input array or Quantity.
  q : array_like
    Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
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
  assert not isinstance(q, Quantity), 'Percentile should be unitless.'
  return _fun_accept_unitless_return_keep_unit(
    jnp.percentile, a, q=q, axis=axis,
    method=method, keepdims=keepdims, unit_to_scale=unit_to_scale
  )


@set_module_as('brainunit.math')
def nanpercentile(
    a: Union[jax.Array, Quantity],
    q: jax.typing.ArrayLike,
    axis: Optional[Union[int, Tuple[int]]] = None,
    method: str = 'linear',
    keepdims: Optional[bool] = False,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the q-th percentile of the data along the specified axis, while ignoring nan values.

  Returns the q-th percentile(s) of the array elements, while ignoring nan values.

  Parameters
  ----------
  a : array_like, Quantity
    Input array or Quantity.
  q : array_like, Quantity
    Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
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
  assert not isinstance(q, Quantity), 'Percentile should be unitless.'
  return _fun_accept_unitless_return_keep_unit(
    jnp.nanpercentile, a, q=q,
    axis=axis,
    method=method, keepdims=keepdims, unit_to_scale=unit_to_scale
  )


@set_module_as('brainunit.math')
def quantile(
    a: Union[jax.Array, Quantity],
    q: jax.typing.ArrayLike,
    axis: Optional[Union[int, Tuple[int]]] = None,
    method: str = 'linear',
    keepdims: Optional[bool] = False,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the q-th percentile of the data along the specified axis.

  Returns the q-th percentile(s) of the array elements.

  Parameters
  ----------
  a : array_like, Quantity
    Input array or Quantity.
  q : array_like, Quantity
    Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
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
  assert not isinstance(q, Quantity), 'Percentile should be unitless.'
  return _fun_accept_unitless_return_keep_unit(
    jnp.quantile, a, q=q,
    axis=axis,
    method=method, keepdims=keepdims, unit_to_scale=unit_to_scale
  )


@set_module_as('brainunit.math')
def nanquantile(
    a: Union[jax.Array, Quantity],
    q: jax.typing.ArrayLike,
    axis: Optional[Union[int, Tuple[int]]] = None,
    method: str = 'linear',
    keepdims: Optional[bool] = False,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
  """
  Compute the q-th percentile of the data along the specified axis, while ignoring nan values.

  Returns the q-th percentile(s) of the array elements, while ignoring nan values.

  Parameters
  ----------
  a : array_like, Quantity
    Input array or Quantity.
  q : array_like, Quantity
    Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
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
  assert not isinstance(q, Quantity), 'Percentile should be unitless.'
  return _fun_accept_unitless_return_keep_unit(
    jnp.nanquantile, a, q=q,
    axis=axis,
    method=method, keepdims=keepdims, unit_to_scale=unit_to_scale
  )


# math funcs only accept unitless (binary)
# ----------------------------------------


def _fun_accept_unitless_binary(
    func: Callable,
    x: jax.typing.ArrayLike | Quantity,
    y: jax.typing.ArrayLike | Quantity,
    *args,
    unit_to_scale: Optional[Unit] = None,
    **kwargs
):
  if isinstance(x, Quantity) and isinstance(y, Quantity):
    if unit_to_scale is None:
      assert x.is_unitless, (f'Input should be unitless for the function "{func}" '
                             f'when scaling "unit_to_scale" is not provided.')
      assert y.is_unitless, (f'Input should be unitless for the function "{func}" '
                             f'when scaling "unit_to_scale" is not provided.')
      x = x.value
      y = y.value
      return func(x, y, *args, **kwargs)
    else:
      fail_for_dimension_mismatch(
        x,
        unit_to_scale,
        error_message="Unit mismatch: {value} != {unit_to_scale}",
        value=x,
        unit_to_scale=unit_to_scale
      )
      fail_for_dimension_mismatch(
        y,
        unit_to_scale,
        error_message="Unit mismatch: {value} != {unit_to_scale}",
        value=y,
        unit_to_scale=unit_to_scale
      )
      return func(x.to_value(unit_to_scale), y.to_value(unit_to_scale), *args, **kwargs)
  else:
    assert unit_to_scale is None, f'Unit should be None for the function "{func}" when "x" and "y" are not Quantities.'
    return func(x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def hypot(
    x: Union[jax.Array, Quantity],
    y: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
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
  return _fun_accept_unitless_binary(jnp.hypot, x, y, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def arctan2(
    x: Union[jax.Array, Quantity],
    y: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
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
  return _fun_accept_unitless_binary(jnp.arctan2, x, y, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def logaddexp(
    x: Union[jax.Array, Quantity],
    y: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
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
  return _fun_accept_unitless_binary(jnp.logaddexp, x, y, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def logaddexp2(
    x: Union[jax.Array, Quantity],
    y: Union[jax.Array, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
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
  return _fun_accept_unitless_binary(jnp.logaddexp2, x, y, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def corrcoef(
    x: Union[jax.Array, Quantity],
    y: Union[jax.Array, Quantity],
    rowvar: bool = True,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
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
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  R : ndarray
    The correlation coefficient matrix of the variables.
  """
  return _fun_accept_unitless_binary(jnp.corrcoef, x, y, rowvar=rowvar, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.math')
def correlate(
    a: Union[jax.Array, Quantity],
    v: Union[jax.Array, Quantity],
    mode: str = 'valid',
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
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
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : ndarray
    Discrete cross-correlation of `a` and `v`.
  """
  return _fun_accept_unitless_binary(
    jnp.correlate, a, v,
    mode=mode, precision=precision,
    preferred_element_type=preferred_element_type,
    unit_to_scale=unit_to_scale
  )


@set_module_as('brainunit.math')
def cov(
    m: Union[jax.Array, Quantity],
    y: Optional[Union[jax.Array, Quantity]] = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[jax.typing.ArrayLike] = None,
    aweights: Optional[jax.typing.ArrayLike] = None,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
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
  unit_to_scale : Unit, optional
    The unit to scale the ``x``.

  Returns
  -------
  out : ndarray
    The covariance matrix of the variables.
  """
  return _fun_accept_unitless_binary(
    jnp.cov, m, y,
    rowvar=rowvar, bias=bias, ddof=ddof, fweights=fweights,
    aweights=aweights, unit_to_scale=unit_to_scale
  )


# Elementwise bit operations (unary)
# ----------------------------------


@set_module_as('brainunit.math')
def bitwise_not(x: Union[Quantity, jax.typing.ArrayLike]) -> jax.Array:
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
  return _fun_accept_unitless_unary(jnp.bitwise_not, x)


@set_module_as('brainunit.math')
def invert(x: Union[Quantity, jax.typing.ArrayLike]) -> jax.Array:
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
  return _fun_accept_unitless_unary(jnp.invert, x)


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
) -> jax.Array:
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
) -> jax.Array:
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
) -> jax.Array:
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
) -> jax.Array:
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
) -> jax.Array:
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
