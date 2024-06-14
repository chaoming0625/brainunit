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

from typing import (Union, Optional, Tuple)

import jax
import jax.numpy as jnp
from jax import Array

from .._base import (Quantity, fail_for_dimension_mismatch, )
from .._misc import set_module_as

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
def exp(x: Union[Quantity, jax.typing.ArrayLike]) -> Array:
  """
  Calculate the exponential of all elements in the input quantity or array.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.exp, x)


@set_module_as('brainunit.math')
def exp2(x: Union[Quantity, jax.typing.ArrayLike]) -> Array:
  """
  Calculate 2**p for all p in the input quantity or array.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.exp2, x)


@set_module_as('brainunit.math')
def expm1(x: Union[Array, Quantity]) -> Array:
  """
  Calculate the exponential of the input elements minus 1.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.expm1, x)


@set_module_as('brainunit.math')
def log(x: Union[Array, Quantity]) -> Array:
  """
  Natural logarithm, element-wise.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.log, x)


@set_module_as('brainunit.math')
def log10(x: Union[Array, Quantity]) -> Array:
  """
  Base-10 logarithm of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.log10, x)


@set_module_as('brainunit.math')
def log1p(x: Union[Array, Quantity]) -> Array:
  """
  Natural logarithm of 1 + the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.log1p, x)


@set_module_as('brainunit.math')
def log2(x: Union[Array, Quantity]) -> Array:
  """
  Base-2 logarithm of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.log2, x)


@set_module_as('brainunit.math')
def arccos(x: Union[Array, Quantity]) -> Array:
  """
  Compute the arccosine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.arccos, x)


@set_module_as('brainunit.math')
def arccosh(x: Union[Array, Quantity]) -> Array:
  """
  Compute the hyperbolic arccosine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.arccosh, x)


@set_module_as('brainunit.math')
def arcsin(x: Union[Array, Quantity]) -> Array:
  """
  Compute the arcsine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.arcsin, x)


@set_module_as('brainunit.math')
def arcsinh(x: Union[Array, Quantity]) -> Array:
  """
  Compute the hyperbolic arcsine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.arcsinh, x)


@set_module_as('brainunit.math')
def arctan(x: Union[Array, Quantity]) -> Array:
  """
  Compute the arctangent of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.arctan, x)


@set_module_as('brainunit.math')
def arctanh(x: Union[Array, Quantity]) -> Array:
  """
  Compute the hyperbolic arctangent of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.arctanh, x)


@set_module_as('brainunit.math')
def cos(x: Union[Array, Quantity]) -> Array:
  """
  Compute the cosine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.cos, x)


@set_module_as('brainunit.math')
def cosh(x: Union[Array, Quantity]) -> Array:
  """
  Compute the hyperbolic cosine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.cosh, x)


@set_module_as('brainunit.math')
def sin(x: Union[Array, Quantity]) -> Array:
  """
  Compute the sine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.sin, x)


@set_module_as('brainunit.math')
def sinc(x: Union[Array, Quantity]) -> Array:
  """
  Compute the sinc function of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.sinc, x)


@set_module_as('brainunit.math')
def sinh(x: Union[Array, Quantity]) -> Array:
  """
  Compute the hyperbolic sine of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.sinh, x)


@set_module_as('brainunit.math')
def tan(x: Union[Array, Quantity]) -> Array:
  """
  Compute the tangent of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.tan, x)


@set_module_as('brainunit.math')
def tanh(x: Union[Array, Quantity]) -> Array:
  """
  Compute the hyperbolic tangent of the input elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.tanh, x)


@set_module_as('brainunit.math')
def deg2rad(x: Union[Array, Quantity]) -> Array:
  """
  Convert angles from degrees to radians.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.deg2rad, x)


@set_module_as('brainunit.math')
def rad2deg(x: Union[Array, Quantity]) -> Array:
  """
  Convert angles from radians to degrees.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.rad2deg, x)


@set_module_as('brainunit.math')
def degrees(x: Union[Array, Quantity]) -> Array:
  """
  Convert angles from radians to degrees.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.degrees, x)


@set_module_as('brainunit.math')
def radians(x: Union[Array, Quantity]) -> Array:
  """
  Convert angles from degrees to radians.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.radians, x)


@set_module_as('brainunit.math')
def angle(x: Union[Array, Quantity]) -> Array:
  """
  Return the angle of the complex argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or Quantity.

  Returns
  -------
  out : jax.Array
    Output array.
  """
  return funcs_only_accept_unitless_unary(jnp.angle, x)


# math funcs only accept unitless (binary)
# ----------------------------------------


def funcs_only_accept_unitless_binary(func, x, y, *args, **kwargs):
  x_value = x.value if isinstance(x, Quantity) else x
  y_value = y.value if isinstance(y, Quantity) else y
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
  return funcs_only_accept_unitless_binary(jnp.hypot, x, y)


@set_module_as('brainunit.math')
def arctan2(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
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
  return funcs_only_accept_unitless_binary(jnp.arctan2, x, y)


@set_module_as('brainunit.math')
def logaddexp(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
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
  return funcs_only_accept_unitless_binary(jnp.logaddexp, x, y)


@set_module_as('brainunit.math')
def logaddexp2(x: Union[Array, Quantity], y: Union[Array, Quantity]) -> Array:
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
  return funcs_only_accept_unitless_binary(jnp.logaddexp2, x, y)


@set_module_as('brainunit.math')
def percentile(
    a: Union[Array, Quantity],
    q: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
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
  return funcs_only_accept_unitless_binary(jnp.percentile, a, q, axis=axis, out=out, overwrite_input=overwrite_input,
                                           method=method, keepdims=keepdims)


@set_module_as('brainunit.math')
def nanpercentile(
    a: Union[Array, Quantity],
    q: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
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
  return funcs_only_accept_unitless_binary(jnp.nanpercentile, a, q, axis=axis, out=out, overwrite_input=overwrite_input,
                                           method=method, keepdims=keepdims)


@set_module_as('brainunit.math')
def quantile(
    a: Union[Array, Quantity],
    q: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
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
  return funcs_only_accept_unitless_binary(jnp.quantile, a, q, axis=axis, out=out, overwrite_input=overwrite_input,
                                           method=method, keepdims=keepdims)


@set_module_as('brainunit.math')
def nanquantile(
    a: Union[Array, Quantity],
    q: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
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
  return funcs_only_accept_unitless_binary(jnp.nanquantile, a, q, axis=axis, out=out, overwrite_input=overwrite_input,
                                           method=method, keepdims=keepdims)
