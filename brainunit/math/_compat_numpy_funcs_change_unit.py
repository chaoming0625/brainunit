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
from functools import wraps
from typing import (Callable, Union, Optional)

import jax
import jax.numpy as jnp
import numpy as np
from brainstate._utils import set_module_as

from ._compat_numpy_get_attribute import isscalar
from .._base import (DIMENSIONLESS,
                     Quantity,
                     )
from .._base import _return_check_unitless

__all__ = [

  # math funcs change unit (unary)
  'reciprocal', 'prod', 'product', 'nancumprod', 'nanprod', 'cumprod',
  'cumproduct', 'var', 'nanvar', 'cbrt', 'square', 'frexp', 'sqrt',

  # math funcs change unit (binary)
  'multiply', 'divide', 'power', 'cross', 'ldexp',
  'true_divide', 'floor_divide', 'float_power',
  'divmod', 'remainder', 'convolve',
]


# math funcs change unit (unary)
# ------------------------------

def wrap_math_funcs_change_unit_unary(change_unit_func: Callable) -> Callable:
  def decorator(func: Callable) -> Callable:
    @wraps(func)
    def f(x, *args, **kwargs):
      if isinstance(x, Quantity):
        return _return_check_unitless(Quantity(func(x.value, *args, **kwargs), dim=change_unit_func(x.dim)))
      elif isinstance(x, (jnp.ndarray, np.ndarray)):
        return func(x, *args, **kwargs)
      else:
        raise ValueError(f'Unsupported type: {type(x)} for {func.__name__}')

    f.__module__ = 'brainunit.math'
    return f

  return decorator


@wrap_math_funcs_change_unit_unary(lambda x: x ** -1)
def reciprocal(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.reciprocal(x)


@wrap_math_funcs_change_unit_unary(lambda x: x ** 2)
def var(x: Union[Quantity, jax.typing.ArrayLike],
        axis: Optional[Union[int, Sequence[int]]] = None,
        ddof: int = 0,
        keepdims: bool = False) -> Union[Quantity, jax.Array]:
  return jnp.var(x, axis=axis, ddof=ddof, keepdims=keepdims)


@wrap_math_funcs_change_unit_unary(lambda x: x ** 2)
def nanvar(x: Union[Quantity, jax.typing.ArrayLike],
           axis: Optional[Union[int, Sequence[int]]] = None,
           ddof: int = 0,
           keepdims: bool = False) -> Union[Quantity, jax.Array]:
  return jnp.nanvar(x, axis=axis, ddof=ddof, keepdims=keepdims)


@wrap_math_funcs_change_unit_unary(lambda x: x * 2 ** -1)
def frexp(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.frexp(x)


@wrap_math_funcs_change_unit_unary(lambda x: x ** 0.5)
def sqrt(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.sqrt(x)


@wrap_math_funcs_change_unit_unary(lambda x: x ** (1 / 3))
def cbrt(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.cbrt(x)


@wrap_math_funcs_change_unit_unary(lambda x: x ** 2)
def square(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  return jnp.square(x)


# docs for the functions above

reciprocal.__doc__ = '''
  Return the reciprocal of the argument.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
'''

var.__doc__ = '''
  Compute the variance along the specified axis.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the square of the unit of `x`, else an array.
'''

nanvar.__doc__ = '''
  Compute the variance along the specified axis, ignoring NaNs.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the square of the unit of `x`, else an array.
'''

frexp.__doc__ = '''
  Decompose a floating-point number into its mantissa and exponent.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Tuple of Quantity if the final unit is the product of the unit of `x` and 2 raised to the power of the exponent, else a tuple of arrays.
'''

sqrt.__doc__ = '''
  Compute the square root of each element.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the square root of the unit of `x`, else an array.
'''

cbrt.__doc__ = '''
  Compute the cube root of each element.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the cube root of the unit of `x`, else an array.
'''

square.__doc__ = '''
  Compute the square of each element.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the square of the unit of `x`, else an array.
'''


@set_module_as('brainunit.math')
def prod(x: Union[Quantity, jax.typing.ArrayLike],
         axis: Optional[int] = None,
         dtype: Optional[jax.typing.DTypeLike] = None,
         out: None = None,
         keepdims: Optional[bool] = False,
         initial: Union[Quantity, jax.typing.ArrayLike] = None,
         where: Union[Quantity, jax.typing.ArrayLike] = None,
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
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  if isinstance(x, Quantity):
    return x.prod(axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where,
                  promote_integers=promote_integers)
  else:
    return jnp.prod(x, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where,
                    promote_integers=promote_integers)


@set_module_as('brainunit.math')
def nanprod(x: Union[Quantity, jax.typing.ArrayLike],
            axis: Optional[int] = None,
            dtype: Optional[jax.typing.DTypeLike] = None,
            out: None = None,
            keepdims: bool = False,
            initial: Union[Quantity, jax.typing.ArrayLike] = None,
            where: Union[Quantity, jax.typing.ArrayLike] = None):
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
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  if isinstance(x, Quantity):
    return x.nanprod(axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)
  else:
    return jnp.nanprod(x, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)


product = prod


@set_module_as('brainunit.math')
def cumprod(x: Union[Quantity, jax.typing.ArrayLike],
            axis: Optional[int] = None,
            dtype: Optional[jax.typing.DTypeLike] = None,
            out: None = None) -> Union[Quantity, jax.typing.ArrayLike]:
  '''
  Return the cumulative product of elements along a given axis.

  Args:
    x: array_like, Quantity
    axis: int, optional
    dtype: dtype, optional
    out: array, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  if isinstance(x, Quantity):
    return x.cumprod(axis=axis, dtype=dtype, out=out)
  else:
    return jnp.cumprod(x, axis=axis, dtype=dtype, out=out)


@set_module_as('brainunit.math')
def nancumprod(x: Union[Quantity, jax.typing.ArrayLike],
               axis: Optional[int] = None,
               dtype: Optional[jax.typing.DTypeLike] = None,
               out: None = None) -> Union[Quantity, jax.typing.ArrayLike]:
  '''
  Return the cumulative product of elements along a given axis treating Not a Numbers (NaNs) as one.

  Args:
    x: array_like, Quantity
    axis: int, optional
    dtype: dtype, optional
    out: array, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` is a Quantity, else an array.
  '''
  if isinstance(x, Quantity):
    return x.nancumprod(axis=axis, dtype=dtype, out=out)
  else:
    return jnp.nancumprod(x, axis=axis, dtype=dtype, out=out)


cumproduct = cumprod


# math funcs change unit (binary)
# -------------------------------

def wrap_math_funcs_change_unit_binary(change_unit_func):
  def decorator(func: Callable) -> Callable:
    @wraps(func)
    def f(x, y, *args, **kwargs):
      if isinstance(x, Quantity) and isinstance(y, Quantity):
        return _return_check_unitless(
          Quantity(func(x.value, y.value, *args, **kwargs), dim=change_unit_func(x.dim, y.dim))
        )
      elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
        return func(x, y, *args, **kwargs)
      elif isinstance(x, Quantity):
        return _return_check_unitless(
          Quantity(func(x.value, y, *args, **kwargs), dim=change_unit_func(x.dim, DIMENSIONLESS)))
      elif isinstance(y, Quantity):
        return _return_check_unitless(
          Quantity(func(x, y.value, *args, **kwargs), dim=change_unit_func(DIMENSIONLESS, y.dim)))
      else:
        raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {func.__name__}')

    f.__module__ = 'brainunit.math'
    return f
  return decorator


@wrap_math_funcs_change_unit_binary(lambda x, y: x * y)
def multiply(x: Union[Quantity, jax.typing.ArrayLike], y: Union[Quantity, jax.typing.ArrayLike]):
  return jnp.multiply(x, y)


@wrap_math_funcs_change_unit_binary(lambda x, y: x / y)
def divide(x: Union[Quantity, jax.typing.ArrayLike], y: Union[Quantity, jax.typing.ArrayLike]):
  return jnp.divide(x, y)


@wrap_math_funcs_change_unit_binary(lambda x, y: x * y)
def cross(x: Union[Quantity, jax.typing.ArrayLike], y: Union[Quantity, jax.typing.ArrayLike]):
  return jnp.cross(x, y)


@wrap_math_funcs_change_unit_binary(lambda x, y: x * 2 ** y)
def ldexp(x: Union[Quantity, jax.typing.ArrayLike], y: Union[Quantity, jax.typing.ArrayLike]):
  return jnp.ldexp(x, y)


@wrap_math_funcs_change_unit_binary(lambda x, y: x / y)
def true_divide(x: Union[Quantity, jax.typing.ArrayLike], y: Union[Quantity, jax.typing.ArrayLike]):
  return jnp.true_divide(x, y)


@wrap_math_funcs_change_unit_binary(lambda x, y: x / y)
def divmod(x: Union[Quantity, jax.typing.ArrayLike], y: Union[Quantity, jax.typing.ArrayLike]):
  return jnp.divmod(x, y)


@wrap_math_funcs_change_unit_binary(lambda x, y: x * y)
def convolve(x: Union[Quantity, jax.typing.ArrayLike], y: Union[Quantity, jax.typing.ArrayLike]):
  return jnp.convolve(x, y)


# docs for the functions above
multiply.__doc__ = '''
  Multiply arguments element-wise.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `x` and the unit of `y`, else an array.
'''

divide.__doc__ = '''
  Divide arguments element-wise.

  Args:
    x: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the quotient of the unit of `x` and the unit of `y`, else an array.
'''

cross.__doc__ = '''
  Return the cross product of two (arrays of) vectors.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `x` and the unit of `y`, else an array.
'''

ldexp.__doc__ = '''
  Return x1 * 2**x2, element-wise.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `x` and 2 raised to the power of the unit of `y`, else an array. 
'''

true_divide.__doc__ = '''
  Returns a true division of the inputs, element-wise.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the quotient of the unit of `x` and the unit of `y`, else an array.
'''

divmod.__doc__ = '''
  Return element-wise quotient and remainder simultaneously.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the quotient of the unit of `x` and the unit of `y`, else an array.
'''

convolve.__doc__ = '''
  Returns the discrete, linear convolution of two one-dimensional sequences.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `x` and the unit of `y`, else an array.
'''


@set_module_as('brainunit.math')
def power(x: Union[Quantity, jax.typing.ArrayLike],
          y: Union[Quantity, jax.typing.ArrayLike], ) -> Union[Quantity, jax.Array]:
  '''
  First array elements raised to powers from second array, element-wise.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `x` and the unit of `y`, else an array.
  '''
  if isinstance(x, Quantity) and isinstance(y, Quantity):
    return _return_check_unitless(Quantity(jnp.power(x.value, y.value), dim=x.dim ** y.dim))
  elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
    return jnp.power(x, y)
  elif isinstance(x, Quantity):
    return _return_check_unitless(Quantity(jnp.power(x.value, y), dim=x.dim ** y))
  elif isinstance(y, Quantity):
    return _return_check_unitless(Quantity(jnp.power(x, y.value), dim=x ** y.dim))
  else:
    raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {jnp.power.__name__}')


@set_module_as('brainunit.math')
def floor_divide(x: Union[Quantity, jax.typing.ArrayLike],
                 y: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return the largest integer smaller or equal to the division of the inputs.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the quotient of the unit of `x` and the unit of `y`, else an array.
  '''
  if isinstance(x, Quantity) and isinstance(y, Quantity):
    return _return_check_unitless(Quantity(jnp.floor_divide(x.value, y.value), dim=x.dim / y.dim))
  elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
    return jnp.floor_divide(x, y)
  elif isinstance(x, Quantity):
    return _return_check_unitless(Quantity(jnp.floor_divide(x.value, y), dim=x.dim / y))
  elif isinstance(y, Quantity):
    return _return_check_unitless(Quantity(jnp.floor_divide(x, y.value), dim=x / y.dim))
  else:
    raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {jnp.floor_divide.__name__}')


@set_module_as('brainunit.math')
def float_power(x: Union[Quantity, jax.typing.ArrayLike],
                y: jax.typing.ArrayLike) -> Union[Quantity, jax.Array]:
  '''
  First array elements raised to powers from second array, element-wise.

  Args:
    x: array_like, Quantity
    y: array_like

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the product of the unit of `x` and the unit of `y`, else an array.
  '''
  if isinstance(y, Quantity):
    assert isscalar(y), f'{jnp.float_power.__name__} only supports scalar exponent'
  if isinstance(x, Quantity):
    return _return_check_unitless(Quantity(jnp.float_power(x.value, y), dim=x.dim ** y))
  elif isinstance(x, (jax.Array, np.ndarray)):
    return jnp.float_power(x, y)
  else:
    raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {jnp.float_power.__name__}')


@set_module_as('brainunit.math')
def remainder(x: Union[Quantity, jax.typing.ArrayLike],
              y: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  '''
  Return element-wise remainder of division.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if the final unit is the remainder of the unit of `x` and the unit of `y`, else an array.
  '''
  if isinstance(x, Quantity) and isinstance(y, Quantity):
    return _return_check_unitless(Quantity(jnp.remainder(x.value, y.value), dim=x.dim / y.dim))
  elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
    return jnp.remainder(x, y)
  elif isinstance(x, Quantity):
    return _return_check_unitless(Quantity(jnp.remainder(x.value, y), dim=x.dim % y))
  elif isinstance(y, Quantity):
    return _return_check_unitless(Quantity(jnp.remainder(x, y.value), dim=x % y.dim))
  else:
    raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {jnp.remainder.__name__}')
