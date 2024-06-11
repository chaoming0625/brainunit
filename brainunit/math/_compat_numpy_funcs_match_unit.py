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
import numpy as np
from jax import Array

from .._base import (Quantity,
                     fail_for_dimension_mismatch,
                     )

__all__ = [
  # math funcs match unit (binary)
  'add', 'subtract', 'nextafter',
]


# math funcs match unit (binary)
# ------------------------------

def wrap_math_funcs_match_unit_binary(func):
  @wraps(func)
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


@wrap_math_funcs_match_unit_binary
def add(x: Union[Quantity, Array], y: Union[Quantity, Array]) -> Union[Quantity, Array]:
  return jnp.add(x, y)


@wrap_math_funcs_match_unit_binary
def subtract(x: Union[Quantity, Array], y: Union[Quantity, Array]) -> Union[Quantity, Array]:
  return jnp.subtract(x, y)


@wrap_math_funcs_match_unit_binary
def nextafter(x: Union[Quantity, Array], y: Union[Quantity, Array]) -> Union[Quantity, Array]:
  return jnp.nextafter(x, y)


# docs for the functions above
add.__doc__ = '''
  Add arguments element-wise.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` and `y` are Quantities that have the same unit, else an array.
'''

subtract.__doc__ = '''
  Subtract arguments element-wise.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` and `y` are Quantities that have the same unit, else an array.
'''

nextafter.__doc__ = '''
  Return the next floating-point value after `x1` towards `x2`.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
'''
