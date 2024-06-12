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
import numpy as np
from jax import Array

from brainunit._misc import set_module_as
from .._base import (Quantity,
                     fail_for_dimension_mismatch, )

__all__ = [
  # math funcs match unit (binary)
  'add', 'subtract', 'nextafter',
]


# math funcs match unit (binary)
# ------------------------------


def funcs_match_unit_binary(func, x, y, *args, **kwargs):
  if isinstance(x, Quantity) and isinstance(y, Quantity):
    fail_for_dimension_mismatch(x, y)
    return Quantity(func(x.value, y.value, *args, **kwargs), dim=x.dim)
  elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
    return func(x, y, *args, **kwargs)
  elif isinstance(x, Quantity):
    if x.is_unitless:
      return Quantity(func(x.value, y, *args, **kwargs), dim=x.dim)
    else:
      raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {func.__name__}')
  elif isinstance(y, Quantity):
    if y.is_unitless:
      return Quantity(func(x, y.value, *args, **kwargs), dim=y.dim)
    else:
      raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {func.__name__}')
  else:
    raise ValueError(f'Unsupported types : {type(x)} abd {type(y)} for {func.__name__}')


@set_module_as('brainunit.math')
def add(
    x: Union[Quantity, Array],
    y: Union[Quantity, Array]
) -> Union[Quantity, Array]:
  """
  Add arguments element-wise.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` and `y` are Quantities that have the same unit, else an array.
  """
  return funcs_match_unit_binary(jnp.add, x, y)


@set_module_as('brainunit.math')
def subtract(
    x: Union[Quantity, Array],
    y: Union[Quantity, Array]
) -> Union[Quantity, Array]:
  """
  Subtract arguments element-wise.

  Args:
    x: array_like, Quantity
    y: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x` and `y` are Quantities that have the same unit, else an array.
  """
  return funcs_match_unit_binary(jnp.subtract, x, y)


@set_module_as('brainunit.math')
def nextafter(
    x: Union[Quantity, Array],
    y: Union[Quantity, Array]
) -> Union[Quantity, Array]:
  """
  Return the next floating-point value after `x1` towards `x2`.

  Args:
    x1: array_like, Quantity
    x2: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  """
  return funcs_match_unit_binary(jnp.nextafter, x, y)
