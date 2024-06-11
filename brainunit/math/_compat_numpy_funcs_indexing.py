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
from typing import (Union, Optional)

import brainstate as bst
import jax
import jax.numpy as jnp
import numpy as np
from brainstate._utils import set_module_as

from .._base import (Quantity,
                     fail_for_dimension_mismatch,
                     is_unitless,
                     )

__all__ = [

  # indexing funcs
  'where', 'tril_indices', 'tril_indices_from', 'triu_indices',
  'triu_indices_from', 'take', 'select',
]


# indexing funcs
# --------------
@set_module_as('brainunit.math')
def where(condition: Union[bool, bst.typing.ArrayLike],
          *args: Union[Quantity, bst.typing.ArrayLike],
          **kwds) -> Union[Quantity, jax.Array]:
  condition = jnp.asarray(condition)
  if len(args) == 0:
    # nothing to do
    return jnp.where(condition, *args, **kwds)
  elif len(args) == 2:
    # check that x and y have the same dimensions
    fail_for_dimension_mismatch(
      args[0], args[1], "x and y need to have the same dimensions"
    )
    new_args = []
    for arg in args:
      if isinstance(arg, Quantity):
        new_args.append(arg.value)
    if is_unitless(args[0]):
      if len(new_args) == 2:
        return jnp.where(condition, *new_args, **kwds)
      else:
        return jnp.where(condition, *args, **kwds)
    else:
      # as both arguments have the same unit, just use the first one's
      dimensionless_args = [jnp.asarray(arg.value) if isinstance(arg, Quantity) else jnp.asarray(arg) for arg in args]
      return Quantity.with_units(
        jnp.where(condition, *dimensionless_args), args[0].unit
      )
  else:
    # illegal number of arguments
    if len(args) == 1:
      raise ValueError("where() takes 2 or 3 positional arguments but 1 was given")
    elif len(args) > 2:
      raise TypeError("where() takes 2 or 3 positional arguments but {} were given".format(len(args)))


tril_indices = jnp.tril_indices
tril_indices.__doc__ = '''
  Return the indices for the lower-triangle of an (n, m) array.

  Args:
    n: int
    m: int
    k: int, optional

  Returns:
    tuple[jax.Array]: tuple[array]
'''


@set_module_as('brainunit.math')
def tril_indices_from(arr: Union[Quantity, bst.typing.ArrayLike],
                      k: Optional[int] = 0) -> tuple[jax.Array, jax.Array]:
  '''
  Return the indices for the lower-triangle of an (n, m) array.

  Args:
    arr: array_like, Quantity
    k: int, optional

  Returns:
    tuple[jax.Array]: tuple[array]
  '''
  if isinstance(arr, Quantity):
    return jnp.tril_indices_from(arr.value, k=k)
  else:
    return jnp.tril_indices_from(arr, k=k)


triu_indices = jnp.triu_indices
triu_indices.__doc__ = '''
  Return the indices for the upper-triangle of an (n, m) array.

  Args:
    n: int
    m: int
    k: int, optional

  Returns:
    tuple[jax.Array]: tuple[array]
'''


@set_module_as('brainunit.math')
def triu_indices_from(arr: Union[Quantity, bst.typing.ArrayLike],
                      k: Optional[int] = 0) -> tuple[jax.Array, jax.Array]:
  '''
  Return the indices for the upper-triangle of an (n, m) array.

  Args:
    arr: array_like, Quantity
    k: int, optional

  Returns:
    tuple[jax.Array]: tuple[array]
  '''
  if isinstance(arr, Quantity):
    return jnp.triu_indices_from(arr.value, k=k)
  else:
    return jnp.triu_indices_from(arr, k=k)


@set_module_as('brainunit.math')
def take(a: Union[Quantity, bst.typing.ArrayLike],
         indices: Union[Quantity, bst.typing.ArrayLike],
         axis: Optional[int] = None,
         mode: Optional[str] = None) -> Union[Quantity, jax.Array]:
  if isinstance(a, Quantity):
    return a.take(indices, axis=axis, mode=mode)
  else:
    return jnp.take(a, indices, axis=axis, mode=mode)


@set_module_as('brainunit.math')
def select(condlist: list[Union[bst.typing.ArrayLike]],
           choicelist: Union[Quantity, bst.typing.ArrayLike],
           default: int = 0) -> Union[Quantity, jax.Array]:
  from builtins import all as origin_all
  from builtins import any as origin_any
  if origin_all(isinstance(choice, Quantity) for choice in choicelist):
    if origin_any(choice.unit != choicelist[0].unit for choice in choicelist):
      raise ValueError("All choices must have the same unit")
    else:
      return Quantity(jnp.select(condlist, [choice.value for choice in choicelist], default=default),
                      unit=choicelist[0].unit)
  elif origin_all(isinstance(choice, (jax.Array, np.ndarray)) for choice in choicelist):
    return jnp.select(condlist, choicelist, default=default)
  else:
    raise ValueError(f"Unsupported types : {type(condlist)} and {type(choicelist)} for select")
