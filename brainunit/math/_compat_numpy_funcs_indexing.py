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

import jax
import jax.numpy as jnp
import numpy as np

from brainunit._misc import set_module_as
from .._base import (Quantity,
                     fail_for_dimension_mismatch,
                     is_unitless, )

__all__ = [

  # indexing funcs
  'where', 'tril_indices', 'tril_indices_from', 'triu_indices',
  'triu_indices_from', 'take', 'select',
]


# indexing funcs
# --------------
@set_module_as('brainunit.math')
def where(
    condition: Union[bool, jax.typing.ArrayLike],
    *args: Union[Quantity, jax.typing.ArrayLike],
    **kwds
) -> Union[Quantity, jax.Array]:
  """
  where(condition, [x, y], /)

  Return elements chosen from `x` or `y` depending on `condition`.

  .. note::
    When only `condition` is provided, this function is a shorthand for
    ``np.asarray(condition).nonzero()``. Using `nonzero` directly should be
    preferred, as it behaves correctly for subclasses. The rest of this
    documentation covers only the case where all three arguments are
    provided.

  Parameters
  ----------
  condition : array_like, bool,
    Where True, yield `x`, otherwise yield `y`.
  x, y : array_like, Quantity
    Values from which to choose. `x`, `y` and `condition` need to be
    broadcastable to some shape.

  Returns
  -------
  out : ndarray
    An array with elements from `x` where `condition` is True, and elements
    from `y` elsewhere.

  See Also
  --------
  choose
  nonzero : The function that is called when x and y are omitted
  """
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
        jnp.where(condition, *dimensionless_args), args[0].dim
      )
  else:
    # illegal number of arguments
    if len(args) == 1:
      raise ValueError("where() takes 2 or 3 positional arguments but 1 was given")
    elif len(args) > 2:
      raise TypeError("where() takes 2 or 3 positional arguments but {} were given".format(len(args)))


tril_indices = jnp.tril_indices
tril_indices.__doc__ = """
  Return the indices for the lower-triangle of an (n, m) array.

  Parameters
  ----------
  n : int
    The row dimension of the arrays for which the returned indices will be valid.
  m : int
    The column dimension of the arrays for which the returned indices will be valid.
  k : int, optional
    Diagonal above which to zero elements. k = 0 is the main diagonal, k < 0 subdiagonal and k > 0 superdiagonal.
    
  Returns
  -------
  out : tuple[jax.Array]
    tuple of arrays  
"""


@set_module_as('brainunit.math')
def tril_indices_from(arr: Union[Quantity, jax.typing.ArrayLike],
                      k: Optional[int] = 0) -> tuple[jax.Array, jax.Array]:
  """
  Return the indices for the lower-triangle of an (n, m) array.

  Parameters
  ----------
  arr : array_like, Quantity
    The arrays for which the returned indices will be valid.
  k : int, optional
    Diagonal above which to zero elements. k = 0 is the main diagonal, k < 0 subdiagonal and k > 0 superdiagonal.

  Returns
  -------
  out : tuple[jax.Array]
    tuple of arrays
  """
  if isinstance(arr, Quantity):
    return jnp.tril_indices_from(arr.value, k=k)
  else:
    return jnp.tril_indices_from(arr, k=k)


triu_indices = jnp.triu_indices
triu_indices.__doc__ = """
  Return the indices for the upper-triangle of an (n, m) array.

  Parameters
  ----------
  n : int
    The row dimension of the arrays for which the returned indices will be valid.
  m : int
    The column dimension of the arrays for which the returned indices will be valid.
  k : int, optional
    Diagonal above which to zero elements. k = 0 is the main diagonal, k < 0 subdiagonal and k > 0 superdiagonal.
    
  Returns
  -------
  out : tuple[jax.Array]
    tuple of arrays
"""


@set_module_as('brainunit.math')
def triu_indices_from(arr: Union[Quantity, jax.typing.ArrayLike],
                      k: Optional[int] = 0) -> tuple[jax.Array, jax.Array]:
  """
  Return the indices for the upper-triangle of an (n, m) array.

  Parameters
  ----------
  arr : array_like, Quantity
    The arrays for which the returned indices will be valid.
  k : int, optional
    Diagonal above which to zero elements. k = 0 is the main diagonal, k < 0 subdiagonal and k > 0 superdiagonal.

  Returns
  -------
  out : tuple[jax.Array]
    tuple of arrays
  """
  if isinstance(arr, Quantity):
    return jnp.triu_indices_from(arr.value, k=k)
  else:
    return jnp.triu_indices_from(arr, k=k)


@set_module_as('brainunit.math')
def take(
    a: Union[Quantity, jax.typing.ArrayLike],
    indices: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    out: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
    mode: Optional[str] = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
) -> Union[Quantity, jax.Array]:
  '''

  '''
  if isinstance(a, Quantity):
    return a.take(indices, axis=axis, out=out, mode=mode, unique_indices=unique_indices,
                  indices_are_sorted=indices_are_sorted, fill_value=fill_value)
  else:
    return jnp.take(a, indices, axis=axis, out=out, mode=mode, unique_indices=unique_indices,
                    indices_are_sorted=indices_are_sorted, fill_value=fill_value)


@set_module_as('brainunit.math')
def select(
    condlist: list[Union[jax.typing.ArrayLike]],
    choicelist: Union[Quantity, jax.typing.ArrayLike],
    default: int = 0
) -> Union[Quantity, jax.Array]:
  '''
  Return an array drawn from elements in choicelist, depending on conditions.

  Parameters
  ----------
  condlist : list of bool ndarrays
    The list of conditions which determine from which array in `choicelist`
    the output elements are taken. When multiple conditions are satisfied,
    the first one encountered in `condlist` is used.
  choicelist : list of ndarrays or Quantity
    The list of arrays from which the output elements are taken. It has
    to be of the same length as `condlist`.
  default : scalar, optional
    The element inserted in `output` when all conditions evaluate to False.

  Returns
  -------
  output : ndarray, Quantity
    The output at position m is the m-th element of the array in
    `choicelist` where the m-th element of the corresponding array in
    `condlist` is True.
  '''
  if all(isinstance(choice, Quantity) for choice in choicelist):
    if any(choice.dim != choicelist[0].dim for choice in choicelist):
      raise ValueError("All choices must have the same unit")
    else:
      return Quantity(jnp.select(condlist, [choice.value for choice in choicelist], default=default),
                      dim=choicelist[0].dim)
  elif all(isinstance(choice, (jax.Array, np.ndarray)) for choice in choicelist):
    return jnp.select(condlist, choicelist, default=default)
  else:
    raise ValueError(f"Unsupported types : {type(condlist)} and {type(choicelist)} for select")
