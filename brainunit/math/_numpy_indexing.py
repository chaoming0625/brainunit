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

import functools
from typing import (Union, Optional)

import jax
import jax.numpy as jnp

from ._numpy_array_manipulation import _fun_keep_unit_sequence
from ._numpy_keep_unit import _fun_keep_unit_binary
from .._base import Quantity
from .._misc import set_module_as

__all__ = [
  # indexing funcs
  'where', 'tril_indices', 'tril_indices_from', 'triu_indices',
  'triu_indices_from', 'take', 'select',
]


# indexing funcs
# --------------
@set_module_as('brainunit.math')
def where(condition, x=None, y=None, /, *, size=None, fill_value=None):
  """
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
  assert not isinstance(condition, Quantity), "condition should not be a Quantity."
  if x is None and y is None:
    return jnp.where(condition, size=size, fill_value=fill_value)
  return _fun_keep_unit_binary(functools.partial(jnp.where, condition, size=size, fill_value=fill_value), x, y)


tril_indices = jnp.tril_indices


@set_module_as('brainunit.math')
def tril_indices_from(
    arr: Union[Quantity, jax.typing.ArrayLike],
    k: Optional[int] = 0
) -> tuple[jax.Array, jax.Array]:
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


@set_module_as('brainunit.math')
def triu_indices_from(
    arr: Union[Quantity, jax.typing.ArrayLike],
    k: Optional[int] = 0
) -> tuple[jax.Array, jax.Array]:
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
    mode: Optional[str] = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
) -> Union[Quantity, jax.Array]:
  """
  Take elements from an array along an axis.

  When axis is not None, this function does the same thing as "fancy"
  indexing (indexing arrays using arrays); however, it can be easier to use
  if you need elements along a given axis. A call such as
  ``np.take(arr, indices, axis=3)`` is equivalent to
  ``arr[:,:,:,indices,...]``.

  Explained without fancy indexing, this is equivalent to the following use
  of `ndindex`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of
  indices::

    Ni, Nk = a.shape[:axis], a.shape[axis+1:]
    Nj = indices.shape
    for ii in ndindex(Ni):
        for jj in ndindex(Nj):
            for kk in ndindex(Nk):
                out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

  Parameters
  ----------
  a : array_like (Ni..., M, Nk...)
    The source array.
  indices : array_like (Nj...)
    The indices of the values to extract.

    Also allow scalars for indices.
  axis : int, optional
    The axis over which to select values. By default, the flattened
    input array is used.
  mode : string, default="fill"
    Out-of-bounds indexing mode. The default mode="fill" returns invalid values
    (e.g. NaN) for out-of bounds indices (see also ``fill_value`` below).
    For more discussion of mode options, see :attr:`jax.numpy.ndarray.at`.
  fill_value : optional
    The fill value to return for out-of-bounds slices when mode is 'fill'. Ignored
    otherwise. Defaults to NaN for inexact types, the largest negative value for
    signed types, the largest positive value for unsigned types, and True for booleans.
  unique_indices : bool, default=False
    If True, the implementation will assume that the indices are unique,
    which can result in more efficient execution on some backends.
  indices_are_sorted : bool, default=False
    If True, the implementation will assume that the indices are sorted in
    ascending order, which can lead to more efficient execution on some backends.

  Returns
  -------
  out : ndarray (Ni..., Nj..., Nk...)
    The returned array has the same type as `a`.
  """
  if isinstance(a, Quantity):
    return a.take(indices, axis=axis, mode=mode, unique_indices=unique_indices,
                  indices_are_sorted=indices_are_sorted, fill_value=fill_value)
  else:
    return jnp.take(a, indices, axis=axis, mode=mode, unique_indices=unique_indices,
                    indices_are_sorted=indices_are_sorted, fill_value=fill_value)


@set_module_as('brainunit.math')
def select(
    condlist: list[Union[jax.typing.ArrayLike]],
    choicelist: Union[Quantity, jax.typing.ArrayLike],
    default: int = 0
) -> Union[Quantity, jax.Array]:
  """
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
  """
  for cond in condlist:
    assert not isinstance(cond, Quantity), "condlist should not contain Quantity."
  return _fun_keep_unit_sequence(functools.partial(jnp.select, condlist), choicelist, default)
