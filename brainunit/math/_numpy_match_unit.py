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

from typing import (Union)

import jax.numpy as jnp
from jax import Array

from .._base import (Quantity,
                     fail_for_dimension_mismatch, )
from .._misc import set_module_as

__all__ = [
  # math funcs match unit (binary)
  'add', 'subtract', 'nextafter',
]


# math funcs match unit (binary)
# ------------------------------


def _fun_match_unit_binary(func, x, y, *args, **kwargs):
  if isinstance(x, Quantity) and isinstance(y, Quantity):
    fail_for_dimension_mismatch(x, y, func.__name__)
    return Quantity(func(x.value, y.value, *args, **kwargs), dim=x.dim)
  elif isinstance(x, Quantity):
    assert x.is_unitless, f'Expected unitless Quantity when y is not a Quantity, got {x}'
    return func(x.value, y, *args, **kwargs)
  elif isinstance(y, Quantity):
    assert y.is_unitless, f'Expected unitless Quantity when x is not a Quantity, got {y}'
    return func(x, y.value, *args, **kwargs)
  else:
    return func(x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def add(
    x: Union[Quantity, Array],
    y: Union[Quantity, Array],
    *args,
    **kwargs
) -> Union[Quantity, Array]:
  """
  Add arguments element-wise.

  Parameters
  ----------
  x, y : array_like, Quantity
    The arrays to be added.
    If ``x.shape != y.shape``, they must be broadcastable to a common
    shape (which becomes the shape of the output).
  where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
  **kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

  Returns
  -------
  add : ndarray or scalar
    The sum of `x` and `y`, element-wise.
    This is a scalar if both `x` and `y` are scalars.
  """
  return _fun_match_unit_binary(jnp.add, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def subtract(
    x: Union[Quantity, Array],
    y: Union[Quantity, Array],
    *args,
    **kwargs
) -> Union[Quantity, Array]:
  """
  subtract(x1, x2, /, out=None, *, where=True, casting='same_kind',
  order='K', dtype=None, subok=True[, signature, extobj])

  Subtract arguments, element-wise.

  Parameters
  ----------
  x, y : array_like
    The arrays to be subtracted from each other.
    If ``x.shape != y.shape``, they must be broadcastable to a common
    shape (which becomes the shape of the output).
  where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
  **kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

  Returns
  -------
  subtract : ndarray
    The difference of `x` and `y`, element-wise.
    This is a scalar if both `x` and `y` are scalars.
  """
  return _fun_match_unit_binary(jnp.subtract, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def nextafter(
    x: Union[Quantity, Array],
    y: Union[Quantity, Array],
    *args,
    **kwargs
) -> Union[Quantity, Array]:
  """
  nextafter(x, y, /, out=None, *, where=True, casting='same_kind',
  order='K', dtype=None, subok=True[, signature, extobj])

  Return the next floating-point value after x1 towards x2, element-wise.

  Parameters
  ----------
  x : array_like, Quantity
    Values to find the next representable value of.
  y : array_like, Quantity
    The direction where to look for the next representable value of `x`.
    If ``x.shape != y.shape``, they must be broadcastable to a common
    shape (which becomes the shape of the output).
  where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
  **kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

  Returns
  -------
  out : ndarray or scalar
    The next representable values of `x` in the direction of `y`.
    This is a scalar if both `x` and `y` are scalars.
  """
  return _fun_match_unit_binary(jnp.nextafter, x, y, *args, **kwargs)
