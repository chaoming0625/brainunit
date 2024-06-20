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

from typing import (Union, Optional)

import jax
import jax.numpy as jnp
from jax import Array

from .._base import Quantity, fail_for_dimension_mismatch
from .._misc import set_module_as

__all__ = [
  'signbit', 'sign', 'bincount', 'digitize',
]


# math funcs remove unit (unary)
# ------------------------------

def funcs_remove_unit_unary(func, x, *args, **kwargs):
  if isinstance(x, Quantity):
    return func(x.value, *args, **kwargs)
  else:
    return func(x, *args, **kwargs)


@set_module_as('brainunit.math')
def signbit(x: Union[Array, Quantity]) -> Array:
  """
  Returns element-wise True where signbit is set (less than zero).

  Parameters
  ----------
  x : array_like, Quantity
      The input value(s).

  Returns
  -------
  result : ndarray of bool
      Output array, or reference to `out` if that was supplied.
      This is a scalar if `x` is a scalar.
  """
  return funcs_remove_unit_unary(jnp.signbit, x)


@set_module_as('brainunit.math')
def sign(x: Union[Array, Quantity]) -> Array:
  """
  Returns the sign of each element in the input array.

  Parameters
  ----------
  x : array_like, Quantity
    Input values.

  Returns
  -------
  y : ndarray
    The sign of `x`.
    This is a scalar if `x` is a scalar.
  """
  return funcs_remove_unit_unary(jnp.sign, x)


@set_module_as('brainunit.math')
def bincount(
    x: Union[Array, Quantity],
    weights: Optional[jax.typing.ArrayLike] = None,
    minlength: int = 0,
    *,
    length: Optional[int] = None
) -> Array:
  """
  Count number of occurrences of each value in array of non-negative ints.

  The number of bins (of size 1) is one larger than the largest value in
  `x`. If `minlength` is specified, there will be at least this number
  of bins in the output array (though it will be longer if necessary,
  depending on the contents of `x`).
  Each bin gives the number of occurrences of its index value in `x`.
  If `weights` is specified the input array is weighted by it, i.e. if a
  value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead
  of ``out[n] += 1``.

  Parameters
  ----------
  x : array_like, Quantity, 1 dimension, nonnegative ints
    Input array.
  weights : array_like, optional
    Weights, array of the same shape as `x`.
  minlength : int, optional
    A minimum number of bins for the output array.

  Returns
  -------
  out : ndarray of ints
    The result of binning the input array.
    The length of `out` is equal to ``bu.amax(x)+1``.
  """
  return funcs_remove_unit_unary(jnp.bincount, x, weights=weights, minlength=minlength, length=length)


@set_module_as('brainunit.math')
def digitize(
    x: Union[Array, Quantity],
    bins: Union[Array, Quantity],
    right: bool = False
) -> Array:
  """
  Return the indices of the bins to which each value in input array belongs.

  =========  =============  ============================
  `right`    order of bins  returned index `i` satisfies
  =========  =============  ============================
  ``False``  increasing     ``bins[i-1] <= x < bins[i]``
  ``True``   increasing     ``bins[i-1] < x <= bins[i]``
  ``False``  decreasing     ``bins[i-1] > x >= bins[i]``
  ``True``   decreasing     ``bins[i-1] >= x > bins[i]``
  =========  =============  ============================

  If values in `x` are beyond the bounds of `bins`, 0 or ``len(bins)`` is
  returned as appropriate.

  Parameters
  ----------
  x : array_like, Quantity
    Input array to be binned. Prior to NumPy 1.10.0, this array had to
    be 1-dimensional, but can now have any shape.
  bins : array_like, Quantity
    Array of bins. It has to be 1-dimensional and monotonic.
  right : bool, optional
    Indicating whether the intervals include the right or the left bin
    edge. Default behavior is (right==False) indicating that the interval
    does not include the right edge. The left bin end is open in this
    case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
    monotonically increasing bins.

  Returns
  -------
  indices : ndarray of ints
      Output array of indices, of same shape as `x`.
  """
  if isinstance(x, Quantity) and isinstance(bins, Quantity):
    fail_for_dimension_mismatch(x, bins, 'digitize requires x and bins to have the same dimension')
    x = x.value
    bins = bins.value
  elif isinstance(x, Quantity):
    assert x.is_unitless, f'Expected unitless Quantity when bins is not a Quantity, got {x}'
    x = x.value
  elif isinstance(bins, Quantity):
    assert bins.is_unitless, f'Expected unitless Quantity when x is not a Quantity, got {bins}'
    bins = bins.value
  return jnp.digitize(x, bins, right=right)
