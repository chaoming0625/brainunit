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
from typing import (Union, Optional, Sequence, Any)

import jax
import jax.numpy as jnp
from jax import Array

from brainunit._misc import set_module_as
from .._base import Quantity

__all__ = [

  # math funcs remove unit (unary)
  'signbit', 'sign', 'histogram', 'bincount',

  # math funcs remove unit (binary)
  'corrcoef', 'correlate', 'cov', 'digitize',
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
def histogram(
    x: Union[Array, Quantity],
    bins: jax.typing.ArrayLike = 10,
    range: Sequence[jax.typing.ArrayLike] | None = None,
    weights: jax.typing.ArrayLike | None = None,
    density: bool | None = None
) -> tuple[Array, Array]:
  """
  Compute the histogram of a set of data.

  Parameters
  ----------
  x : array_like, Quantity
    Input data. The histogram is computed over the flattened array.
  bins : int or sequence of scalars or str, optional
    If `bins` is an int, it defines the number of equal-width
    bins in the given range (10, by default). If `bins` is a
    sequence, it defines a monotonically increasing array of bin edges,
    including the rightmost edge, allowing for non-uniform bin widths.

    If `bins` is a string, it defines the method used to calculate the
    optimal bin width, as defined by `histogram_bin_edges`.

  range : (float, float), optional
    The lower and upper range of the bins.  If not provided, range
    is simply ``(a.min(), a.max())``.  Values outside the range are
    ignored. The first element of the range must be less than or
    equal to the second. `range` affects the automatic bin
    computation as well. While bin width is computed to be optimal
    based on the actual data within `range`, the bin count will fill
    the entire range including portions containing no data.
  weights : array_like, optional
    An array of weights, of the same shape as `a`.  Each value in
    `a` only contributes its associated weight towards the bin count
    (instead of 1). If `density` is True, the weights are
    normalized, so that the integral of the density over the range
    remains 1.
  density : bool, optional
    If ``False``, the result will contain the number of samples in
    each bin. If ``True``, the result is the value of the
    probability *density* function at the bin, normalized such that
    the *integral* over the range is 1. Note that the sum of the
    histogram values will not be equal to 1 unless bins of unity
    width are chosen; it is not a probability *mass* function.

  Returns
  -------
  hist : array
    The values of the histogram. See `density` and `weights` for a
    description of the possible semantics.
  bin_edges : array of dtype float
    Return the bin edges ``(length(hist)+1)``.
  """
  return funcs_remove_unit_unary(jnp.histogram, x, bins=bins, range=range, weights=weights, density=density)


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


# math funcs remove unit (binary)
# -------------------------------
def funcs_remove_unit_binary(func, x, y, *args, **kwargs):
  if isinstance(x, Quantity):
    x_value = x.value
  if isinstance(y, Quantity):
    y_value = y.value
  if isinstance(x, Quantity) or isinstance(y, Quantity):
    return func(jnp.array(x_value), jnp.array(y_value), *args, **kwargs)
  else:
    return func(x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def corrcoef(
    x: Union[Array, Quantity],
    y: Union[Array, Quantity],
    rowvar: bool = True
) -> Array:
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

  Returns
  -------
  R : ndarray
    The correlation coefficient matrix of the variables.
  """
  return funcs_remove_unit_binary(jnp.corrcoef, x, y, rowvar=rowvar)


@set_module_as('brainunit.math')
def correlate(
    a: Union[Array, Quantity],
    v: Union[Array, Quantity],
    mode: str = 'valid',
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Array:
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

  Returns
  -------
  out : ndarray
    Discrete cross-correlation of `a` and `v`.
  """
  return funcs_remove_unit_binary(jnp.correlate, a, v, mode=mode, precision=precision,
                                  preferred_element_type=preferred_element_type)


@set_module_as('brainunit.math')
def cov(
    m: Union[Array, Quantity],
    y: Optional[Union[Array, Quantity]] = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: int | None = None,
    fweights: jax.typing.ArrayLike | None = None,
    aweights: jax.typing.ArrayLike | None = None
) -> Array:
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

  Returns
  -------
  out : ndarray
    The covariance matrix of the variables.
  """
  return funcs_remove_unit_binary(jnp.cov, m, y, rowvar=rowvar, bias=bias, ddof=ddof, fweights=fweights,
                                  aweights=aweights)


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
  return funcs_remove_unit_binary(jnp.digitize, x, bins, right=right)
