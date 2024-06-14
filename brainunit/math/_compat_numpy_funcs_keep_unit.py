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
from typing import (Union, Optional, Sequence)

import jax
import jax.numpy as jnp
import numpy as np

from brainunit._misc import set_module_as
from .._base import Quantity, fail_for_dimension_mismatch

__all__ = [
  # math funcs keep unit (unary)
  'real', 'imag', 'conj', 'conjugate', 'negative', 'positive',
  'abs', 'round', 'around', 'round_', 'rint',
  'floor', 'ceil', 'trunc', 'fix', 'sum', 'nancumsum', 'nansum',
  'cumsum', 'ediff1d', 'absolute', 'fabs', 'median',
  'nanmin', 'nanmax', 'ptp', 'average', 'mean', 'std',
  'nanmedian', 'nanmean', 'nanstd', 'diff', 'modf',

  # math funcs keep unit (binary)
  'fmod', 'mod', 'copysign', 'heaviside',
  'maximum', 'minimum', 'fmax', 'fmin', 'lcm', 'gcd',

  # math funcs keep unit (n-ary)
  'interp', 'clip',
]


# math funcs keep unit (unary)
# ----------------------------


def funcs_keep_unit_unary(func, x, *args, **kwargs):
  if isinstance(x, Quantity):
    return Quantity(func(x.value, *args, **kwargs), dim=x.dim)
  elif isinstance(x, (jax.Array, np.ndarray)):
    return func(x, *args, **kwargs)
  else:
    raise ValueError(f'Unsupported type: {type(x)} for {func.__name__}')


@set_module_as('brainunit.math')
def real(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Return the real part of the complex argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.real, x)


@set_module_as('brainunit.math')
def imag(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Return the imaginary part of the complex argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.imag, x)


@set_module_as('brainunit.math')
def conj(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Return the complex conjugate of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.conj, x)


@set_module_as('brainunit.math')
def conjugate(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Return the complex conjugate of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.conjugate, x)


@set_module_as('brainunit.math')
def negative(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Return the negative of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.negative, x)


@set_module_as('brainunit.math')
def positive(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Return the positive of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.positive, x)


@set_module_as('brainunit.math')
def abs(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Return the absolute value of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.abs, x)


@set_module_as('brainunit.math')
def round_(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Round an array to the nearest integer.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.round_, x)


@set_module_as('brainunit.math')
def around(
    x: Union[Quantity, jax.typing.ArrayLike],
    decimals: int = 0,
    out: Optional[Union[Quantity, jax.typing.ArrayLike]] = None
) -> Union[Quantity, jax.Array]:
  """
  Round an array to the nearest integer.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  decimals : int, optional
    Number of decimal places to round to (default is 0).
  out : array_like, Quantity, optional
    Output array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.around, x)


@set_module_as('brainunit.math')
def round(
    x: Union[Quantity, jax.typing.ArrayLike],
    decimals: int = 0,
) -> Union[Quantity, jax.Array]:
  """
  Round an array to the nearest integer.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  decimals : int, optional
    Number of decimal places to round to (default is 0).
  out : array_like, Quantity, optional
    Output array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.round, x, decimals=decimals)


@set_module_as('brainunit.math')
def rint(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Round an array to the nearest integer.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.rint, x)


@set_module_as('brainunit.math')
def floor(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Return the floor of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.floor, x)


@set_module_as('brainunit.math')
def ceil(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Return the ceiling of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.ceil, x)


@set_module_as('brainunit.math')
def trunc(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Return the truncated value of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.trunc, x)


@set_module_as('brainunit.math')
def fix(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
  """
  Return the nearest integer towards zero.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.fix, x)


@set_module_as('brainunit.math')
def sum(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
    keepdims: bool = False,
    initial: Union[jax.typing.ArrayLike, None] = None,
    where: Union[jax.typing.ArrayLike, None] = None,
    promote_integers: bool = True
) -> Union[Quantity, jax.Array]:
  """
  Return the sum of the array elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : None or int or tuple of ints, optional
    Axis or axes along which a sum is performed.  The default,
    axis=None, will sum all of the elements of the input array.  If
    axis is negative it counts from the last to the first axis.

    If axis is a tuple of ints, a sum is performed on all of the axes
    specified in the tuple instead of a single axis or all the axes as
    before.
  dtype : dtype, optional
    The type of the returned array and of the accumulator in which the
    elements are summed.  The dtype of `a` is used by default unless `a`
    has an integer dtype of less precision than the default platform
    integer.  In that case, if `a` is signed then the platform integer
    is used while if `a` is unsigned then an unsigned integer of the
    same precision as the platform integer is used.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `sum` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.
  initial : scalar, optional
    Starting value for the sum. See `~numpy.ufunc.reduce` for details.
  where : array_like of bool, optional
    Elements to include in the sum. See `~numpy.ufunc.reduce` for details.
  promote_integers : bool, optional
    If True, and if the accumulator is an integer type, then the

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.sum, x,
                               axis=axis, dtype=dtype, keepdims=keepdims, initial=initial,
                               where=where, promote_integers=promote_integers)


@set_module_as('brainunit.math')
def nancumsum(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
) -> Union[Quantity, jax.Array]:
  """
  Return the cumulative sum of the array elements, ignoring NaNs.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : int, optional
    Axis along which the cumulative sum is computed. The default
    (None) is to compute the cumsum over the flattened array.
  dtype : dtype, optional
    Type of the returned array and of the accumulator in which the
    elements are summed.  If `dtype` is not specified, it defaults
    to the dtype of `a`, unless `a` has an integer dtype with a
    precision less than that of the default platform integer.  In
    that case, the default platform integer is used.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.nancumsum, x, axis=axis, dtype=dtype)


@set_module_as('brainunit.math')
def nansum(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
    keepdims: bool = False,
    initial: Union[jax.typing.ArrayLike, None] = None,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> Union[Quantity, jax.Array]:
  """
  Return the sum of the array elements, ignoring NaNs.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : {int, tuple of int, None}, optional
    Axis or axes along which the sum is computed. The default is to compute the
    sum of the flattened array.
  dtype : data-type, optional
    The type of the returned array and of the accumulator in which the
    elements are summed.  By default, the dtype of `a` is used.  An
    exception is when `a` has an integer type with less precision than
    the platform (u)intp. In that case, the default will be either
    (u)int32 or (u)int64 depending on whether the platform is 32 or 64
    bits. For inexact inputs, dtype must be inexact.
  keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the original `a`.

      If the value is anything but the default, then
      `keepdims` will be passed through to the `mean` or `sum` methods
      of sub-classes of `ndarray`.  If the sub-classes methods
      does not implement `keepdims` any exceptions will be raised.
  initial : scalar, optional
      Starting value for the sum. See `~numpy.ufunc.reduce` for details.
  where : array_like of bool, optional
      Elements to include in the sum. See `~numpy.ufunc.reduce` for details.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.nansum, x,
                               axis=axis, dtype=dtype, keepdims=keepdims, initial=initial,
                               where=where)


@set_module_as('brainunit.math')
def cumsum(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
) -> Union[Quantity, jax.Array]:
  """
  Return the cumulative sum of the array elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : int, optional
    Axis along which the cumulative sum is computed. The default
    (None) is to compute the cumsum over the flattened array.
  dtype : dtype, optional
    Type of the returned array and of the accumulator in which the
    elements are summed.  If `dtype` is not specified, it defaults
    to the dtype of `a`, unless `a` has an integer dtype with a
    precision less than that of the default platform integer.  In
    that case, the default platform integer is used.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.cumsum, x, axis=axis, dtype=dtype)


@set_module_as('brainunit.math')
def ediff1d(
    x: Union[Quantity, jax.typing.ArrayLike],
    to_end: jax.typing.ArrayLike = None,
    to_begin: jax.typing.ArrayLike = None
) -> Union[Quantity, jax.Array]:
  """
  Return the differences between consecutive elements of the array.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  to_end : array_like, optional
    Number(s) to append at the end of the returned differences.
  to_begin : array_like, optional
    Number(s) to prepend at the beginning of the returned differences.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.ediff1d, x, to_end=to_end, to_begin=to_begin)


@set_module_as('brainunit.math')
def absolute(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Return the absolute value of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.absolute, x)


@set_module_as('brainunit.math')
def fabs(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
  """
  Return the absolute value of the argument.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.fabs, x)


@set_module_as('brainunit.math')
def median(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    overwrite_input: bool = False,
    keepdims: bool = False
) -> Union[Quantity, jax.Array]:
  """
  Return the median of the array elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : {int, sequence of int, None}, optional
    Axis or axes along which the medians are computed. The default
    is to compute the median along a flattened version of the array.
    A sequence of axes is supported since version 1.9.0.
  overwrite_input : bool, optional
   If True, then allow use of memory of input array `a` for
   calculations. The input array will be modified by the call to
   `median`. This will save memory when you do not need to preserve
   the contents of the input array. Treat the input as undefined,
   but it will probably be fully or partially sorted. Default is
   False. If `overwrite_input` is ``True`` and `a` is not already an
   `ndarray`, an error will be raised.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the original `arr`.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.median, x, axis=axis, overwrite_input=overwrite_input, keepdims=keepdims)


@set_module_as('brainunit.math')
def nanmin(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    keepdims: bool = False,
    initial: Union[jax.typing.ArrayLike, None] = None,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> Union[Quantity, jax.Array]:
  """
  Return the minimum of the array elements, ignoring NaNs.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : {int, tuple of int, None}, optional
    Axis or axes along which the minimum is computed. The default is to compute
    the minimum of the flattened array.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the original `a`.

    If the value is anything but the default, then
    `keepdims` will be passed through to the `min` method
    of sub-classes of `ndarray`.  If the sub-classes methods
    does not implement `keepdims` any exceptions will be raised.
  initial : scalar, optional
    The maximum value of an output element. Must be present to allow
    computation on empty slice. See `~numpy.ufunc.reduce` for details.
  where : array_like of bool, optional
    Elements to compare for the minimum. See `~numpy.ufunc.reduce`
    for details.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.nanmin, x, axis=axis, keepdims=keepdims, initial=initial, where=where)


@set_module_as('brainunit.math')
def nanmax(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    keepdims: bool = False,
    initial: Union[jax.typing.ArrayLike, None] = None,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> Union[Quantity, jax.Array]:
  """
  Return the maximum of the array elements, ignoring NaNs.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : {int, tuple of int, None}, optional
    Axis or axes along which the minimum is computed. The default is to compute
    the minimum of the flattened array.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the original `a`.

    If the value is anything but the default, then
    `keepdims` will be passed through to the `min` method
    of sub-classes of `ndarray`.  If the sub-classes methods
    does not implement `keepdims` any exceptions will be raised.
  initial : scalar, optional
    The maximum value of an output element. Must be present to allow
    computation on empty slice. See `~numpy.ufunc.reduce` for details.
  where : array_like of bool, optional
    Elements to compare for the minimum. See `~numpy.ufunc.reduce`
    for details.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.nanmax, x, axis=axis, keepdims=keepdims, initial=initial, where=where)


@set_module_as('brainunit.math')
def ptp(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    keepdims: bool = False,
) -> Union[Quantity, jax.Array]:
  """
  Return the range of the array elements (maximum - minimum).

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : None or int or tuple of ints, optional
    Axis along which to find the peaks.  By default, flatten the
    array.  `axis` may be negative, in
    which case it counts from the last to the first axis.

    If this is a tuple of ints, a reduction is performed on multiple
    axes, instead of a single axis or all the axes as before.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `ptp` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.ptp, x, axis=axis, keepdims=keepdims)


@set_module_as('brainunit.math')
def average(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    weights: Union[jax.typing.ArrayLike, None] = None,
    returned: bool = False,
    keepdims: bool = False
) -> Union[Quantity, jax.Array]:
  """
  Return the weighted average of the array elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : None or int or tuple of ints, optional
    Axis or axes along which to average `a`.  The default,
    axis=None, will average over all of the elements of the input array.
    If axis is negative it counts from the last to the first axis.

    If axis is a tuple of ints, averaging is performed on all of the axes
    specified in the tuple instead of a single axis or all the axes as
    before.
  weights : array_like, optional
    An array of weights associated with the values in `a`. Each value in
    `a` contributes to the average according to its associated weight.
    The weights array can either be 1-D (in which case its length must be
    the size of `a` along the given axis) or of the same shape as `a`.
    If `weights=None`, then all data in `a` are assumed to have a
    weight equal to one.  The 1-D calculation is::

        avg = sum(a * weights) / sum(weights)

    The only constraint on `weights` is that `sum(weights)` must not be 0.
  returned : bool, optional
    Default is `False`. If `True`, the tuple (`average`, `sum_of_weights`)
    is returned, otherwise only the average is returned.
    If `weights=None`, `sum_of_weights` is equivalent to the number of
    elements over which the average is taken.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the original `a`.
    *Note:* `keepdims` will not work with instances of `numpy.matrix`
    or other classes whose methods do not support `keepdims`.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.average, x, axis=axis, weights=weights, returned=returned, keepdims=keepdims)


@set_module_as('brainunit.math')
def mean(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
    keepdims: bool = False, *,
    where: Union[jax.typing.ArrayLike, None] = None
) -> Union[Quantity, jax.Array]:
  """
  Return the mean of the array elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : None or int or tuple of ints, optional
    Axis or axes along which the means are computed. The default is to
    compute the mean of the flattened array.

    If this is a tuple of ints, a mean is performed over multiple axes,
    instead of a single axis or all the axes as before.
  dtype : data-type, optional
    Type to use in computing the mean.  For integer inputs, the default
    is `float64`; for floating point inputs, it is the same as the
    input dtype.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `mean` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.
  where : array_like of bool, optional
      Elements to include in the mean.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.mean, x, axis=axis, dtype=dtype, keepdims=keepdims, where=where)


@set_module_as('brainunit.math')
def std(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
    ddof: int = 0,
    keepdims: bool = False, *,
    where: Union[jax.typing.ArrayLike, None] = None
) -> Union[Quantity, jax.Array]:
  """
  Return the standard deviation of the array elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : None or int or tuple of ints, optional
    Axis or axes along which the standard deviation is computed. The
    default is to compute the standard deviation of the flattened array.

    If this is a tuple of ints, a standard deviation is performed over
    multiple axes, instead of a single axis or all the axes as before.
  dtype : dtype, optional
    Type to use in computing the standard deviation. For arrays of
    integer type the default is float64, for arrays of float types it is
    the same as the array type.
  ddof : int, optional
    Means Delta Degrees of Freedom.  The divisor used in calculations
    is ``N - ddof``, where ``N`` represents the number of elements.
    By default `ddof` is zero.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `std` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.
  where : array_like of bool, optional
    Elements to include in the standard deviation.
    See `~numpy.ufunc.reduce` for details.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.std, x, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)


@set_module_as('brainunit.math')
def nanmedian(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, tuple[int, ...], None] = None,
    overwrite_input: bool = False,
    keepdims: bool = False
) -> Union[Quantity, jax.Array]:
  """
  Return the median of the array elements, ignoring NaNs.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : {int, sequence of int, None}, optional
    Axis or axes along which the medians are computed. The default
    is to compute the median along a flattened version of the array.
    A sequence of axes is supported since version 1.9.0.
  overwrite_input : bool, optional
   If True, then allow use of memory of input array `a` for
   calculations. The input array will be modified by the call to
   `median`. This will save memory when you do not need to preserve
   the contents of the input array. Treat the input as undefined,
   but it will probably be fully or partially sorted. Default is
   False. If `overwrite_input` is ``True`` and `a` is not already an
   `ndarray`, an error will be raised.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the original `a`.

    If this is anything but the default value it will be passed
    through (in the special case of an empty array) to the
    `mean` function of the underlying array.  If the array is
    a sub-class and `mean` does not have the kwarg `keepdims` this
    will raise a RuntimeError.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.nanmedian, x, axis=axis, overwrite_input=overwrite_input, keepdims=keepdims)


@set_module_as('brainunit.math')
def nanmean(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
    keepdims: bool = False, *,
    where: Union[jax.typing.ArrayLike, None] = None
) -> Union[Quantity, jax.Array]:
  """
  Return the mean of the array elements, ignoring NaNs.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : None or int or tuple of ints, optional
    Axis or axes along which the means are computed. The default is to
    compute the mean of the flattened array.

    If this is a tuple of ints, a mean is performed over multiple axes,
    instead of a single axis or all the axes as before.
  dtype : data-type, optional
    Type to use in computing the mean.  For integer inputs, the default
    is `float64`; for floating point inputs, it is the same as the
    input dtype.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `mean` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.
  where : array_like of bool, optional
      Elements to include in the mean.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.nanmean, x, axis=axis, dtype=dtype, keepdims=keepdims, where=where)


@set_module_as('brainunit.math')
def nanstd(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
    ddof: int = 0,
    keepdims: bool = False, *,
    where: Union[jax.typing.ArrayLike, None] = None
) -> Union[Quantity, jax.Array]:
  """
  Return the standard deviation of the array elements, ignoring NaNs.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  axis : None or int or tuple of ints, optional
    Axis or axes along which the standard deviation is computed. The
    default is to compute the standard deviation of the flattened array.

    If this is a tuple of ints, a standard deviation is performed over
    multiple axes, instead of a single axis or all the axes as before.
  dtype : dtype, optional
    Type to use in computing the standard deviation. For arrays of
    integer type the default is float64, for arrays of float types it is
    the same as the array type.
  ddof : int, optional
    Means Delta Degrees of Freedom.  The divisor used in calculations
    is ``N - ddof``, where ``N`` represents the number of elements.
    By default `ddof` is zero.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `std` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.
  where : array_like of bool, optional
    Elements to include in the standard deviation.
    See `~numpy.ufunc.reduce` for details.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.nanstd, x, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims,
                               where=where)


@set_module_as('brainunit.math')
def diff(
    x: Union[Quantity, jax.typing.ArrayLike],
    n: int = 1, axis: int = -1,
    prepend: Union[jax.typing.ArrayLike, None] = None,
    append: Union[jax.typing.ArrayLike, None] = None
) -> Union[Quantity, jax.Array]:
  """
  Return the differences between consecutive elements of the array.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.
  n : int, optional
    The number of times values are differenced. If zero, the input
    is returned as-is.
  axis : int, optional
    The axis along which the difference is taken, default is the
    last axis.
  prepend, append : array_like, optional
    Values to prepend or append to `a` along axis prior to
    performing the difference.  Scalar values are expanded to
    arrays with length 1 in the direction of axis and the shape
    of the input array in along all other axes.  Otherwise the
    dimension and shape must match `a` except along axis.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.diff, x, n=n, axis=axis, prepend=prepend, append=append)


@set_module_as('brainunit.math')
def modf(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
  """
  Return the fractional and integer parts of the array elements.

  Parameters
  ----------
  x : array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x` is a Quantity, else an array.
  """
  return funcs_keep_unit_unary(jnp.modf, x)


# math funcs keep unit (binary)
# -----------------------------

def funcs_keep_unit_binary(func, x1, x2, *args, **kwargs):
  if isinstance(x1, Quantity) and isinstance(x2, Quantity):
    fail_for_dimension_mismatch(x1, x2, func.__name__)
    return Quantity(func(x1.value, x2.value, *args, **kwargs), dim=x1.dim)
  elif isinstance(x1, (jax.Array, np.ndarray)) and isinstance(x2, (jax.Array, np.ndarray)):
    return func(x1, x2, *args, **kwargs)
  else:
    raise ValueError(f'Unsupported type: {type(x1)} and {type(x2)} for {func.__name__}')


@set_module_as('brainunit.math')
def fmod(x1: Union[Quantity, jax.Array],
         x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  """
  Return the element-wise remainder of division.

  Parameters
  ----------
  x1: array_like, Quantity
    Input array.
  x2: array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  """
  return funcs_keep_unit_binary(jnp.fmod, x1, x2)


@set_module_as('brainunit.math')
def mod(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  """
  Return the element-wise modulus of division.

  Parameters
  ----------
  x1: array_like, Quantity
    Input array.
  x2: array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  """
  return funcs_keep_unit_binary(jnp.mod, x1, x2)


@set_module_as('brainunit.math')
def copysign(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  """
  Return a copy of the first array elements with the sign of the second array.

  Parameters
  ----------
  x1: array_like, Quantity
    Input array.
  x2: array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  """
  return funcs_keep_unit_binary(jnp.copysign, x1, x2)


@set_module_as('brainunit.math')
def heaviside(x1: Union[Quantity, jax.Array],
              x2: jax.typing.ArrayLike) -> Union[Quantity, jax.Array]:
  """
  Compute the Heaviside step function.

  Parameters
  ----------
  x1: array_like, Quantity
    Input array.
  x2: array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  """
  x1 = x1.value if isinstance(x1, Quantity) else x1
  return jnp.heaviside(x1, x2)


@set_module_as('brainunit.math')
def maximum(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  """
  Element-wise maximum of array elements.

  Parameters
  ----------
  x1: array_like, Quantity
    Input array.
  x2: array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  """
  return funcs_keep_unit_binary(jnp.maximum, x1, x2)


@set_module_as('brainunit.math')
def minimum(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  """
  Element-wise minimum of array elements.

  Parameters
  ----------
  x1: array_like, Quantity
    Input array.
  x2: array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  """
  return funcs_keep_unit_binary(jnp.minimum, x1, x2)


@set_module_as('brainunit.math')
def fmax(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  """
  Element-wise maximum of array elements ignoring NaNs.

  Parameters
  ----------
  x1: array_like, Quantity
    Input array.
  x2: array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  """
  return funcs_keep_unit_binary(jnp.fmax, x1, x2)


@set_module_as('brainunit.math')
def fmin(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  """
  Element-wise minimum of array elements ignoring NaNs.

  Parameters
  ----------
  x1: array_like, Quantity
    Input array.
  x2: array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  """
  return funcs_keep_unit_binary(jnp.fmin, x1, x2)


@set_module_as('brainunit.math')
def lcm(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  """
  Return the least common multiple of `x1` and `x2`.

  Parameters
  ----------
  x1: array_like, Quantity
    Input array.
  x2: array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  """
  return funcs_keep_unit_binary(jnp.lcm, x1, x2)


@set_module_as('brainunit.math')
def gcd(x1: Union[Quantity, jax.Array], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
  """
  Return the greatest common divisor of `x1` and `x2`.

  Parameters
  ----------
  x1: array_like, Quantity
    Input array.
  x2: array_like, Quantity
    Input array.

  Returns
  -------
  out : jax.Array, Quantity
    Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
  """
  return funcs_keep_unit_binary(jnp.gcd, x1, x2)


# math funcs keep unit (n-ary)
# ----------------------------
@set_module_as('brainunit.math')
def interp(
    x: Union[Quantity, jax.typing.ArrayLike],
    xp: Union[Quantity, jax.typing.ArrayLike],
    fp: Union[Quantity, jax.typing.ArrayLike],
    left: Union[Quantity, jax.typing.ArrayLike] = None,
    right: Union[Quantity, jax.typing.ArrayLike] = None,
    period: Union[Quantity, jax.typing.ArrayLike] = None
) -> Union[Quantity, jax.Array]:
  """
  One-dimensional linear interpolation.

  Args:
    x: array_like, Quantity
    xp: array_like, Quantity
    fp: array_like, Quantity
    left: array_like, Quantity, optional
    right: array_like, Quantity, optional
    period: array_like, Quantity, optional

  Returns:
    Union[jax.Array, Quantity]: Quantity if `x`, `xp`, and `fp` are Quantities that have the same unit, else an array.
  """
  unit = None
  if isinstance(x, Quantity) or isinstance(xp, Quantity) or isinstance(fp, Quantity):
    unit = x.dim if isinstance(x, Quantity) else xp.dim if isinstance(xp, Quantity) else fp.dim
  if isinstance(x, Quantity):
    x_value = x.value
  else:
    x_value = x
  if isinstance(xp, Quantity):
    xp_value = xp.value
  else:
    xp_value = xp
  if isinstance(fp, Quantity):
    fp_value = fp.value
  else:
    fp_value = fp
  result = jnp.interp(x_value, xp_value, fp_value, left=left, right=right, period=period)
  if unit is not None:
    return Quantity(result, dim=unit)
  else:
    return result


@set_module_as('brainunit.math')
def clip(
    a: Union[Quantity, jax.typing.ArrayLike],
    a_min: Union[Quantity, jax.typing.ArrayLike],
    a_max: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.Array]:
  """
  Clip (limit) the values in an array.

  Args:
    a: array_like, Quantity
    a_min: array_like, Quantity
    a_max: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity]: Quantity if `a`, `a_min`, and `a_max` are Quantities that have the same unit, else an array.
  """
  unit = None
  if isinstance(a, Quantity) or isinstance(a_min, Quantity) or isinstance(a_max, Quantity):
    unit = a.dim if isinstance(a, Quantity) else a_min.dim if isinstance(a_min, Quantity) else a_max.dim
  if isinstance(a, Quantity):
    a_value = a.value
  else:
    a_value = a
  if isinstance(a_min, Quantity):
    a_min_value = a_min.value
  else:
    a_min_value = a_min
  if isinstance(a_max, Quantity):
    a_max_value = a_max.value
  else:
    a_max_value = a_max
  result = jnp.clip(a_value, a_min_value, a_max_value)
  if unit is not None:
    return Quantity(result, dim=unit)
  else:
    return result
