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
from jax import Array

from .._base import (Quantity, fail_for_dimension_mismatch)
from .._misc import set_module_as

__all__ = [
  # logic funcs (unary)
  'all', 'any', 'logical_not',

  # logic funcs (binary)
  'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
  'array_equal', 'isclose', 'allclose', 'logical_and',
  'logical_or', 'logical_xor', "alltrue", 'sometrue',
]


# logic funcs (unary)
# -------------------


def logic_func_unary(func, x, *args, **kwargs):
  if isinstance(x, Quantity):
    raise ValueError(f'Expected arrays, got {x}')
  elif isinstance(x, (jax.Array, np.ndarray)):
    return func(x, *args, **kwargs)
  else:
    raise ValueError(f'Unsupported types {type(x)} for {func.__name__}')


@set_module_as('brainunit.math')
def all(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    keepdims: bool = False,
    where: Optional[Array] = None
) -> Union[bool, Array]:
  """
  Test whether all array elements along a given axis evaluate to True.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or object that can be converted to an array.
  axis : None or int or tuple of ints, optional
    Axis or axes along which a logical AND reduction is performed.
    The default (``axis=None``) is to perform a logical AND over all
    the dimensions of the input array. `axis` may be negative, in
    which case it counts from the last to the first axis.

    If this is a tuple of ints, a reduction is performed on multiple
    axes, instead of a single axis or all the axes as before.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `all` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.
  where : array_like of bool, optional
    Elements to include in checking for all `True` values.

  Returns
  -------
  all : ndarray, bool
    A new boolean or array is returned unless `out` is specified,
    in which case a reference to `out` is returned.
  """
  return logic_func_unary(jnp.all, x, axis=axis, keepdims=keepdims, where=where)


@set_module_as('brainunit.math')
def any(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    keepdims: bool = False,
    where: Optional[Array] = None
) -> Union[bool, Array]:
  """
  Test whether any array element along a given axis evaluates to True.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or object that can be converted to an array.
  axis : None or int or tuple of ints, optional
    Axis or axes along which a logical AND reduction is performed.
    The default (``axis=None``) is to perform a logical AND over all
    the dimensions of the input array. `axis` may be negative, in
    which case it counts from the last to the first axis.

    If this is a tuple of ints, a reduction is performed on multiple
    axes, instead of a single axis or all the axes as before.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `all` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.
  where : array_like of bool, optional
    Elements to include in checking for all `True` values.

  Returns
  -------
  any : ndarray, bool
    A new boolean or array is returned unless `out` is specified,
    in which case a reference to `out` is returned.
  """
  return logic_func_unary(jnp.any, x, axis=axis, keepdims=keepdims, where=where)


@set_module_as('brainunit.math')
def logical_not(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, Array]:
  """
  Compute the truth value of NOT x element-wise.

  Parameters
  ----------
  x : array_like, Quantity
    Input array or object that can be converted to an array.

  Returns
  -------
  logical_not : ndarray, bool
    A new boolean or array is returned unless `out` is specified,
    in which case a reference to `out` is returned.
  """
  return logic_func_unary(jnp.logical_not, x)


alltrue = all
sometrue = any


# logic funcs (binary)
# --------------------


def logic_func_binary(func, x, y, *args, **kwargs):
  if isinstance(x, Quantity) and isinstance(y, Quantity):
    fail_for_dimension_mismatch(x, y)
    return func(x.value, y.value, *args, **kwargs)
  elif isinstance(x, (jax.Array, np.ndarray)) and isinstance(y, (jax.Array, np.ndarray)):
    return func(x, y, *args, **kwargs)
  else:
    raise ValueError(f'Unsupported types {type(x)} and {type(y)} for {func.__name__}')


@set_module_as('brainunit.math')
def equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, Array]:
  """
  equal(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

  Return (x == y) element-wise.

  Parameters
  ----------
  x, y : array_like, Quantity
    Input arrays.
    If ``x.shape != y.shape``, they must be broadcastable to a common
    shape (which becomes the shape of the output).
  out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
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
    Output array, element-wise comparison of `x` and `y`.
    Typically of type bool, unless ``dtype=object`` is passed.
    This is a scalar if both `x` and `y` are scalars.
  """
  return logic_func_binary(jnp.equal, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def not_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, Array]:
  """
  not_equal(x, y, /, out=None, *, where=True, casting='same_kind',
  order='K', dtype=None, subok=True[, signature, extobj])

  Return (x != y) element-wise.

  Parameters
  ----------
  x, y : array_like, Quantity
    Input arrays.
    If ``x.shape != y.shape``, they must be broadcastable to a common
    shape (which becomes the shape of the output).
  out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
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
    Output array, element-wise comparison of `x` and `y`.
    Typically of type bool, unless ``dtype=object`` is passed.
    This is a scalar if both `x` and `y` are scalars.
  """
  return logic_func_binary(jnp.not_equal, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def greater(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, Array]:
  """
  greater(x, y, /, out=None, *, where=True, casting='same_kind',
  order='K', dtype=None, subok=True[, signature, extobj])

  Return the truth value of (x > y) element-wise.

  Parameters
  ----------
  x, y : array_like, Quantity
    Input arrays.
    If ``x.shape != y.shape``, they must be broadcastable to a common
    shape (which becomes the shape of the output).
  out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
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
    Output array, element-wise comparison of `x` and `y`.
    Typically of type bool, unless ``dtype=object`` is passed.
    This is a scalar if both `x` and `y` are scalars.
  """
  return logic_func_binary(jnp.greater, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def greater_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[
  bool, Array]:
  """
  greater_equal(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

  Return the truth value of (x >= y) element-wise.

  Parameters
  ----------
  x, y : array_like, Quantity
    Input arrays.
    If ``x.shape != y.shape``, they must be broadcastable to a common
    shape (which becomes the shape of the output).
  out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
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
  out : bool or ndarray of bool
    Output array, element-wise comparison of `x` and `y`.
    Typically of type bool, unless ``dtype=object`` is passed.
    This is a scalar if both `x` and `y` are scalars.
  """
  return logic_func_binary(jnp.greater_equal, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def less(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, Array]:
  """
  less(x, y, /, out=None, *, where=True, casting='same_kind',
  order='K', dtype=None, subok=True[, signature, extobj])

  Return the truth value of (x < y) element-wise.

  Parameters
  ----------
  x, y : array_like, Quantity
      Input arrays.
      If ``x1.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).
  out : ndarray, None, or tuple of ndarray and None, optional
      A location into which the result is stored. If provided, it must have
      a shape that the inputs broadcast to. If not provided or None,
      a freshly-allocated array is returned. A tuple (possible only as a
      keyword argument) must have length equal to the number of outputs.
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
      Output array, element-wise comparison of `x` and `y`.
      Typically of type bool, unless ``dtype=object`` is passed.
      This is a scalar if both `x` and `y` are scalars.
  """
  return logic_func_binary(jnp.less, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def less_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[
  bool, Array]:
  """
  less_equal(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

  Return the truth value of (x <= y) element-wise.

  Parameters
  ----------
  x, y : array_like, Quantity
    Input arrays.
    If ``x.shape != y.shape``, they must be broadcastable to a common
    shape (which becomes the shape of the output).
  out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
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
    Output array, element-wise comparison of `x` and `y`.
    Typically of type bool, unless ``dtype=object`` is passed.
    This is a scalar if both `x` and `y` are scalars.
  """
  return logic_func_binary(jnp.less_equal, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def array_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[
  bool, Array]:
  """
  True if two arrays have the same shape and elements, False otherwise.

  Parameters
  ----------
  x, y : array_like, Quantity
    Input arrays.
  equal_nan : bool
    Whether to compare NaN's as equal. If the dtype of a1 and a2 is
    complex, values will be considered equal if either the real or the
    imaginary component of a given value is ``nan``.

  Returns
  -------
  b : bool
    Returns True if the arrays are equal.
  """
  return logic_func_binary(jnp.array_equal, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def isclose(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False
) -> Union[bool, Array]:
  """
  Returns a boolean array where two arrays are element-wise equal within a
  tolerance.

  The tolerance values are positive, typically very small numbers.  The
  relative difference (`rtol` * abs(`b`)) and the absolute difference
  `atol` are added together to compare against the absolute difference
  between `a` and `b`.

  Parameters
  ----------
  x, y : array_like, Quantity
    Input arrays to compare.
  rtol : float
    The relative tolerance parameter (see Notes).
  atol : float
    The absolute tolerance parameter (see Notes).
  equal_nan : bool
    Whether to compare NaN's as equal.  If True, NaN's in `a` will be
    considered equal to NaN's in `b` in the output array.

  Returns
  -------
  out : array_like
    Returns a boolean array of where `a` and `b` are equal within the
    given tolerance. If both `a` and `b` are scalars, returns a single
    boolean value.
  """
  return logic_func_binary(jnp.isclose, x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)


@set_module_as('brainunit.math')
def allclose(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False
) -> Union[bool, Array]:
  """
  Returns True if two arrays are element-wise equal within a tolerance.

  The tolerance values are positive, typically very small numbers.  The
  relative difference (`rtol` * abs(`b`)) and the absolute difference
  `atol` are added together to compare against the absolute difference
  between `a` and `b`.

  NaNs are treated as equal if they are in the same place and if
  ``equal_nan=True``.  Infs are treated as equal if they are in the same
  place and of the same sign in both arrays.

  Parameters
  ----------
  x, y : array_like, Quantity
    Input arrays to compare.
  rtol : float
    The relative tolerance parameter (see Notes).
  atol : float
    The absolute tolerance parameter (see Notes).
  equal_nan : bool
    Whether to compare NaN's as equal.  If True, NaN's in `a` will be
    considered equal to NaN's in `b` in the output array.

  Returns
  -------
  allclose : bool
    Returns True if the two arrays are equal within the given
    tolerance; False otherwise.
  """
  return logic_func_binary(jnp.allclose, x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)


@set_module_as('brainunit.math')
def logical_and(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[
  bool, Array]:
  """
  logical_and(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

  Compute the truth value of x AND y element-wise.

  Parameters
  ----------
  x, y : array_like, Quantity
    Input arrays.
    If ``x.shape != y.shape``, they must be broadcastable to a common
    shape (which becomes the shape of the output).
  out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
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
  out : ndarray or bool
    Boolean result of the logical AND operation applied to the elements
    of `x` and `y`; the shape is determined by broadcasting.
    This is a scalar if both `x` and `y` are scalars.
  """
  return logic_func_binary(jnp.logical_and, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def logical_or(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[
  bool, Array]:
  """
  logical_or(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

  Compute the truth value of x OR y element-wise.

  Parameters
  ----------
  x, y : array_like, Quantity
    Logical OR is applied to the elements of `x` and `y`.
    If ``x.shape != y.shape``, they must be broadcastable to a common
    shape (which becomes the shape of the output).
  out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
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
  out : ndarray or bool
    Boolean result of the logical OR operation applied to the elements
    of `x` and `y`; the shape is determined by broadcasting.
    This is a scalar if both `x` and `y` are scalars.
  """
  return logic_func_binary(jnp.logical_or, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def logical_xor(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[
  bool, Array]:
  """
  logical_xor(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

  Compute the truth value of x XOR y, element-wise.

  Parameters
  ----------
  x, y : array_like, Quantity
    Logical XOR is applied to the elements of `x` and `y`.
    If ``x.shape != y.shape``, they must be broadcastable to a common
    shape (which becomes the shape of the output).
  out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
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
  out : bool or ndarray of bool
    Boolean result of the logical XOR operation applied to the elements
    of `x` and `y`; the shape is determined by broadcasting.
    This is a scalar if both `x` and `y` are scalars.
  """
  return logic_func_binary(jnp.logical_xor, x, y, *args, **kwargs)
