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

from typing import (Union, Any, Optional, Tuple)

import jax
import jax.numpy as jnp
from jax import Array

from ._numpy_change_unit import _fun_change_unit_binary
from ._numpy_keep_unit import _fun_keep_unit_unary
from .._base import Quantity
from .._misc import set_module_as

__all__ = [
  # linear algebra
  'dot', 'vdot', 'inner', 'outer', 'kron', 'matmul', 'tensordot', 'trace',

]


# linear algebra
# --------------

@set_module_as('brainunit.math')
def dot(
    a: Union[Array, Quantity],
    b: Union[Array, Quantity],
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Union[Array, Quantity]:
  """
  Dot product of two arrays or quantities.

  Parameters
  ----------
  a : array_like, Quantity
    First argument.
  b : array_like, Quantity
    Second argument.
  precision : either ``None`` (default),
    which means the default precision for the backend, a :class:`~jax.lax.Precision`
    enum value (``Precision.DEFAULT``, ``Precision.HIGH`` or ``Precision.HIGHEST``)
    or a tuple of two such values indicating precision of ``a`` and ``b``.
  preferred_element_type : either ``None`` (default)
    which means the default accumulation type for the input types, or a datatype,
    indicating to accumulate results to and return a result with that datatype.

  Returns
  -------
  output : ndarray, Quantity
    array containing the dot product of the inputs, with batch dimensions of
    ``a`` and ``b`` stacked rather than broadcast.

    This is a Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
  """
  return _fun_change_unit_binary(jnp.dot,
                                 lambda x, y: x * y,
                                 a, b,
                                 precision=precision,
                                 preferred_element_type=preferred_element_type)


@set_module_as('brainunit.math')
def vdot(
    a: Union[Array, Quantity],
    b: Union[Array, Quantity],
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Union[Array, Quantity]:
  """
  Perform a conjugate multiplication of two 1D vectors.

  Parameters
  ----------
  a : array_like, Quantity
    First argument.
  b : array_like, Quantity
    Second argument.
  precision : either ``None`` (default),
    which means the default precision for the backend, a :class:`~jax.lax.Precision`
    enum value (``Precision.DEFAULT``, ``Precision.HIGH`` or ``Precision.HIGHEST``)
    or a tuple of two such values indicating precision of ``a`` and ``b``.
  preferred_element_type : either ``None`` (default)
    which means the default accumulation type for the input types, or a datatype,
    indicating to accumulate results to and return a result with that datatype.

  Returns
  -------
  output : ndarray, Quantity
    array containing the dot product of the inputs, with batch dimensions of
    ``a`` and ``b`` stacked rather than broadcast.

    This is a Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
  """
  return _fun_change_unit_binary(jnp.vdot,
                                 lambda x, y: x * y,
                                 a, b,
                                 precision=precision,
                                 preferred_element_type=preferred_element_type)


@set_module_as('brainunit.math')
def inner(
    a: Union[Array, Quantity],
    b: Union[Array, Quantity],
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Union[Array, Quantity]:
  """
  Inner product of two arrays or quantities.

  Parameters
  ----------
  a : array_like, Quantity
    First argument.
  b : array_like, Quantity
    Second argument.
  precision : either ``None`` (default),
    which means the default precision for the backend, a :class:`~jax.lax.Precision`
    enum value (``Precision.DEFAULT``, ``Precision.HIGH`` or ``Precision.HIGHEST``)
    or a tuple of two such values indicating precision of ``a`` and ``b``.
  preferred_element_type : either ``None`` (default)
    which means the default accumulation type for the input types, or a datatype,
    indicating to accumulate results to and return a result with that datatype.

  Returns
  -------
  output : ndarray, Quantity
    array containing the inner product of the inputs, with batch dimensions of
    ``a`` and ``b`` stacked rather than broadcast.

    This is a Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
  """
  return _fun_change_unit_binary(jnp.inner,
                                 lambda x, y: x * y,
                                 a, b,
                                 precision=precision,
                                 preferred_element_type=preferred_element_type)


@set_module_as('brainunit.math')
def outer(
    a: Union[Array, Quantity],
    b: Union[Array, Quantity],
    out: Optional[Any] = None
) -> Union[Array, Quantity]:
  """
  Compute the outer product of two vectors or quantities.

  Parameters
  ----------
  a : array_like, Quantity
    First argument.
  b : array_like, Quantity
    Second argument.
  out : ndarray, optional
    A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
    If not provided or None, a freshly-allocated array is returned.

  Returns
  -------
  output : ndarray, Quantity
    array containing the outer product of the inputs, with batch dimensions of
    ``a`` and ``b`` stacked rather than broadcast.

    This is a Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
  """
  return _fun_change_unit_binary(jnp.outer,
                                 lambda x, y: x * y,
                                 a, b,
                                 out=out)


@set_module_as('brainunit.math')
def kron(
    a: Union[Array, Quantity],
    b: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  """
  Compute the Kronecker product of two arrays or quantities.

  Parameters
  ----------
  a : array_like, Quantity
    First input.
  b : array_like, Quantity
    Second input.

  Returns
  -------
  output : ndarray, Quantity
    Kronecker product of `a` and `b`.

    This is a Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
  """
  return _fun_change_unit_binary(jnp.kron,
                                 lambda x, y: x * y,
                                 a, b)


@set_module_as('brainunit.math')
def matmul(
    a: Union[Array, Quantity],
    b: Union[Array, Quantity],
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Union[Array, Quantity]:
  """
  Matrix product of two arrays or quantities.

  Parameters
  ----------
  a : array_like, Quantity
    First argument.
  b : array_like, Quantity
    Second argument.
  precision : either ``None`` (default),
    which means the default precision for the backend, a :class:`~jax.lax.Precision`
    enum value (``Precision.DEFAULT``, ``Precision.HIGH`` or ``Precision.HIGHEST``)
    or a tuple of two such values indicating precision of ``a`` and ``b``.
  preferred_element_type : either ``None`` (default)
    which means the default accumulation type for the input types, or a datatype,
    indicating to accumulate results to and return a result with that datatype.

  Returns
  -------
  output : ndarray, Quantity
    array containing the matrix product of the inputs, with batch dimensions of
    ``a`` and ``b`` stacked rather than broadcast.

    This is a Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
  """
  return _fun_change_unit_binary(jnp.matmul,
                                 lambda x, y: x * y,
                                 a, b,
                                 precision=precision,
                                 preferred_element_type=preferred_element_type)


@set_module_as('brainunit.math')
def tensordot(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    axes: Union[int, Tuple[int, int]] = 2,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Union[jax.Array, Quantity]:
  """
  Compute tensor dot product along specified axes.

  Given two tensors, `a` and `b`, and an array_like object containing
  two array_like objects, ``(a_axes, b_axes)``, sum the products of
  `a`'s and `b`'s elements (components) over the axes specified by
  ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
  integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
  of `a` and the first ``N`` dimensions of `b` are summed over.

  Parameters
  ----------
  a, b : array_like, Quantity
    Tensors to "dot".

  axes : int or (2,) array_like
    * integer_like
      If an int N, sum over the last N axes of `a` and the first N axes
      of `b` in order. The sizes of the corresponding axes must match.
    * (2,) array_like
      Or, a list of axes to be summed over, first sequence applying to `a`,
      second to `b`. Both elements array_like must be of the same length.
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
  output : ndarray, Quantity
    The tensor dot product of the input.

    This is a quantity if the product of the units of `a` and `b` is not dimensionless.
  """
  return _fun_change_unit_binary(jnp.tensordot,
                                 lambda x, y: x * y,
                                 a, b,
                                 axes=axes,
                                 precision=precision,
                                 preferred_element_type=preferred_element_type)


@set_module_as('brainunit.math')
def trace(
    a: Union[Array, Quantity],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: Optional[jax.typing.DTypeLike] = None,
) -> Union[Array, Quantity]:
  """
  Return the sum along diagonals of the array.

  If `a` is 2-D, the sum along its diagonal with the given offset
  is returned, i.e., the sum of elements ``a[i,i+offset]`` for all i.

  If `a` has more than two dimensions, then the axes specified by axis1 and
  axis2 are used to determine the 2-D sub-arrays whose traces are returned.
  The shape of the resulting array is the same as that of `a` with `axis1`
  and `axis2` removed.

  Parameters
  ----------
  a : array_like, Quantity
    Input array, from which the diagonals are taken.
  offset : int, optional
    Offset of the diagonal from the main diagonal. Can be both positive
    and negative. Defaults to 0.
  axis1, axis2 : int, optional
    Axes to be used as the first and second axis of the 2-D sub-arrays
    from which the diagonals should be taken. Defaults are the first two
    axes of `a`.
  dtype : dtype, optional
    Determines the data-type of the returned array and of the accumulator
    where the elements are summed. If dtype has the value None and `a` is
    of integer type of precision less than the default integer
    precision, then the default integer precision is used. Otherwise,
    the precision is the same as that of `a`.

  Returns
  -------
  sum_along_diagonals : ndarray
    If `a` is 2-D, the sum along the diagonal is returned.  If `a` has
    larger dimensions, then an array of sums along diagonals is returned.

    This is a Quantity if `a` is a Quantity, else an array.
  """
  return _fun_keep_unit_unary(jnp.trace, a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)
