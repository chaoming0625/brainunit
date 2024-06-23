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

import collections
from collections.abc import Sequence
from typing import (Callable, Union, Tuple, Any, Optional)

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum
from jax import Array, core, jit, lax
from jax._src import config, dtypes, api_util
from jax._src.lax import lax as lax_internal
from jax._src.lax.lax import DotDimensionNumbers, PrecisionLike, dot_general_p, canonicalize_precision, select_n_p
from jax._src.numpy.lax_numpy import _einsum, asarray, bool_, _removechars, shape
from jax._src.util import partition_list, unzip2

from brainunit.math import squeeze, transpose, zeros_like
from ._compat_numpy_array_manipulation import func_array_manipulation
from ._compat_numpy_funcs_change_unit import funcs_change_unit_binary
from ._compat_numpy_funcs_keep_unit import funcs_keep_unit_unary
from .._base import (DIMENSIONLESS,
                     Quantity,
                     fail_for_dimension_mismatch,
                     is_unitless,
                     get_dim, _return_check_unitless, )
from .._misc import set_module_as

__all__ = [

  # constants
  'e', 'pi', 'inf',

  # data types
  'dtype', 'finfo', 'iinfo',

  # more
  'broadcast_arrays', 'broadcast_shapes',
  'einsum', 'gradient', 'intersect1d', 'nan_to_num', 'nanargmax', 'nanargmin',
  'rot90', 'tensordot', 'frexp',
]

# constants
# ---------
e = jnp.e
pi = jnp.pi
inf = jnp.inf

# data types
# ----------
dtype = jnp.dtype


@set_module_as('brainunit.math')
def finfo(a: Union[Quantity, jax.typing.ArrayLike]) -> jnp.finfo:
  if isinstance(a, Quantity):
    return jnp.finfo(a.value)
  else:
    return jnp.finfo(a)


@set_module_as('brainunit.math')
def iinfo(a: Union[Quantity, jax.typing.ArrayLike]) -> jnp.iinfo:
  if isinstance(a, Quantity):
    return jnp.iinfo(a.value)
  else:
    return jnp.iinfo(a)


# more
# ----
@set_module_as('brainunit.math')
def broadcast_arrays(*args: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, list[Array]]:
  """
  Broadcast any number of arrays against each other.

  Parameters
  ----------
  `*args` : array_likes
      The arrays to broadcast.

  Returns
  -------
  broadcasted : list of arrays
      These arrays are views on the original arrays.  They are typically
      not contiguous.  Furthermore, more than one element of a
      broadcasted array may refer to a single memory location. If you need
      to write to the arrays, make copies first. While you can set the
      ``writable`` flag True, writing to a single output value may end up
      changing more than one location in the output array.
  """
  leaves, tree = jax.tree.flatten(args)
  leaves = jnp.broadcast_arrays(*leaves)
  return jax.tree.unflatten(tree, leaves)


@set_module_as('brainunit.math')
def broadcast_shapes(*shapes):
  """
  Broadcast a sequence of array shapes.

  Parameters
  ----------
  *shapes : tuple of ints
      The shapes of the arrays to broadcast.

  Returns
  -------
  broadcast_shape : tuple of ints
      The shape of the broadcasted arrays.
  """
  return jnp.broadcast_shapes(*shapes)


# Enable other modules to override einsum_contact_path.
# Indexed by the type of the non constant dimension
_poly_einsum_handlers = {}  # type: ignore


def _default_poly_einsum_handler(*operands, **kwargs):
  dummy = collections.namedtuple('dummy', ['shape', 'dtype'])
  dummies = [dummy(tuple(d if type(d) is int else 8 for d in x.shape), x.dtype)
             if hasattr(x, 'dtype') else x for x in operands]
  mapping = {id(d): i for i, d in enumerate(dummies)}
  out_dummies, contractions = opt_einsum.contract_path(*dummies, **kwargs)
  contract_operands = [operands[mapping[id(d)]] for d in out_dummies]
  return contract_operands, contractions


def _quantity_dot_general(
    lhs: jax.ArrayLike | Quantity,
    rhs: jax.ArrayLike | Quantity,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike = None,
    preferred_element_type: jax.DTypeLike | None = None
) -> Array | Quantity:
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  cdims = (api_util._ensure_index_tuple(lhs_contract),
           api_util._ensure_index_tuple(rhs_contract))
  bdims = (api_util._ensure_index_tuple(lhs_batch),
           api_util._ensure_index_tuple(rhs_batch))
  preferred_element_type = (
    None if preferred_element_type is None else
    dtypes.canonicalize_dtype(np.dtype(preferred_element_type)))
  if isinstance(lhs, Quantity) and isinstance(rhs, Quantity):
    return _return_check_unitless(
      Quantity(
        dot_general_p.bind(lhs.value, rhs.value,
                           dimension_numbers=(cdims, bdims),
                           precision=canonicalize_precision(precision),
                           preferred_element_type=preferred_element_type),
        dim=lhs.dim * rhs.dim))
  elif isinstance(lhs, Quantity):
    return Quantity(
      dot_general_p.bind(lhs.value, rhs,
                         dimension_numbers=(cdims, bdims),
                         precision=canonicalize_precision(precision),
                         preferred_element_type=preferred_element_type),
      dim=lhs.dim)
  elif isinstance(rhs, Quantity):
    return Quantity(
      dot_general_p.bind(lhs, rhs.value,
                         dimension_numbers=(cdims, bdims),
                         precision=canonicalize_precision(precision),
                         preferred_element_type=preferred_element_type),
      dim=rhs.dim)
  else:
    return dot_general_p.bind(lhs, rhs,
                              dimension_numbers=(cdims, bdims),
                              precision=canonicalize_precision(precision),
                              preferred_element_type=preferred_element_type)


def _select(
    pred: jax.ArrayLike,
    on_true: jax.ArrayLike | Quantity,
    on_false: jax.ArrayLike | Quantity
) -> Array | Quantity:
  if isinstance(on_true, Quantity) and isinstance(on_false, Quantity):
    fail_for_dimension_mismatch(on_true, on_false, 'select')
    return Quantity(select_n_p.bind(pred, on_true.value, on_false.value), dim=on_true.dim)
  else:
    return select_n_p.bind(pred, on_false, on_true)


def _einsum(
    operands: Sequence[Quantity | Array],
    contractions: Sequence[tuple[tuple[int, ...], frozenset[str], str]],
    precision,
    preferred_element_type,
    _dot_general=_quantity_dot_general,
):
  dtypes.check_user_dtype_supported(preferred_element_type, "einsum")
  new_operands = []
  for operand in operands:
    if isinstance(operand, Array):
      new_operands.append(asarray(operand))
    elif isinstance(operand, Quantity):
      new_operands.append(operand)

  operands = new_operands
  if preferred_element_type is None:
    preferred_element_type, output_weak_type = dtypes.result_type(*operands, return_weak_type_flag=True)
  else:
    output_weak_type = False

  def sum(x, axes):
    if dtypes.result_type(x, preferred_element_type) != x.dtype:
      x = x.astype(preferred_element_type)
    dim = None
    if isinstance(x, Quantity):
      dim = x.dim
      x = x.value
    x = lax.reduce(x, np.array(0, x.dtype), lax.add if x.dtype != bool_ else lax.bitwise_or, axes)
    return Quantity(x, dim=dim) if dim is not None else x

  def sum_uniques(operand, names, uniques):
    if uniques:
      axes = [names.index(name) for name in uniques]
      operand = sum(operand, axes)
      names = _removechars(names, uniques)
    return operand, names

  def sum_repeats(operand, names, counts, keep_names):
    for name, count in counts.items():
      if count > 1:
        axes = [i for i, n in enumerate(names) if n == name]
        eye = lax_internal._delta(np.dtype('bool'), operand.shape, axes)
        operand = _select(eye,
                          operand,
                          zeros_like(operand))
        if name not in keep_names:
          operand = sum(operand, axes)
          names = names.replace(name, '')
        else:
          operand = sum(operand, axes[:-1])
          names = names.replace(name, '', count - 1)
    return operand, names

  def filter_singleton_dims(operand, names, other_shape, other_names):
    eq = core.definitely_equal
    keep = [not eq(operand.shape[i], 1) or j == -1 or eq(other_shape[j], 1)
            for i, j in enumerate(map(other_names.find, names))]
    sqez_axes, keep_axes = partition_list(keep, list(range(operand.ndim)))
    return squeeze(operand, sqez_axes), "".join(names[i] for i in keep_axes)

  for operand_indices, contracted_names_set, einstr in contractions:
    contracted_names = sorted(contracted_names_set)
    input_str, result_names = einstr.split('->')
    input_names = input_str.split(',')

    # switch on the number of operands to be processed in this loop iteration.
    # every case here sets 'operand' and 'names'.
    if len(operand_indices) == 1:
      operand = operands.pop(operand_indices[0])
      names, = input_names
      counts = collections.Counter(names)

      # sum out unique contracted indices with a single reduce-sum
      uniques = [name for name in contracted_names if counts[name] == 1]
      operand, names = sum_uniques(operand, names, uniques)

      # for every repeated index, do a contraction against an identity matrix
      operand, names = sum_repeats(operand, names, counts, result_names)

    elif len(operand_indices) == 2:
      lhs, rhs = map(operands.pop, operand_indices)
      lhs_names, rhs_names = input_names

      # handle cases where one side of a contracting or batch dimension is 1
      # but its counterpart is not.
      lhs, lhs_names = filter_singleton_dims(lhs, lhs_names, shape(rhs),
                                             rhs_names)
      rhs, rhs_names = filter_singleton_dims(rhs, rhs_names, shape(lhs),
                                             lhs_names)

      lhs_counts = collections.Counter(lhs_names)
      rhs_counts = collections.Counter(rhs_names)

      # sum out unique contracted indices in lhs and rhs
      lhs_uniques = [name for name in contracted_names
                     if lhs_counts[name] == 1 and rhs_counts[name] == 0]
      lhs, lhs_names = sum_uniques(lhs, lhs_names, lhs_uniques)

      rhs_uniques = [name for name in contracted_names
                     if rhs_counts[name] == 1 and lhs_counts[name] == 0]
      rhs, rhs_names = sum_uniques(rhs, rhs_names, rhs_uniques)

      # for every repeated index, contract against an identity matrix
      lhs, lhs_names = sum_repeats(lhs, lhs_names, lhs_counts,
                                   result_names + rhs_names)
      rhs, rhs_names = sum_repeats(rhs, rhs_names, rhs_counts,
                                   result_names + lhs_names)

      lhs_or_rhs_names = set(lhs_names) | set(rhs_names)
      contracted_names = [x for x in contracted_names if x in lhs_or_rhs_names]
      lhs_and_rhs_names = set(lhs_names) & set(rhs_names)
      batch_names = [x for x in result_names if x in lhs_and_rhs_names]

      lhs_batch, rhs_batch = unzip2((lhs_names.find(n), rhs_names.find(n))
                                    for n in batch_names)

      # NOTE(mattjj): this can fail non-deterministically in python3, maybe
      # due to opt_einsum
      assert config.dynamic_shapes.value or all(
        name in lhs_names and name in rhs_names and
        lhs.shape[lhs_names.index(name)] == rhs.shape[rhs_names.index(name)]
        for name in contracted_names), (
        "Incompatible reduction dimensions: "
        f"lhs.shape={lhs.shape} lhs_names={lhs_names} "
        f"rhs.shape={rhs.shape} rhs_names={rhs_names}")

      # contract using dot_general
      batch_names_str = ''.join(batch_names)
      lhs_cont, rhs_cont = unzip2((lhs_names.index(n), rhs_names.index(n))
                                  for n in contracted_names)
      deleted_names = batch_names_str + ''.join(contracted_names)
      remaining_lhs_names = _removechars(lhs_names, deleted_names)
      remaining_rhs_names = _removechars(rhs_names, deleted_names)
      # Try both orders of lhs and rhs, in the hope that one of them means we
      # don't need an explicit transpose. opt_einsum likes to contract from
      # right to left, so we expect (rhs,lhs) to have the best chance of not
      # needing a transpose.
      names = batch_names_str + remaining_rhs_names + remaining_lhs_names
      if names == result_names:
        dimension_numbers = ((rhs_cont, lhs_cont), (rhs_batch, lhs_batch))
        operand = _dot_general(rhs, lhs, dimension_numbers, precision,
                               preferred_element_type=preferred_element_type)
      else:
        names = batch_names_str + remaining_lhs_names + remaining_rhs_names
        dimension_numbers = ((lhs_cont, rhs_cont), (lhs_batch, rhs_batch))
        operand = _dot_general(lhs, rhs, dimension_numbers, precision,
                               preferred_element_type=preferred_element_type)
    else:
      raise NotImplementedError  # if this is actually reachable, open an issue!

    # the resulting 'operand' with axis labels 'names' should be a permutation
    # of the desired result
    assert len(names) == len(result_names) == len(set(names))
    assert set(names) == set(result_names)
    if names != result_names:
      perm = tuple(names.index(name) for name in result_names)
      operand = transpose(operand, perm)
    operands.append(operand)  # used in next iteration

  if isinstance(operands[0], Quantity):
    return Quantity(lax_internal._convert_element_type(operands[0].value, preferred_element_type, output_weak_type),
                    dim=operands[0].dim)
  else:
    return lax_internal._convert_element_type(operands[0], preferred_element_type, output_weak_type)


def einsum(
    subscripts: str,
    /,
    *operands: Union[Quantity, jax.Array],
    out: None = None,
    optimize: Union[str, bool] = "optimal",
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None,
    _dot_general: Callable[..., jax.Array] = jax.lax.dot_general,
) -> Union[jax.Array, Quantity]:
  """
  Evaluates the Einstein summation convention on the operands.

  Parameters
  ----------
  subscripts : str
    string containing axes names separated by commas.
  *operands : array_like, Quantity, optional
    sequence of one or more arrays or quantities corresponding to the subscripts.
  optimize : {False, True, 'optimal'}, optional
    determine whether to optimize the order of computation. In JAX
    this defaults to ``"optimize"`` which produces optimized expressions via
    the opt_einsum_ package.
  precision : either ``None`` (default),
    which means the default precision for the backend
    a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
    ``Precision.HIGH`` or ``Precision.HIGHEST``).
  preferred_element_type : either ``None`` (default)
    which means the default accumulation type for the input types,
    or a datatype, indicating to accumulate results to and return a result with that datatype.
  out : {None}, optional
    This parameter is not supported in JAX.
  _dot_general : callable, optional
    optionally override the ``dot_general`` callable used by ``einsum``.
    This parameter is experimental, and may be removed without warning at any time.

  Returns
  -------
  output : Quantity or jax.Array
    The calculation based on the Einstein summation convention.
  """
  operands = (subscripts, *operands)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.einsum is not supported.")
  spec = operands[0] if isinstance(operands[0], str) else None
  optimize = 'optimal' if optimize is True else optimize

  # Allow handling of shape polymorphism
  non_constant_dim_types = {
    type(d) for op in operands if not isinstance(op, str)
    for d in np.shape(op) if not core.is_constant_dim(d)
  }
  if not non_constant_dim_types:
    contract_path = opt_einsum.contract_path
  else:
    ty = next(iter(non_constant_dim_types))
    contract_path = _poly_einsum_handlers.get(ty, _default_poly_einsum_handler)
  # using einsum_call=True here is an internal api for opt_einsum... sorry
  operands, contractions = contract_path(
    *operands, einsum_call=True, use_blas=True, optimize=optimize)

  contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)

  einsum = jit(_einsum, static_argnums=(1, 2, 3, 4), inline=True)
  if spec is not None:
    einsum = jax.named_call(einsum, name=spec)
  return einsum(operands, contractions, precision,  # type: ignore[operator]
                preferred_element_type, _quantity_dot_general)


@set_module_as('brainunit.math')
def gradient(
    f: Union[jax.typing.ArrayLike, Quantity],
    *varargs: Union[jax.typing.ArrayLike, Quantity],
    axis: Union[int, Sequence[int], None] = None,
    edge_order: Union[int, None] = None,
) -> Union[jax.Array, list[jax.Array], Quantity, list[Quantity]]:
  """
  Computes the gradient of a scalar field.

  Return the gradient of an N-dimensional array.

  The gradient is computed using second order accurate central differences
  in the interior points and either first or second order accurate one-sides
  (forward or backwards) differences at the boundaries.
  The returned gradient hence has the same shape as the input array.

  Parameters
  ----------
  f : array_like, Quantity
    An N-dimensional array containing samples of a scalar function.
  varargs : list of scalar or array, optional
    Spacing between f values. Default unitary spacing for all dimensions.
    Spacing can be specified using:

    1. single scalar to specify a sample distance for all dimensions.
    2. N scalars to specify a constant sample distance for each dimension.
       i.e. `dx`, `dy`, `dz`, ...
    3. N arrays to specify the coordinates of the values along each
       dimension of F. The length of the array must match the size of
       the corresponding dimension
    4. Any combination of N scalars/arrays with the meaning of 2. and 3.

    If `axis` is given, the number of varargs must equal the number of axes.
    Default: 1.
  edge_order : {1, 2}, optional
    Gradient is calculated using N-th order accurate differences
    at the boundaries. Default: 1.
  axis : None or int or tuple of ints, optional
    Gradient is calculated only along the given axis or axes
    The default (axis = None) is to calculate the gradient for all the axes
    of the input array. axis may be negative, in which case it counts from
    the last to the first axis.

  Returns
  -------
  gradient : ndarray or list of ndarray or Quantity
    A list of ndarrays (or a single ndarray if there is only one dimension)
    corresponding to the derivatives of f with respect to each dimension.
    Each derivative has the same shape as f.
  """
  if edge_order is not None:
    raise NotImplementedError("The 'edge_order' argument to jnp.gradient is not supported.")

  if len(varargs) == 0:
    if isinstance(f, Quantity) and not is_unitless(f):
      return Quantity(jnp.gradient(f.value, axis=axis), dim=f.dim)
    else:
      return jnp.gradient(f)
  elif len(varargs) == 1:
    unit = get_dim(f) / get_dim(varargs[0])
    if unit is None or unit == DIMENSIONLESS:
      return jnp.gradient(f, varargs[0], axis=axis)
    else:
      return [Quantity(r, dim=unit) for r in jnp.gradient(f.value, varargs[0].value, axis=axis)]
  else:
    unit_list = [get_dim(f) / get_dim(v) for v in varargs]
    f = f.value if isinstance(f, Quantity) else f
    varargs = [v.value if isinstance(v, Quantity) else v for v in varargs]
    result_list = jnp.gradient(f, *varargs, axis=axis)
    return [Quantity(r, dim=unit) if unit is not None else r for r, unit in zip(result_list, unit_list)]


@set_module_as('brainunit.math')
def intersect1d(
    ar1: Union[jax.typing.ArrayLike, Quantity],
    ar2: Union[jax.typing.ArrayLike, Quantity],
    assume_unique: bool = False,
    return_indices: bool = False
) -> Union[jax.Array, Quantity, tuple[Union[jax.Array, Quantity], jax.Array, jax.Array]]:
  """
  Find the intersection of two arrays.

  Return the sorted, unique values that are in both of the input arrays.

  Parameters
  ----------
  ar1, ar2 : array_like, Quantity
    Input arrays. Will be flattened if not already 1D.
  assume_unique : bool
    If True, the input arrays are both assumed to be unique, which
    can speed up the calculation.  If True but ``ar1`` or ``ar2`` are not
    unique, incorrect results and out-of-bounds indices could result.
    Default is False.
  return_indices : bool
    If True, the indices which correspond to the intersection of the two
    arrays are returned. The first instance of a value is used if there are
    multiple. Default is False.

  Returns
  -------
  intersect1d : ndarray, Quantity
    Sorted 1D array of common and unique elements.
  comm1 : ndarray
    The indices of the first occurrences of the common values in `ar1`.
    Only provided if `return_indices` is True.
  comm2 : ndarray
    The indices of the first occurrences of the common values in `ar2`.
    Only provided if `return_indices` is True.
  """
  fail_for_dimension_mismatch(ar1, ar2, 'intersect1d')
  unit = None
  if isinstance(ar1, Quantity):
    unit = ar1.dim
  ar1 = ar1.value if isinstance(ar1, Quantity) else ar1
  ar2 = ar2.value if isinstance(ar2, Quantity) else ar2
  result = jnp.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
  if return_indices:
    if unit is not None:
      return Quantity(result[0], dim=unit), result[1], result[2]
    else:
      return result
  else:
    if unit is not None:
      return Quantity(result, dim=unit)
    else:
      return result


@set_module_as('brainunit.math')
def nan_to_num(
    x: Union[jax.typing.ArrayLike, Quantity],
    nan: float | Quantity = None,
    posinf: float | Quantity = None,
    neginf: float | Quantity = None
) -> Union[jax.Array, Quantity]:
  """
  Replace NaN with zero and infinity with large finite numbers (default
  behaviour) or with the numbers defined by the user using the `nan`,
  `posinf` and/or `neginf` keywords.

  If `x` is inexact, NaN is replaced by zero or by the user defined value in
  `nan` keyword, infinity is replaced by the largest finite floating point
  values representable by ``x.dtype`` or by the user defined value in
  `posinf` keyword and -infinity is replaced by the most negative finite
  floating point values representable by ``x.dtype`` or by the user defined
  value in `neginf` keyword.

  For complex dtypes, the above is applied to each of the real and
  imaginary components of `x` separately.

  If `x` is not inexact, then no replacements are made.

  Parameters
  ----------
  x : scalar, array_like or Quantity
    Input data.
  nan : int, float, optional
    Value to be used to fill NaN values. If no value is passed
    then NaN values will be replaced with 0.0.
  posinf : int, float, optional
    Value to be used to fill positive infinity values. If no value is
    passed then positive infinity values will be replaced with a very
    large number.
  neginf : int, float, optional
    Value to be used to fill negative infinity values. If no value is
    passed then negative infinity values will be replaced with a very
    small (or negative) number.

  Returns
  -------
  out : ndarray, Quantity
    `x`, with the non-finite values replaced. If `copy` is False, this may
    be `x` itself.
  """
  if isinstance(x, Quantity):
    nan = Quantity(0.0, dim=x.dim) if nan is None else nan
    posinf = Quantity(jnp.finfo(x.dtype).max, dim=x.dim) if posinf is None else posinf
    neginf = Quantity(jnp.finfo(x.dtype).min, dim=x.dim) if neginf is None else neginf
    return Quantity(jnp.nan_to_num(x.value, nan=nan.value, posinf=posinf.value, neginf=neginf.value), dim=x.dim)
  else:
    nan = 0.0 if nan is None else nan
    posinf = jnp.inf if posinf is None else posinf
    neginf = -jnp.inf if neginf is None else neginf
    return jnp.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


@set_module_as('brainunit.math')
def rot90(
    m: Union[jax.typing.ArrayLike, Quantity],
    k: int = 1,
    axes: Tuple[int, int] = (0, 1)
) -> Union[
  jax.Array, Quantity]:
  """
  Rotate an array by 90 degrees in the plane specified by axes.

  Rotation direction is from the first towards the second axis.

  Parameters
  ----------
  m : array_like, Quantity
    Array of two or more dimensions.
  k : integer
    Number of times the array is rotated by 90 degrees.
  axes : (2,) array_like
    The array is rotated in the plane defined by the axes.
    Axes must be different.

  Returns
  -------
  y : ndarray, Quantity
    A rotated view of `m`.

    This is a quantity if `m` is a quantity.
  """
  return funcs_keep_unit_unary(jnp.rot90, m, k=k, axes=axes)


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
  return funcs_change_unit_binary(jnp.tensordot,
                                  lambda x, y: x * y,
                                  a, b,
                                  axes=axes,
                                  precision=precision,
                                  preferred_element_type=preferred_element_type)


@set_module_as('brainunit.math')
def nanargmax(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: int = None,
    keepdims: bool = False
) -> jax.Array:
  """
  Return the indices of the maximum values in the specified axis ignoring
  NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the
  results cannot be trusted if a slice contains only NaNs and -Infs.


  Parameters
  ----------
  a : array_like, Quantity
    Input data.
  axis : int, optional
    Axis along which to operate.  By default flattened input is used.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the array.

  Returns
  -------
  index_array : ndarray
    An array of indices or a single index value.
  """
  return func_array_manipulation(jnp.nanargmax,
                                 a,
                                 return_quantity=False,
                                 axis=axis,
                                 keepdims=keepdims)


@set_module_as('brainunit.math')
def nanargmin(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: int = None,
    keepdims: bool = False
) -> jax.Array:
  """
  Return the indices of the minimum values in the specified axis ignoring
  NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the results
  cannot be trusted if a slice contains only NaNs and Infs.

  Parameters
  ----------
  a : array_like, Quantity
    Input data.
  axis : int, optional
    Axis along which to operate.  By default flattened input is used.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the array.

  Returns
  -------
  index_array : ndarray
    An array of indices or a single index value.
  """
  return func_array_manipulation(jnp.nanargmin,
                                 a,
                                 return_quantity=False,
                                 axis=axis,
                                 keepdims=keepdims)


@set_module_as('brainunit.math')
def frexp(
    x: Union[Quantity, jax.typing.ArrayLike]
) -> Tuple[jax.Array, jax.Array]:
  """
  Decompose the elements of x into mantissa and twos exponent.

  Returns (`mantissa`, `exponent`), where ``x = mantissa * 2**exponent``.
  The mantissa lies in the open interval(-1, 1), while the twos
  exponent is a signed integer.

  Parameters
  ----------
  x : array_like, Quantity
    Array of numbers to be decomposed.

  Returns
  -------
  mantissa : ndarray
    Floating values between -1 and 1.
    This is a scalar if `x` is a scalar.
  exponent : ndarray
    Integer exponents of 2.
    This is a scalar if `x` is a scalar.
  """
  assert not isinstance(x, Quantity) or is_unitless(x), "Input must be unitless"
  x = x.value if isinstance(x, Quantity) else x
  return jnp.frexp(x)
