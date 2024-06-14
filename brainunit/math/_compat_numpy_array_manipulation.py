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

from collections.abc import Sequence
from typing import (Union, Optional, Tuple, List, Any)

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import tree_map

from .._base import Quantity
from .._misc import set_module_as

__all__ = [
  # array manipulation
  'reshape', 'moveaxis', 'transpose', 'swapaxes', 'row_stack',
  'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',
  'split', 'dsplit', 'hsplit', 'vsplit', 'tile', 'repeat', 'unique',
  'append', 'flip', 'fliplr', 'flipud', 'roll', 'atleast_1d', 'atleast_2d',
  'atleast_3d', 'expand_dims', 'squeeze', 'sort', 'argsort', 'argmax', 'argmin',
  'argwhere', 'nonzero', 'flatnonzero', 'searchsorted', 'extract',
  'count_nonzero', 'max', 'min', 'amax', 'amin', 'block', 'compress',
  'diagflat', 'diagonal', 'choose', 'ravel',
]


# array manipulation
# ------------------


def _as_jax_array_(obj):
  return obj.value if isinstance(obj, Quantity) else obj


def _is_leaf(a):
  return isinstance(a, Quantity)


def func_array_manipulation(fun, *args, return_quantity=True, **kwargs) -> Any:
  unit = None
  if isinstance(args[0], Quantity):
    unit = args[0].dim
  elif isinstance(args[0], tuple):
    if len(args[0]) == 1:
      unit = args[0][0].dim if isinstance(args[0][0], Quantity) else None
    elif len(args[0]) == 2:
      # check all args[0] have the same unit
      if all(isinstance(a, Quantity) for a in args[0]):
        if all(a.dim == args[0][0].dim for a in args[0]):
          unit = args[0][0].dim
        else:
          raise ValueError(f'Units do not match for {fun.__name__} operation.')
      elif all(not isinstance(a, Quantity) for a in args[0]):
        unit = None
      else:
        raise ValueError(f'Units do not match for {fun.__name__} operation.')
  args = tree_map(_as_jax_array_, args, is_leaf=_is_leaf)
  out = None
  if len(kwargs):
    # compatible with PyTorch syntax
    if 'dim' in kwargs:
      kwargs['axis'] = kwargs.pop('dim')
    if 'keepdim' in kwargs:
      kwargs['keepdims'] = kwargs.pop('keepdim')
    # compatible with TensorFlow syntax
    if 'keep_dims' in kwargs:
      kwargs['keepdims'] = kwargs.pop('keep_dims')
    # compatible with NumPy/PyTorch syntax
    if 'out' in kwargs:
      out = kwargs.pop('out')
      if out is not None and not isinstance(out, Quantity):
        raise TypeError(f'"out" must be an instance of brainpy Array. While we got {type(out)}')
    # format
    kwargs = tree_map(_as_jax_array_, kwargs, is_leaf=_is_leaf)

  if not return_quantity:
    unit = None

  r = fun(*args, **kwargs)
  if unit is not None:
    if isinstance(r, (list, tuple)):
      return [Quantity(rr, dim=unit) for rr in r]
    else:
      if out is None:
        return Quantity(r, dim=unit)
      else:
        out.value = r
  if out is None:
    return r
  else:
    out.value = r


@set_module_as('brainunit.math')
def reshape(
    a: Union[Array, Quantity],
    shape: Union[int, Tuple[int, ...]],
    order: str = 'C'
) -> Union[Array, Quantity]:
  """
  Gives a new shape to a quantity or an array without changing its data.

  Parameters
  ----------
  a : array_like, Quantity
    Array to be reshaped.
  shape : int or tuple of ints
    The new shape should be compatible with the original shape. If
    an integer, then the result will be a 1-D array of that length.
    One shape dimension can be -1. In this case, the value is
    inferred from the length of the array and remaining dimensions.
  order : {'C', 'F', 'A'}, optional
    Read the elements of `a` using this index order, and place the
    elements into the reshaped array using this index order.  'C'
    means to read / write the elements using C-like index order,
    with the last axis index changing fastest, back to the first
    axis index changing slowest. 'F' means to read / write the
    elements using Fortran-like index order, with the first index
    changing fastest, and the last index changing slowest. Note that
    the 'C' and 'F' options take no account of the memory layout of
    the underlying array, and only refer to the order of indexing.
    'A' means to read / write the elements in Fortran-like index
    order if `a` is Fortran *contiguous* in memory, C-like order
    otherwise.

  Returns
  -------
  reshaped_array : ndarray, Quantity
    This will be a new view object if possible; otherwise, it will
    be a copy.  Note there is no guarantee of the *memory layout* (C- or
    Fortran- contiguous) of the returned array.
  """
  return func_array_manipulation(jnp.reshape, a, shape, order=order)


@set_module_as('brainunit.math')
def moveaxis(
    a: Union[Array, Quantity],
    source: Union[int, Tuple[int, ...]],
    destination: Union[int, Tuple[int, ...]]
) -> Union[Array, Quantity]:
  """
  Moves axes of a quantity or an array to new positions.
  Other axes remain in their original order.

  Parameters
  ----------
  a : array_like, Quantity
    The array whose axes should be reordered.
  source : int or sequence of int
    Original positions of the axes to move. These must be unique.
  destination : int or sequence of int
    Destination positions for each of the original axes. These must also be
    unique.

  Returns
  -------
  result : ndarray, Quantity
    Array with moved axes. This array is a view of the input array.
  """
  return func_array_manipulation(jnp.moveaxis, a, source, destination)


@set_module_as('brainunit.math')
def transpose(
    a: Union[Array, Quantity],
    axes: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[Array, Quantity]:
  """
  Permute the dimensions of a quantity or an array.

  Parameters
  ----------
  a : array_like, Quantity
    Input array.
  axes : list of ints, optional
    By default, reverse the dimensions, otherwise permute the axes
    according to the values given.

  Returns
  -------
  p : ndarray, Quantity
    `a` with its axes permuted.  A view is returned whenever
    possible.
  """
  return func_array_manipulation(jnp.transpose, a, axes)


@set_module_as('brainunit.math')
def swapaxes(
    a: Union[Array, Quantity],
    axis1: int,
    axis2: int
) -> Union[Array, Quantity]:
  """
  Interchange two axes of a quantity or an array.

  Parameters
  ----------
  a : array_like, Quantity
    Input array.
  axis1 : int
    First axis.
  axis2 : int
    Second axis.

  Returns
  -------
  a_swapped : ndarray, Quantity
    a new array where the axes are swapped.
  """
  return func_array_manipulation(jnp.swapaxes, a, axis1, axis2)


@set_module_as('brainunit.math')
def concatenate(
    arrays: Union[Sequence[Array], Sequence[Quantity]],
    axis: Optional[int] = None,
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[Array, Quantity]:
  """
  Join a sequence of quantities or arrays along an existing axis.

  Parameters
  ----------
  arrays : sequence of array_like, Quantity
    The arrays must have the same shape, except in the dimension corresponding
    to `axis` (the first, by default).
  axis : int, optional
    The axis along which the arrays will be joined.  Default is 0.
  dtype : dtype, optional
    If provided, the concatenation will be done using this dtype. Otherwise, the
    array with the highest precision will be used.

  Returns
  -------
  res : ndarray, Quantity
    The concatenated array. The type of the array is the same as that of the
    first array passed in.
  """
  return func_array_manipulation(jnp.concatenate, arrays, axis=axis, dtype=dtype)


@set_module_as('brainunit.math')
def stack(
    arrays: Union[Sequence[Array], Sequence[Quantity]],
    axis: int = 0,
    out: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[Array, Quantity]:
  """
  Join a sequence of quantities or arrays along a new axis.

  Parameters
  ----------
  arrays : sequence of array_like, Quantity
    The arrays must have the same shape.
  axis : int, optional
    The axis in the result array along which the input arrays are stacked.
  out : Quantity, jax.typing.ArrayLike, optional
    If provided, the destination to place the result. The shape must be
    correct, matching that of what stack would have returned if no out
    argument were specified.
  dtype : dtype, optional
    If provided, the concatenation will be done using this dtype. Otherwise, the
    array with the highest precision will be used.

  Returns
  -------
  res : ndarray, Quantity
    The stacked array has one more dimension than the input arrays.
  """
  return func_array_manipulation(jnp.stack, arrays, axis=axis, out=out, dtype=dtype)


@set_module_as('brainunit.math')
def vstack(
    tup: Union[Sequence[Array], Sequence[Quantity]],
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[Array, Quantity]:
  """
  Stack quantities or arrays in sequence vertically (row wise).

  Parameters
  ----------
  tup : sequence of array_like, Quantity
    The arrays must have the same shape along all but the first axis.
  dtype : dtype, optional
    If provided, the concatenation will be done using this dtype. Otherwise, the
    array with the highest precision will be used.

  Returns
  -------
  res : ndarray, Quantity
    The array formed by stacking the given arrays.
  """
  return func_array_manipulation(jnp.vstack, tup, dtype=dtype)


row_stack = vstack


@set_module_as('brainunit.math')
def hstack(
    arrays: Union[Sequence[Array], Sequence[Quantity]],
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[Array, Quantity]:
  """
  Stack quantities arrays in sequence horizontally (column wise).

  Parameters
  ----------
  arrays : sequence of array_like, Quantity
    The arrays must have the same shape along all but the second axis.
  dtype : dtype, optional
    If provided, the concatenation will be done using this dtype. Otherwise, the
    array with the highest precision will be used.

  Returns
  -------
  res : ndarray, Quantity
    The array formed by stacking the given arrays.
  """
  return func_array_manipulation(jnp.hstack, arrays, dtype=dtype)


@set_module_as('brainunit.math')
def dstack(
    arrays: Union[Sequence[Array], Sequence[Quantity]],
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[Array, Quantity]:
  """
  Stack quantities or arrays in sequence depth wise (along third axis).

  Parameters
  ----------
  arrays : sequence of array_like, Quantity
    The arrays must have the same shape along all but the third axis.
  dtype : dtype, optional
    If provided, the concatenation will be done using this dtype. Otherwise, the
    array with the highest precision will be used.

  Returns
  -------
  res : ndarray, Quantity
    The array formed by stacking the given arrays.
  """
  return func_array_manipulation(jnp.dstack, arrays, dtype=dtype)


@set_module_as('brainunit.math')
def column_stack(
    tup: Union[Sequence[Array], Sequence[Quantity]]
) -> Union[Array, Quantity]:
  """
  Stack 1-D arrays as columns into a 2-D array.

  Take a sequence of 1-D arrays and stack them as columns to make a single
  2-D array. 2-D arrays are stacked as-is, just like with hstack.

  Parameters
  ----------
  tup : sequence of 1-D array_like, Quantity
    1-D arrays to stack as columns.

  Returns
  -------
  res : ndarray, Quantity
    The array formed by stacking the given arrays.
  """
  return func_array_manipulation(jnp.column_stack, tup)


@set_module_as('brainunit.math')
def split(
    a: Union[Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]],
    axis: int = 0
) -> Union[List[Array], List[Quantity]]:
  """
  Split quantity or array into a list of multiple sub-arrays.

  Parameters
  ----------
  a : array_like, Quantity
    Array to be divided into sub-arrays.
  indices_or_sections : int or 1-D array
    If `indices_or_sections` is an integer, N, the array will be divided into
    N equal arrays along `axis`. If such a split is not possible, an error is
    raised. If `indices_or_sections` is a 1-D array of sorted integers, the
    entries indicate where along `axis` the array is split. For example,
    `[2, 3]` would, for `axis=0`, result in
    - `a[:2]`
    - `a[2:3]`
    - `a[3:]`
  axis : int, optional
    The axis along which to split, default is 0.

  Returns
  -------
  res : list of ndarrays, Quantity
    A list of sub-arrays.
  """
  return func_array_manipulation(jnp.split, a, indices_or_sections, axis=axis)


@set_module_as('brainunit.math')
def dsplit(
    a: Union[Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]]
) -> Union[List[Array], List[Quantity]]:
  """
  Split a quantity or an array into multiple sub-arrays along the 3rd axis (depth).

  Parameters
  ----------
  a : array_like, Quantity
    Array to be divided into sub-arrays.
  indices_or_sections : int or 1-D array
    If `indices_or_sections` is an integer, N, the array will be divided into
    N equal arrays along the third axis (depth). If such a split is not possible,
    an error is raised. If `indices_or_sections` is a 1-D array of sorted integers,
    the entries indicate where along the third axis the array is split.

  Returns
  -------
  res : list of ndarrays, Quantity
    A list of sub-arrays.
  """
  return func_array_manipulation(jnp.dsplit, a, indices_or_sections)


@set_module_as('brainunit.math')
def hsplit(
    a: Union[Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]]
) -> Union[List[Array], List[Quantity]]:
  """
  Split a quantity or an array into multiple sub-arrays horizontally (column-wise).

  Parameters
  ----------
  a : array_like, Quantity
    Array to be divided into sub-arrays.
  indices_or_sections : int or 1-D array
    If `indices_or_sections` is an integer, N, the array will be divided into
    N equal arrays along the second axis. If such a split is not possible, an
    error is raised. If `indices_or_sections` is a 1-D array of sorted integers,
    the entries indicate where along the second axis the array is split.

  Returns
  -------
  res : list of ndarrays, Quantity
    A list of sub-arrays.
  """
  return func_array_manipulation(jnp.hsplit, a, indices_or_sections)


@set_module_as('brainunit.math')
def vsplit(
    a: Union[Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]]
) -> Union[List[Array], List[Quantity]]:
  """
  Split a quantity or an array into multiple sub-arrays vertically (row-wise).

  Parameters
  ----------
  a : array_like, Quantity
    Array to be divided into sub-arrays.
  indices_or_sections : int or 1-D array
    If `indices_or_sections` is an integer, N, the array will be divided into
    N equal arrays along the first axis. If such a split is not possible, an
    error is raised. If `indices_or_sections` is a 1-D array of sorted integers,
    the entries indicate where along the first axis the array is split.

  Returns
  -------
  res : list of ndarrays, Quantity
    A list of sub-arrays.
  """
  return func_array_manipulation(jnp.vsplit, a, indices_or_sections)


@set_module_as('brainunit.math')
def tile(
    A: Union[Array, Quantity],
    reps: Union[int, Tuple[int, ...]]
) -> Union[Array, Quantity]:
  """
  Construct a quantity or an array by repeating A the number of times given by reps.

  Parameters
  ----------
  A : array_like, Quantity
    The input array.
  reps : array_like
    The number of repetitions of A along each axis.

  Returns
  -------
  res : ndarray, Quantity
    The tiled output array.
  """
  return func_array_manipulation(jnp.tile, A, reps)


@set_module_as('brainunit.math')
def repeat(
    a: Union[Array, Quantity],
    repeats: Union[int, Tuple[int, ...]],
    axis: Optional[int] = None,
    total_repeat_length: Optional[int] = None
) -> Union[Array, Quantity]:
  """
  Repeat elements of a quantity or an array.

  Parameters
  ----------
  a : array_like, Quantity
    Input array.
  repeats : int or tuple of ints
    The number of repetitions for each element. `repeats` is broadcasted to fit the shape of the given axis.
  axis : int, optional
    The axis along which to repeat values. By default, use the flattened input array, and return a flat output array.
  total_repeat_length : int, optional
    The total length of the repeated array. If `total_repeat_length` is not None, the output array
    will have the length of `total_repeat_length`.

  Returns
  -------
  res : ndarray, Quantity
    Output array which has the same shape as `a`, except along the given axis.
  """
  return func_array_manipulation(jnp.repeat, a, repeats, axis=axis, total_repeat_length=total_repeat_length)


@set_module_as('brainunit.math')
def unique(
    a: Union[Array, Quantity],
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: Optional[int] = None,
    *,
    equal_nan: bool = False,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = None
) -> Union[Array, Quantity]:
  """
  Find the unique elements of a quantity or an array.

  Parameters
  ----------
  a : array_like, Quantity
    Input array.
  return_index : bool, optional
    If True, also return the indices of `a` (along the specified axis, if provided) that result in the unique array.
  return_inverse : bool, optional
    If True, also return the indices of the unique array (for the specified axis, if provided)
    that can be used to reconstruct `a`.
  return_counts : bool, optional
    If True, also return the number of times each unique item appears in `a`.
  axis : int, optional
    The axis along which to operate. If None, the array is flattened before use. Default is None.
  equal_nan : bool, optional
    Whether to compare NaN's as equal. If True, NaN's in `a` will be considered equal to each other in the unique array.
  size : int, optional
    The length of the output array. If `size` is not None, the output array will have the length of `size`.
  fill_value : scalar, optional
    The value to use for missing values. If `fill_value` is not None, the output array will have the length of `size`.

  Returns
  -------
  res : ndarray, Quantity
    The sorted unique values.
  """
  return func_array_manipulation(jnp.unique,
                                 a,
                                 return_index=return_index,
                                 return_inverse=return_inverse,
                                 return_counts=return_counts,
                                 axis=axis,
                                 equal_nan=equal_nan,
                                 size=size,
                                 fill_value=fill_value)


@set_module_as('brainunit.math')
def append(
    arr: Union[Array, Quantity],
    values: Union[Array, Quantity],
    axis: Optional[int] = None
) -> Union[Array, Quantity]:
  """
  Append values to the end of a quantity or an array.

  Parameters
  ----------
  arr : array_like, Quantity
    Values are appended to a copy of this array.
  values : array_like, Quantity
    These values are appended to a copy of `arr`.
    It must be of the correct shape (the same shape as `arr`, excluding `axis`).
  axis : int, optional
    The axis along which `values` are appended. If `axis` is None, `values` is flattened before use.

  Returns
  -------
  res : ndarray, Quantity
    A copy of `arr` with `values` appended to `axis`. Note that `append` does not occur in-place:
    a new array is allocated and filled.
  """
  return func_array_manipulation(jnp.append, arr, values, axis=axis)


@set_module_as('brainunit.math')
def flip(
    m: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[Array, Quantity]:
  """
  Reverse the order of elements in a quantity or an array along the given axis.

  Parameters
  ----------
  m : array_like, Quantity
    Input array.
  axis : int or tuple of ints, optional
    Axis or axes along which to flip over. The default, axis=None, will flip over all of the axes of the input array.

  Returns
  -------
  res : ndarray, Quantity
    A view of `m` with the entries of axis reversed. Since a view is returned, this operation is done in constant time.
  """
  return func_array_manipulation(jnp.flip, m, axis=axis)


@set_module_as('brainunit.math')
def fliplr(
    m: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  """
  Flip quantity or array in the left/right direction.

  Parameters
  ----------
  m : array_like, Quantity
    Input array.

  Returns
  -------
  res : ndarray, Quantity
    A view of `m` with the columns reversed. Since a view is returned, this operation is done in constant time.
  """
  return func_array_manipulation(jnp.fliplr, m)


@set_module_as('brainunit.math')
def flipud(
    m: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  """
  Flip quantity or array in the up/down direction.

  Parameters
  ----------
  m : array_like, Quantity
    Input array.

  Returns
  -------
  res : ndarray, Quantity
    A view of `m` with the rows reversed.
  """
  return func_array_manipulation(jnp.flipud, m)


@set_module_as('brainunit.math')
def roll(
    a: Union[Array, Quantity],
    shift: Union[int, Tuple[int, ...]],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[Array, Quantity]:
  """
  Roll quantity or array elements along a given axis.

  Parameters
  ----------
  a : array_like, Quantity
    Input array.
  shift : int or tuple of ints
    The number of places by which elements are shifted. If a tuple, then `axis` must be a tuple of the same size,
    and each of the given axes is shifted by the corresponding number. If an int while `axis` is a tuple of ints,
    then the same value is used for all given axes.
  axis : int or tuple of ints, optional
    Axis or axes along which elements are shifted. By default, the array is flattened before shifting, after which
    the original shape is restored.

  Returns
  -------
  res : ndarray, Quantity
    Output array, with the same shape as `a`.
  """
  return func_array_manipulation(jnp.roll, a, shift, axis=axis)


@set_module_as('brainunit.math')
def atleast_1d(
    *arys: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  """
  View inputs as quantities or arrays with at least one dimension.

  Parameters
  ----------
  *args : array_like, Quantity
    One or more input arrays or quantities.

  Returns
  -------
  res : ndarray, Quantity
    An array or a quantity, or a tuple of arrays or quantities, each with `a.ndim >= 1`.
  """
  return func_array_manipulation(jnp.atleast_1d, *arys)


@set_module_as('brainunit.math')
def atleast_2d(
    *arys: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  """
  View inputs as quantities or arrays with at least two dimensions.

  Parameters
  ----------
  *args : array_like, Quantity
    One or more input arrays or quantities.

  Returns
  -------
  res : ndarray, Quantity
    An array or a quantity, or a tuple of arrays or quantities, each with `a.ndim >= 2`.
  """
  return func_array_manipulation(jnp.atleast_2d, *arys)


@set_module_as('brainunit.math')
def atleast_3d(
    *arys: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  """
  View inputs as quantities or arrays with at least three dimensions.

  Parameters
  ----------
  *args : array_like, Quantity
    One or more input arrays or quantities.

  Returns
  -------
  res : ndarray, Quantity
    An array or a quantity, or a tuple of arrays or quantities, each with `a.ndim >= 3`.
  """
  return func_array_manipulation(jnp.atleast_3d, *arys)


@set_module_as('brainunit.math')
def expand_dims(
    a: Union[Array, Quantity],
    axis: int
) -> Union[Array, Quantity]:
  """
  Expand the shape of a quantity or an array.

  Parameters
  ----------
  a : array_like, Quantity
    Input array.
  axis : int
    Position in the expanded axes where the new axis is placed.

  Returns
  -------
  res : ndarray, Quantity
    View of `a` with the number of dimensions increased by one.
  """
  return func_array_manipulation(jnp.expand_dims, a, axis)


@set_module_as('brainunit.math')
def squeeze(
    a: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[Array, Quantity]:
  """
  Remove single-dimensional entries from the shape of a quantity or an array.

  Parameters
  ----------
  a : array_like, Quantity
    Input data.
  axis : None or int or tuple of ints, optional
    Selects a subset of the single-dimensional entries in the shape. If an axis is selected with shape entry greater
    than one, an error is raised.

  Returns
  -------
  res : ndarray, Quantity
    An array with the same data as `a`, but with a lower dimension.
  """
  return func_array_manipulation(jnp.squeeze, a, axis)


@set_module_as('brainunit.math')
def sort(
    a: Union[Array, Quantity],
    axis: Optional[int] = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Union[Array, Quantity]:
  """
  Return a sorted copy of a quantity or an array.

  Parameters
  ----------
  a : array_like, Quantity
    Array or quantity to be sorted.
  axis : int or None, optional
    Axis along which to sort. If None, the array is flattened before sorting. The default is -1, which sorts along
    the last axis.
  kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    Sorting algorithm. The default is 'quicksort'.
  order : str or list of str, optional
    When `a` is a quantity, it can be a string or a sequence of strings, which is interpreted as an order the quantity
    should be sorted. The default is None.
  stable : bool, optional
    Whether to use a stable sorting algorithm. The default is True.
  descending : bool, optional
    Whether to sort in descending order. The default is False.

  Returns
  -------
  res : ndarray, Quantity
    Sorted copy of the input array.
  """
  return func_array_manipulation(jnp.sort, a, axis=axis, kind=kind, order=order, stable=stable, descending=descending)


@set_module_as('brainunit.math')
def argsort(
    a: Union[Array, Quantity],
    axis: Optional[int] = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Array:
  """
  Returns the indices that would sort an array or a quantity.

  Parameters
  ----------
  a : array_like, Quantity
    Array or quantity to be sorted.
  axis : int or None, optional
    Axis along which to sort. If None, the array is flattened before sorting. The default is -1, which sorts along
    the last axis.
  kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    Sorting algorithm. The default is 'None'.
  order : str or list of str, optional
    When `a` is a quantity, it can be a string or a sequence of strings, which is interpreted as an order the quantity
    should be sorted. The default is None.
  stable : bool, optional
    Whether to use a stable sorting algorithm. The default is True.
  descending : bool, optional
    Whether to sort in descending order. The default is False.

  Returns
  -------
  res : ndarray
    Array of indices that sort the array.
  """
  return func_array_manipulation(jnp.argsort,
                                 a,
                                 axis=axis,
                                 kind=kind,
                                 order=order,
                                 stable=stable,
                                 descending=descending)


@set_module_as('brainunit.math')
def max(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float]] = None,
    where: Optional[Array] = None,
) -> Union[Array, Quantity]:
  """
  Return the maximum of a quantity or an array or maximum along an axis.

  Parameters
  ----------
  a : array_like, Quantity
    Array or quantity containing numbers whose maximum is desired.
  axis : int or None, optional
    Axis or axes along which to operate. By default, flattened input is used.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
    option, the result will broadcast correctly against the input array.
  initial : scalar, optional
    The minimum value of an output element. Must be present to allow computation on empty slice.
    See `numpy.ufunc.reduce`.
  where : array_like, optional
    Values of True indicate to calculate the ufunc at that position, values of False indicate to leave the value in the
    output alone.

  Returns
  -------
  res : ndarray, Quantity
    Maximum of `a`. If `axis` is None, the result is a scalar value. If `axis` is given, the result is an array of
    dimension `a.ndim - 1`.
  """
  return func_array_manipulation(jnp.max, a, axis=axis, keepdims=keepdims, initial=initial, where=where)


@set_module_as('brainunit.math')
def min(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float]] = None,
    where: Optional[Array] = None,
) -> Union[Array, Quantity]:
  """
  Return the minimum of a quantity or an array or minimum along an axis.

  Parameters
  ----------
  a : array_like, Quantity
    Array or quantity containing numbers whose minimum is desired.
  axis : int or None, optional
    Axis or axes along which to operate. By default, flattened input is used.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
    option, the result will broadcast correctly against the input array.
  initial : scalar, optional
    The maximum value of an output element. Must be present to allow computation on empty slice.
    See `numpy.ufunc.reduce`.
  where : array_like, optional
    Values of True indicate to calculate the ufunc at that position, values of False indicate to leave the value in the
    output alone.

  Returns
  -------
  res : ndarray, Quantity
    Minimum of `a`. If `axis` is None, the result is a scalar value. If `axis` is given, the result is an array of
    dimension `a.ndim - 1`.
  """
  return func_array_manipulation(jnp.min, a, axis=axis, keepdims=keepdims, initial=initial, where=where)


@set_module_as('brainunit.math')
def choose(
    a: Union[Array, Quantity],
    choices: Sequence[Union[Array, Quantity]],
    mode: str = 'raise',
) -> Union[Array, Quantity]:
  """
  Construct a quantity or an array from an index array and a set of arrays to choose from.

  Parameters
  ----------
  a : array_like, Quantity
    This array must be an integer array of the same shape as `choices`. The elements of `a` are used to select elements
    from `choices`.
  choices : sequence of array_like, Quantity
    Choice arrays. `a` and all `choices` must be broadcastable to the same shape.
  mode : {'raise', 'wrap', 'clip'}, optional
    Specifies how indices outside [0, n-1] will be treated:
    - 'raise' : raise an error (default)
    - 'wrap' : wrap around
    - 'clip' : clip to the range [0, n-1]

  Returns
  -------
  res : ndarray, Quantity
    The constructed array. The shape is identical to the shape of `a`, and the data type is the data type of `choices`.
  """
  return func_array_manipulation(jnp.choose, a, choices, mode=mode)


@set_module_as('brainunit.math')
def block(
    arrays: Sequence[Union[Array, Quantity]]
) -> Union[Array, Quantity]:
  """
  Assemble a quantity or an array from nested lists of blocks.

  Parameters
  ----------
  arrays : sequence of array_like, Quantity
    Each element in `arrays` can itself be a nested sequence of arrays, in which case the blocks in the corresponding
    cells are recursively stacked as the elements of the resulting array.

  Returns
  -------
  res : ndarray, Quantity
    The array constructed from the given blocks.
  """
  return func_array_manipulation(jnp.block, arrays)


@set_module_as('brainunit.math')
def compress(
    condition: Union[Array, Quantity],
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = 0,
) -> Union[Array, Quantity]:
  """
  Return selected slices of a quantity or an array along given axis.

  Parameters
  ----------
  condition : array_like, Quantity
    An array of boolean values that selects which slices to return. If the shape of condition is not the same as `a`,
    it must be broadcastable to `a`.
  a : array_like, Quantity
    Input array.
  axis : int or None, optional
    The axis along which to take slices. If axis is None, `condition` must be a 1-D array with the same length as `a`.
    If axis is an integer, `condition` must be broadcastable to the same shape as `a` along all axes except `axis`.
  size : int, optional
    The length of the returned axis. By default, the length of the input array along the axis is used.
  fill_value : scalar, optional
    The value to use for elements in the output array that are not selected. If None, the output array has the same
    type as `a` and is filled with zeros.

  Returns
  -------
  res : ndarray, Quantity
    A new array that has the same number of dimensions as `a`, and the same shape as `a` with axis `axis` removed.
  """
  return func_array_manipulation(jnp.compress, condition, a, axis, size=size, fill_value=fill_value)


@set_module_as('brainunit.math')
def diagflat(
    v: Union[Array, Quantity],
    k: int = 0
) -> Union[Array, Quantity]:
  """
  Create a two-dimensional a quantity or array with the flattened input as a diagonal.

  Parameters
  ----------
  v : array_like, Quantity
    Input data, which is flattened and set as the `k`-th diagonal of the output.
  k : int, optional
    Diagonal in question. The default is 0.

  Returns
  -------
  res : ndarray, Quantity
    The 2-D output array.
  """
  return func_array_manipulation(jnp.diagflat, v, k)


# return jax.numpy.Array, not Quantity

@set_module_as('brainunit.math')
def argmax(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = None
) -> Array:
  """
  Returns indices of the max value along an axis.

  Parameters
  ----------
  a : array_like, Quantity
    Input data.
  axis : int, optional
    By default, the index is into the flattened array, otherwise along the specified axis.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
    option, the result will broadcast correctly against the input array.

  Returns
  -------
  res : ndarray
    Array of indices into the array. It has the same shape as `a.shape` with the dimension along `axis` removed.
  """
  return func_array_manipulation(jnp.argmax, a, axis=axis, keepdim=keepdims, return_quantity=False)


@set_module_as('brainunit.math')
def argmin(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = None
) -> Array:
  """
  Returns indices of the min value along an axis.

  Parameters
  ----------
  a : array_like, Quantity
    Input data.
  axis : int, optional
    By default, the index is into the flattened array, otherwise along the specified axis.
  keepdims : bool, optional
    If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
    option, the result will broadcast correctly against the input array.

  Returns
  -------
  res : ndarray
    Array of indices into the array. It has the same shape as `a.shape` with the dimension along `axis` removed.
  """
  return func_array_manipulation(jnp.argmin, a, axis=axis, keepdims=keepdims, return_quantity=False)


@set_module_as('brainunit.math')
def argwhere(
    a: Union[Array, Quantity],
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = None,
) -> Array:
  """
  Find the indices of array elements that are non-zero, grouped by element.

  Parameters
  ----------
  a : array_like, Quantity
    Input data.
  size : int, optional
    The length of the returned axis. By default, the length of the input array along the axis is used.
  fill_value : scalar, optional
    The value to use for elements in the output array that are not selected. If None, the output array has the same
    type as `a` and is filled with zeros.

  Returns
  -------
  res : ndarray
    The indices of elements that are non-zero. The indices are grouped by element.
  """
  return func_array_manipulation(jnp.argwhere, a, size=size, fill_value=fill_value, return_quantity=False)


@set_module_as('brainunit.math')
def nonzero(
    a: Union[Array, Quantity],
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = None,
) -> Tuple[Array, ...]:
  """
  Return the indices of the elements that are non-zero.

  Parameters
  ----------
  a : array_like, Quantity
    Input data.
  size : int, optional
    The length of the returned axis. By default, the length of the input array along the axis is used.
  fill_value : scalar, optional
    The value to use for elements in the output array that are not selected. If None, the output array has the same
    type as `a` and is filled with zeros.

  Returns
  -------
  res : tuple of ndarrays
    Indices of elements that are non-zero along the specified axis. Each array in the tuple has the same shape as the
    input array.
  """
  return func_array_manipulation(jnp.nonzero, a, size=size, fill_value=fill_value, return_quantity=False)


@set_module_as('brainunit.math')
def flatnonzero(
    a: Union[Array, Quantity],
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = None,
) -> Array:
  """
  Return indices that are non-zero in the flattened version of the input quantity or array.

  Parameters
  ----------
  a : array_like, Quantity
    Input data.
  size : int, optional
    The length of the returned axis. By default, the length of the input array along the axis is used.
  fill_value : scalar, optional
    The value to use for elements in the output array that are not selected. If None, the output array has the same
    type as `a` and is filled with zeros.

  Returns
  -------
  res : ndarray
    Output array, containing the indices of the elements of `a.ravel()` that are non-zero.
  """
  return func_array_manipulation(jnp.flatnonzero, a, size=size, fill_value=fill_value, return_quantity=False)


@set_module_as('brainunit.math')
def searchsorted(
    a: Union[Array, Quantity],
    v: Union[Array, Quantity],
    side: str = 'left',
    sorter: Optional[Array] = None,
    *,
    method: Optional[str] = 'scan'
) -> Array:
  """
  Find indices where elements should be inserted to maintain order.

  Find the indices into a sorted array `a` such that, if the corresponding elements in `v` were inserted before the
  indices, the order of `a` would be preserved.

  Parameters
  ----------
  a : array_like, Quantity
    Input array. It must be sorted in ascending order.
  v : array_like, Quantity
    Values to insert into `a`.
  side : {'left', 'right'}, optional
    If 'left', the index of the first suitable location found is given. If 'right', return the last such index. If
    there is no suitable index, return either 0 or N (where N is the length of `a`).
  sorter : 1-D array_like, optional
    Optional array of integer indices that sort array `a` into ascending order. They are typically the result of
    `argsort`.
  method : str
    One of 'scan' (default), 'scan_unrolled', 'sort' or 'compare_all'. Controls the method used by the
    implementation: 'scan' tends to be more performant on CPU (particularly when ``a`` is
    very large), 'scan_unrolled' is more performant on GPU at the expense of additional compile time,
    'sort' is often more performant on accelerator backends like GPU and TPU
    (particularly when ``v`` is very large), and 'compare_all' can be most performant
    when ``a`` is very small. The default is 'scan'.

  Returns
  -------
  out : ndarray
    Array of insertion points with the same shape as `v`.
  """
  return func_array_manipulation(jnp.searchsorted, a, v, side=side, sorter=sorter, method=method, return_quantity=False)


@set_module_as('brainunit.math')
def extract(
    condition: Array,
    arr: Union[Array, Quantity],
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = 0,
) -> Array:
  """
  Return the elements of an array that satisfy some condition.

  Parameters
  ----------
  condition : array_like, Quantity
    An array of boolean values that selects which elements to extract.
  arr : array_like, Quantity
    The array from which to extract elements.
  size: int
    optional static size for output. Must be specified in order for ``extract``
    to be compatible with JAX transformations like :func:`~jax.jit` or :func:`~jax.vmap`.
  fill_value: array_like
    if ``size`` is specified, fill padded entries with this value (default: 0).

  Returns
  -------
  res : ndarray
    The extracted elements. The shape of `res` is the same as that of `condition`.
  """
  return func_array_manipulation(jnp.extract, condition, arr, size=size, fill_value=fill_value, return_quantity=False)


@set_module_as('brainunit.math')
def count_nonzero(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = None
) -> Array:
  """
  Count the number of non-zero values in the quantity or array `a`.

  Parameters
  ----------
  a : array_like, Quantity
    The array for which to count non-zeros.
  axis : int, optional
    The axis along which to count the non-zeros. If `None`, count non-zeros over the entire array.
  keepdims : bool, optional
    If this is set to `True`, the axes which are counted are left in the result as dimensions with size one. With this
    option, the result will broadcast correctly against the original array.

  Returns
  -------
  res : ndarray
    Number of non-zero values in the quantity or array along a given axis.
  """
  return func_array_manipulation(jnp.count_nonzero, a, axis=axis, keepdims=keepdims, return_quantity=False)


amax = max
amin = min


def function_to_method(func, x, *args, **kwargs):
  if isinstance(x, Quantity):
    return Quantity(func(x.value, *args, **kwargs), dim=x.dim)
  else:
    return func(x, *args, **kwargs)


@set_module_as('brainunit.math')
def diagonal(
    a: Union[jax.Array, Quantity],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1
) -> Union[jax.Array, Quantity]:
  """
  Return specified diagonals.

  Parameters
  ----------
  a : array_like, Quantity
    Array from which the diagonals are taken.
  offset : int, optional
    Offset of the diagonal from the main diagonal. Can be positive or negative. Defaults to main diagonal (0).
  axis1 : int, optional
    Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken. Defaults to first
    axis (0).
  axis2 : int, optional
    Axis to be used as the second axis of the 2-D sub-arrays from which the diagonals should be taken. Defaults to
    second axis (1).

  Returns
  -------
  res : ndarray
    The extracted diagonals. The shape of the output is determined by considering the shape of the input array with
    the specified axis removed.
  """
  return function_to_method(jnp.diagonal, a, offset, axis1, axis2)


@set_module_as('brainunit.math')
def ravel(
    a: Union[jax.Array, Quantity],
    order: str = 'C'
) -> Union[jax.Array, Quantity]:
  """
  Return a contiguous flattened quantity or array.

  Parameters
  ----------
  a : array_like, Quantity
    Input array. The elements in `a` are read in the order specified by `order`, and packed as a 1-D array.
  order : {'C', 'F', 'A', 'K'}, optional
    The elements of `a` are read using this index order. 'C' means to index the elements in row-major, C-style order,
    with the last axis index changing fastest, back to the first axis index changing slowest. 'F' means to index the
    elements in column-major, Fortran-style order, with the first index changing fastest, and the last index changing
    slowest. 'A' means to read the elements in Fortran-like index order if `a` is Fortran contiguous in memory, C-like
    order otherwise. 'K' means to read the elements in the order they occur in memory, except for reversing the data
    when strides are negative. By default, 'C' index order is used.

  Returns
  -------
  res : ndarray, Quantity
    The flattened quantity or array. The shape of the output is the same as `a`, but the array is 1-D.
  """
  return function_to_method(jnp.ravel, a, order)
