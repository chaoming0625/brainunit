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
from typing import (Union, Optional, Tuple, List)

import jax
import jax.numpy as jnp
from jax import Array

from brainunit._misc import set_module_as
from .._base import Quantity

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
from jax.tree_util import tree_map


def _as_jax_array_(obj):
  return obj.value if isinstance(obj, Quantity) else obj


def _is_leaf(a):
  return isinstance(a, Quantity)


def func_array_manipulation(fun, *args, return_quantity=True, **kwargs) -> Union[list[Quantity], Quantity, jax.Array]:
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
  Return a reshaped copy of an array or a Quantity.

  Args:
    a: input array or Quantity to reshape
    shape: integer or sequence of integers giving the new shape, which must match the
      size of the input array. If any single dimension is given size ``-1``, it will be
      replaced with a value such that the output has the correct size.
    order: ``'F'`` or ``'C'``, specifies whether the reshape should apply column-major
      (fortran-style, ``"F"``) or row-major (C-style, ``"C"``) order; default is ``"C"``.
      brainunit does not support ``order="A"``.

  Returns:
    reshaped copy of input array with the specified shape.
  """
  return func_array_manipulation(jnp.reshape, a, shape, order=order)


@set_module_as('brainunit.math')
def moveaxis(
    a: Union[Array, Quantity],
    source: Union[int, Tuple[int, ...]],
    destination: Union[int, Tuple[int, ...]]
) -> Union[Array, Quantity]:
  """
  Moves axes of an array to new positions. Other axes remain in their original order.

  Args:
    a: array_like, Quantity
    source: int or sequence of ints
    destination: int or sequence of ints

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.moveaxis, a, source, destination)


@set_module_as('brainunit.math')
def transpose(
    a: Union[Array, Quantity],
    axes: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[Array, Quantity]:
  """
  Returns a view of the array with axes transposed.

  Args:
    a: array_like, Quantity
    axes: tuple or list of ints, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.transpose, a, axes)


@set_module_as('brainunit.math')
def swapaxes(
    a: Union[Array, Quantity], axis1: int, axis2: int
) -> Union[Array, Quantity]:
  """
  Interchanges two axes of an array.

  Args:
    a: array_like, Quantity
    axis1: int
    axis2: int

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.swapaxes, a, axis1, axis2)


@set_module_as('brainunit.math')
def concatenate(
    arrays: Union[Sequence[Array], Sequence[Quantity]],
    axis: Optional[int] = None
) -> Union[Array, Quantity]:
  """
  Join a sequence of arrays along an existing axis.

  Args:
    arrays: sequence of array_like, Quantity
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.concatenate, arrays, axis=axis)


@set_module_as('brainunit.math')
def stack(
    arrays: Union[Sequence[Array], Sequence[Quantity]],
    axis: int = 0
) -> Union[Array, Quantity]:
  """
  Join a sequence of arrays along a new axis.

  Args:
    arrays: sequence of array_like, Quantity
    axis: int

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.stack, arrays, axis=axis)


@set_module_as('brainunit.math')
def vstack(
    arrays: Union[Sequence[Array], Sequence[Quantity]]
) -> Union[Array, Quantity]:
  """
  Stack arrays in sequence vertically (row wise).

  Args:
    arrays: sequence of array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.numpy.Array
  """
  return func_array_manipulation(jnp.vstack, arrays)


row_stack = vstack


@set_module_as('brainunit.math')
def hstack(
    arrays: Union[Sequence[Array], Sequence[Quantity]]
) -> Union[Array, Quantity]:
  """
  Stack arrays in sequence horizontally (column wise).

  Args:
    arrays: sequence of array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.hstack, arrays)


@set_module_as('brainunit.math')
def dstack(
    arrays: Union[Sequence[Array], Sequence[Quantity]]
) -> Union[Array, Quantity]:
  """
  Stack arrays in sequence depth wise (along third axis).

  Args:
    arrays: sequence of array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.dstack, arrays)


@set_module_as('brainunit.math')
def column_stack(
    arrays: Union[Sequence[Array], Sequence[Quantity]]
) -> Union[Array, Quantity]:
  """
  Stack 1-D arrays as columns into a 2-D array.

  Args:
    arrays: sequence of 1-D or 2-D array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.column_stack, arrays)


@set_module_as('brainunit.math')
def split(
    a: Union[Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]],
    axis: int = 0
) -> Union[List[Array], List[Quantity]]:
  """
  Split an array into multiple sub-arrays.

  Args:
    a: array_like, Quantity
    indices_or_sections: int or 1-D array
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a list of Quantity if a is a Quantity, otherwise a list of jax.Array
  """
  return func_array_manipulation(jnp.split, a, indices_or_sections, axis=axis)


@set_module_as('brainunit.math')
def dsplit(
    a: Union[Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]]
) -> Union[List[Array], List[Quantity]]:
  """
  Split array along third axis (depth).

  Args:
    a: array_like, Quantity
    indices_or_sections: int or 1-D array

  Returns:
    Union[jax.Array, Quantity] a list of Quantity if a is a Quantity, otherwise a list of jax.Array
  """
  return func_array_manipulation(jnp.dsplit, a, indices_or_sections)


@set_module_as('brainunit.math')
def hsplit(
    a: Union[Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]]
) -> Union[List[Array], List[Quantity]]:
  """
  Split an array into multiple sub-arrays horizontally (column-wise).

  Args:
    a: array_like, Quantity
    indices_or_sections: int or 1-D array

  Returns:
    Union[jax.Array, Quantity] a list of Quantity if a is a Quantity, otherwise a list of jax.Array
  """
  return func_array_manipulation(jnp.hsplit, a, indices_or_sections)


@set_module_as('brainunit.math')
def vsplit(
    a: Union[Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]]
) -> Union[List[Array], List[Quantity]]:
  """
  Split an array into multiple sub-arrays vertically (row-wise).

  Args:
    a: array_like, Quantity
    indices_or_sections: int or 1-D array

  Returns:
    Union[jax.Array, Quantity] a list of Quantity if a is a Quantity, otherwise a list of jax.Array
  """
  return func_array_manipulation(jnp.vsplit, a, indices_or_sections)


@set_module_as('brainunit.math')
def tile(
    A: Union[Array, Quantity],
    reps: Union[int, Tuple[int, ...]]
) -> Union[Array, Quantity]:
  """
  Construct an array by repeating A the number of times given by reps.

  Args:
    A: array_like, Quantity
    reps: array_like

  Returns:
    Union[jax.Array, Quantity] a Quantity if A is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.tile, A, reps)


@set_module_as('brainunit.math')
def repeat(
    a: Union[Array, Quantity],
    repeats: Union[int, Tuple[int, ...]],
    axis: Optional[int] = None
) -> Union[Array, Quantity]:
  """
  Repeat elements of an array.

  Args:
    a: array_like, Quantity
    repeats: array_like
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.repeat, a, repeats, axis=axis)


@set_module_as('brainunit.math')
def unique(
    a: Union[Array, Quantity],
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: Optional[int] = None
) -> Union[Array, Quantity]:
  """
  Find the unique elements of an array.

  Args:
    a: array_like, Quantity
    return_index: bool, optional
    return_inverse: bool, optional
    return_counts: bool, optional
    axis: int or None, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.unique, a, return_index=return_index, return_inverse=return_inverse,
                                 return_counts=return_counts, axis=axis)


@set_module_as('brainunit.math')
def append(
    arr: Union[Array, Quantity],
    values: Union[Array, Quantity],
    axis: Optional[int] = None
) -> Union[Array, Quantity]:
  """
  Append values to the end of an array.

  Args:
    arr: array_like, Quantity
    values: array_like, Quantity
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if arr and values are Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.append, arr, values, axis=axis)


@set_module_as('brainunit.math')
def flip(
    m: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[Array, Quantity]:
  """
  Reverse the order of elements in an array along the given axis.

  Args:
    m: array_like, Quantity
    axis: int or tuple of ints, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if m is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.flip, m, axis=axis)


@set_module_as('brainunit.math')
def fliplr(
    m: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  """
  Flip array in the left/right direction.

  Args:
    m: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if m is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.fliplr, m)


@set_module_as('brainunit.math')
def flipud(
    m: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  """
  Flip array in the up/down direction.

  Args:
    m: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if m is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.flipud, m)


@set_module_as('brainunit.math')
def roll(
    a: Union[Array, Quantity],
    shift: Union[int, Tuple[int, ...]],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[Array, Quantity]:
  """
  Roll array elements along a given axis.

  Args:
    a: array_like, Quantity
    shift: int or tuple of ints
    axis: int or tuple of ints, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.roll, a, shift, axis=axis)


@set_module_as('brainunit.math')
def atleast_1d(
    *arys: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  """
  View inputs as arrays with at least one dimension.

  Args:
    *args: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if any input is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.atleast_1d, *arys)


@set_module_as('brainunit.math')
def atleast_2d(
    *arys: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  """
  View inputs as arrays with at least two dimensions.

  Args:
    *args: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if any input is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.atleast_2d, *arys)


@set_module_as('brainunit.math')
def atleast_3d(
    *arys: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  """
  View inputs as arrays with at least three dimensions.

  Args:
    *args: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if any input is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.atleast_3d, *arys)


@set_module_as('brainunit.math')
def expand_dims(
    a: Union[Array, Quantity],
    axis: int
) -> Union[Array, Quantity]:
  """
  Expand the shape of an array.

  Args:
    a: array_like, Quantity
    axis: int or tuple of ints

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.expand_dims, a, axis)


@set_module_as('brainunit.math')
def squeeze(
    a: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[Array, Quantity]:
  """
  Remove single-dimensional entries from the shape of an array.

  Args:
    a: array_like, Quantity
    axis: None or int or tuple of ints, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.squeeze, a, axis)


@set_module_as('brainunit.math')
def sort(
    a: Union[Array, Quantity],
    axis: Optional[int] = -1,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Union[Array, Quantity]:
  """
  Return a sorted copy of an array.

  Args:
    a: array_like, Quantity
    axis: int or None, optional
    kind: {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    order: str or list of str, optional
    stable: bool, optional
    descending: bool, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.sort, a, axis=axis, kind=kind, order=order, stable=stable, descending=descending)


@set_module_as('brainunit.math')
def argsort(
    a: Union[Array, Quantity],
    axis: Optional[int] = -1,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Array:
  """
  Returns the indices that would sort an array.

  Args:
    a: array_like, Quantity
    axis: int or None, optional
    kind: {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    order: str or list of str, optional
    stable: bool, optional
    descending: bool, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.argsort, a, axis=axis, kind=kind, order=order, stable=stable,
                                 descending=descending)


@set_module_as('brainunit.math')
def max(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    keepdims: bool = False
) -> Union[Array, Quantity]:
  """
  Return the maximum of an array or maximum along an axis.

  Args:
    a: array_like, Quantity
    axis: int or tuple of ints, optional
    keepdims: bool, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.max, a, axis=axis, keepdims=keepdims)


@set_module_as('brainunit.math')
def min(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    keepdims: bool = False
) -> Union[Array, Quantity]:
  """
  Return the minimum of an array or minimum along an axis.

  Args:
    a: array_like, Quantity
    axis: int or tuple of ints, optional
    keepdims: bool, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.min, a, axis=axis, keepdims=keepdims)


@set_module_as('brainunit.math')
def choose(
    a: Union[Array, Quantity],
    choices: Sequence[Union[Array, Quantity]]
) -> Union[Array, Quantity]:
  """
  Use an index array to construct a new array from a set of choices.

  Args:
    a: array_like, Quantity
    choices: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if a and choices are Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.choose, a, choices)


@set_module_as('brainunit.math')
def block(
    arrays: Sequence[Union[Array, Quantity]]
) -> Union[Array, Quantity]:
  """
  Assemble an nd-array from nested lists of blocks.

  Args:
    arrays: sequence of array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.block, arrays)


@set_module_as('brainunit.math')
def compress(
    condition: Union[Array, Quantity],
    a: Union[Array, Quantity],
    axis: Optional[int] = None
) -> Union[Array, Quantity]:
  """
  Return selected slices of an array along given axis.

  Args:
    condition: array_like, Quantity
    a: array_like, Quantity
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.compress, condition, a, axis=axis)


@set_module_as('brainunit.math')
def diagflat(
    v: Union[Array, Quantity],
    k: int = 0
) -> Union[Array, Quantity]:
  """
  Create a two-dimensional array with the flattened input as a diagonal.

  Args:
    v: array_like, Quantity
    k: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  """
  return func_array_manipulation(jnp.diagflat, v, k)


# return jax.numpy.Array, not Quantity

@set_module_as('brainunit.math')
def argmax(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    out: Optional[Array] = None
) -> Array:
  """
  Returns indices of the max value along an axis.

  Args:
    a: array_like, Quantity
    axis: int, optional
    out: array, optional

  Returns:
    jax.Array: an array (does not return a Quantity)
  """
  return func_array_manipulation(jnp.argmax, a, axis=axis, out=out, return_quantity=False)


@set_module_as('brainunit.math')
def argmin(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    out: Optional[Array] = None
) -> Array:
  """
  Returns indices of the min value along an axis.

  Args:
    a: array_like, Quantity
    axis: int, optional
    out: array, optional

  Returns:
    jax.Array: an array (does not return a Quantity)
  """
  return func_array_manipulation(jnp.argmin, a, axis=axis, out=out, return_quantity=False)


@set_module_as('brainunit.math')
def argwhere(
    a: Union[Array, Quantity]
) -> Array:
  """
  Find indices of non-zero elements.

  Args:
    a: array_like, Quantity

  Returns:
    jax.Array: an array (does not return a Quantity)
  """
  return func_array_manipulation(jnp.argwhere, a, return_quantity=False)


@set_module_as('brainunit.math')
def nonzero(
    a: Union[Array, Quantity]
) -> Tuple[Array, ...]:
  """
  Return the indices of the elements that are non-zero.

  Args:
    a: array_like, Quantity

  Returns:
    jax.Array: an array (does not return a Quantity)
  """
  return func_array_manipulation(jnp.nonzero, a, return_quantity=False)


@set_module_as('brainunit.math')
def flatnonzero(
    a: Union[Array, Quantity]
) -> Array:
  """
  Return indices that are non-zero in the flattened version of a.

  Args:
    a: array_like, Quantity

  Returns:
    jax.Array: an array (does not return a Quantity)
  """
  return func_array_manipulation(jnp.flatnonzero, a, return_quantity=False)


@set_module_as('brainunit.math')
def searchsorted(
    a: Union[Array, Quantity],
    v: Union[Array, Quantity],
    side: str = 'left',
    sorter: Optional[Array] = None
) -> Array:
  """
  Find indices where elements should be inserted to maintain order.

  Args:
    a: array_like, Quantity
    v: array_like, Quantity
    side: {'left', 'right'}, optional

  Returns:
    jax.Array: an array (does not return a Quantity)
  """
  return func_array_manipulation(jnp.searchsorted, a, v, side=side, sorter=sorter, return_quantity=False)


@set_module_as('brainunit.math')
def extract(
    condition: Union[Array, Quantity],
    arr: Union[Array, Quantity]
) -> Array:
  """
  Return the elements of an array that satisfy some condition.

  Args:
    condition: array_like, Quantity
    arr: array_like, Quantity

  Returns:
    jax.Array: an array (does not return a Quantity)
  """
  return func_array_manipulation(jnp.extract, condition, arr, return_quantity=False)


@set_module_as('brainunit.math')
def count_nonzero(
    a: Union[Array, Quantity], axis: Optional[int] = None
) -> Array:
  """
  Counts the number of non-zero values in the array a.

  Args:
    a: array_like, Quantity
    axis: int or tuple of ints, optional

  Returns:
    jax.Array: an array (does not return a Quantity)
  """
  return func_array_manipulation(jnp.count_nonzero, a, axis=axis, return_quantity=False)


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

  Args:
    a: array_like, Quantity
    offset: int, optional
    axis1: int, optional
    axis2: int, optional

  Returns:
    Union[jax.Array, Quantity]: a Quantity if a is a Quantity, otherwise a jax.numpy.Array
  """
  return function_to_method(jnp.diagonal, a, offset, axis1, axis2)


@set_module_as('brainunit.math')
def ravel(
    a: Union[jax.Array, Quantity],
    order: str = 'C'
) -> Union[jax.Array, Quantity]:
  """
  Return a contiguous flattened array.

  Args:
    a: array_like, Quantity
    order: {'C', 'F', 'A', 'K'}, optional

  Returns:
    Union[jax.Array, Quantity]: a Quantity if a is a Quantity, otherwise a jax.numpy.Array
  """
  return function_to_method(jnp.ravel, a, order)
