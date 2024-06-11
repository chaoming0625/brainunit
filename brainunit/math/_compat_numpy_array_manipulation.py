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
from functools import wraps
from typing import (Union, Optional, Tuple, List)

import jax
import jax.numpy as jnp
from jax import Array

from ._utils import _compatible_with_quantity
from .._base import (Quantity,
                     )

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


@_compatible_with_quantity()
def reshape(a: Union[Array, Quantity], shape: Union[int, Tuple[int, ...]], order: str = 'C') -> Union[Array, Quantity]:
  return jnp.reshape(a, shape, order)


@_compatible_with_quantity()
def moveaxis(a: Union[Array, Quantity], source: Union[int, Tuple[int, ...]],
             destination: Union[int, Tuple[int, ...]]) -> Union[Array, Quantity]:
  return jnp.moveaxis(a, source, destination)


@_compatible_with_quantity()
def transpose(a: Union[Array, Quantity], axes: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[Array, Quantity]:
  return jnp.transpose(a, axes)


@_compatible_with_quantity()
def swapaxes(a: Union[Array, Quantity], axis1: int, axis2: int) -> Union[Array, Quantity]:
  return jnp.swapaxes(a, axis1, axis2)


@_compatible_with_quantity()
def concatenate(arrays: Union[Sequence[Array], Sequence[Quantity]], axis: Optional[int] = None) -> Union[
  Array, Quantity]:
  return jnp.concatenate(arrays, axis)


@_compatible_with_quantity()
def stack(arrays: Union[Sequence[Array], Sequence[Quantity]], axis: int = 0) -> Union[Array, Quantity]:
  return jnp.stack(arrays, axis)


@_compatible_with_quantity()
def vstack(arrays: Union[Sequence[Array], Sequence[Quantity]]) -> Union[Array, Quantity]:
  return jnp.vstack(arrays)


row_stack = vstack


@_compatible_with_quantity()
def hstack(arrays: Union[Sequence[Array], Sequence[Quantity]]) -> Union[Array, Quantity]:
  return jnp.hstack(arrays)


@_compatible_with_quantity()
def dstack(arrays: Union[Sequence[Array], Sequence[Quantity]]) -> Union[Array, Quantity]:
  return jnp.dstack(arrays)


@_compatible_with_quantity()
def column_stack(arrays: Union[Sequence[Array], Sequence[Quantity]]) -> Union[Array, Quantity]:
  return jnp.column_stack(arrays)


@_compatible_with_quantity()
def split(a: Union[Array, Quantity], indices_or_sections: Union[int, Sequence[int]], axis: int = 0) -> Union[
  List[Array], List[Quantity]]:
  return jnp.split(a, indices_or_sections, axis)


@_compatible_with_quantity()
def dsplit(a: Union[Array, Quantity], indices_or_sections: Union[int, Sequence[int]]) -> Union[
  List[Array], List[Quantity]]:
  return jnp.dsplit(a, indices_or_sections)


@_compatible_with_quantity()
def hsplit(a: Union[Array, Quantity], indices_or_sections: Union[int, Sequence[int]]) -> Union[
  List[Array], List[Quantity]]:
  return jnp.hsplit(a, indices_or_sections)


@_compatible_with_quantity()
def vsplit(a: Union[Array, Quantity], indices_or_sections: Union[int, Sequence[int]]) -> Union[
  List[Array], List[Quantity]]:
  return jnp.vsplit(a, indices_or_sections)


@_compatible_with_quantity()
def tile(A: Union[Array, Quantity], reps: Union[int, Tuple[int, ...]]) -> Union[Array, Quantity]:
  return jnp.tile(A, reps)


@_compatible_with_quantity()
def repeat(a: Union[Array, Quantity], repeats: Union[int, Tuple[int, ...]], axis: Optional[int] = None) -> Union[
  Array, Quantity]:
  return jnp.repeat(a, repeats, axis)


@_compatible_with_quantity()
def unique(a: Union[Array, Quantity], return_index: bool = False, return_inverse: bool = False,
           return_counts: bool = False, axis: Optional[int] = None) -> Union[Array, Quantity]:
  return jnp.unique(a, return_index, return_inverse, return_counts, axis)


@_compatible_with_quantity()
def append(arr: Union[Array, Quantity], values: Union[Array, Quantity], axis: Optional[int] = None) -> Union[
  Array, Quantity]:
  return jnp.append(arr, values, axis)


@_compatible_with_quantity()
def flip(m: Union[Array, Quantity], axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[Array, Quantity]:
  return jnp.flip(m, axis)


@_compatible_with_quantity()
def fliplr(m: Union[Array, Quantity]) -> Union[Array, Quantity]:
  return jnp.fliplr(m)


@_compatible_with_quantity()
def flipud(m: Union[Array, Quantity]) -> Union[Array, Quantity]:
  return jnp.flipud(m)


@_compatible_with_quantity()
def roll(a: Union[Array, Quantity], shift: Union[int, Tuple[int, ...]],
         axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[Array, Quantity]:
  return jnp.roll(a, shift, axis)


@_compatible_with_quantity()
def atleast_1d(*arys: Union[Array, Quantity]) -> Union[Array, Quantity]:
  return jnp.atleast_1d(*arys)


@_compatible_with_quantity()
def atleast_2d(*arys: Union[Array, Quantity]) -> Union[Array, Quantity]:
  return jnp.atleast_2d(*arys)


@_compatible_with_quantity()
def atleast_3d(*arys: Union[Array, Quantity]) -> Union[Array, Quantity]:
  return jnp.atleast_3d(*arys)


@_compatible_with_quantity()
def expand_dims(a: Union[Array, Quantity], axis: int) -> Union[Array, Quantity]:
  return jnp.expand_dims(a, axis)


@_compatible_with_quantity()
def squeeze(a: Union[Array, Quantity], axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[Array, Quantity]:
  return jnp.squeeze(a, axis)


@_compatible_with_quantity()
def sort(a: Union[Array, Quantity],
         axis: Optional[int] = -1,
         kind: None = None,
         order: None = None,
         stable: bool = True,
         descending: bool = False, ) -> Union[Array, Quantity]:
  return jnp.sort(a, axis, kind=kind, order=order, stable=stable, descending=descending)


@_compatible_with_quantity()
def argsort(a: Union[Array, Quantity],
            axis: Optional[int] = -1,
            kind: None = None,
            order: None = None,
            stable: bool = True,
            descending: bool = False, ) -> Array:
  return jnp.argsort(a, axis, kind=kind, order=order, stable=stable, descending=descending)


@_compatible_with_quantity()
def max(a: Union[Array, Quantity], axis: Optional[int] = None, out: Optional[Array] = None,
        keepdims: bool = False) -> Union[Array, Quantity]:
  return jnp.max(a, axis, out, keepdims)


@_compatible_with_quantity()
def min(a: Union[Array, Quantity], axis: Optional[int] = None, out: Optional[Array] = None,
        keepdims: bool = False) -> Union[Array, Quantity]:
  return jnp.min(a, axis, out, keepdims)


@_compatible_with_quantity()
def choose(a: Union[Array, Quantity], choices: Sequence[Union[Array, Quantity]]) -> Union[Array, Quantity]:
  return jnp.choose(a, choices)


@_compatible_with_quantity()
def block(arrays: Sequence[Union[Array, Quantity]]) -> Union[Array, Quantity]:
  return jnp.block(arrays)


@_compatible_with_quantity()
def compress(condition: Union[Array, Quantity], a: Union[Array, Quantity], axis: Optional[int] = None) -> Union[
  Array, Quantity]:
  return jnp.compress(condition, a, axis)


@_compatible_with_quantity()
def diagflat(v: Union[Array, Quantity], k: int = 0) -> Union[Array, Quantity]:
  return jnp.diagflat(v, k)


# return jax.numpy.Array, not Quantity

@_compatible_with_quantity(return_quantity=False)
def argmax(a: Union[Array, Quantity], axis: Optional[int] = None, out: Optional[Array] = None) -> Array:
  return jnp.argmax(a, axis, out)


@_compatible_with_quantity(return_quantity=False)
def argmin(a: Union[Array, Quantity], axis: Optional[int] = None, out: Optional[Array] = None) -> Array:
  return jnp.argmin(a, axis, out)


@_compatible_with_quantity(return_quantity=False)
def argwhere(a: Union[Array, Quantity]) -> Array:
  return jnp.argwhere(a)


@_compatible_with_quantity(return_quantity=False)
def nonzero(a: Union[Array, Quantity]) -> Tuple[Array, ...]:
  return jnp.nonzero(a)


@_compatible_with_quantity(return_quantity=False)
def flatnonzero(a: Union[Array, Quantity]) -> Array:
  return jnp.flatnonzero(a)


@_compatible_with_quantity(return_quantity=False)
def searchsorted(a: Union[Array, Quantity], v: Union[Array, Quantity], side: str = 'left',
                 sorter: Optional[Array] = None) -> Array:
  return jnp.searchsorted(a, v, side, sorter)


@_compatible_with_quantity(return_quantity=False)
def extract(condition: Union[Array, Quantity], arr: Union[Array, Quantity]) -> Array:
  return jnp.extract(condition, arr)


@_compatible_with_quantity(return_quantity=False)
def count_nonzero(a: Union[Array, Quantity], axis: Optional[int] = None) -> Array:
  return jnp.count_nonzero(a, axis)


amax = max
amin = min

# docs for the functions above
reshape.__doc__ = '''
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
'''

moveaxis.__doc__ = '''
  Moves axes of an array to new positions. Other axes remain in their original order.

  Args:
    a: array_like, Quantity
    source: int or sequence of ints
    destination: int or sequence of ints

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''

transpose.__doc__ = '''
  Returns a view of the array with axes transposed.

  Args:
    a: array_like, Quantity
    axes: tuple or list of ints, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''

swapaxes.__doc__ = '''
  Interchanges two axes of an array.

  Args:
    a: array_like, Quantity
    axis1: int
    axis2: int

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''

concatenate.__doc__ = '''
  Join a sequence of arrays along an existing axis.

  Args:
    arrays: sequence of array_like, Quantity
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
'''

stack.__doc__ = '''
  Join a sequence of arrays along a new axis.

  Args:
    arrays: sequence of array_like, Quantity
    axis: int

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
'''

vstack.__doc__ = '''
  Stack arrays in sequence vertically (row wise).

  Args:
    arrays: sequence of array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.numpy.Array
'''

hstack.__doc__ = '''
  Stack arrays in sequence horizontally (column wise).

  Args:
    arrays: sequence of array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
'''

dstack.__doc__ = '''
  Stack arrays in sequence depth wise (along third axis).

  Args:
    arrays: sequence of array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
'''

column_stack.__doc__ = '''
  Stack 1-D arrays as columns into a 2-D array.

  Args:
    arrays: sequence of 1-D or 2-D array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
'''

split.__doc__ = '''
  Split an array into multiple sub-arrays.

  Args:
    a: array_like, Quantity
    indices_or_sections: int or 1-D array
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a list of Quantity if a is a Quantity, otherwise a list of jax.Array
'''

dsplit.__doc__ = '''
  Split array along third axis (depth).

  Args:
    a: array_like, Quantity
    indices_or_sections: int or 1-D array

  Returns:
    Union[jax.Array, Quantity] a list of Quantity if a is a Quantity, otherwise a list of jax.Array
'''

hsplit.__doc__ = '''
  Split an array into multiple sub-arrays horizontally (column-wise).

  Args:
    a: array_like, Quantity
    indices_or_sections: int or 1-D array

  Returns:
    Union[jax.Array, Quantity] a list of Quantity if a is a Quantity, otherwise a list of jax.Array
'''

vsplit.__doc__ = '''
  Split an array into multiple sub-arrays vertically (row-wise).

  Args:
    a: array_like, Quantity
    indices_or_sections: int or 1-D array

  Returns:
    Union[jax.Array, Quantity] a list of Quantity if a is a Quantity, otherwise a list of jax.Array
'''

tile.__doc__ = '''
  Construct an array by repeating A the number of times given by reps.

  Args:
    A: array_like, Quantity
    reps: array_like

  Returns:
    Union[jax.Array, Quantity] a Quantity if A is a Quantity, otherwise a jax.Array
'''

repeat.__doc__ = '''
  Repeat elements of an array.

  Args:
    a: array_like, Quantity
    repeats: array_like
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''

unique.__doc__ = '''
  Find the unique elements of an array.

  Args:
    a: array_like, Quantity
    return_index: bool, optional
    return_inverse: bool, optional
    return_counts: bool, optional
    axis: int or None, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''

append.__doc__ = '''
  Append values to the end of an array.

  Args:
    arr: array_like, Quantity
    values: array_like, Quantity
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if arr and values are Quantity, otherwise a jax.Array
'''

flip.__doc__ = '''
  Reverse the order of elements in an array along the given axis.

  Args:
    m: array_like, Quantity
    axis: int or tuple of ints, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if m is a Quantity, otherwise a jax.Array
'''

fliplr.__doc__ = '''
  Flip array in the left/right direction.

  Args:
    m: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if m is a Quantity, otherwise a jax.Array
'''

flipud.__doc__ = '''
  Flip array in the up/down direction.

  Args:
    m: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if m is a Quantity, otherwise a jax.Array
'''

roll.__doc__ = '''
  Roll array elements along a given axis.

  Args:
    a: array_like, Quantity
    shift: int or tuple of ints
    axis: int or tuple of ints, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''

atleast_1d.__doc__ = '''
  View inputs as arrays with at least one dimension.

  Args:
    *args: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if any input is a Quantity, otherwise a jax.Array
'''

atleast_2d.__doc__ = '''
  View inputs as arrays with at least two dimensions.

  Args:
    *args: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if any input is a Quantity, otherwise a jax.Array
'''

atleast_3d.__doc__ = '''
  View inputs as arrays with at least three dimensions.

  Args:
    *args: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if any input is a Quantity, otherwise a jax.Array
'''

expand_dims.__doc__ = '''
  Expand the shape of an array.

  Args:
    a: array_like, Quantity
    axis: int or tuple of ints

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''

squeeze.__doc__ = '''
  Remove single-dimensional entries from the shape of an array.

  Args:
    a: array_like, Quantity
    axis: None or int or tuple of ints, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''

sort.__doc__ = '''
  Return a sorted copy of an array.

  Args:
    a: array_like, Quantity
    axis: int or None, optional
    kind: {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    order: str or list of str, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''
max.__doc__ = '''
  Return the maximum of an array or maximum along an axis.

  Args:
    a: array_like, Quantity
    axis: int or tuple of ints, optional
    keepdims: bool, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''

min.__doc__ = '''
  Return the minimum of an array or minimum along an axis.

  Args:
    a: array_like, Quantity
    axis: int or tuple of ints, optional
    keepdims: bool, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''

choose.__doc__ = '''
  Use an index array to construct a new array from a set of choices.

  Args:
    a: array_like, Quantity
    choices: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if a and choices are Quantity, otherwise a jax.Array
'''

block.__doc__ = '''
  Assemble an nd-array from nested lists of blocks.

  Args:
    arrays: sequence of array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
'''

compress.__doc__ = '''
  Return selected slices of an array along given axis.

  Args:
    condition: array_like, Quantity
    a: array_like, Quantity
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''

diagflat.__doc__ = '''
  Create a two-dimensional array with the flattened input as a diagonal.

  Args:
    a: array_like, Quantity
    offset: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
'''

argsort.__doc__ = '''
  Returns the indices that would sort an array.

  Args:
    a: array_like, Quantity
    axis: int or None, optional
    kind: {'quicksort', 'mergesort', 'heapsort'}, optional
    order: str or list of str, optional

  Returns:
    jax.Array jax.numpy.Array (does not return a Quantity)
'''

argmax.__doc__ = '''
  Returns indices of the max value along an axis.

  Args:
    a: array_like, Quantity
    axis: int, optional
    out: array, optional

  Returns:
    jax.Array: an array (does not return a Quantity)
'''

argmin.__doc__ = '''
  Returns indices of the min value along an axis.

  Args:
    a: array_like, Quantity
    axis: int, optional
    out: array, optional

  Returns:
    jax.Array: an array (does not return a Quantity)
'''

argwhere.__doc__ = '''
  Find indices of non-zero elements.

  Args:
    a: array_like, Quantity

  Returns:
    jax.Array: an array (does not return a Quantity)
'''

nonzero.__doc__ = '''
  Return the indices of the elements that are non-zero.

  Args:
    a: array_like, Quantity

  Returns:
    jax.Array: an array (does not return a Quantity)
'''

flatnonzero.__doc__ = '''
  Return indices that are non-zero in the flattened version of a.

  Args:
    a: array_like, Quantity

  Returns:
    jax.Array: an array (does not return a Quantity)
'''

searchsorted.__doc__ = '''
  Find indices where elements should be inserted to maintain order.

  Args:
    a: array_like, Quantity
    v: array_like, Quantity
    side: {'left', 'right'}, optional

  Returns:
    jax.Array: an array (does not return a Quantity)
'''

extract.__doc__ = '''
  Return the elements of an array that satisfy some condition.

  Args:
    condition: array_like, Quantity
    a: array_like, Quantity

  Returns:
    jax.Array: an array (does not return a Quantity)
'''

count_nonzero.__doc__ = '''
  Counts the number of non-zero values in the array a.

  Args:
    a: array_like, Quantity
    axis: int or tuple of ints, optional

  Returns:
    jax.Array: an array (does not return a Quantity)
'''


def wrap_function_to_method(func):
  @wraps(func)
  def f(x, *args, **kwargs):
    if isinstance(x, Quantity):
      return Quantity(func(x.value, *args, **kwargs), unit=x.unit)
    else:
      return func(x, *args, **kwargs)

  f.__module__ = 'brainunit.math'
  return f


@wrap_function_to_method
def diagonal(a: Union[jax.Array, Quantity], offset: int = 0, axis1: int = 0, axis2: int = 1) -> Union[
  jax.Array, Quantity]:
  return jnp.diagonal(a, offset, axis1, axis2)


@wrap_function_to_method
def ravel(a: Union[jax.Array, Quantity], order: str = 'C') -> Union[jax.Array, Quantity]:
  return jnp.ravel(a, order)


diagonal.__doc__ = '''
  Return specified diagonals.

  Args:
    a: array_like, Quantity
    offset: int, optional
    axis1: int, optional
    axis2: int, optional

  Returns:
    Union[jax.Array, Quantity]: a Quantity if a is a Quantity, otherwise a jax.numpy.Array
'''

ravel.__doc__ = '''
  Return a contiguous flattened array.

  Args:
    a: array_like, Quantity
    order: {'C', 'F', 'A', 'K'}, optional

  Returns:
    Union[jax.Array, Quantity]: a Quantity if a is a Quantity, otherwise a jax.numpy.Array
'''
