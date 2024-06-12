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
from typing import (Union, Optional, Tuple, List, Callable)

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


@_compatible_with_quantity(jnp.reshape)
def reshape(
    a: Union[Array, Quantity],
    shape: Union[int, Tuple[int, ...]],
    order: str = 'C'
) -> Union[Array, Quantity]:
  '''
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
  ...


@_compatible_with_quantity(jnp.moveaxis)
def moveaxis(
    a: Union[Array, Quantity],
    source: Union[int, Tuple[int, ...]],
    destination: Union[int, Tuple[int, ...]]
) -> Union[Array, Quantity]:
  '''
  Moves axes of an array to new positions. Other axes remain in their original order.

  Args:
    a: array_like, Quantity
    source: int or sequence of ints
    destination: int or sequence of ints

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.transpose)
def transpose(
    a: Union[Array, Quantity],
    axes: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[Array, Quantity]:
  '''
  Returns a view of the array with axes transposed.

  Args:
    a: array_like, Quantity
    axes: tuple or list of ints, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.swapaxes)
def swapaxes(
    a: Union[Array, Quantity], axis1: int, axis2: int
) -> Union[Array, Quantity]:
  '''
  Interchanges two axes of an array.

  Args:
    a: array_like, Quantity
    axis1: int
    axis2: int

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.concatenate)
def concatenate(
    arrays: Union[Sequence[Array],
    Sequence[Quantity]], axis: Optional[int] = None
) -> Union[Array, Quantity]:
  '''
  Join a sequence of arrays along an existing axis.

  Args:
    arrays: sequence of array_like, Quantity
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.stack)
def stack(
    arrays: Union[Sequence[Array],
    Sequence[Quantity]], axis: int = 0
) -> Union[Array, Quantity]:
  '''
  Join a sequence of arrays along a new axis.

  Args:
    arrays: sequence of array_like, Quantity
    axis: int

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.vstack)
def vstack(
    arrays: Union[Sequence[Array],
    Sequence[Quantity]]
) -> Union[Array, Quantity]:
  '''
  Stack arrays in sequence vertically (row wise).

  Args:
    arrays: sequence of array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.numpy.Array
  '''
  ...


row_stack = vstack


@_compatible_with_quantity(jnp.hstack)
def hstack(
    arrays: Union[Sequence[Array],
    Sequence[Quantity]]
) -> Union[Array, Quantity]:
  '''
  Stack arrays in sequence horizontally (column wise).

  Args:
    arrays: sequence of array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.dstack)
def dstack(
    arrays: Union[Sequence[Array],
    Sequence[Quantity]]
) -> Union[Array, Quantity]:
  '''
  Stack arrays in sequence depth wise (along third axis).

  Args:
    arrays: sequence of array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.column_stack)
def column_stack(
    arrays: Union[Sequence[Array],
    Sequence[Quantity]]
) -> Union[Array, Quantity]:
  '''
  Stack 1-D arrays as columns into a 2-D array.

  Args:
    arrays: sequence of 1-D or 2-D array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.split)
def split(
    a: Union[Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]],
    axis: int = 0
) -> Union[List[Array], List[Quantity]]:
  '''
  Split an array into multiple sub-arrays.

  Args:
    a: array_like, Quantity
    indices_or_sections: int or 1-D array
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a list of Quantity if a is a Quantity, otherwise a list of jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.dsplit)
def dsplit(
    a: Union[Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]]
) -> Union[List[Array], List[Quantity]]:
  '''
  Split array along third axis (depth).

  Args:
    a: array_like, Quantity
    indices_or_sections: int or 1-D array

  Returns:
    Union[jax.Array, Quantity] a list of Quantity if a is a Quantity, otherwise a list of jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.hsplit)
def hsplit(
    a: Union[Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]]
) -> Union[List[Array], List[Quantity]]:
  '''
  Split an array into multiple sub-arrays horizontally (column-wise).

  Args:
    a: array_like, Quantity
    indices_or_sections: int or 1-D array

  Returns:
    Union[jax.Array, Quantity] a list of Quantity if a is a Quantity, otherwise a list of jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.vsplit)
def vsplit(
    a: Union[Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]]
) -> Union[List[Array], List[Quantity]]:
  '''
  Split an array into multiple sub-arrays vertically (row-wise).

  Args:
    a: array_like, Quantity
    indices_or_sections: int or 1-D array

  Returns:
    Union[jax.Array, Quantity] a list of Quantity if a is a Quantity, otherwise a list of jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.tile)
def tile(
    A: Union[Array, Quantity],
    reps: Union[int, Tuple[int, ...]]
) -> Union[Array, Quantity]:
  '''
  Construct an array by repeating A the number of times given by reps.

  Args:
    A: array_like, Quantity
    reps: array_like

  Returns:
    Union[jax.Array, Quantity] a Quantity if A is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.repeat)
def repeat(
    a: Union[Array, Quantity],
    repeats: Union[int, Tuple[int, ...]],
    axis: Optional[int] = None
) -> Union[Array, Quantity]:
  '''
  Repeat elements of an array.

  Args:
    a: array_like, Quantity
    repeats: array_like
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.unique)
def unique(
    a: Union[Array, Quantity],
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: Optional[int] = None
) -> Union[Array, Quantity]:
  '''
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
  ...


@_compatible_with_quantity(jnp.append)
def append(
    arr: Union[Array, Quantity],
    values: Union[Array, Quantity],
    axis: Optional[int] = None
) -> Union[Array, Quantity]:
  '''
  Append values to the end of an array.

  Args:
    arr: array_like, Quantity
    values: array_like, Quantity
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if arr and values are Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.flip)
def flip(
    m: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[Array, Quantity]:
  '''
  Reverse the order of elements in an array along the given axis.

  Args:
    m: array_like, Quantity
    axis: int or tuple of ints, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if m is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.fliplr)
def fliplr(
    m: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  '''
  Flip array in the left/right direction.

  Args:
    m: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if m is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.flipud)
def flipud(
    m: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  '''
  Flip array in the up/down direction.

  Args:
    m: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if m is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.roll)
def roll(
    a: Union[Array, Quantity],
    shift: Union[int, Tuple[int, ...]],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[Array, Quantity]:
  '''
  Roll array elements along a given axis.

  Args:
    a: array_like, Quantity
    shift: int or tuple of ints
    axis: int or tuple of ints, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.atleast_1d)
def atleast_1d(
    *arys: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  '''
  View inputs as arrays with at least one dimension.

  Args:
    *args: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if any input is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.atleast_2d)
def atleast_2d(
    *arys: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  '''
  View inputs as arrays with at least two dimensions.

  Args:
    *args: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if any input is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.atleast_3d)
def atleast_3d(
    *arys: Union[Array, Quantity]
) -> Union[Array, Quantity]:
  '''
  View inputs as arrays with at least three dimensions.

  Args:
    *args: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if any input is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.expand_dims)
def expand_dims(
    a: Union[Array, Quantity],
    axis: int
) -> Union[Array, Quantity]:
  '''
  Expand the shape of an array.

  Args:
    a: array_like, Quantity
    axis: int or tuple of ints

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.squeeze)
def squeeze(
    a: Union[Array, Quantity],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[Array, Quantity]:
  '''
  Remove single-dimensional entries from the shape of an array.

  Args:
    a: array_like, Quantity
    axis: None or int or tuple of ints, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.sort)
def sort(
    a: Union[Array, Quantity],
    axis: Optional[int] = -1,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Union[Array, Quantity]:
  '''
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
  '''
  ...


@_compatible_with_quantity(jnp.argsort)
def argsort(
    a: Union[Array, Quantity],
    axis: Optional[int] = -1,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Array:
  '''
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
  '''
  ...


@_compatible_with_quantity(jnp.max)
def max(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    out: Optional[Array] = None,
    keepdims: bool = False
) -> Union[Array, Quantity]:
  '''
  Return the maximum of an array or maximum along an axis.

  Args:
    a: array_like, Quantity
    axis: int or tuple of ints, optional
    keepdims: bool, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.min)
def min(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    out: Optional[Array] = None,
    keepdims: bool = False
) -> Union[Array, Quantity]:
  '''
  Return the minimum of an array or minimum along an axis.

  Args:
    a: array_like, Quantity
    axis: int or tuple of ints, optional
    keepdims: bool, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.choose)
def choose(
    a: Union[Array, Quantity],
    choices: Sequence[Union[Array, Quantity]]
) -> Union[Array, Quantity]:
  '''
  Use an index array to construct a new array from a set of choices.

  Args:
    a: array_like, Quantity
    choices: array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if a and choices are Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.block)
def block(
    arrays: Sequence[Union[Array, Quantity]]
) -> Union[Array, Quantity]:
  '''
  Assemble an nd-array from nested lists of blocks.

  Args:
    arrays: sequence of array_like, Quantity

  Returns:
    Union[jax.Array, Quantity] a Quantity if all input arrays are Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.compress)
def compress(
    condition: Union[Array, Quantity],
    a: Union[Array, Quantity],
    axis: Optional[int] = None
) -> Union[Array, Quantity]:
  '''
  Return selected slices of an array along given axis.

  Args:
    condition: array_like, Quantity
    a: array_like, Quantity
    axis: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  '''
  ...


@_compatible_with_quantity(jnp.diagflat)
def diagflat(
    v: Union[Array, Quantity],
    k: int = 0
) -> Union[Array, Quantity]:
  '''
  Create a two-dimensional array with the flattened input as a diagonal.

  Args:
    v: array_like, Quantity
    k: int, optional

  Returns:
    Union[jax.Array, Quantity] a Quantity if a is a Quantity, otherwise a jax.Array
  '''
  ...


# return jax.numpy.Array, not Quantity

@_compatible_with_quantity(jnp.argmax, return_quantity=False)
def argmax(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    out: Optional[Array] = None
) -> Array:
  '''
  Returns indices of the max value along an axis.

  Args:
    a: array_like, Quantity
    axis: int, optional
    out: array, optional

  Returns:
    jax.Array: an array (does not return a Quantity)
  '''
  ...


@_compatible_with_quantity(jnp.argmin, return_quantity=False)
def argmin(
    a: Union[Array, Quantity],
    axis: Optional[int] = None,
    out: Optional[Array] = None
) -> Array:
  '''
  Returns indices of the min value along an axis.

  Args:
    a: array_like, Quantity
    axis: int, optional
    out: array, optional

  Returns:
    jax.Array: an array (does not return a Quantity)
  '''
  ...


@_compatible_with_quantity(jnp.argwhere, return_quantity=False)
def argwhere(
    a: Union[Array, Quantity]
) -> Array:
  '''
  Find indices of non-zero elements.

  Args:
    a: array_like, Quantity

  Returns:
    jax.Array: an array (does not return a Quantity)
  '''
  ...


@_compatible_with_quantity(jnp.nonzero, return_quantity=False)
def nonzero(
    a: Union[Array, Quantity]
) -> Tuple[Array, ...]:
  '''
  Return the indices of the elements that are non-zero.

  Args:
    a: array_like, Quantity

  Returns:
    jax.Array: an array (does not return a Quantity)
  '''
  ...


@_compatible_with_quantity(jnp.flatnonzero, return_quantity=False)
def flatnonzero(
    a: Union[Array, Quantity]
) -> Array:
  '''
  Return indices that are non-zero in the flattened version of a.

  Args:
    a: array_like, Quantity

  Returns:
    jax.Array: an array (does not return a Quantity)
  '''
  ...


@_compatible_with_quantity(jnp.searchsorted, return_quantity=False)
def searchsorted(
    a: Union[Array, Quantity], v: Union[Array, Quantity],
    side: str = 'left',
    sorter: Optional[Array] = None
) -> Array:
  '''
  Find indices where elements should be inserted to maintain order.

  Args:
    a: array_like, Quantity
    v: array_like, Quantity
    side: {'left', 'right'}, optional

  Returns:
    jax.Array: an array (does not return a Quantity)
  '''
  ...


@_compatible_with_quantity(jnp.extract, return_quantity=False)
def extract(
    condition: Union[Array, Quantity],
    arr: Union[Array, Quantity]
) -> Array:
  '''
  Return the elements of an array that satisfy some condition.

  Args:
    condition: array_like, Quantity
    arr: array_like, Quantity

  Returns:
    jax.Array: an array (does not return a Quantity)
  '''
  ...


@_compatible_with_quantity(jnp.count_nonzero, return_quantity=False)
def count_nonzero(
    a: Union[Array, Quantity], axis: Optional[int] = None
) -> Array:
  '''
  Counts the number of non-zero values in the array a.

  Args:
    a: array_like, Quantity
    axis: int or tuple of ints, optional

  Returns:
    jax.Array: an array (does not return a Quantity)
  '''
  ...


amax = max
amin = min


def wrap_function_to_method(func: Callable):
  @wraps(func)
  def decorator(*args, **kwargs) -> Callable:
    def f(x, *args, **kwargs):
      if isinstance(x, Quantity):
        return Quantity(func(x.value, *args, **kwargs), dim=x.dim)
      else:
        return func(x, *args, **kwargs)

    f.__module__ = 'brainunit.math'
    return f

  return decorator


@wrap_function_to_method(jnp.diagonal)
def diagonal(a: Union[jax.Array, Quantity], offset: int = 0, axis1: int = 0, axis2: int = 1) -> Union[
  jax.Array, Quantity]:
  '''
  Return specified diagonals.

  Args:
    a: array_like, Quantity
    offset: int, optional
    axis1: int, optional
    axis2: int, optional

  Returns:
    Union[jax.Array, Quantity]: a Quantity if a is a Quantity, otherwise a jax.numpy.Array
  '''
  ...


@wrap_function_to_method(jnp.ravel)
def ravel(a: Union[jax.Array, Quantity], order: str = 'C') -> Union[jax.Array, Quantity]:
  '''
  Return a contiguous flattened array.

  Args:
    a: array_like, Quantity
    order: {'C', 'F', 'A', 'K'}, optional

  Returns:
    Union[jax.Array, Quantity]: a Quantity if a is a Quantity, otherwise a jax.numpy.Array
  '''
  ...
