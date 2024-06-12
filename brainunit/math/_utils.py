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


import functools
from typing import Callable, Union

import jax
from jax.tree_util import tree_map

from .._base import Quantity


def _as_jax_array_(obj):
  return obj.value if isinstance(obj, Quantity) else obj


def _is_leaf(a):
  return isinstance(a, Quantity)


def _compatible_with_quantity(
    fun: Callable,
    return_quantity: bool = True,
):
  @functools.wraps(fun)
  def decorator(*args, **kwargs) -> Callable:
    def new_fun(*args, **kwargs) -> Union[list[Quantity], Quantity, jax.Array]:
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
          if not isinstance(out, Quantity):
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

    return new_fun

  return decorator
