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
from typing import (Callable, Union, Optional)

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum
from jax import Array

from .._base import (DIMENSIONLESS,
                     Quantity,
                     fail_for_dimension_mismatch,
                     is_unitless,
                     get_dim, )
from .._misc import set_module_as

__all__ = [
  # constants
  'e', 'pi', 'inf',

  # data types
  'dtype', 'finfo', 'iinfo',

  # more
  'broadcast_arrays', 'broadcast_shapes', 'einsum', 'gradient',

  # window funcs
  'bartlett', 'blackman', 'hamming', 'hanning', 'kaiser',
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


def _default_poly_einsum_handler(*operands, **kwargs):
  dummy = collections.namedtuple('dummy', ['shape', 'dtype'])
  dummies = [dummy(tuple(d if type(d) is int else 8 for d in x.shape), x.dtype)
             if hasattr(x, 'dtype') else x for x in operands]
  mapping = {id(d): i for i, d in enumerate(dummies)}
  out_dummies, contractions = opt_einsum.contract_path(*dummies, **kwargs)
  contract_operands = [operands[mapping[id(d)]] for d in out_dummies]
  return contract_operands, contractions


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
    for d in np.shape(op) if not jax.core.is_constant_dim(d)
  }
  if not non_constant_dim_types:
    contract_path = opt_einsum.contract_path
  else:
    contract_path = _default_poly_einsum_handler

  operands, contractions = contract_path(*operands, einsum_call=True, use_blas=True, optimize=optimize)

  unit = None
  for i in range(len(contractions) - 1):
    if contractions[i][4] == 'False':
      fail_for_dimension_mismatch(
        Quantity([], dim=unit), operands[i + 1], 'einsum'
      )
    elif contractions[i][4] == 'DOT' or \
        contractions[i][4] == 'TDOT' or \
        contractions[i][4] == 'GEMM' or \
        contractions[i][4] == 'OUTER/EINSUM':
      if i == 0:
        if isinstance(operands[i], Quantity) and isinstance(operands[i + 1], Quantity):
          unit = operands[i].dim * operands[i + 1].dim
        elif isinstance(operands[i], Quantity):
          unit = operands[i].dim
        elif isinstance(operands[i + 1], Quantity):
          unit = operands[i + 1].dim
      else:
        if isinstance(operands[i + 1], Quantity):
          unit = unit * operands[i + 1].dim
  operands = [op.value if isinstance(op, Quantity) else op for op in operands]

  r = jnp.einsum(subscripts,
                 *operands,
                 precision=precision,
                 preferred_element_type=preferred_element_type,
                 _dot_general=_dot_general)

  # contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)
  #
  # einsum = jax.jit(_einsum, static_argnums=(1, 2, 3, 4), inline=True)
  # if spec is not None:
  #   einsum = jax.named_call(einsum, name=spec)

  # r = einsum(operands, contractions, precision,  # type: ignore[operator]
  #            preferred_element_type, _dot_general)
  if unit is not None:
    return Quantity(r, dim=unit)
  else:
    return r


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


# window funcs
# ------------

bartlett = jnp.bartlett
blackman = jnp.blackman
hamming = jnp.hamming
hanning = jnp.hanning
kaiser = jnp.kaiser
