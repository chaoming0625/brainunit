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
from typing import (Callable, Union, Tuple)

import brainstate as bst
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum
from brainstate._utils import set_module_as
from jax import Array
from jax._src.numpy.lax_numpy import _einsum

from ._compat_numpy_funcs_change_unit import wrap_math_funcs_change_unit_binary
from ._compat_numpy_funcs_keep_unit import wrap_math_funcs_keep_unit_unary
from ._utils import _compatible_with_quantity
from .._base import (DIMENSIONLESS,
                     Quantity,
                     fail_for_dimension_mismatch,
                     is_unitless,
                     get_unit, )

__all__ = [

  # constants
  'e', 'pi', 'inf',

  # data types
  'dtype', 'finfo', 'iinfo',

  # more
  'broadcast_arrays', 'broadcast_shapes',
  'einsum', 'gradient', 'intersect1d', 'nan_to_num', 'nanargmax', 'nanargmin',
  'rot90', 'tensordot',
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
def finfo(a: Union[Quantity, bst.typing.ArrayLike]) -> jnp.finfo:
  if isinstance(a, Quantity):
    return jnp.finfo(a.value)
  else:
    return jnp.finfo(a)


@set_module_as('brainunit.math')
def iinfo(a: Union[Quantity, bst.typing.ArrayLike]) -> jnp.iinfo:
  if isinstance(a, Quantity):
    return jnp.iinfo(a.value)
  else:
    return jnp.iinfo(a)


# more
# ----
@set_module_as('brainunit.math')
def broadcast_arrays(*args: Union[Quantity, bst.typing.ArrayLike]) -> Union[Quantity, list[Array]]:
  from builtins import all as origin_all
  from builtins import any as origin_any
  if origin_all(isinstance(arg, Quantity) for arg in args):
    if origin_any(arg.dim != args[0].dim for arg in args):
      raise ValueError("All arguments must have the same unit")
    return Quantity(jnp.broadcast_arrays(*[arg.value for arg in args]), dim=args[0].dim)
  elif origin_all(isinstance(arg, (jax.Array, np.ndarray)) for arg in args):
    return jnp.broadcast_arrays(*args)
  else:
    raise ValueError(f"Unsupported types : {type(args)} for broadcast_arrays")


broadcast_shapes = jnp.broadcast_shapes


@set_module_as('brainunit.math')
def einsum(
    subscripts: str,
    /,
    *operands: Union[Quantity, jax.Array],
    out: None = None,
    optimize: Union[str, bool] = "optimal",
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: Union[jax.typing.DTypeLike, None] = None,
    _dot_general: Callable[..., jax.Array] = jax.lax.dot_general,
) -> Union[jax.Array, Quantity]:
  '''
  Evaluates the Einstein summation convention on the operands.

  Args:
      subscripts: string containing axes names separated by commas.
      *operands: sequence of one or more arrays or quantities corresponding to the subscripts.
      optimize: determine whether to optimize the order of computation. In JAX
        this defaults to ``"optimize"`` which produces optimized expressions via
        the opt_einsum_ package.
      precision: either ``None`` (default), which means the default precision for
        the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
        ``Precision.HIGH`` or ``Precision.HIGHEST``).
      preferred_element_type: either ``None`` (default), which means the default
        accumulation type for the input types, or a datatype, indicating to
        accumulate results to and return a result with that datatype.
      out: unsupported by JAX
      _dot_general: optionally override the ``dot_general`` callable used by ``einsum``.
        This parameter is experimental, and may be removed without warning at any time.

    Returns:
      array containing the result of the einstein summation.
  '''
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
    from jax._src.numpy.lax_numpy import _default_poly_einsum_handler
    contract_path = _default_poly_einsum_handler

  operands, contractions = contract_path(
    *operands, einsum_call=True, use_blas=True, optimize=optimize)

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

  contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)

  einsum = jax.jit(_einsum, static_argnums=(1, 2, 3, 4), inline=True)
  if spec is not None:
    einsum = jax.named_call(einsum, name=spec)
  operands = [op.value if isinstance(op, Quantity) else op for op in operands]
  r = einsum(operands, contractions, precision,  # type: ignore[operator]
             preferred_element_type, _dot_general)
  if unit is not None:
    return Quantity(r, dim=unit)
  else:
    return r


@set_module_as('brainunit.math')
def gradient(
    f: Union[bst.typing.ArrayLike, Quantity],
    *varargs: Union[bst.typing.ArrayLike, Quantity],
    axis: Union[int, Sequence[int], None] = None,
    edge_order: Union[int, None] = None,
) -> Union[jax.Array, list[jax.Array], Quantity, list[Quantity]]:
  '''
  Computes the gradient of a scalar field.

  Args:
    f: input array.
    *varargs: list of scalar fields to compute the gradient.
    axis: axis or axes along which to compute the gradient. The default is to compute the gradient along all axes.
    edge_order: order of the edge used for the finite difference computation. The default is 1.

  Returns:
    array containing the gradient of the scalar field.
  '''
  if edge_order is not None:
    raise NotImplementedError("The 'edge_order' argument to jnp.gradient is not supported.")

  if len(varargs) == 0:
    if isinstance(f, Quantity) and not is_unitless(f):
      return Quantity(jnp.gradient(f.value, axis=axis), dim=f.dim)
    else:
      return jnp.gradient(f)
  elif len(varargs) == 1:
    unit = get_unit(f) / get_unit(varargs[0])
    if unit is None or unit == DIMENSIONLESS:
      return jnp.gradient(f, varargs[0], axis=axis)
    else:
      return [Quantity(r, dim=unit) for r in jnp.gradient(f.value, varargs[0].value, axis=axis)]
  else:
    unit_list = [get_unit(f) / get_unit(v) for v in varargs]
    f = f.value if isinstance(f, Quantity) else f
    varargs = [v.value if isinstance(v, Quantity) else v for v in varargs]
    result_list = jnp.gradient(f, *varargs, axis=axis)
    return [Quantity(r, dim=unit) if unit is not None else r for r, unit in zip(result_list, unit_list)]


@set_module_as('brainunit.math')
def intersect1d(
    ar1: Union[bst.typing.ArrayLike],
    ar2: Union[bst.typing.ArrayLike],
    assume_unique: bool = False,
    return_indices: bool = False
) -> Union[jax.Array, Quantity, tuple[Union[jax.Array, Quantity], jax.Array, jax.Array]]:
  '''
  Find the intersection of two arrays.

  Args:
    ar1: input array.
    ar2: input array.
    assume_unique: if True, the input arrays are both assumed to be unique.
    return_indices: if True, the indices which correspond to the intersection of the two arrays are returned.

  Returns:
    array containing the intersection of the two arrays.
  '''
  fail_for_dimension_mismatch(ar1, ar2, 'intersect1d')
  unit = None
  if isinstance(ar1, Quantity):
    unit = ar1.dim
  ar1 = ar1.value if isinstance(ar1, Quantity) else ar1
  ar2 = ar2.value if isinstance(ar2, Quantity) else ar2
  result = jnp.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
  if return_indices:
    if unit is not None:
      return (Quantity(result[0], dim=unit), result[1], result[2])
    else:
      return result
  else:
    if unit is not None:
      return Quantity(result, dim=unit)
    else:
      return result


@wrap_math_funcs_keep_unit_unary
def nan_to_num(x: Union[bst.typing.ArrayLike, Quantity], nan: float = 0.0, posinf: float = jnp.inf,
               neginf: float = -jnp.inf) -> Union[jax.Array, Quantity]:
  return jnp.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


@wrap_math_funcs_keep_unit_unary
def rot90(m: Union[bst.typing.ArrayLike, Quantity], k: int = 1, axes: Tuple[int, int] = (0, 1)) -> Union[
  jax.Array, Quantity]:
  return jnp.rot90(m, k=k, axes=axes)


@wrap_math_funcs_change_unit_binary(lambda x, y: x * y)
def tensordot(a: Union[bst.typing.ArrayLike, Quantity], b: Union[bst.typing.ArrayLike, Quantity],
              axes: Union[int, Tuple[int, int]] = 2) -> Union[jax.Array, Quantity]:
  return jnp.tensordot(a, b, axes=axes)


@_compatible_with_quantity(return_quantity=False)
def nanargmax(a: Union[bst.typing.ArrayLike, Quantity], axis: int = None) -> jax.Array:
  return jnp.nanargmax(a, axis=axis)


@_compatible_with_quantity(return_quantity=False)
def nanargmin(a: Union[bst.typing.ArrayLike, Quantity], axis: int = None) -> jax.Array:
  return jnp.nanargmin(a, axis=axis)


# docs for functions above
nan_to_num.__doc__ = '''
  Replace NaN with zero and infinity with large finite numbers (default behaviour) or with the numbers defined by the user using the `nan`, `posinf` and `neginf` arguments.

  Args:
    x: input array.
    nan: value to replace NaNs with.
    posinf: value to replace positive infinity with.
    neginf: value to replace negative infinity with.

  Returns:
    array with NaNs replaced by zero and infinities replaced by large finite numbers.
'''

nanargmax.__doc__ = '''
  Return the index of the maximum value in an array, ignoring NaNs.

  Args:
    a: array like, Quantity.
    axis: axis along which to operate. The default is to compute the index of the maximum over all the dimensions of the input array.
    out: output array, optional.
    keepdims: if True, the result is broadcast to the input array with the same number of dimensions.

  Returns:
    index of the maximum value in the array.
'''

nanargmin.__doc__ = '''
  Return the index of the minimum value in an array, ignoring NaNs.

  Args:
    a: array like, Quantity.
    axis: axis along which to operate. The default is to compute the index of the minimum over all the dimensions of the input array.
    out: output array, optional.
    keepdims: if True, the result is broadcast to the input array with the same number of dimensions.

  Returns:
    index of the minimum value in the array.
'''

rot90.__doc__ = '''
  Rotate an array by 90 degrees in the plane specified by axes.

  Args:
    m: array like, Quantity.
    k: number of times the array is rotated by 90 degrees.
    axes: plane of rotation. Default is the last two axes.

  Returns:
    rotated array.
'''

tensordot.__doc__ = '''
  Compute tensor dot product along specified axes for arrays.

  Args:
    a: array like, Quantity.
    b: array like, Quantity.
    axes: axes along which to compute the tensor dot product.

  Returns:
    tensor dot product of the two arrays.
'''
