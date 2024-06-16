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
import itertools
import numbers
import operator
from contextlib import contextmanager
from typing import Union, Optional, Sequence, Callable, Tuple, Any, List

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters.partial_eval import DynamicJaxprTracer
from jax.tree_util import register_pytree_node_class

__all__ = [
  'Quantity',
  'Unit',
  'UnitRegistry',
  'Dimension',
  'DIMENSIONLESS',
  'DimensionMismatchError',
  'get_or_create_dimension',
  'get_dim',
  'get_basic_unit',
  'is_unitless',
  'have_same_unit',
  'in_unit',
  'in_best_unit',
  'register_new_unit',
  'check_units',
  'is_scalar_type',
  'fail_for_dimension_mismatch',
]

_all_slice = slice(None, None, None)
_unit_checking = True
_allow_python_scalar_value = False
_auto_register_unit = True


@contextmanager
def turn_off_auto_unit_register():
  try:
    global _auto_register_unit
    _auto_register_unit = False
    yield
  finally:
    _auto_register_unit = True


@contextmanager
def allow_python_scalar():
  try:
    global _allow_python_scalar_value
    _allow_python_scalar_value = True
    yield
  finally:
    _allow_python_scalar_value = False


@contextmanager
def turn_off_unit_checking():
  try:
    global _unit_checking
    _unit_checking = False
    yield
  finally:
    _unit_checking = True


def _to_quantity(array):
  if isinstance(array, Quantity):
    return array
  elif isinstance(array, (numbers.Number, jax.Array, np.number, np.ndarray, list, tuple)):
    return Quantity(value=array)
  else:
    raise TypeError('Input array should be an instance of Array.')


def _assert_not_quantity(array):
  if isinstance(array, Quantity):
    raise ValueError('Input array should not be an instance of Array.')
  return array


def _short_str(arr):
  """
  Return a short string representation of an array, suitable for use in
  error messages.
  """
  arr = arr.value if isinstance(arr, Quantity) else arr
  arr = np.asanyarray(arr)
  old_printoptions = jnp.get_printoptions()
  jnp.set_printoptions(edgeitems=2, threshold=5)
  arr_string = str(arr)
  jnp.set_printoptions(**old_printoptions)
  return arr_string


def get_unit_for_display(d):
  """
  Return a string representation of an appropriate unscaled unit or ``'1'``
  for a dimensionless array.

  Parameters
  ----------
  d : Dimension or int
      The dimension to find a unit for.

  Returns
  -------
  s : str
      A string representation of the respective unit or the string ``'1'``.
  """
  if (isinstance(d, int) and d == 1) or d is DIMENSIONLESS:
    return "1"
  else:
    return str(get_dim(d))


# SI dimensions (see table at the top of the file) and various descriptions,
# each description maps to an index i, and the power of each dimension
# is stored in the variable dims[i]
_di = {
  "Length": 0,
  "length": 0,
  "metre": 0,
  "metres": 0,
  "meter": 0,
  "meters": 0,
  "m": 0,
  "Mass": 1,
  "mass": 1,
  "kilogram": 1,
  "kilograms": 1,
  "kg": 1,
  "Time": 2,
  "time": 2,
  "second": 2,
  "seconds": 2,
  "s": 2,
  "Electric Current": 3,
  "electric current": 3,
  "Current": 3,
  "current": 3,
  "ampere": 3,
  "amperes": 3,
  "A": 3,
  "Temperature": 4,
  "temperature": 4,
  "kelvin": 4,
  "kelvins": 4,
  "K": 4,
  "Quantity of Substance": 5,
  "Quantity of substance": 5,
  "quantity of substance": 5,
  "Substance": 5,
  "substance": 5,
  "mole": 5,
  "moles": 5,
  "mol": 5,
  "Luminosity": 6,
  "luminosity": 6,
  "candle": 6,
  "candles": 6,
  "cd": 6,
}

_ilabel = ["m", "kg", "s", "A", "K", "mol", "cd"]

# The same labels with the names used for constructing them in Python code
_iclass_label = ["metre", "kilogram", "second", "amp", "kelvin", "mole", "candle"]

# SI unit _prefixes as integer exponents of 10, see table at end of file.
_siprefixes = {
  "y": -24,
  "z": -21,
  "a": -18,
  "f": -15,
  "p": -12,
  "n": -9,
  "u": -6,
  "m": -3,
  "c": -2,
  "d": -1,
  "": 0,
  "da": 1,
  "h": 2,
  "k": 3,
  "M": 6,
  "G": 9,
  "T": 12,
  "P": 15,
  "E": 18,
  "Z": 21,
  "Y": 24,
}


class Dimension:
  """
  Stores the indices of the 7 basic SI unit dimension (length, mass, etc.).

  Provides a subset of arithmetic operations appropriate to dimensions:
  multiplication, division and powers, and equality testing.

  Parameters
  ----------
  dims : sequence of `float`
      The dimension indices of the 7 basic SI unit dimensions.

  Notes
  -----
  Users shouldn't use this class directly, it is used internally in Array
  and Unit. Even internally, never use ``Dimension(...)`` to create a new
  instance, use `get_or_create_dimension` instead. This function makes
  sure that only one Dimension instance exists for every combination of
  indices, allowing for a very fast dimensionality check with ``is``.
  """

  __module__ = "brainunit"
  __slots__ = ["_dims"]
  __array_priority__ = 1000

  # ---- INITIALISATION ---- #

  def __init__(self, dims):
    self._dims = dims

  # ---- METHODS ---- #
  def get_dimension(self, d):
    """
    Return a specific dimension.

    Parameters
    ----------
    d : `str`
        A string identifying the SI basic unit dimension. Can be either a
        description like "length" or a basic unit like "m" or "metre".

    Returns
    -------
    dim : `float`
        The dimensionality of the dimension `d`.
    """
    return self._dims[_di[d]]

  @property
  def is_unitless(self):
    """
    Whether this Dimension is dimensionless.

    Notes
    -----
    Normally, instead one should check dimension for being identical to
    `DIMENSIONLESS`.
    """
    return all([x == 0 for x in self._dims])

  @property
  def dim(self):
    """
    Returns the `Dimension` object itself. This can be useful, because it
    allows to check for the dimension of an object by checking its ``dim``
    attribute -- this will return a `Dimension` object for `Array`,
    `Unit` and `Dimension`.
    """
    return self

  @property
  def unit(self):
    return self

  # ---- REPRESENTATION ---- #
  def _str_representation(self, python_code=False):
    """
    String representation in basic SI units, or ``"1"`` for dimensionless.
    Use ``python_code=False`` for display purposes and ``True`` for valid
    Python code.
    """

    if python_code:
      power_operator = " ** "
    else:
      power_operator = "^"

    parts = []
    for i in range(len(self._dims)):
      if self._dims[i]:
        if python_code:
          s = _iclass_label[i]
        else:
          s = _ilabel[i]
        if self._dims[i] != 1:
          s += power_operator + str(self._dims[i])
        parts.append(s)
    if python_code:
      s = " * ".join(parts)
      if not len(s):
        return f"{self.__class__.__name__}()"
    else:
      s = " ".join(parts)
      if not len(s):
        return "1"
    return s.strip()

  def __repr__(self):
    return self._str_representation(python_code=True)

  def __str__(self):
    return self._str_representation(python_code=False)

  # ---- ARITHMETIC ---- #
  # Note that none of the dimension arithmetic objects do sanity checking
  # on their inputs, although most will throw an exception if you pass the
  # wrong sort of input
  def __mul__(self, value):
    return get_or_create_dimension([x + y for x, y in zip(self._dims, value._dims)])

  def __div__(self, value):
    return get_or_create_dimension([x - y for x, y in zip(self._dims, value._dims)])

  def __truediv__(self, value):
    return self.__div__(value)

  def __pow__(self, value: numbers.Number | jax.Array):
    if value is DIMENSIONLESS:
      return self
    value = np.array(value)
    if value.size > 1:
      raise TypeError("Too many exponents")
    return get_or_create_dimension([x * value for x in self._dims])

  def __imul__(self, value):
    raise TypeError("Dimension object is immutable")

  def __idiv__(self, value):
    raise TypeError("Dimension object is immutable")

  def __itruediv__(self, value):
    raise TypeError("Dimension object is immutable")

  def __ipow__(self, value):
    raise TypeError("Dimension object is immutable")

  # ---- COMPARISON ---- #
  def __eq__(self, value):
    try:
      return np.allclose(self._dims, value._dims)
    except AttributeError:
      # Only compare equal to another Dimensions object
      return False

  def __ne__(self, value):
    return not self.__eq__(value)

  def __hash__(self):
    return hash(self._dims)

  # MAKE DIMENSION PICKABLE #
  def __getstate__(self):
    return self._dims

  def __setstate__(self, state):
    self._dims = state

  def __reduce__(self):
    # Make sure that unpickling Dimension objects does not bypass the singleton system
    return get_or_create_dimension, (self._dims,)

  # --- Dimension objects are singletons and deepcopy is therefore not necessary
  def __deepcopy__(self, memodict):
    return self


def get_or_create_dimension(*args, **kwds):
  """
  Create a new Dimension object or get a reference to an existing one.
  This function takes care of only creating new objects if they were not
  created before and otherwise returning a reference to an existing object.
  This allows to compare dimensions very efficiently using ``is``.

  Parameters
  ----------
  args : sequence of `float`
      A sequence with the indices of the 7 elements of an SI dimension.
  kwds : keyword arguments
      a sequence of ``keyword=value`` pairs where the keywords are the names of
      the SI dimensions, or the standard unit.

  Examples
  --------
  The following are all definitions of the dimensions of force

  >>> from brainunit import *
  >>> get_or_create_dimension(length=1, mass=1, time=-2)
  metre * kilogram * second ** -2
  >>> get_or_create_dimension(m=1, kg=1, s=-2)
  metre * kilogram * second ** -2
  >>> get_or_create_dimension([1, 1, -2, 0, 0, 0, 0])
  metre * kilogram * second ** -2

  Notes
  -----
  The 7 units are (in order):

  Length, Mass, Time, Electric Current, Temperature,
  Quantity of Substance, Luminosity

  and can be referred to either by these names or their SI unit names,
  e.g. length, metre, and m all refer to the same thing here.
  """
  if len(args):
    # initialisation by list
    dims = args[0]
    try:
      if len(dims) != 7:
        raise TypeError()
    except TypeError:
      raise TypeError("Need a sequence of exactly 7 items")
  else:
    # initialisation by keywords
    dims = [0, 0, 0, 0, 0, 0, 0]
    for k in kwds:
      # _di stores the index of the dimension with name 'k'
      dims[_di[k]] = kwds[k]

  dims = tuple(dims)

  # check whether this Dimension object has already been created
  if dims in _dimensions:
    return _dimensions[dims]
  else:
    new_dim = Dimension(dims)
    _dimensions[dims] = new_dim
    return new_dim


'''The dimensionless unit, used for quantities without a unit.'''
DIMENSIONLESS = Dimension((0, 0, 0, 0, 0, 0, 0))
_dimensions = {(0, 0, 0, 0, 0, 0, 0): DIMENSIONLESS}


class DimensionMismatchError(Exception):
  """
  Exception class for attempted operations with inconsistent dimensions.

  For example, ``3*mvolt + 2*amp`` raises this exception. The purpose of this
  class is to help catch errors based on incorrect units. The exception will
  print a representation of the dimensions of the two inconsistent objects
  that were operated on.

  Parameters
  ----------
  description : ``str``
      A description of the type of operation being performed, e.g. Addition,
      Multiplication, etc.
  dims : Dimension
      The physical dimensions of the objects involved in the operation, any
      number of them is possible
  """
  __module__ = "brainunit"

  def __init__(self, description, *dims):
    # Call the base class constructor to make Exception pickable, see:
    # http://bugs.python.org/issue1692335
    super().__init__(description, *dims)
    self.dims: Tuple = dims
    self.desc = description

  def __repr__(self):
    dims_repr = [repr(dim) for dim in self.dims]
    return f"{self.__class__.__name__}({self.desc!r}, {', '.join(dims_repr)})"

  def __str__(self):
    s = self.desc
    if len(self.dims) == 0:
      pass
    elif len(self.dims) == 1:
      s += f" (unit is {get_unit_for_display(self.dims[0])}"
    elif len(self.dims) == 2:
      d1, d2 = self.dims
      s += (
        f" (units are {get_unit_for_display(d1)} and {get_unit_for_display(d2)}"
      )
    else:
      s += (
        " (units are"
        f" {' '.join([f'({get_unit_for_display(d)})' for d in self.dims])}"
      )
    if len(self.dims):
      s += ")."
    return s


def get_dim(obj) -> Dimension:
  """
  Return the unit of any object that has them.

  Slightly more general than `Array.dimensions` because it will
  return `DIMENSIONLESS` if the object is of number type but not a `Array`
  (e.g. a `float` or `int`).

  Parameters
  ----------
  obj : `object`
      The object to check.

  Returns
  -------
  dim : Dimension
      The physical dimensions of the `obj`.
  """
  try:
    return obj.dim
  except AttributeError:
    # The following is not very pretty, but it will avoid the costly
    # isinstance check for the common types
    if isinstance(obj, (numbers.Number, jax.Array, np.number, np.ndarray)):
      return DIMENSIONLESS
    try:
      return Quantity(obj).dim
    except TypeError:
      raise TypeError(f"Object of type {type(obj)} does not have dimensions")


def have_same_unit(obj1, obj2) -> bool:
  """Test if two values have the same dimensions.

  Parameters
  ----------
  obj1, obj2 : {`Array`, array-like, number}
      The values of which to compare the dimensions.

  Returns
  -------
  same : `bool`
      ``True`` if `obj1` and `obj2` have the same dimensions.
  """

  if not _unit_checking:
    return True  # ignore units when unit checking is disabled

  # If dimensions are consistently created using get_or_create_dimensions,
  #   the fast "is" comparison should always return the correct result.
  #   To be safe, we also do an equals comparison in case it fails. This
  #   should only add a small amount of unnecessary computation for cases in
  #   which this function returns False which very likely leads to a
  #   DimensionMismatchError anyway.
  dim1 = get_dim(obj1)
  dim2 = get_dim(obj2)
  return (dim1 is dim2) or (dim1 == dim2) or dim1 is None or dim2 is None


def fail_for_dimension_mismatch(
    obj1, obj2=None, error_message=None, **error_arrays
):
  """
  Compare the dimensions of two objects.

  Parameters
  ----------
  obj1, obj2 : {array-like, `Array`}
      The object to compare. If `obj2` is ``None``, assume it to be
      dimensionless
  error_message : str, optional
      An error message that is used in the DimensionMismatchError
  error_arrays : dict mapping str to `Array`, optional
      Arrays in this dictionary will be converted using the `_short_str`
      helper method and inserted into the ``error_message`` (which should
      have placeholders with the corresponding names). The reason for doing
      this in a somewhat complicated way instead of directly including all the
      details in ``error_messsage`` is that converting large arrays
      to strings can be rather costly and we don't want to do it if no error
      occured.

  Returns
  -------
  dim1, dim2 : Dimension, `Dimension`
      The physical dimensions of the two arguments (so that later code does
      not need to get the dimensions again).

  Raises
  ------
  DimensionMismatchError
      If the dimensions of `obj1` and `obj2` do not match (or, if `obj2` is
      ``None``, in case `obj1` is not dimensionsless).

  Notes
  -----
  Implements special checking for ``0``, treating it as having "any
  dimensions".
  """
  if not _unit_checking:
    return None, None

  dim1 = get_dim(obj1)
  if obj2 is None:
    dim2 = DIMENSIONLESS
  else:
    dim2 = get_dim(obj2)

  if dim1 is not dim2 and not (dim1 is None or dim2 is None):
    # Special treatment for "0":
    #     if it is not a Array, it has "any dimension".
    #     This allows expressions like 3*mV + 0 to pass (useful in cases where
    #     zero is treated as the neutral element, e.g. in the Python sum
    #     builtin) or comparisons like 3 * mV == 0 to return False instead of
    #     failing # with a DimensionMismatchError. Note that 3*mV == 0*second
    #     is not allowed, though.

    if (dim1 is DIMENSIONLESS and jnp.all(obj1 == 0)) or (
        dim2 is DIMENSIONLESS and jnp.all(obj2 == 0)
    ):
      return dim1, dim2

    # We do another check here, this should allow Brian1 units to pass as
    # having the same dimensions as a Brian2 unit
    if dim1 == dim2:
      return dim1, dim2

    if error_message is None:
      error_message = "Dimension mismatch"
    else:
      error_arrays = {
        name: _short_str(q) for name, q in error_arrays.items()
      }
      error_message = error_message.format(**error_arrays)
    # If we are comparing an object to a specific unit, we don't want to
    # restate this unit (it is probably mentioned in the text already)
    if obj2 is None or isinstance(obj2, (Dimension, Unit)):
      raise DimensionMismatchError(error_message, dim1)
    else:
      raise DimensionMismatchError(error_message, dim1, dim2)
  else:
    return dim1, dim2


def in_unit(x, u, precision=None) -> str:
  """
  Display a value in a certain unit with a given precision.

  Parameters
  ----------
  x : {`Array`, array-like, number}
      The value to display
  u : {`Array`, `Unit`}
      The unit to display the value `x` in.
  precision : `int`, optional
      The number of digits of precision (in the given unit, see Examples).
      If no value is given, numpy's `get_printoptions` value is used.

  Returns
  -------
  s : `str`
      A string representation of `x` in units of `u`.

  Examples
  --------
  >>> from brainunit import *
  >>> in_unit(3 * volt, mvolt)
  '3000. mV'
  >>> in_unit(123123 * msecond, second, 2)
  '123.12 s'
  >>> in_unit(10 * uA/cm**2, nA/um**2)
  '1.00000000e-04 nA/(um^2)'
  >>> in_unit(10 * mV, ohm * amp)
  '0.01 ohm A'
  >>> in_unit(10 * nS, ohm) # doctest: +NORMALIZE_WHITESPACE
  ...                       # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
      ...
  DimensionMismatchError: Non-matching unit for method "in_unit",
  dimensions were (m^-2 kg^-1 s^3 A^2) (m^2 kg s^-3 A^-2)

  See Also
  --------
  Array.in_unit
  """
  if is_unitless(x):
    fail_for_dimension_mismatch(x, u, 'Non-matching unit for function "in_unit"')
    return str(jnp.array(x / u))
  else:
    return x.repr_in_unit(u, precision=precision)


def in_best_unit(x, precision=None):
  """
  Represent the value in the "best" unit.

  Parameters
  ----------
  x : {`Array`, array-like, number}
      The value to display
  precision : `int`, optional
      The number of digits of precision (in the best unit, see Examples).
      If no value is given, numpy's `get_printoptions` value is used.

  Returns
  -------
  representation : `str`
      A string representation of this `Array`.

  Examples
  --------
  >>> from brainunit import *
  >>> in_best_unit(0.00123456 * volt)
  '1.23456 mV'
  >>> in_best_unit(0.00123456 * volt, 2)
  '1.23 mV'
  >>> in_best_unit(0.123456)
  '0.123456'
  >>> in_best_unit(0.123456, 2)
  '0.12'

  See Also
  --------
  Array.in_best_unit
  """
  if is_unitless(x):
    if precision is None:
      precision = jnp.get_printoptions()["precision"]
    return str(jnp.round(x, precision))

  u = x.get_best_unit()
  return x.repr_in_unit(u, precision=precision)


def array_with_unit(
    floatval,
    unit: Dimension,
    dtype: jax.typing.DTypeLike = None
) -> 'Quantity':
  """
  Create a new `Array` with the given dimensions. Calls
  `get_or_create_dimension` with the dimension tuple of the `dims`
  argument to make sure that unpickling (which calls this function) does not
  accidentally create new Dimension objects which should instead refer to
  existing ones.

  Parameters
  ----------
  floatval : `float`
      The floating point value of the array.
  unit: Dimension
      The unit dimensions of the array.
  dtype: `dtype`, optional
      The data type of the array.

  Returns
  -------
  array : `Quantity`
      The new `Array` object.

  Examples
  --------
  >>> from brainunit import *
  >>> array_with_unit(0.001, volt.dim)
  1. * mvolt
  """
  return Quantity(floatval, dim=get_or_create_dimension(unit._dims), dtype=dtype)


def is_unitless(obj) -> bool:
  """
  Test if a value is dimensionless or not.

  Parameters
  ----------
  obj : `object`
      The object to check.

  Returns
  -------
  dimensionless : `bool`
      ``True`` if `obj` is dimensionless.
  """
  return get_dim(obj) is DIMENSIONLESS


def is_scalar_type(obj) -> bool:
  """
  Tells you if the object is a 1d number type.

  Parameters
  ----------
  obj : `object`
      The object to check.

  Returns
  -------
  scalar : `bool`
      ``True`` if `obj` is a scalar that can be interpreted as a
      dimensionless `Array`.
  """
  try:
    return obj.ndim == 0 and is_unitless(obj)
  except AttributeError:
    return jnp.isscalar(obj) and not isinstance(obj, str)


def wrap_function_keep_dimensions(func):
  """
  Returns a new function that wraps the given function `func` so that it
  keeps the dimensions of its input. Arrays are transformed to
  unitless jax numpy arrays before calling `func`, the output is a array
  with the original dimensions re-attached.

  These transformations apply only to the very first argument, all
  other arguments are ignored/untouched, allowing to work functions like
  ``sum`` to work as expected with additional ``axis`` etc. arguments.
  """

  def f(x, *args, **kwds):  # pylint: disable=C0111
    return Quantity(func(x.value, *args, **kwds), dim=x.dim)

  f._arg_units = [None]
  f._return_unit = lambda u: u
  f.__name__ = func.__name__
  f.__doc__ = func.__doc__
  f._do_not_run_doctests = True
  return f


def wrap_function_change_dimensions(func, change_dim_func):
  """
  Returns a new function that wraps the given function `func` so that it
  changes the dimensions of its input. Arrays are transformed to
  unitless jax numpy arrays before calling `func`, the output is a array
  with the original dimensions passed through the function
  `change_dim_func`. A typical use would be a ``sqrt`` function that uses
  ``lambda d: d ** 0.5`` as ``change_dim_func``.

  These transformations apply only to the very first argument, all
  other arguments are ignored/untouched.
  """

  def f(x, *args, **kwds):  # pylint: disable=C0111
    assert isinstance(x, Quantity), "Only Quantity objects can be passed to this function"
    return _return_check_unitless(Quantity(func(x.value, *args, **kwds), dim=change_dim_func(x.value, x.dim)))

  f._arg_units = [None]
  f._return_unit = change_dim_func
  f.__name__ = func.__name__
  f.__doc__ = func.__doc__
  f._do_not_run_doctests = True
  return f


def wrap_function_remove_dimensions(func):
  """
  Returns a new function that wraps the given function `func` so that it
  removes any dimensions from its input. Useful for functions that are
  returning integers (indices) or booleans, irrespective of the datatype
  contained in the array.

  These transformations apply only to the very first argument, all
  other arguments are ignored/untouched.
  """

  def f(x, *args, **kwds):  # pylint: disable=C0111
    assert isinstance(x, Quantity), "Only Quantity objects can be passed to this function"
    return func(x.value, *args, **kwds)

  f._arg_units = [None]
  f._return_unit = 1
  f.__name__ = func.__name__
  f.__doc__ = func.__doc__
  f._do_not_run_doctests = True
  return f


def _return_check_unitless(q):
  if q.is_unitless:
    return q.value
  else:
    return q


def process_list_with_units(value):
  def check_units_and_collect_values(lst):
    all_dims = []
    values = []

    for item in lst:
      if isinstance(item, list):
        val, dim = check_units_and_collect_values(item)
        values.append(val)
        if dim is not None:
          all_dims.append(dim)
      elif isinstance(item, Quantity):
        values.append(item.value)
        all_dims.append(item.dim)
      else:
        values.append(item)
        all_dims.append(DIMENSIONLESS)

    if all_dims:
      first_unit = all_dims[0]
      if not all(unit == first_unit for unit in all_dims):
        raise TypeError("All elements must have the same unit")
      return values, first_unit
    else:
      return values, DIMENSIONLESS

  values, dim = check_units_and_collect_values(value)
  return values, dim


def _get_dim(dim: Dimension, unit: 'Unit'):
  if dim != DIMENSIONLESS and unit is not None:
    raise ValueError("Cannot specify both a dimension and a unit")
  if dim == DIMENSIONLESS:
    if unit is None:
      return None, DIMENSIONLESS
    else:
      return unit.value, unit.dim
  else:
    return None, dim


@register_pytree_node_class
class Quantity(object):
  """
  The `Quantity` class represents a physical quantity with a value and a
  unit. It is used to represent all physical quantities in ``BrainCore``.

  """
  __module__ = "brainunit"
  __slots__ = ('_value', '_dim')
  _value: Union[jax.Array, numbers.Number]
  _dim: Dimension
  __array_priority__ = 1000

  def __init__(
      self,
      value: Any,
      dtype: Optional[jax.typing.DTypeLike] = None,
      dim: Dimension = DIMENSIONLESS,
      unit: Optional['Unit'] = None,
  ):
    scale, dim = _get_dim(dim, unit)

    # always allow python scalar
    if isinstance(value, numbers.Number):
      self._dim = dim
      self._value = (value if scale is None else (value * scale))
      return

    if isinstance(value, (list, tuple)):
      value, new_dim = process_list_with_units(value)
      if dim == DIMENSIONLESS:
        dim = new_dim
      elif new_dim != DIMENSIONLESS:
        if dim != new_dim:
          raise TypeError(f"All elements must have the same unit. But got {dim} != {new_dim}")
      try:
        # Transform to jnp array
        value = jnp.array(value, dtype=dtype)
      except ValueError:
        raise TypeError("All elements must be convertible to a jax array")

    # array value
    if isinstance(value, Quantity):
      self._dim = value.dim
      self._value = jnp.array(value.value, dtype=dtype)
      return

    elif isinstance(value, (np.ndarray, jax.Array)):
      value = jnp.array(value, dtype=dtype)

    elif isinstance(value, (jnp.number, numbers.Number)):
      value = jnp.array(value, dtype=dtype)

    elif isinstance(value, (jax.core.ShapedArray, jax.ShapeDtypeStruct)):
      value = value

    else:
      value = value

    # value
    self._value = (value if scale is None else (value * scale))

    # dimension
    self._dim = dim

  @property
  def value(self) -> jax.Array | numbers.Number:
    # return the value
    return self._value

  @value.setter
  def value(self, value):
    # Do not support setting the value directly
    raise NotImplementedError("Cannot set the value of a Quantity object directly,"
                              "Please create a new Quantity object with the value you want.")

  def update_value(self, value):
    """
    Set the value of the array.

    Examples::

    >>> a = jax.numpy.array([1, 2, 3]) * mV
    >>> a[:] = jax.numpy.array([4, 5, 6]) * mV
    >>> a.value = jax.numpy.array([7, 8, 9])

    Args:
      value: The new value of the array.
    """
    self_value = self._check_tracer()
    if isinstance(value, Quantity):
      raise ValueError("Cannot set the value of an Array object to another Array object.")
    if isinstance(value, np.ndarray):
      value = jnp.asarray(value, dtype=self.dtype)
    elif isinstance(value, jax.Array):
      pass
    else:
      value = jnp.asarray(value, dtype=self.dtype)
    # check
    if value.shape != jnp.shape(self_value):
      raise ValueError(f"The shape of the original data is {jnp.shape(self_value)}, "
                       f"while we got {value.shape}.")
    if value.dtype != jax.dtypes.result_type(self_value):
      raise ValueError(f"The dtype of the original data is {jax.dtypes.result_type(self_value)}, "
                       f"while we got {value.dtype}.")
    self._value = value

  @property
  def dim(self) -> Dimension:
    """
    The physical unit dimensions of this Array
    """
    return self._dim

  @dim.setter
  def dim(self, *args):
    # Do not support setting the unit directly
    raise NotImplementedError("Cannot set the dimension of a Quantity object directly,"
                              "Please create a new Quantity object with the value you want.")

  @property
  def unit(self) -> 'Unit':
    return Unit(1., self.dim, register=False)

  @unit.setter
  def unit(self, *args):
    # Do not support setting the unit directly
    raise NotImplementedError("Cannot set the unit of a Quantity object directly,"
                              "Please create a new Quantity object with the unit you want.")

  def to_value(self, unit: 'Unit') -> jax.Array | numbers.Number:
    """
    Convert the value of the array to a new unit.

    Examples::

    >>> a = jax.numpy.array([1, 2, 3]) * mV
    >>> a.to_value(volt)
    array([0.001, 0.002, 0.003])

    Args:
      unit: The new unit to convert the value of the array to.

    Returns:
      The value of the array in the new unit.
    """
    return self.value / unit.value

  @staticmethod
  def with_units(value, *args, **keywords):
    """
    Create a `Array` object with the given units.

    Parameters
    ----------
    value : {array_like, number}
        The value of the dimension
    args : {`Dimension`, sequence of float}
        Either a single argument (a `Dimension`) or a sequence of 7 values.
    keywords
        Keywords defining the dim, see `Dimension` for details.

    Returns
    -------
    q : `Quantity`
        A `Array` object with the given dim

    Examples
    --------
    All of these define an equivalent `Array` object:

    >>> from brainunit import *
    >>> Quantity.with_units(2, get_or_create_dimension(length=1))
    2. * metre
    >>> Quantity.with_units(2, length=1)
    2. * metre
    >>> 2 * metre
    2. * metre
    """
    if len(args) and isinstance(args[0], Dimension):
      dimensions = args[0]
    else:
      dimensions = get_or_create_dimension(*args, **keywords)
    return Quantity(value, dim=dimensions)

  @property
  def is_unitless(self) -> bool:
    """
    Whether the array does not have unit.

    Returns:
      bool: True if the array does not have unit.
    """
    return self.dim.is_unitless

  def has_same_unit(self, other):
    """
    Whether this Array has the same unit dimensions as another Array

    Parameters
    ----------
    other : Quantity
        The other Array to compare with

    Returns
    -------
    bool
        Whether the two Arrays have the same unit dimensions
    """
    if not _unit_checking:
      return True
    other_unit = get_dim(other.dim)
    return (get_dim(self.dim) is other_unit) or (get_dim(self.dim) == other_unit)

  def get_best_unit(self, *regs) -> 'Quantity':
    """
    Return the best unit for this `Array`.

    Parameters
    ----------
    regs : any number of `UnitRegistry objects
        The registries that are searched for units. If none are provided, it
        will check the standard, user and additional unit registers in turn.

    Returns
    -------
    u : `Quantity` or `Unit`
        The best unit for this `Array`.
    """
    if self.is_unitless:
      return Unit(1)
    if len(regs):
      for r in regs:
        try:
          return r[self]
        except KeyError:
          pass
      return Quantity(1, dim=self.dim)
    else:
      return self.get_best_unit(standard_unit_register, user_unit_register, additional_unit_register)

  def repr_in_unit(
      self,
      u: 'Unit',
      precision: int | None = None,
      python_code: bool = False
  ) -> str:
    """
    Represent the Array in a given unit.

    Parameters
    ----------
    u : `Unit`
        The unit in which to show the ar.
    precision : `int`, optional
        The number of digits of precision (in the given unit)
        If no value is given, numpy's `get_printoptions` is used.
    python_code : `bool`, optional
        Whether to return a string that can be used as python code.
        If True, the string will be formatted as a python expression.
        If False, the string will be formatted as a human-readable string.

    Returns
    -------
    s : `str`
        The string representation of the Array in the given unit.

    Examples
    --------
    >>> from brainunit import *
    >>> x = 25.123456 * mV
    >>> x.repr_in_unit(volt)
    '0.02512346 V'
    >>> x.repr_in_unit(volt, 3)
    '0.025 V'
    >>> x.repr_in_unit(mV, 3)
    '25.123 mV'
    """
    fail_for_dimension_mismatch(self, u, 'Non-matching unit for method "in_unit"')
    value = jnp.asarray(self.value / u.value)
    if isinstance(value, (jax.ShapeDtypeStruct, jax.core.ShapedArray, DynamicJaxprTracer)):
      s = str(value)
    else:
      if value.shape == ():
        s = jnp.array_str(jnp.array([value]), precision=precision)
        s = s.replace("[", "").replace("]", "").strip()
      else:
        if value.size > 100:
          if python_code:
            s = jnp.array_repr(value, precision=precision)[:100]
            s += "..."
          else:
            s = jnp.array_str(value, precision=precision)[:100]
            s += "..."
        else:
          if python_code:
            s = jnp.array_repr(value, precision=precision)
          else:
            s = jnp.array_str(value, precision=precision)

    if not u.is_unitless:
      if isinstance(u, Unit):
        if python_code:
          s += f" * {repr(u)}"
        else:
          s += f" {str(u)}"
      else:
        if python_code:
          s += f" * {repr(u.dim)}"
        else:
          s += f" {str(u.dim)}"
    elif python_code:  # Make a array without unit recognisable
      return f"{self.__class__.__name__}({s.strip()})"
    return s.strip()

  def repr_in_best_unit(self, precision: int = None, python_code: bool = False, *regs):
    """
    Represent the array in the "best" unit.

    Parameters
    ----------
    precision : `int`, optional
        The number of digits of precision (in the best unit, see
        Examples). If no value is given, numpy's
        `get_printoptions` value is used.
    python_code : `bool`, optional
        Whether to return a string that can be used as python code.
        If True, the string will be formatted as a python expression.
        If False, the string will be formatted as a human-readable string.
    regs : `UnitRegistry` objects
        The registries where to search for units. If none are given, the
        standard, user-defined and additional registries are searched in
        that order.

    Returns
    -------
    representation : `str`
        A string representation of this `Array`.

    Examples
    --------
    >>> from brainunit import *
    >>> x = 0.00123456 * volt
    >>> x.repr_in_best_unit()
    '1.23456 mV'
    >>> x.repr_in_best_unit(3)
    '1.23 mV'
    """
    u = self.get_best_unit(*regs)
    return self.repr_in_unit(u, precision, python_code)

  def _check_tracer(self):
    self_value = self.value
    # if hasattr(self_value, '_trace') and hasattr(self_value._trace.main, 'jaxpr_stack'):
    #   if len(self_value._trace.main.jaxpr_stack) == 0:
    #     raise RuntimeError('This Array is modified during the transformation. '
    #                        'BrainPy only supports transformations for Variable. '
    #                        'Please declare it as a Variable.') from jax.core.escaped_tracer_error(self_value, None)
    return self_value

  @property
  def dtype(self):
    """Variable dtype."""
    a = self._value
    if hasattr(a, 'dtype'):
      return a.dtype
    else:
      if isinstance(a, bool):
        return bool
      elif isinstance(a, int):
        return jax.dtypes.canonicalize_dtype(int)
      elif isinstance(a, float):
        return jax.dtypes.canonicalize_dtype(float)
      elif isinstance(a, complex):
        return jax.dtypes.canonicalize_dtype(complex)
      else:
        raise TypeError(f'Can not get dtype of {a}.')

  @property
  def shape(self) -> Tuple[int, ...]:
    """Variable shape."""
    return jnp.shape(self._value)

  @property
  def ndim(self) -> int:
    return jnp.ndim(self.value)

  @property
  def imag(self) -> 'Quantity':
    return Quantity(jnp.imag(self.value), dim=self.dim)

  @property
  def real(self) -> 'Quantity':
    return Quantity(jnp.real(self.value), dim=self.dim)

  @property
  def size(self) -> int:
    return jnp.size(self.value)

  @property
  def T(self) -> 'Quantity':
    return Quantity(jnp.asarray(self.value).T, dim=self.dim)

  @property
  def isreal(self) -> jax.Array:
    return jnp.isreal(self.value)

  @property
  def isscalar(self) -> bool:
    return is_scalar_type(self)

  @property
  def isfinite(self) -> jax.Array:
    return jnp.isfinite(self.value)

  @property
  def isinfnite(self) -> jax.Array:
    return jnp.isinf(self.value)

  @property
  def isinf(self) -> jax.Array:
    return jnp.isinf(self.value)

  @property
  def isnan(self) -> jax.Array:
    return jnp.isnan(self.value)

  # ----------------------- #
  # Python inherent methods #
  # ----------------------- #

  def __repr__(self) -> str:
    if isinstance(self.value, (jax.ShapeDtypeStruct, jax.core.ShapedArray, DynamicJaxprTracer)):
      return f'{self.value} * {Quantity(1, dim=self.dim)}'
    return self.repr_in_best_unit(python_code=True)

  def __str__(self) -> str:
    return self.repr_in_best_unit()

  def __format__(self, format_spec: str) -> str:
    # Avoid that formatted strings like f"{q}" use floating point formatting for the
    # array, i.e. discard the unit
    if format_spec == "":
      return str(self)
    else:
      return self.value.__format__(format_spec)

  def __iter__(self):
    """Solve the issue of DeviceArray.__iter__.

    Details please see JAX issues:

    - https://github.com/google/jax/issues/7713
    - https://github.com/google/jax/pull/3821
    """
    if self.ndim == 0:
      yield self
    else:
      for i in range(self.shape[0]):
        yield Quantity(self.value[i], dim=self.dim)

  def __getitem__(self, index) -> 'Quantity':
    if isinstance(index, slice) and (index == _all_slice):
      return Quantity(self.value, dim=self.dim)
    elif isinstance(index, tuple):
      for x in index:
        assert not isinstance(x, Quantity), "Array indices must be integers or slices, not Array"
    elif isinstance(index, Quantity):
      raise TypeError("Array indices must be integers or slices, not Array")
    return Quantity(self.value[index], dim=self.dim)

  def __setitem__(self, index, value: 'Quantity'):
    if not isinstance(value, Quantity):
      raise DimensionMismatchError("Only Array can be assigned to Array.")
    fail_for_dimension_mismatch(self, value, "Inconsistent units in assignment")
    value = value.value

    # index is a tuple
    _assert_not_quantity(index)
    if isinstance(index, (tuple, list)):
      index = tuple(_assert_not_quantity(x) for x in index)
    # index is numpy.ndarray
    elif isinstance(index, np.ndarray):
      index = jnp.asarray(index)

    # update
    self_value = self._check_tracer()
    self.update_value(self_value.at[index].set(value))

  # ---------- #
  # operations #
  # ---------- #

  def __len__(self) -> int:
    return len(self.value)

  def __neg__(self) -> 'Quantity':
    return Quantity(self.value.__neg__(), dim=self.dim)

  def __pos__(self) -> 'Quantity':
    return Quantity(self.value.__pos__(), dim=self.dim)

  def __abs__(self) -> 'Quantity':
    return Quantity(self.value.__abs__(), dim=self.dim)

  def __invert__(self) -> 'Quantity':
    return Quantity(self.value.__invert__(), dim=self.dim)

  def _comparison(self, other: Any, operator_str: str, operation: Callable):
    is_scalar = is_scalar_type(other)
    if not is_scalar and not isinstance(other, (jax.Array, Quantity, np.ndarray)):
      return NotImplemented
    if not is_scalar or not jnp.isinf(other):
      other = _to_quantity(other)
      message = "Cannot perform comparison {value1} %s {value2}, units do not match" % operator_str
      fail_for_dimension_mismatch(self, other, message, value1=self, value2=other)
    other = _to_quantity(other)
    return operation(self.value, other.value)

  def __eq__(self, oc):
    return self._comparison(oc, "==", operator.eq)

  def __ne__(self, oc):
    return self._comparison(oc, "!=", operator.ne)

  def __lt__(self, oc):
    return self._comparison(oc, "<", operator.lt)

  def __le__(self, oc):
    return self._comparison(oc, "<=", operator.le)

  def __gt__(self, oc):
    return self._comparison(oc, ">", operator.gt)

  def __ge__(self, oc):
    return self._comparison(oc, ">=", operator.ge)

  def _binary_operation(
      self,
      other,
      value_operation: Callable,
      unit_operation: Callable = lambda a, b: a,
      fail_for_mismatch: bool = False,
      operator_str: str = None,
      inplace: bool = False,
  ):
    """
    General implementation for binary operations.

    Parameters
    ----------
    other : {`Array`, `ndarray`, scalar}
        The object with which the operation should be performed.
    value_operation : function of two variables
        The function with which the two objects are combined. For example,
        `operator.mul` for a multiplication.
    unit_operation : function of two variables, optional
        The function with which the dimension of the resulting object is
        calculated (as a function of the dimensions of the two involved
        objects). For example, `operator.mul` for a multiplication. If not
        specified, the dimensions of `self` are used for the resulting
        object.
    fail_for_mismatch : bool, optional
        Whether to fail for a dimension mismatch between `self` and `other`
        (defaults to ``False``)
    operator_str : str, optional
        The string to use for the operator in an error message.
    inplace: bool, optional
        Whether to do the operation in-place (defaults to ``False``).
    """
    other = _to_quantity(other)
    other_dim = None

    if fail_for_mismatch:
      if inplace:
        message = "Cannot calculate ... %s {value}, units do not match" % operator_str
        _, other_dim = fail_for_dimension_mismatch(self, other, message, value=other)
      else:
        message = "Cannot calculate {value1} %s {value2}, units do not match" % operator_str
        _, other_dim = fail_for_dimension_mismatch(self, other, message, value1=self, value2=other)

    if other_dim is None:
      other_dim = get_dim(other)

    new_dim = unit_operation(self.dim, other_dim)
    result = value_operation(self.value, other.value)
    r = Quantity(result, dim=new_dim)
    if inplace:
      self.update_value(r.value)
      return self
    return r

  def __add__(self, oc):
    return self._binary_operation(oc, operator.add, fail_for_mismatch=True, operator_str="+")

  def __radd__(self, oc):
    return self.__add__(oc)

  def __iadd__(self, oc):
    # a += b
    return self._binary_operation(oc, operator.add, fail_for_mismatch=True, operator_str="+=", inplace=True)

  def __sub__(self, oc):
    return self._binary_operation(oc, operator.sub, fail_for_mismatch=True, operator_str="-")

  def __rsub__(self, oc):
    return Quantity(oc).__sub__(self)

  def __isub__(self, oc):
    # a -= b
    return self._binary_operation(oc, operator.sub, fail_for_mismatch=True, operator_str="-=", inplace=True)

  def __mul__(self, oc):
    r = self._binary_operation(oc, operator.mul, operator.mul)
    return _return_check_unitless(r)

  def __rmul__(self, oc):
    return self.__mul__(oc)

  def __imul__(self, oc):
    # a *= b
    raise NotImplementedError("In-place multiplication is not supported")

  def __div__(self, oc):
    # self / oc
    r = self._binary_operation(oc, operator.truediv, operator.truediv)
    return _return_check_unitless(r)

  def __idiv__(self, oc):
    raise NotImplementedError("In-place division is not supported")

  def __truediv__(self, oc):
    # self / oc
    return self.__div__(oc)

  def __rdiv__(self, oc):
    # oc / self
    # division with swapped arguments
    rdiv = lambda a, b: operator.truediv(b, a)
    r = self._binary_operation(oc, rdiv, rdiv)
    return _return_check_unitless(r)

  def __rtruediv__(self, oc):
    # oc / self
    return self.__rdiv__(oc)

  def __itruediv__(self, oc):
    # a /= b
    raise NotImplementedError("In-place true division is not supported")

  def __floordiv__(self, oc):
    # self // oc
    r = self._binary_operation(oc, operator.floordiv, operator.truediv)
    return _return_check_unitless(r)

  def __rfloordiv__(self, oc):
    # oc // self
    rdiv = lambda a, b: operator.truediv(b, a)
    rfloordiv = lambda a, b: operator.truediv(b, a)
    r = self._binary_operation(oc, rfloordiv, rdiv)
    return _return_check_unitless(r)

  def __ifloordiv__(self, oc):
    # a //= b
    raise NotImplementedError("In-place floor division is not supported")

  def __mod__(self, oc):
    # self % oc
    r = self._binary_operation(oc, operator.mod, operator_str=r"%")
    return _return_check_unitless(r)

  def __rmod__(self, oc):
    # oc % self
    oc = _to_quantity(oc)
    r = oc._binary_operation(self, operator.mod, operator_str=r"%")
    return _return_check_unitless(r)

  def __imod__(self, oc):
    raise NotImplementedError("In-place mod is not supported")

  def __divmod__(self, oc):
    return self.__floordiv__(oc), self.__mod__(oc)

  def __rdivmod__(self, oc):
    return self.__rfloordiv__(oc), self.__rmod__(oc)

  def __matmul__(self, oc):
    r = self._binary_operation(oc, operator.matmul, operator.mul)
    return _return_check_unitless(r)

  def __rmatmul__(self, oc):
    oc = _to_quantity(oc)
    r = oc._binary_operation(self, operator.matmul, operator.mul)
    return _return_check_unitless(r)

  def __imatmul__(self, oc):
    # a @= b
    raise NotImplementedError("In-place matrix multiplication is not supported")

  # -------------------- #

  def __pow__(self, oc):
    # self ** oc
    if isinstance(oc, (jax.Array, np.ndarray, numbers.Number, Quantity)) or is_scalar_type(oc):
      fail_for_dimension_mismatch(
        oc,
        error_message=(
          "Cannot calculate "
          "{base} ** {exponent}, "
          "the exponent has to be "
          "dimensionless"
        ),
        base=self,
        exponent=oc,
      )
      if isinstance(oc, Quantity):
        oc = oc.value
      r = Quantity(jnp.array(self.value) ** oc, dim=self.dim ** oc)
      return _return_check_unitless(r)
    else:
      return TypeError('Cannot calculate {base} ** {exponent}, the '
                       'exponent has to be dimensionless'.format(base=self, exponent=oc))

  def __rpow__(self, oc):
    # oc ** self
    if self.is_unitless:
      if isinstance(oc, (jax.Array, np.ndarray, numbers.Number)):
        return oc ** self.value
      else:
        return oc.__pow__(self.value)
    else:
      raise DimensionMismatchError(f"Cannot calculate {_short_str(oc)} ** {_short_str(self)}, "
                                   f"the base has to be dimensionless",
                                   self.dim)

  def __ipow__(self, oc):
    # a **= b
    raise NotImplementedError("In-place power is not supported")

  def __and__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __rand__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __iand__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __or__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __ror__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __ior__(self, oc):
    # Remove the unit from the result
    # a |= b
    raise NotImplementedError("Bitwise operations are not supported")

  def __xor__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __rxor__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __ixor__(self, oc) -> 'Quantity':
    # Remove the unit from the result
    # a ^= b
    raise NotImplementedError("Bitwise operations are not supported")

  def __lshift__(self, oc) -> 'Quantity':
    # self << oc
    if is_scalar_type(oc) and isinstance(oc, Quantity):
      oc = oc.value
    r = Quantity(self.value << oc, dim=self.dim)
    return _return_check_unitless(r)

  def __rlshift__(self, oc) -> 'Quantity':
    # oc << self
    if isinstance(oc, (jax.Array, np.ndarray, numbers.Number)):
      oc = Quantity(oc, dim=DIMENSIONLESS)
    r = oc.__lshift__(self.value)
    return _return_check_unitless(r)

  def __ilshift__(self, oc) -> 'Quantity':
    # a <<= b
    r = self.__lshift__(oc)
    self.update_value(r.value)
    return self

  def __rshift__(self, oc) -> 'Quantity':
    # self >> oc
    if isinstance(oc, Quantity):
      oc = oc.value
    r = Quantity(self.value >> oc, dim=self.dim)
    return _return_check_unitless(r)

  def __rrshift__(self, oc) -> 'Quantity':
    # oc >> self
    if isinstance(oc, (jax.Array, np.ndarray, numbers.Number)):
      oc = Quantity(oc, dim=DIMENSIONLESS)
    r = oc.__rshift__(self.value)
    return _return_check_unitless(r)

  def __irshift__(self, oc) -> 'Quantity':
    # a >>= b
    r = self.__rshift__(oc)
    self.update_value(r.value)
    return self

  def __round__(self, ndigits: int = None) -> 'Quantity':
    return Quantity(self.value.__round__(ndigits), dim=self.dim)

  def __reduce__(self):
    return array_with_unit, (self.value, self.dim, None)

  # ----------------------- #
  #      NumPy methods      #
  # ----------------------- #

  all = wrap_function_remove_dimensions(jnp.all)
  any = wrap_function_remove_dimensions(jnp.any)
  nonzero = wrap_function_remove_dimensions(jnp.nonzero)
  argmax = wrap_function_remove_dimensions(jnp.argmax)
  argmin = wrap_function_remove_dimensions(jnp.argmin)
  argsort = wrap_function_remove_dimensions(jnp.argsort)

  var = wrap_function_change_dimensions(jnp.var, lambda v, d: d ** 2)

  round = wrap_function_keep_dimensions(jnp.round)
  std = wrap_function_keep_dimensions(jnp.std)
  sum = wrap_function_keep_dimensions(jnp.sum)
  trace = wrap_function_keep_dimensions(jnp.trace)
  cumsum = wrap_function_keep_dimensions(jnp.cumsum)
  diagonal = wrap_function_keep_dimensions(jnp.diagonal)
  max = wrap_function_keep_dimensions(jnp.max)
  mean = wrap_function_keep_dimensions(jnp.mean)
  min = wrap_function_keep_dimensions(jnp.min)
  ptp = wrap_function_keep_dimensions(jnp.ptp)
  ravel = wrap_function_keep_dimensions(jnp.ravel)

  def astype(self, dtype) -> 'Quantity':
    """Copy of the array, cast to a specified type.

    Parameters
    ----------
    dtype: str, dtype
      Typecode or data-type to which the array is cast.
    """
    if dtype is None:
      return Quantity(self.value, dim=self.dim)
    else:
      return Quantity(jnp.astype(self.value, dtype), dim=self.dim)

  def clip(self, min: Quantity = None, max: Quantity = None, *args, **kwds) -> 'Quantity':
    """Return an array whose values are limited to [min, max]. One of max or min must be given."""

    fail_for_dimension_mismatch(self, min, "clip")
    fail_for_dimension_mismatch(self, max, "clip")
    return Quantity(
      jnp.clip(
        jnp.array(self.value),
        jnp.array(min.value),
        jnp.array(max.value),
        *args,
        **kwds,
      ),
      dim=self.dim,
    )

  def conj(self) -> 'Quantity':
    """Complex-conjugate all elements."""
    return Quantity(jnp.conj(self.value), dim=self.dim)

  def conjugate(self) -> 'Quantity':
    """Return the complex conjugate, element-wise."""
    return Quantity(jnp.conjugate(self.value), dim=self.dim)

  def copy(self) -> 'Quantity':
    """Return a copy of the array."""
    return Quantity(jnp.copy(self.value), dim=self.dim)

  def dot(self, b) -> 'Quantity':
    """Dot product of two arrays."""
    r = self._binary_operation(b, jnp.dot, operator.mul)
    return _return_check_unitless(r)

  def fill(self, value: Quantity) -> 'Quantity':
    """Fill the array with a scalar value."""
    fail_for_dimension_mismatch(self, value, "fill")
    self.update_value((jnp.ones_like(self.value) * value).value)
    return self

  def flatten(self) -> 'Quantity':
    return Quantity(jnp.reshape(self.value, -1), dim=self.dim)

  def item(self, *args) -> 'Quantity':
    """Copy an element of an array to a standard Python scalar and return it."""
    if isinstance(self.value, jax.Array):
      return Quantity(self.value.item(*args), dim=self.dim)
    else:
      return Quantity(self.value, dim=self.dim)

  def prod(self, *args, **kwds) -> 'Quantity':
    """Return the product of the array elements over the given axis."""
    prod_res = jnp.prod(self.value, *args, **kwds)
    # Calculating the correct dimensions is not completly trivial (e.g.
    # like doing self.dim**self.size) because prod can be called on
    # multidimensional arrays along a certain axis.
    # Our solution: Use a "dummy matrix" containing a 1 (without units) at
    # each entry and sum it, using the same keyword arguments as provided.
    # The result gives the exponent for the dimensions.
    # This relies on sum and prod having the same arguments, which is true
    # now and probably remains like this in the future
    dim_exponent = jnp.ones_like(self.value).sum(*args, **kwds)
    # The result is possibly multidimensional but all entries should be
    # identical
    if dim_exponent.size > 1:
      dim_exponent = dim_exponent[-1]
    r = Quantity(jnp.array(prod_res), dim=self.dim ** dim_exponent)
    return _return_check_unitless(r)

  def nanprod(self, *args, **kwds) -> 'Quantity':
    """Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones."""
    prod_res = jnp.nanprod(self.value, *args, **kwds)
    nan_mask = jnp.isnan(self.value)
    dim_exponent = jnp.cumsum(jnp.where(nan_mask, 0, 1), *args)
    if dim_exponent.size > 1:
      dim_exponent = dim_exponent[-1]
    r = Quantity(jnp.array(prod_res), dim=self.dim ** dim_exponent)
    return _return_check_unitless(r)

  def cumprod(self, *args, **kwds):  # pylint: disable=C0111
    prod_res = jnp.cumprod(self.value, *args, **kwds)
    dim_exponent = jnp.ones_like(self.value).cumsum(*args, **kwds)
    if dim_exponent.size > 1:
      dim_exponent = dim_exponent[-1]
    r = Quantity(jnp.array(prod_res), dim=self.dim ** dim_exponent)
    return _return_check_unitless(r)

  def nancumprod(self, *args, **kwds):  # pylint: disable=C0111
    prod_res = jnp.nancumprod(self.value, *args, **kwds)
    nan_mask = jnp.isnan(self.value)
    dim_exponent = jnp.cumsum(jnp.where(nan_mask, 0, 1), *args)
    if dim_exponent.size > 1:
      dim_exponent = dim_exponent[-1]
    r = Quantity(jnp.array(prod_res), dim=self.dim ** dim_exponent)
    return _return_check_unitless(r)

  def put(self, indices, values) -> 'Quantity':
    """Replaces specified elements of an array with given values.

    Parameters
    ----------
    indices: array_like
      Target indices, interpreted as integers.
    values: array_like
      Values to place in the array at target indices.
    """
    fail_for_dimension_mismatch(self, values, "put")
    self.__setitem__(indices, values)
    return self

  def repeat(self, repeats, axis=None) -> 'Quantity':
    """Repeat elements of an array."""
    r = jnp.repeat(self.value, repeats=repeats, axis=axis)
    return Quantity(r, dim=self.dim)

  def reshape(self, *shape, order='C') -> 'Quantity':
    """Returns an array containing the same data with a new shape."""
    return Quantity(jnp.reshape(self.value, shape, order=order), dim=self.dim)

  def resize(self, new_shape) -> 'Quantity':
    """Change shape and size of array in-place."""
    self.update_value(jnp.resize(self.value, new_shape))
    return self

  def sort(self, axis=-1, stable=True, order=None) -> 'Quantity':
    """Sort an array in-place.

    Parameters
    ----------
    axis : int, optional
        Axis along which to sort. Default is -1, which means sort along the
        last axis.
    stable : bool, optional
        Whether to use a stable sorting algorithm. The default is True.
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.
    """
    self.update_value(jnp.sort(self.value, axis=axis, stable=stable, order=order))
    return self

  def squeeze(self, axis=None) -> 'Quantity':
    """Remove axes of length one from ``a``."""
    return Quantity(jnp.squeeze(self.value, axis=axis), dim=self.dim)

  def swapaxes(self, axis1, axis2) -> 'Quantity':
    """Return a view of the array with `axis1` and `axis2` interchanged."""
    return Quantity(jnp.swapaxes(self.value, axis1, axis2), dim=self.dim)

  def split(self, indices_or_sections, axis=0) -> List['Quantity']:
    """Split an array into multiple sub-arrays as views into ``ary``.

    Parameters
    ----------
    indices_or_sections : int, 1-D array
      If `indices_or_sections` is an integer, N, the array will be divided
      into N equal arrays along `axis`.  If such a split is not possible,
      an error is raised.

      If `indices_or_sections` is a 1-D array of sorted integers, the entries
      indicate where along `axis` the array is split.  For example,
      ``[2, 3]`` would, for ``axis=0``, result in

        - ary[:2]
        - ary[2:3]
        - ary[3:]

      If an index exceeds the dimension of the array along `axis`,
      an empty sub-array is returned correspondingly.
    axis : int, optional
      The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list of ndarrays
      A list of sub-arrays as views into `ary`.
    """
    return [Quantity(a, dim=self.dim) for a in jnp.split(self.value, indices_or_sections, axis=axis)]

  def take(
      self,
      indices,
      axis=None,
      mode=None,
      unique_indices=False,
      indices_are_sorted=False,
      fill_value=None,
  ) -> 'Quantity':
    """Return an array formed from the elements of a at the given indices."""
    if isinstance(fill_value, Quantity):
      fail_for_dimension_mismatch(self, fill_value, "take")
      fill_value = fill_value.value
    elif fill_value is not None:
      if not self.is_unitless:
        raise TypeError(f"fill_value must be a Quantity when the unit {self.unit}. But got {fill_value}")
    return Quantity(
      jnp.take(self.value,
               indices=indices, axis=axis, mode=mode,
               unique_indices=unique_indices,
               indices_are_sorted=indices_are_sorted,
               fill_value=fill_value),
      dim=self.dim
    )

  def tolist(self):
    """Return the array as an ``a.ndim``-levels deep nested list of Python scalars.

    Return a copy of the array data as a (nested) Python list.
    Data items are converted to the nearest compatible builtin Python type, via
    the `~numpy.ndarray.item` function.

    If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will
    not be a list at all, but a simple Python scalar.
    """

    def replace_with_array(seq, unit):
      """
      Replace all the elements in the list with an equivalent `Array`
      with the given `unit`.
      """
      # No recursion needed for single values
      if not isinstance(seq, list):
        return Quantity(seq, dim=unit)

      def top_replace(s):
        """
        Recursively descend into the list.
        """
        for i in s:
          if not isinstance(i, list):
            yield Quantity(i, dim=unit)
          else:
            yield type(i)(top_replace(i))

      return type(seq)(top_replace(seq))

    if isinstance(self.value, jax.Array):
      return replace_with_array(self.value.tolist(), self.dim)
    else:
      return Quantity(self.value, dim=self.dim)

  def transpose(self, *axes) -> 'Quantity':
    """Returns a view of the array with axes transposed.

    For a 1-D array this has no effect, as a transposed vector is simply the
    same vector. To convert a 1-D array into a 2D column vector, an additional
    dimension must be added. `jnp.atleast2d(a).T` achieves this, as does
    `a[:, jnp.newaxis]`.
    For a 2-D array, this is a standard matrix transpose.
    For an n-D array, if axes are given, their order indicates how the
    axes are permuted (see Examples). If axes are not provided and
    ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
    ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

    Parameters
    ----------
    axes : None, tuple of ints, or `n` ints

     * None or no argument: reverses the order of the axes.

     * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
       `i`-th axis becomes `a.transpose()`'s `j`-th axis.

     * `n` ints: same as an n-tuple of the same ints (this form is
       intended simply as a "convenience" alternative to the tuple form)

    Returns
    -------
    out : ndarray
        View of `a`, with axes suitably permuted.
    """
    return Quantity(jnp.transpose(self.value, *axes), dim=self.dim)

  def tile(self, reps) -> 'Quantity':
    """Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Note : Although tile may be used for broadcasting, it is strongly
    recommended to use numpy's broadcasting operations and functions.

    Parameters
    ----------
    reps : array_like
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.
    """
    return Quantity(jnp.tile(self.value, reps), dim=self.dim)

  def view(self, *args, dtype=None) -> 'Quantity':
    r"""New view of array with the same data.

    This function is compatible with pytorch syntax.

    Returns a new tensor with the same data as the :attr:`self` tensor but of a
    different :attr:`shape`.

    The returned tensor shares the same data and must have the same number
    of elements, but may have a different size. For a tensor to be viewed, the new
    view size must be compatible with its original size and stride, i.e., each new
    view dimension must either be a subspace of an original dimension, or only span
    across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
    contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

    .. math::

      \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

    Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
    without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
    :meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
    returns a view if the shapes are compatible, and copies (equivalent to calling
    :meth:`contiguous`) otherwise.

    Args:
        shape (int...): the desired size

    Example::

        >>> import brainstate, brainunit
        >>> x = brainstate.random.randn(4, 4)
        >>> x.size
       [4, 4]
        >>> y = x.view(16)
        >>> y.size
        [16]
        >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
        >>> z.size
        [2, 8]

        >>> a = brainstate.random.randn(1, 2, 3, 4)
        >>> a.size
        [1, 2, 3, 4]
        >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
        >>> b.size
        [1, 3, 2, 4]
        >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
        >>> c.size
        [1, 3, 2, 4]
        >>> brainunit.math.equal(b, c)
        False


    .. method:: view(dtype) -> Tensor
       :noindex:

    Returns a new tensor with the same data as the :attr:`self` tensor but of a
    different :attr:`dtype`.

    If the element size of :attr:`dtype` is different than that of ``self.dtype``,
    then the size of the last dimension of the output will be scaled
    proportionally.  For instance, if :attr:`dtype` element size is twice that of
    ``self.dtype``, then each pair of elements in the last dimension of
    :attr:`self` will be combined, and the size of the last dimension of the output
    will be half that of :attr:`self`. If :attr:`dtype` element size is half that
    of ``self.dtype``, then each element in the last dimension of :attr:`self` will
    be split in two, and the size of the last dimension of the output will be
    double that of :attr:`self`. For this to be possible, the following conditions
    must be true:

        * ``self.dim()`` must be greater than 0.
        * ``self.stride(-1)`` must be 1.

    Additionally, if the element size of :attr:`dtype` is greater than that of
    ``self.dtype``, the following conditions must be true as well:

        * ``self.size(-1)`` must be divisible by the ratio between the element
          sizes of the dtypes.
        * ``self.storage_offset()`` must be divisible by the ratio between the
          element sizes of the dtypes.
        * The strides of all dimensions, except the last dimension, must be
          divisible by the ratio between the element sizes of the dtypes.

    If any of the above conditions are not met, an error is thrown.


    Args:
        dtype (:class:`dtype`): the desired dtype

    Example::

        >>> x = brainstate.random.randn(4, 4)
        >>> x
        Array([[ 0.9482, -0.0310,  1.4999, -0.5316],
                [-0.1520,  0.7472,  0.5617, -0.8649],
                [-2.4724, -0.0334, -0.2976, -0.8499],
                [-0.2109,  1.9913, -0.9607, -0.6123]])
        >>> x.dtype
        brainstate.math.float32

        >>> y = x.view(numpy.int32)
        >>> y
        tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
                [-1105482831,  1061112040,  1057999968, -1084397505],
                [-1071760287, -1123489973, -1097310419, -1084649136],
                [-1101533110,  1073668768, -1082790149, -1088634448]],
            dtype=numpy.int32)
        >>> y[0, 0] = 1000000000
        >>> x
        tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
                [-0.1520,  0.7472,  0.5617, -0.8649],
                [-2.4724, -0.0334, -0.2976, -0.8499],
                [-0.2109,  1.9913, -0.9607, -0.6123]])

        >>> x.view(numpy.complex64)
        tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
                [-0.1520+0.7472j,  0.5617-0.8649j],
                [-2.4724-0.0334j, -0.2976-0.8499j],
                [-0.2109+1.9913j, -0.9607-0.6123j]])
        >>> x.view(numpy.complex64).size
        [4, 2]

        >>> x.view(numpy.uint8)
        tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
                   8, 191],
                [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
                  93, 191],
                [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
                  89, 191],
                [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
                  28, 191]], dtype=uint8)
        >>> x.view(numpy.uint8).size
        [4, 16]

    """
    if isinstance(self.value, jax.Array):
      if len(args) == 0:
        if dtype is None:
          raise ValueError('Provide dtype or shape.')
        else:
          return Quantity(self.value.view(dtype), dim=self.dim)
      else:
        if isinstance(args[0], int):  # shape
          if dtype is not None:
            raise ValueError('Provide one of dtype or shape. Not both.')
          return Quantity(self.value.reshape(*args), dim=self.dim)
        else:  # dtype
          assert not isinstance(args[0], int)
          assert dtype is None
          return Quantity(self.value.view(args[0]), dim=self.dim)
    else:
      return Quantity(jnp.asarray(self.value, dtype=dtype), dim=self.dim)

  # ------------------
  # NumPy support
  # ------------------

  def to_numpy(
      self,
      unit: Optional['Unit'] = None,
      dtype: Optional[jax.typing.DTypeLike] = None,
  ) -> np.ndarray:
    """
    Remove the unit and convert to ``numpy.ndarray``.

    Args:
      dtype: The data type of the output array.
      unit: The unit of the output array.

    Returns:
      The numpy.ndarray.
    """
    if unit is None:
      assert self.dim == DIMENSIONLESS, (f"only dimensionless quantities can be converted to "
                                         f"NumPy arrays when 'unit' is not provided. But got {self}")
      return np.asarray(self.value, dtype=dtype)
    else:
      fail_for_dimension_mismatch(self, unit, "to_numpy")
      assert isinstance(unit, Unit), f"unit must be a Unit object, but got {type(unit)}"
      return np.asarray(self / unit, dtype=dtype)

  def to_jax(
      self,
      unit: Optional['Unit'] = None,
      dtype: Optional[jax.typing.DTypeLike] = None,
  ) -> jax.Array:
    """
    Remove the unit and convert to ``jax.Array``.

    Args:
      dtype: The data type of the output array.
      unit: The unit of the output array.

    Returns:
      The jax.Array.
    """
    if unit is None:
      assert self.dim == DIMENSIONLESS, (f"only dimensionless quantities can be converted to "
                                         f"JAX arrays when 'unit' is not provided. But got {self}")
      return jnp.asarray(self.value, dtype=dtype)
    else:
      fail_for_dimension_mismatch(self, unit, "to_jax")
      assert isinstance(unit, Unit), f"unit must be a Unit object, but got {type(unit)}"
      return jnp.asarray(self / unit, dtype=dtype)

  def __array__(self, dtype: Optional[jax.typing.DTypeLike] = None) -> np.ndarray:
    """Support ``numpy.array()`` and ``numpy.asarray()`` functions."""
    if self.dim == DIMENSIONLESS:
      return np.asarray(self.value, dtype=dtype)
    else:
      raise TypeError(
        f"only dimensionless quantities can be "
        f"converted to NumPy arrays. But got {self}"
      )

  def __float__(self):
    if self.dim == DIMENSIONLESS and self.ndim == 0:
      return float(self.value)
    else:
      raise TypeError(
        "only dimensionless scalar quantities can be "
        f"converted to Python scalars. But got {self}"
      )

  def __int__(self):
    if self.dim == DIMENSIONLESS and self.ndim == 0:
      return int(self.value)
    else:
      raise TypeError(
        "only dimensionless scalar quantities can be "
        f"converted to Python scalars. But got {self}"
      )

  def __index__(self):
    if self.dim == DIMENSIONLESS:
      return operator.index(self.value)
    else:
      raise TypeError(
        "only dimensionless quantities can be "
        f"converted to a Python index. But got {self}"
      )

  # ----------------------
  # PyTorch compatibility
  # ----------------------

  def unsqueeze(self, axis: int) -> 'Quantity':
    """
    Array.unsqueeze(dim) -> Array, or so called Tensor
    equals
    Array.expand_dims(dim)

    See :func:`brainstate.math.unsqueeze`
    """
    return Quantity(jnp.expand_dims(self.value, axis), dim=self.dim)

  def expand_dims(self, axis: Union[int, Sequence[int]]) -> 'Quantity':
    """
    self.expand_dims(axis: int|Sequence[int])

    1. 如果axis类型为int：
    返回一个在self基础上的第axis维度前插入一个维度Array，
    axis<0表示倒数第|axis|维度，
    令n=len(self._value.shape)，则axis的范围为[-(n+1),n]

    2. 如果axis类型为Sequence[int]：
    则返回依次扩展axis[i]的结果，
    即self.expand_dims(axis)==self.expand_dims(axis[0]).expand_dims(axis[1])...expand_dims(axis[len(axis)-1])


    1. If the type of axis is int:

    Returns an Array of dimensions inserted before the axis dimension based on self,

    The first | axis < 0 indicates the bottom axis | dimensions,

    Set n=len(self._value.shape), then axis has the range [-(n+1),n]


    2. If the type of axis is Sequence[int] :

    Returns the result of extending axis[i] in sequence,

    self.expand_dims(axis)==self.expand_dims(axis[0]).expand_dims(axis[1])... expand_dims(axis[len(axis)-1])

    """
    return Quantity(jnp.expand_dims(self.value, axis), dim=self.dim)

  def expand_as(self, array: Union['Quantity', jax.Array, np.ndarray]) -> 'Quantity':
    """
    Expand an array to a shape of another array.

    Parameters
    ----------
    array : Quantity

    Returns
    -------
    expanded : Quantity
        A readonly view on the original array with the given shape of array. It is
        typically not contiguous. Furthermore, more than one element of a
        expanded array may refer to a single memory location.
    """
    if isinstance(array, Quantity):
      array = array.value
    return Quantity(jnp.broadcast_to(self.value, array), dim=self.dim)

  def pow(self, oc) -> 'Quantity':
    return self.__pow__(oc)

  def clamp(
      self,
      min_value: Optional['Quantity'] = None,
      max_value: Optional['Quantity'] = None,
  ) -> 'Quantity':
    """
    return the value between min_value and max_value,
    if min_value is None, then no lower bound,
    if max_value is None, then no upper bound.
    """
    return self.clip(min_value, max_value)

  def clone(self) -> 'Quantity':
    if isinstance(self.value, jax.Array):
      return self.copy()
    return type(self)(self.value, dim=self.dim)

  def tree_flatten(self) -> Tuple[jax.Array | numbers.Number, Any]:
    """
    Tree flattens the data.

    Returns:
      The data and the dimension.
    """
    return (self.value,), self.dim

  @classmethod
  def tree_unflatten(cls, dim, value) -> 'Quantity':
    """
    Tree unflattens the data.

    Args:
      dim: The dimension.
      value: The data.

    Returns:
      The Quantity object.
    """
    return cls(*value, dim=dim)

  def cuda(self, deice=None) -> 'Quantity':
    deice = jax.devices('cuda')[0] if deice is None else deice
    self.update_value(jax.device_put(self.value, deice))
    return self

  def cpu(self, device=None) -> 'Quantity':
    device = jax.devices('cpu')[0] if device is None else device
    self.update_value(jax.device_put(self.value, device))
    return self

  # dtype exchanging #
  # ---------------- #

  def int(self) -> 'Quantity':
    return Quantity(jnp.asarray(self.value, dtype=jnp.int32), dim=self.dim)

  def long(self) -> 'Quantity':
    return Quantity(jnp.asarray(self.value, dtype=jnp.int64), dim=self.dim)

  def half(self) -> 'Quantity':
    return Quantity(jnp.asarray(self.value, dtype=jnp.float16), dim=self.dim)

  def float(self) -> 'Quantity':
    return Quantity(jnp.asarray(self.value, dtype=jnp.float32), dim=self.dim)

  def double(self) -> 'Quantity':
    return Quantity(jnp.asarray(self.value, dtype=jnp.float64), dim=self.dim)


class Unit(Quantity):
  r"""
   A physical unit.

   Normally, you do not need to worry about the implementation of
   units. They are derived from the `Array` object with
   some additional information (name and string representation).

   Basically, a unit is just a number with given dimensions, e.g.
   mvolt = 0.001 with the dimensions of voltage. The units module
   defines a large number of standard units, and you can also define
   your own (see below).

   The unit class also keeps track of various things that were used
   to define it so as to generate a nice string representation of it.
   See below.

   When creating scaled units, you can use the following prefixes:

    ======     ======  ==============
    Factor     Name    Prefix
    ======     ======  ==============
    10^24      yotta   Y
    10^21      zetta   Z
    10^18      exa     E
    10^15      peta    P
    10^12      tera    T
    10^9       giga    G
    10^6       mega    M
    10^3       kilo    k
    10^2       hecto   h
    10^1       deka    da
    1
    10^-1      deci    d
    10^-2      centi   c
    10^-3      milli   m
    10^-6      micro   u (\mu in SI)
    10^-9      nano    n
    10^-12     pico    p
    10^-15     femto   f
    10^-18     atto    a
    10^-21     zepto   z
    10^-24     yocto   y
    ======     ======  ==============

  **Defining your own**

   It can be useful to define your own units for printing
   purposes. So for example, to define the newton metre, you
   write

   >>> import brainunit as U
   >>> Nm = U.newton * U.metre

   You can then do

   >>> (1*Nm).in_unit(Nm)
   '1. N m'

   New "compound units", i.e. units that are composed of other units will be
   automatically registered and from then on used for display. For example,
   imagine you define total conductance for a membrane, and the total area of
   that membrane:

   >>> conductance = 10.*U.nS
   >>> area = 20000*U.um**2

   If you now ask for the conductance density, you will get an "ugly" display
   in basic SI dimensions, as Brian does not know of a corresponding unit:

   >>> conductance/area
   0.5 * metre ** -4 * kilogram ** -1 * second ** 3 * amp ** 2

   By using an appropriate unit once, it will be registered and from then on
   used for display when appropriate:

   >>> U.usiemens/U.cm**2
   usiemens / (cmetre ** 2)
   >>> conductance/area  # same as before, but now Brian knows about uS/cm^2
   50. * usiemens / (cmetre ** 2)

   Note that user-defined units cannot override the standard units (`volt`,
   `second`, etc.) that are predefined by Brian. For example, the unit
   ``Nm`` has the dimensions "length²·mass/time²", and therefore the same
   dimensions as the standard unit `joule`. The latter will be used for display
   purposes:

   >>> 3*U.joule
   3. * joule
   >>> 3*Nm
   3. * joule

  """

  __module__ = "brainunit"
  __slots__ = ["_value", "_unit", "scale", "_dispname", "_name", "iscompound"]
  __array_priority__ = 1000

  def __init__(
      self,
      value,
      dim: Dimension = None,
      scale: int = 0,
      name: str = None,
      dispname: str = None,
      iscompound: bool = None,
      dtype: jax.typing.DTypeLike = None,
      register: bool = True,
  ):
    if dim is None:
      dim = DIMENSIONLESS
    if value != 10.0 ** scale:
      raise AssertionError(f"Unit value has to be 10**scale (scale={scale}, value={value})")

    # The scale for this unit (as the integer exponent of 10), i.e.
    # a scale of 3 means 10^3, for a "k" prefix.
    self.scale = scale
    if name is None:
      if dim is DIMENSIONLESS:
        name = "Unit(1)"
      else:
        name = repr(dim)
    # The full name of this unit
    self._name = name
    # The display name of this unit
    if dispname is None:
      dispname = name
    self._dispname = dispname
    # Whether this unit is a combination of other units
    self.iscompound = iscompound

    super().__init__(value, dtype=dtype, dim=dim)

    if _auto_register_unit and register:
      register_new_unit(self)

  @staticmethod
  def create(unit: Dimension, name: str, dispname: str, scale: int = 0):
    """
    Create a new named unit.

    Parameters
    ----------
    unit : Dimension
        The dimensions of the unit.
    name : `str`
        The full name of the unit, e.g. ``'volt'``
    dispname : `str`
        The display name, e.g. ``'V'``
    scale : int, optional
        The scale of this unit as an exponent of 10, e.g. -3 for a unit that
        is 1/1000 of the base scale. Defaults to 0 (i.e. a base unit).

    Returns
    -------
    u : `Unit`
        The new unit.
    """
    name = str(name)
    dispname = str(dispname)

    u = Unit(
      10.0 ** scale,
      dim=unit,
      scale=scale,
      name=name,
      dispname=dispname,
    )

    return u

  @staticmethod
  def create_scaled_unit(baseunit, scalefactor):
    """
    Create a scaled unit from a base unit.

    Parameters
    ----------
    baseunit : `Unit`
        The unit of which to create a scaled version, e.g. ``volt``,
        ``amp``.
    scalefactor : `str`
        The scaling factor, e.g. ``"m"`` for mvolt, mamp

    Returns
    -------
    u : `Unit`
        The new unit.
    """
    name = scalefactor + baseunit.name
    dispname = scalefactor + baseunit.dispname
    scale = _siprefixes[scalefactor] + baseunit.scale

    u = Unit(
      10.0 ** scale,
      dim=baseunit.dim,
      name=name,
      dispname=dispname,
      scale=scale,
    )

    return u

  name = property(fget=lambda self: self._name, doc="The name of the unit")

  dispname = property(fget=lambda self: self._dispname, doc="The display name of the unit")

  def __repr__(self):
    return self.name

  def __str__(self):
    return self.dispname

  def __mul__(self, other):
    if isinstance(other, Unit):
      name = f"{self.name} * {other.name}"
      dispname = f"{self.dispname} * {other.dispname}"
      scale = self.scale + other.scale
      u = Unit(
        10.0 ** scale,
        dim=self.dim * other.dim,
        name=name,
        dispname=dispname,
        iscompound=True,
        scale=scale,
      )
      return u
    else:
      return super().__mul__(other)

  def __div__(self, other):
    if isinstance(other, Unit):
      if self.iscompound:
        dispname = f"({self.dispname})"
        name = f"({self.name})"
      else:
        dispname = self.dispname
        name = self.name
      dispname += "/"
      name += " / "
      if other.iscompound:
        dispname += f"({other.dispname})"
        name += f"({other.name})"
      else:
        dispname += other.dispname
        name += other.name

      scale = self.scale - other.scale
      u = Unit(
        10.0 ** scale,
        dim=self.dim / other.dim,
        name=name,
        dispname=dispname,
        scale=scale,
        iscompound=True,
      )
      return u
    else:
      return super().__div__(other)

  def __rdiv__(self, other):
    if is_scalar_type(other) and other == 1:
      dispname = self.dispname
      name = self.name
      if self.iscompound:
        dispname = f"({self.dispname})"
        name = f"({self.name})"
      u = Unit(
        self.value,
        dim=self.dim ** -1,
        name=f"1 / {name}",
        dispname=f"1 / {dispname}",
        scale=-self.scale,
        iscompound=True,
      )
      return u
    else:
      return super().__rdiv__(other)

  def __pow__(self, other):
    if is_scalar_type(other):
      if self.iscompound:
        dispname = f"({self.dispname})"
        name = f"({self.name})"
      else:
        dispname = self.dispname
        name = self.name
      dispname += f"^{str(other)}"
      name += f" ** {repr(other)}"
      scale = self.scale * other
      u = Unit(
        10.0 ** scale,
        dim=self.dim ** other,
        name=name,
        dispname=dispname,
        scale=scale,
        iscompound=True,
      )  # To avoid issues with units like (second ** -1) ** -1
      return u
    else:
      return super().__pow__(other)

  def __iadd__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __isub__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __imul__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __idiv__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __itruediv__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __ifloordiv__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __imod__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __ipow__(self, other, modulo=None):
    raise TypeError("Units cannot be modified in-place")

  def __eq__(self, other):
    if isinstance(other, Unit):
      return other.dim is self.dim and other.scale == self.scale
    else:
      return Quantity.__eq__(self, other)

  def __neq__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash((self.dim, self.scale))


class UnitRegistry:
  """
  Stores known units for printing in best units.

  All a user needs to do is to use the `register_new_unit`
  function.

  Default registries:

  The units module defines three registries, the standard units,
  user units, and additional units. Finding best units is done
  by first checking standard, then user, then additional. New
  user units are added by using the `register_new_unit` function.

  Standard units includes all the basic non-compound unit names
  built in to the module, including volt, amp, etc. Additional
  units defines some compound units like newton metre (Nm) etc.

  Methods
  -------
  add
  __getitem__
  """

  __module__ = "brainunit"

  def __init__(self):
    self.units_for_dimensions = collections.defaultdict(dict)

  def add(self, u: Unit):
    """Add a unit to the registry"""
    if isinstance(u.value, (jax.ShapeDtypeStruct, jax.core.ShapedArray, DynamicJaxprTracer)):
      self.units_for_dimensions[u.dim][1.] = u
    else:
      self.units_for_dimensions[u.dim][float(u.value)] = u

  def __getitem__(self, x):
    """
    Returns the best unit for array x

    The algorithm is to consider the value:

    m=abs(x/u)

    for all matching units u. We select the unit where this ratio is the
    closest to 10 (if it is an array with several values, we select the
    unit where the deviations from that are the smallest. More precisely,
    the unit that minimizes the sum of (log10(m)-1)**2 over all entries).
    """
    matching = self.units_for_dimensions.get(x.dim, {})
    if len(matching) == 0:
      raise KeyError("Unit not found in registry.")

    matching_values = np.array(list(matching.keys()))
    if isinstance(x.value, (jax.ShapeDtypeStruct, jax.core.ShapedArray, DynamicJaxprTracer)):
      return matching[1.0]
    print_opts = np.get_printoptions()
    edgeitems, threshold = print_opts["edgeitems"], print_opts["threshold"]
    if x.size > threshold:
      # Only care about optimizing the units for the values that will
      # actually be shown later
      # The code looks a bit complex, but should return the same numbers
      # that are shown by numpy's string conversion
      slices = []
      for shape in x.shape:
        if shape > 2 * edgeitems:
          slices.append((slice(0, edgeitems), slice(-edgeitems, None)))
        else:
          slices.append((slice(None),))
      x_flat = np.hstack([np.array(x[use_slices].flatten().value)
                          for use_slices in itertools.product(*slices)])
    else:
      x_flat = np.array(x.value).flatten()
    floatreps = np.tile(np.abs(x_flat), (len(matching), 1)).T / matching_values
    # ignore zeros, they are well represented in any unit
    floatreps[floatreps == 0] = np.nan
    if np.all(np.isnan(floatreps)):
      return matching[1.0]  # all zeros, use the base unit
    deviations = np.nansum((np.log10(floatreps) - 1) ** 2, axis=0)
    return list(matching.values())[deviations.argmin()]


def register_new_unit(u):
  """Register a new unit for automatic displaying of arrays

  Parameters
  ----------
  u : `Unit`
      The unit that should be registered.

  Examples
  --------
  >>> from brainunit import *
  >>> 2.0*farad/metre**2
  2. * metre ** -4 * kilogram ** -1 * second ** 4 * amp ** 2
  >>> register_new_unit(pfarad / mmetre**2)
  >>> 2.0*farad/metre**2
  2000000. * pfarad / (mmetre ** 2)
  """
  user_unit_register.add(u)


#: `UnitRegistry` containing all the standard units (metre, kilogram, um2...)
standard_unit_register = UnitRegistry()

#: `UnitRegistry` containing additional units (newton*metre, farad / metre, ...)
additional_unit_register = UnitRegistry()

#: `UnitRegistry` containing all units defined by the user
user_unit_register = UnitRegistry()


def get_basic_unit(d):
  """
  Find an unscaled unit (e.g. `volt` but not `mvolt`) for a `Dimension`.

  Parameters
  ----------
  d : Dimension
      The dimension to find a unit for.

  Returns
  -------
  u : `Unit`
      A registered unscaled `Unit` for the dimensions ``d``, or a new `Unit`
      if no unit was found.
  """
  for unit_register in [
    standard_unit_register,
    user_unit_register,
    additional_unit_register,
  ]:
    if 1.0 in unit_register.units_for_dimensions[d]:
      return unit_register.units_for_dimensions[d][1.0]
  return Unit(1.0, dim=d)


def check_units(**au):
  """Decorator to check units of arguments passed to a function

  Examples
  --------
  >>> from brainunit import *
  >>> @check_units(I=amp, R=ohm, wibble=metre, result=volt)
  ... def getvoltage(I, R, **k):
  ...     return I*R

  You don't have to check the units of every variable in the function, and
  you can define what the units should be for variables that aren't
  explicitly named in the definition of the function. For example, the code
  above checks that the variable wibble should be a length, so writing

  >>> getvoltage(1*amp, 1*ohm, wibble=1)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  ...
  DimensionMismatchError: Function "getvoltage" variable "wibble" has wrong dimensions, dimensions were (1) (m)

  fails, but

  >>> getvoltage(1*amp, 1*ohm, wibble=1*metre)
  1. * volt

  passes. String arguments or ``None`` are not checked

  >>> getvoltage(1*amp, 1*ohm, wibble='hello')
  1. * volt

  By using the special name ``result``, you can check the return value of the
  function.

  You can also use ``1`` or ``bool`` as a special value to check for a
  unitless number or a boolean value, respectively:

  >>> @check_units(value=1, absolute=bool, result=bool)
  ... def is_high(value, absolute=False):
  ...     if absolute:
  ...         return abs(value) >= 5
  ...     else:
  ...         return value >= 5

  This will then again raise an error if the argument if not of the expected
  type:

  >>> is_high(7)
  True
  >>> is_high(-7, True)
  True
  >>> is_high(3, 4)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  ...
  TypeError: Function "is_high" expected a boolean value for argument "absolute" but got 4.

  If the return unit depends on the unit of an argument, you can also pass
  a function that takes the units of all the arguments as its inputs (in the
  order specified in the function header):

  >>> @check_units(result=lambda d: d**2)
  ... def square(value):
  ...     return value**2

  If several arguments take arbitrary units but they have to be
  consistent among each other, you can state the name of another argument as
  a string to state that it uses the same unit as that argument.

  >>> @check_units(summand_1=None, summand_2='summand_1')
  ... def multiply_sum(multiplicand, summand_1, summand_2):
  ...     "Calculates multiplicand*(summand_1 + summand_2)"
  ...     return multiplicand*(summand_1 + summand_2)
  >>> multiply_sum(3, 4*mV, 5*mV)
  27. * mvolt
  >>> multiply_sum(3*nA, 4*mV, 5*mV)
  27. * pwatt
  >>> multiply_sum(3*nA, 4*mV, 5*nA)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  ...
  DimensionMismatchError: Function 'multiply_sum' expected the same arguments for arguments 'summand_1', 'summand_2', but argument 'summand_1' has unit V, while argument 'summand_2' has unit A.

  Raises
  ------

  DimensionMismatchError
      In case the input arguments or the return value do not have the
      expected dimensions.
  TypeError
      If an input argument or return value was expected to be a boolean but
      is not.

  Notes
  -----
  This decorator will destroy the signature of the original function, and
  replace it with the signature ``(*args, **kwds)``. Other decorators will
  do the same thing, and this decorator critically needs to know the signature
  of the function it is acting on, so it is important that it is the first
  decorator to act on a function. It cannot be used in combination with
  another decorator that also needs to know the signature of the function.

  Note that the ``bool`` type is "strict", i.e. it expects a proper
  boolean value and does not accept 0 or 1. This is not the case the other
  way round, declaring an argument or return value as "1" *does* allow for a
  ``True`` or ``False`` value.
  """

  def do_check_units(f):
    def new_f(*args, **kwds):
      newkeyset = kwds.copy()
      arg_names = f.__code__.co_varnames[0: f.__code__.co_argcount]
      for n, v in zip(arg_names, args[0: f.__code__.co_argcount]):
        if (
            not isinstance(v, (Quantity, str, bool))
            and v is not None
            and n in au
        ):
          try:
            # allow e.g. to pass a Python list of values
            v = Quantity(v)
          except TypeError:
            if have_same_unit(au[n], 1):
              raise TypeError(
                f"Argument {n} is not a unitless value/array."
              )
            else:
              raise TypeError(
                f"Argument '{n}' is not a array, "
                "expected a array with dimensions "
                f"{au[n]}"
              )
        newkeyset[n] = v

      for k in newkeyset:
        # string variables are allowed to pass, the presumption is they
        # name another variable. None is also allowed, useful for
        # default parameters
        if (
            k in au
            and not isinstance(newkeyset[k], str)
            and not newkeyset[k] is None
            and not au[k] is None
        ):
          if au[k] == bool:
            if not isinstance(newkeyset[k], bool):
              value = newkeyset[k]
              error_message = (
                f"Function '{f.__name__}' "
                "expected a boolean value "
                f"for argument '{k}' but got "
                f"'{value}'"
              )
              raise TypeError(error_message)
          elif isinstance(au[k], str):
            if not au[k] in newkeyset:
              error_message = (
                f"Function '{f.__name__}' "
                "expected its argument to have the "
                f"same units as argument '{k}', but "
                "there is no argument of that name"
              )
              raise TypeError(error_message)
            if not have_same_unit(newkeyset[k], newkeyset[au[k]]):
              d1 = get_dim(newkeyset[k])
              d2 = get_dim(newkeyset[au[k]])
              error_message = (
                f"Function '{f.__name__}' expected "
                f"the argument '{k}' to have the same "
                f"units as argument '{au[k]}', but "
                f"argument '{k}' has "
                f"unit {get_unit_for_display(d1)}, "
                f"while argument '{au[k]}' "
                f"has unit {get_unit_for_display(d2)}."
              )
              raise DimensionMismatchError(error_message)
          elif not have_same_unit(newkeyset[k], au[k]):
            unit = repr(au[k])
            value = newkeyset[k]
            error_message = (
              f"Function '{f.__name__}' "
              "expected a array with unit "
              f"{unit} for argument '{k}' but got "
              f"'{value}'"
            )
            raise DimensionMismatchError(
              error_message, get_dim(newkeyset[k])
            )

      result = f(*args, **kwds)
      if "result" in au:
        if isinstance(au["result"], Callable) and au["result"] != bool:
          expected_result = au["result"](*[get_dim(a) for a in args])
        else:
          expected_result = au["result"]
        if au["result"] == bool:
          if not isinstance(result, bool):
            error_message = (
              "The return value of function "
              f"'{f.__name__}' was expected to be "
              "a boolean value, but was of type "
              f"{type(result)}"
            )
            raise TypeError(error_message)
        elif not have_same_unit(result, expected_result):
          unit = get_unit_for_display(expected_result)
          error_message = (
            "The return value of function "
            f"'{f.__name__}' was expected to have "
            f"unit {unit} but was "
            f"'{result}'"
          )
          raise DimensionMismatchError(error_message, get_dim(result))
      return result

    new_f._orig_func = f
    new_f.__doc__ = f.__doc__
    new_f.__name__ = f.__name__
    # store the information in the function, necessary when using the
    # function in expressions or equations
    if hasattr(f, "_orig_arg_names"):
      arg_names = f._orig_arg_names
    else:
      arg_names = f.__code__.co_varnames[: f.__code__.co_argcount]
    new_f._arg_names = arg_names
    new_f._arg_units = [au.get(name, None) for name in arg_names]
    return_unit = au.get("result", None)
    if return_unit is None:
      new_f._return_unit = None
    else:
      new_f._return_unit = return_unit
    if return_unit == bool:
      new_f._returns_bool = True
    else:
      new_f._returns_bool = False
    new_f._orig_arg_names = arg_names

    # copy any annotation attributes
    if hasattr(f, "_annotation_attributes"):
      for attrname in f._annotation_attributes:
        setattr(new_f, attrname, getattr(f, attrname))
    new_f._annotation_attributes = getattr(f, "_annotation_attributes", []) + [
      "_arg_units",
      "_arg_names",
      "_return_unit",
      "_orig_func",
      "_returns_bool",
    ]
    return new_f

  return do_check_units
