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

import itertools
import unittest
import warnings
from copy import deepcopy

import brainstate as bst
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_equal

import brainunit as bu
from brainunit._base import (
  DIMENSIONLESS,
  UNITLESS,
  DimensionMismatchError,
  Quantity,
  Unit,
  check_units,
  fail_for_dimension_mismatch,
  get_or_create_dimension,
  get_dim,
  have_same_dim,
  display_in_unit,
  is_scalar_type,
  assert_quantity,
)
from brainunit._unit_common import *
from brainunit._unit_shortcuts import kHz, ms, mV, nS


class TestDimension(unittest.TestCase):

  def test_inplace_operations(self):
    for inplace_op in [
      volt.dim.__imul__,
      volt.dim.__idiv__,
      volt.dim.__itruediv__,
      volt.dim.__ipow__,
    ]:
      with pytest.raises(NotImplementedError):
        inplace_op(volt.dim)


class TestUnit(unittest.TestCase):
  def test_div(self):
    print()

    a = 1. * bu.second
    b = 1. * bu.ms
    print(a / b)

    a = 1. * bu.ms
    print(a / b)

    c = bu.ms / bu.ms
    assert c.is_unitless

    print(bu.Unit((bu.ms / bu.ms).dim, scale=2))
    print(bu.Unit(bu.ms.dim, scale=2))

  def test_mul(self):
    a = bu.Unit(base=2)
    b = bu.Unit(base=10)
    with pytest.raises(AssertionError):
      a * b

  def test_inplace_operations(self):
    # make sure that inplace operations do not work on units/dimensions at all
    for inplace_op in [
      volt.__iadd__,
      volt.__isub__,
      volt.__imul__,
      volt.__idiv__,
      volt.__itruediv__,
      volt.__ifloordiv__,
      volt.__imod__,
      volt.__ipow__,
    ]:
      with pytest.raises(NotImplementedError):
        inplace_op(volt)


class TestQuantity(unittest.TestCase):
  def test_dim(self):
    a = [1, 2.] * bu.ms

    with self.assertRaises(NotImplementedError):
      a.dim = bu.mV.dim

  def test_clip(self):
    a = [1, 2.] * bu.ms
    self.assertTrue(bu.math.allclose(a.clip(1.5 * bu.ms, 2.5 * bu.ms), [1.5, 2.] * bu.ms))

    b = bu.Quantity([1, 2.])
    self.assertTrue(bu.math.allclose(b.clip(1.5, 2.5), bu.math.asarray([1.5, 2.])))

  def test_round(self):
    for u in [bu.ms, bu.joule, bu.mV]:
      a = [1.1, 2.2] * u
      self.assertTrue(bu.math.allclose(a.round(), [1, 2] * u))

    b = bu.Quantity([1.1, 2.2])
    self.assertTrue(bu.math.allclose(b.round(), bu.math.asarray([1, 2])))

  def test_astype(self):
    a = [1, 2.] * bu.ms
    self.assertTrue(a.astype(jnp.float16).dtype == jnp.float16)

  def test___array__(self):
    a = bu.Quantity([1, 2.])
    self.assertTrue(bu.math.allclose(np.asarray(a), np.asarray([1, 2.])))

    with self.assertRaises(TypeError):
      a = [1, 2.] * bu.ms
      self.assertTrue(bu.math.allclose(np.asarray(a), np.asarray([1, 2.])))

  def test__float__(self):
    a = bu.Quantity(1.)
    self.assertTrue(bu.math.allclose(float(a), 1.))

    a = bu.Quantity([1, 2.])
    with self.assertRaises(TypeError):
      self.assertTrue(bu.math.allclose(float(a), 1.5))

    with self.assertRaises(TypeError):
      a = [1, 2.] * bu.ms
      self.assertTrue(bu.math.allclose(float(a), 1.5))

  def test_construction(self):
    """Test the construction of Array objects"""
    q = 500 * ms
    assert_quantity(q, 0.5, second)
    q = np.float64(500) * ms
    assert_quantity(q, 0.5, second)
    q = np.array(500) * ms
    assert_quantity(q, 0.5, second)
    q = np.array([500, 1000]) * ms
    assert_quantity(q, np.array([0.5, 1]), second)
    q = Quantity(500)
    assert_quantity(q, 500)
    q = Quantity(500, unit=second)
    assert_quantity(q, 500, second)
    q = Quantity([0.5, 1], unit=second)
    assert_quantity(q, np.array([0.5, 1]), second)
    q = Quantity(np.array([0.5, 1]), unit=second)
    assert_quantity(q, np.array([0.5, 1]), second)
    q = Quantity([500 * ms, 1 * second])
    assert_quantity(q, np.array([0.5, 1]), second)
    q = Quantity.with_unit(np.array([0.5, 1]), unit=second)
    assert_quantity(q, np.array([0.5, 1]), second)
    q = [0.5, 1] * second
    assert_quantity(q, np.array([0.5, 1]), second)

    # dimensionless quantities
    q = Quantity([1, 2, 3])
    assert_quantity(q, np.array([1, 2, 3]), Unit())
    q = Quantity(np.array([1, 2, 3]))
    assert_quantity(q, np.array([1, 2, 3]), Unit())
    q = Quantity([])
    assert_quantity(q, np.array([]), Unit())

    # Illegal constructor calls
    with pytest.raises(TypeError):
      Quantity([500 * ms, 1])
    with pytest.raises(TypeError):
      Quantity(["some", "nonsense"])
    with pytest.raises(TypeError):
      Quantity([500 * ms, 1 * volt])

  def test_construction2(self):
    a = np.array([1, 2, 3]) * bu.mV
    b = bu.Quantity(a)
    self.assertTrue(bu.math.allclose(a, b))

    c = bu.Quantity(a, unit=bu.volt)
    self.assertTrue(bu.math.allclose(c.mantissa, np.asarray([1, 2, 3]) * 1e-3))
    self.assertTrue(bu.math.allclose(c, a))
    print(c)

  def test_get_dimensions(self):
    """
    Test various ways of getting/comparing the dimensions of a Array.
    """
    q = 500 * ms
    assert get_dim(q) == get_or_create_dimension(q.dim._dims)
    assert get_dim(q) is q.dim
    assert q.has_same_unit(3 * second)
    dims = q.dim
    assert_equal(dims.get_dimension("time"), 1.0)
    assert_equal(dims.get_dimension("length"), 0)

    assert get_dim(5) is DIMENSIONLESS
    assert get_dim(5.0) is DIMENSIONLESS
    assert get_dim(np.array(5, dtype=np.int32)) is DIMENSIONLESS
    assert get_dim(np.array(5.0)) is DIMENSIONLESS
    assert get_dim(np.float32(5.0)) is DIMENSIONLESS
    assert get_dim(np.float64(5.0)) is DIMENSIONLESS
    assert is_scalar_type(5)
    assert is_scalar_type(5.0)
    assert is_scalar_type(np.array(5, dtype=np.int32))
    assert is_scalar_type(np.array(5.0))
    assert is_scalar_type(np.float32(5.0))
    assert is_scalar_type(np.float64(5.0))
    # wrong number of indices
    with pytest.raises(TypeError):
      get_or_create_dimension([1, 2, 3, 4, 5, 6])
    # not a sequence
    with pytest.raises(TypeError):
      get_or_create_dimension(42)

  def test_display(self):
    """
    Test displaying a Array in different units
    """

    assert_equal(display_in_unit(3. * volt, mvolt), "3000. * mvolt")
    # assert_equal(display_in_unit(10. * mV, ohm * amp), "0.01 ohm * A")
    assert_equal(display_in_unit(10. * mV, ohm * amp), "0.01 * volt")
    with pytest.raises(bu.UnitMismatchError):
      display_in_unit(10 * nS, ohm)
    with bst.environ.context(precision=32):
      assert_equal(display_in_unit(3. * volt, mvolt), "3000. * mvolt")
      assert_equal(display_in_unit(10. * mV, ohm * amp), "0.01 * volt")
      with pytest.raises(bu.UnitMismatchError):
        display_in_unit(10 * nS, ohm)
    assert_equal(display_in_unit(10.0, Unit(scale=1)), "1. * Unit(10.0)")


  def test_display2(self):

    @jax.jit
    def f(s):
      a = bu.ms ** s
      print(a)
      return a

    with self.assertRaises(jax.errors.TracerBoolConversionError):
      f(2)


  def test_unary_operations(self):
    q = Quantity(5, unit=mV)
    assert_quantity(-q, -5, mV)
    assert_quantity(+q, 5, mV)
    assert_quantity(abs(Quantity(-5, unit=mV)), 5, mV)
    assert_quantity(~Quantity(0b101), -0b110, UNITLESS)

  def test_operations(self):
    q1 = 5 * second
    q2 = 10 * second
    assert_quantity(q1 + q2, 15, second)
    assert_quantity(q1 - q2, -5, second)
    assert_quantity(q1 * q2, 50, second * second)
    assert_quantity(q2 / q1, 2)
    assert_quantity(q2 // q1, 2)
    assert_quantity(q2 % q1, 0, second)
    assert_quantity(divmod(q2, q1)[0], 2)
    assert_quantity(divmod(q2, q1)[1], 0, second)
    assert_quantity(q1 ** 2, 25, second ** 2)
    assert_quantity(round(q1, 0), 5, second)

    # matmul
    q1 = [1, 2] * second
    q2 = [3, 4] * second
    assert_quantity(q1 @ q2, 11, second ** 2)
    q1 = Quantity([1, 2], unit=second)
    q2 = Quantity([3, 4], unit=second)
    assert_quantity(q1 @ q2, 11, second ** 2)

    # shift
    q1 = Quantity(0b1100, dtype=jnp.int32)
    assert_quantity(q1 << 1, 0b11000)
    assert_quantity(q1 >> 1, 0b110)

  def test_numpy_methods(self):
    q = [[1, 2], [3, 4]] * second
    assert q.all()
    assert q.any()
    assert q.nonzero()[0].tolist() == [0, 0, 1, 1]
    assert q.argmax() == 3
    assert q.argmin() == 0
    assert q.argsort(axis=None).tolist() == [0, 1, 2, 3]
    assert_quantity(q.var(), 1.25, second ** 2)
    assert_quantity(q.round(), [[1, 2], [3, 4]], second)
    assert_quantity(q.std(), 1.11803398875, second)
    assert_quantity(q.sum(), 10, second)
    assert_quantity(q.trace(), 5, second)
    assert_quantity(q.cumsum(), [1, 3, 6, 10], second)
    assert_quantity(q.cumprod(), [1, 2, 6, 24], second ** 4)
    assert_quantity(q.diagonal(), [1, 4], second)
    assert_quantity(q.max(), 4, second)
    assert_quantity(q.mean(), 2.5, second)
    assert_quantity(q.min(), 1, second)
    assert_quantity(q.ptp(), 3, second)
    assert_quantity(q.ravel(), [1, 2, 3, 4], second)

  def test_shape_manipulation(self):
    q = [[1, 2], [3, 4]] * volt

    # Test flatten
    assert_quantity(q.flatten(), [1, 2, 3, 4], volt)

    # Test swapaxes
    assert_quantity(q.swapaxes(0, 1), [[1, 3], [2, 4]], volt)

    # Test take
    assert_quantity(q.take(jnp.array([0, 2])), [1, 3], volt)

    # Test transpose
    assert_quantity(q.transpose(), [[1, 3], [2, 4]], volt)

    # Test tile
    assert_quantity(q.tile(2), [[1, 2, 1, 2], [3, 4, 3, 4]], volt)

    # Test unsqueeze
    assert_quantity(q.unsqueeze(0), [[[1, 2], [3, 4]]], volt)

    # Test expand_dims
    assert_quantity(q.expand_dims(0), [[[1, 2], [3, 4]]], volt)

    # Test expand_as
    expand_as_shape = (1, 2, 2)
    assert_quantity(q.expand_as(jnp.zeros(expand_as_shape).shape), [[[1, 2], [3, 4]]], volt)

    # Test put
    q_put = [[1, 2], [3, 4]] * volt
    q_put.put(((1, 0), (0, 1)), [10, 30] * volt)
    assert_quantity(q_put, [[1, 30], [10, 4]], volt)

    # Test squeeze (no axes to squeeze in this case, so the array remains the same)
    q_squeeze = [[1, 2], [3, 4]] * volt
    assert_quantity(q_squeeze.squeeze(), [[1, 2], [3, 4]], volt)

    # Test array_split
    q_spilt = [[10, 2], [30, 4]] * volt
    assert_quantity(np.array_split(q_spilt, 2)[0], [[10, 2]], volt)

  def test_misc_methods(self):
    q = [5, 10, 15] * volt

    # Test astype
    assert_quantity(q.astype(np.float32), [5, 10, 15], volt)

    # Test clip
    min_val = [6, 6, 6] * volt
    max_val = [14, 14, 14] * volt
    assert_quantity(q.clip(min_val, max_val), [6, 10, 14], volt)

    # Test conj
    assert_quantity(q.conj(), [5, 10, 15], volt)

    # Test conjugate
    assert_quantity(q.conjugate(), [5, 10, 15], volt)

    # Test copy
    assert_quantity(q.copy(), [5, 10, 15], volt)

    # Test dot
    assert_quantity(q.dot(Quantity([2, 2, 2])), 60, volt)

    # Test fill
    q_filled = [5, 10, 15] * volt
    q_filled.fill(2 * volt)
    assert_quantity(q_filled, [2, 2, 2], volt)

    # Test item
    assert_quantity(q.item(0), 5, volt)

    # Test prod
    assert_quantity(q.prod(), 750, volt ** 3)

    # Test repeat
    assert_quantity(q.repeat(2), [5, 5, 10, 10, 15, 15], volt)

    # Test clamp (same as clip, but using min and max values directly)
    assert_quantity(q.clip(6 * volt, 14 * volt), [6, 10, 14], volt)

    # Test sort
    q = [15, 5, 10] * volt
    assert_quantity(q.sort(), [5, 10, 15], volt)

  def test_slicing(self):
    # Slicing and indexing, setting items
    a = np.reshape(np.arange(6), (2, 3))
    q = a * mV
    assert bu.math.allclose(q[:].mantissa, q.mantissa)
    assert bu.math.allclose(q[0].mantissa, (a[0] * volt).mantissa)
    assert bu.math.allclose(q[0:1].mantissa, (a[0:1] * volt).mantissa)
    assert bu.math.allclose(q[0, 1].mantissa, (a[0, 1] * volt).mantissa)
    assert bu.math.allclose(q[0:1, 1:].mantissa, (a[0:1, 1:] * volt).mantissa)
    bool_matrix = np.array([[True, False, False], [False, False, True]])
    assert bu.math.allclose(q[bool_matrix].mantissa, (a[bool_matrix] * volt).mantissa)

  def test_setting(self):
    quantity = np.reshape(np.arange(6), (2, 3)) * mV
    quantity[0, 1] = 10 * mV
    assert quantity[0, 1] == 10 * mV
    quantity[:, 1] = 20 * mV
    assert np.all(quantity[:, 1] == 20 * mV)
    quantity[1, :] = np.ones((3,)) * volt
    assert np.all(quantity[1, :] == 1 * volt)

    quantity[1, 2] = 0 * mV
    assert quantity[1, 2] == 0 * mV

    def set_to_value(key, value):
      quantity[key] = value

    with pytest.raises(TypeError):
      set_to_value(0, 1)
    with pytest.raises(bu.UnitMismatchError):
      set_to_value(0, 1 * second)
    with pytest.raises(TypeError):
      set_to_value((slice(2), slice(3)), np.ones((2, 3)))

    quantity = Quantity(bst.random.rand(10))
    quantity[0] = 1

  def test_multiplication_division(self):
    u = mV
    quantities = [3 * mV, np.array([1, 2]) * u, np.ones((3, 3)) * u]
    q2 = 5 * second

    for q in quantities:
      # Scalars and array scalars
      assert_quantity(q / 3, q.mantissa / 3, u)
      assert_quantity(3 / q, 3 / q.mantissa, 1 / u)
      assert_quantity(q * 3, q.mantissa * 3, u)
      assert_quantity(3 * q, 3 * q.mantissa, u)
      assert_quantity(q / np.float64(3), q.mantissa / 3, u)
      assert_quantity(np.float64(3) / q, 3 / q.mantissa, 1 / u)
      assert_quantity(q * np.float64(3), q.mantissa * 3, u)
      assert_quantity(np.float64(3) * q, 3 * q.mantissa, u)
      assert_quantity(q / jnp.array(3), q.mantissa / 3, u)
      assert_quantity(np.array(3) / q, 3 / q.mantissa, 1 / u)
      assert_quantity(q * jnp.array(3), q.mantissa * 3, u)
      assert_quantity(np.array(3) * q, 3 * q.mantissa, u)

      # (unitless) arrays
      assert_quantity(q / np.array([3]), q.mantissa / 3, u)
      assert_quantity(np.array([3]) / q, 3 / q.mantissa, 1 / u)
      assert_quantity(q * np.array([3]), q.mantissa * 3, u)
      assert_quantity(np.array([3]) * q, 3 * q.mantissa, u)

      # arrays with units
      assert_quantity(q / q, q.mantissa / q.mantissa)
      assert_quantity(q * q, q.mantissa ** 2, u ** 2)
      assert_quantity(q / q2, q.mantissa / q2.mantissa, u / second)
      assert_quantity(q2 / q, q2.mantissa / q.mantissa, second / u)
      assert_quantity(q * q2, q.mantissa * q2.mantissa, u * second)

      # # using unsupported objects should fail
      # with pytest.raises(TypeError):
      #   q / "string"
      # with pytest.raises(TypeError):
      #   "string" / q
      # with pytest.raises(TypeError):
      #   "string" * q
      # with pytest.raises(TypeError):
      #   q * "string"

  def test_addition_subtraction(self):
    u = mV
    quantities = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
    q2 = 5 * volt
    q2_mantissa = q2.in_unit(u).mantissa

    for q in quantities:
      # arrays with units
      assert_quantity(q + q, q.mantissa + q.mantissa, u)
      assert_quantity(q - q, 0, u)
      assert_quantity(q + q2, q.mantissa + q2_mantissa, u)
      assert_quantity(q2 + q, q2_mantissa + q.mantissa, u)
      assert_quantity(q - q2, q.mantissa - q2_mantissa, u)
      assert_quantity(q2 - q, q2_mantissa - q.mantissa, u)

      # mismatching units
      with pytest.raises(bu.UnitMismatchError):
        q + 5 * second
      with pytest.raises(bu.UnitMismatchError):
        5 * second + q
      with pytest.raises(bu.UnitMismatchError):
        q - 5 * second
      with pytest.raises(bu.UnitMismatchError):
        5 * second - q

      # scalar
      with pytest.raises(bu.UnitMismatchError):
        q + 5
      with pytest.raises(bu.UnitMismatchError):
        5 + q
      with pytest.raises(bu.UnitMismatchError):
        q + np.float64(5)
      with pytest.raises(bu.UnitMismatchError):
        np.float64(5) + q
      with pytest.raises(bu.UnitMismatchError):
        q - 5
      with pytest.raises(bu.UnitMismatchError):
        5 - q
      with pytest.raises(bu.UnitMismatchError):
        q - np.float64(5)
      with pytest.raises(bu.UnitMismatchError):
        np.float64(5) - q

      # unitless array
      with pytest.raises(bu.UnitMismatchError):
        q + np.array([5])
      with pytest.raises(bu.UnitMismatchError):
        np.array([5]) + q
      with pytest.raises(bu.UnitMismatchError):
        q + np.array([5], dtype=np.float64)
      with pytest.raises(bu.UnitMismatchError):
        np.array([5], dtype=np.float64) + q
      with pytest.raises(bu.UnitMismatchError):
        q - np.array([5])
      with pytest.raises(bu.UnitMismatchError):
        np.array([5]) - q
      with pytest.raises(bu.UnitMismatchError):
        q - np.array([5], dtype=np.float64)
      with pytest.raises(bu.UnitMismatchError):
        np.array([5], dtype=np.float64) - q

      # Check that operations with 0 work
      with pytest.raises(bu.UnitMismatchError):
        assert_quantity(q + 0, q.mantissa, u)
      with pytest.raises(bu.UnitMismatchError):
        assert_quantity(0 + q, q.mantissa, u)
      with pytest.raises(bu.UnitMismatchError):
        assert_quantity(q - 0, q.mantissa, u)
      with pytest.raises(bu.UnitMismatchError):
        # Doesn't support 0 - Quantity
        # assert_quantity(0 - q, -q.mantissa, volt)
        assert_quantity(q + np.float64(0), q.mantissa, u)
      with pytest.raises(bu.UnitMismatchError):
        assert_quantity(np.float64(0) + q, q.mantissa, u)
      with pytest.raises(bu.UnitMismatchError):
        assert_quantity(q - np.float64(0), q.mantissa, u)

      # using unsupported objects should fail
      with pytest.raises(bu.UnitMismatchError):
        "string" + q
      with pytest.raises(bu.UnitMismatchError):
        q + "string"
      with pytest.raises(bu.UnitMismatchError):
        q - "string"
      with pytest.raises(bu.UnitMismatchError):
        "string" - q

  def test_binary_operations(self):
    """Test whether binary operations work when they should and raise
    DimensionMismatchErrors when they should.
    Does not test for the actual result.
    """
    from operator import add, eq, ge, gt, le, lt, ne, sub

    def assert_operations_work(a, b):
      try:
        # Test python builtins
        tryops = [add, sub, lt, le, gt, ge, eq, ne]
        for op in tryops:
          op(a, b)
          op(b, a)

        # Test equivalent numpy functions
        numpy_funcs = [
          bu.math.add,
          bu.math.subtract,
          bu.math.less,
          bu.math.less_equal,
          bu.math.greater,
          bu.math.greater_equal,
          bu.math.equal,
          bu.math.not_equal,
          bu.math.maximum,
          bu.math.minimum,
        ]
        for numpy_func in numpy_funcs:
          numpy_func(a, b)
          numpy_func(b, a)
      except DimensionMismatchError as ex:
        raise AssertionError(f"Operation raised unexpected exception: {ex}")

    def assert_operations_do_not_work(a, b):
      # Test python builtins
      tryops = [add, sub, lt, le, gt, ge, eq, ne]
      for op in tryops:
        with pytest.raises(bu.UnitMismatchError):
          op(a, b)
        with pytest.raises(bu.UnitMismatchError):
          op(b, a)

    #
    # Check that consistent units work
    #

    # unit arrays
    a = 1 * kilogram
    for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
      assert_operations_work(a, b)

    # dimensionless units and scalars
    a = 1
    for b in [
      2 * kilogram / kilogram,
      np.array([2]) * kilogram / kilogram,
      np.array([1, 2]) * kilogram / kilogram,
    ]:
      assert_operations_work(a, b)

    # dimensionless units and unitless arrays
    a = np.array([1])
    for b in [
      2 * kilogram / kilogram,
      np.array([2]) * kilogram / kilogram,
      np.array([1, 2]) * kilogram / kilogram,
    ]:
      assert_operations_work(a, b)

    #
    # Check that inconsistent units do not work
    #

    # unit arrays
    a = np.array([1]) * second
    for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
      assert_operations_do_not_work(a, b)

    # unitless array
    a = np.array([1])
    for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
      assert_operations_do_not_work(a, b)

    # scalar
    a = 1
    for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
      assert_operations_do_not_work(a, b)

    # Check that comparisons with inf/-inf always work
    values = [
      2 * kilogram / kilogram,
      2 * kilogram,
      np.array([2]) * kilogram,
      np.array([1, 2]) * kilogram,
    ]
    for value in values:
      assert bu.math.all(value < np.inf * bu.get_unit(value))
      assert bu.math.all(np.inf * bu.get_unit(value) > value)
      assert bu.math.all(value <= np.inf * bu.get_unit(value))
      assert bu.math.all(np.inf * bu.get_unit(value) >= value)
      assert bu.math.all(value != np.inf * bu.get_unit(value))
      assert bu.math.all(np.inf * bu.get_unit(value) != value)
      assert bu.math.all(value >= -np.inf * bu.get_unit(value))
      assert bu.math.all(-np.inf * bu.get_unit(value) <= value)
      assert bu.math.all(value > -np.inf * bu.get_unit(value))
      assert bu.math.all(-np.inf * bu.get_unit(value) < value)

  def test_power(self):
    """
    Test raising quantities to a power.
    """
    arrs = [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]
    for a in arrs:
      assert_quantity(a ** 3, a.mantissa ** 3, kilogram ** 3)
      # Test raising to a dimensionless Array
      assert_quantity(a ** (3 * volt / volt), a.mantissa ** 3, kilogram ** 3)
      with pytest.raises(AssertionError):
        a ** (2 * volt)
      with pytest.raises(TypeError):
        a ** np.array([2, 3])

  def test_inplace_operations(self):
    q = np.arange(10) * volt
    q_orig = q.copy()
    q_id = id(q)

    # Doesn't support in-place operations which change unit
    # q *= 2
    # assert np.all(q == 2 * q_orig) and id(q) == q_id
    # q /= 2
    # assert np.all(q == q_orig) and id(q) == q_id
    q += 1 * volt
    assert np.all(q == q_orig + 1 * volt) and id(q) == q_id
    q -= 1 * volt
    assert np.all(q == q_orig) and id(q) == q_id

    # q **= 2
    # assert np.all(q == q_orig ** 2) and id(q) == q_id
    # q **= 0.5
    # assert np.all(q == q_orig) and id(q) == q_id

    def illegal_add(q2):
      q = np.arange(10) * volt
      q += q2

    with pytest.raises(bu.UnitMismatchError):
      illegal_add(1 * second)
    with pytest.raises(bu.UnitMismatchError):
      illegal_add(1)

    def illegal_sub(q2):
      q = np.arange(10) * volt
      q -= q2

    with pytest.raises(bu.UnitMismatchError):
      illegal_sub(1 * second)
    with pytest.raises(bu.UnitMismatchError):
      illegal_sub(1)

    def illegal_pow(q2):
      q = np.arange(10) * volt
      q **= q2

    with pytest.raises(NotImplementedError):
      illegal_pow(1 * volt)
    with pytest.raises(NotImplementedError):
      illegal_pow(np.arange(10))

    # inplace operations with unsupported objects should fail
    for inplace_op in [
      q.__iadd__,
      q.__isub__,
      q.__imul__,
      q.__idiv__,
      q.__itruediv__,
      q.__ifloordiv__,
      q.__imod__,
      q.__ipow__,
    ]:
      try:
        result = inplace_op("string")
      except (bu.UnitMismatchError, NotImplementedError):
        pass

  def test_deepcopy(self):
    d = {"x": 1 * second}

    d_copy = deepcopy(d)
    assert d_copy["x"] == 1 * second
    d_copy["x"] += 1 * second
    assert d_copy["x"] == 2 * second
    assert d["x"] == 1 * second

  def test_numpy_functions_indices(self):
    """
    Check numpy functions that return indices.
    """
    values = [np.array([-4, 3, -2, 1, 0]), np.ones((3, 3)), np.array([17])]
    units = [volt, second, siemens, mV, kHz]

    # numpy functions
    keep_dim_funcs = [np.argmin, np.argmax, np.argsort, np.nonzero]

    for value, unit in itertools.product(values, units):
      q_ar = value * unit
      for func in keep_dim_funcs:
        test_ar = func(q_ar)
        # Compare it to the result on the same value without units
        comparison_ar = func(value)
        test_ar = np.asarray(test_ar)
        comparison_ar = np.asarray(comparison_ar)
        assert_equal(
          test_ar,
          comparison_ar,
          (
              "function %s returned an incorrect result when used on quantities "
              % func.__name__
          ),
        )

  def test_list(self):
    """
    Test converting to and from a list.
    """
    values = [3 * mV, np.array([1, 2]) * mV, np.arange(12).reshape(4, 3) * mV]
    for value in values:
      l = value.tolist()
      from_list = Quantity(l)
      assert have_same_dim(from_list, value)
      assert bu.math.allclose(from_list.mantissa, value.mantissa)

  def test_units_vs_quantities(self):
    # Unit objects should stay Unit objects under certain operations
    # (important e.g. in the unit definition of Equations, where only units but
    # not quantities are allowed)
    assert isinstance(meter ** 2, Unit)
    assert isinstance(meter ** -1, Unit)
    assert isinstance(meter ** 0.5, Unit)
    assert isinstance(meter / second, Unit)
    assert isinstance(amp / meter ** 2, Unit)
    assert isinstance(1 / meter, Unit)
    assert isinstance(1.0 / meter, Unit)

    # Using the unconventional type(x) == y since we want to test that
    # e.g. meter**2 stays a Unit and does not become a Array however Unit
    # inherits from Array and therefore both would pass the isinstance test
    assert type(2 / meter) == Quantity
    assert type(2 * meter) == Quantity
    assert type(meter + meter) == Unit
    assert type(meter - meter) == Unit

  def test_jit_array(self):
    @jax.jit
    def f1(a):
      b = a * bu.siemens / bu.cm ** 2
      print(b)
      return b

    val = np.random.rand(3)
    r = f1(val)
    bu.math.allclose(val * bu.siemens / bu.cm ** 2, r)

    @jax.jit
    def f2(a):
      a = a + 1. * bu.siemens / bu.cm ** 2
      return a

    val = np.random.rand(3) * bu.siemens / bu.cm ** 2
    r = f2(val)
    bu.math.allclose(val + 1 * bu.siemens / bu.cm ** 2, r)

    @jax.jit
    def f3(a):
      b = a * bu.siemens / bu.cm ** 2
      print(display_in_unit(b, bu.siemens / bu.meter ** 2))
      return b

    val = np.random.rand(3)
    r = f3(val)
    bu.math.allclose(val * bu.siemens / bu.cm ** 2, r)

  def test_jit_array2(self):
    a = 2.0 * (bu.farad / bu.metre ** 2)
    print(a)

    @jax.jit
    def f(b):
      print(b)
      return b

    f(a)

  def test_setiterm(self):
    u = bu.Quantity([0, 0, 0.])
    u[jnp.asarray([0, 1, 1])] += jnp.asarray([1., 1., 1.])
    assert_quantity(u, [1., 1., 0.])

    u = bu.Quantity([0, 0, 0.])
    u = u.scatter_add(jnp.asarray([0, 1, 1]), jnp.asarray([1., 1., 1.]))
    assert_quantity(u, [1., 2., 0.])

    nu = np.asarray([0, 0, 0.])
    nu[np.asarray([0, 1, 1])] += np.asarray([1., 1., 1.])
    self.assertTrue(np.allclose(nu, np.asarray([1., 1., 0.])))
  
  def test_at(self):
    x = jnp.arange(5.0) * bu.mV
    with self.assertRaises(bu.UnitMismatchError):
      x.at[2].add(10)
    x.at[2].add(10 * bu.mV)
    x.at[10].add(10 * bu.mV)  # out-of-bounds indices are ignored
    x.at[20].add(10 * bu.mV, mode='clip')
    x.at[2].get()
    x.at[20].get()  # out-of-bounds indices clipped
    x.at[20].get(mode='fill')  # out-of-bounds indices filled with NaN
    with self.assertRaises(bu.UnitMismatchError):
      x.at[20].get(mode='fill', fill_value=-1)  # custom fill value
    x.at[20].get(mode='fill', fill_value=-1 * bu.mV)  # custom fill value


class TestNumPyFunctions(unittest.TestCase):
  def test_special_case_numpy_functions(self):
    """
    Test a couple of functions/methods that need special treatment.
    """
    from brainunit.math import diagonal, dot, ravel, trace, where

    quadratic_matrix = np.reshape(np.arange(9), (3, 3)) * mV

    # Temporarily suppress warnings related to the matplotlib 1.3 bug
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      # Check that function and method do the same
      assert bu.math.allclose(ravel(quadratic_matrix).mantissa, quadratic_matrix.ravel().mantissa)
      # Check that function gives the same result as on unitless arrays
      assert bu.math.allclose(
        np.asarray(ravel(quadratic_matrix).mantissa),
        ravel(np.asarray(quadratic_matrix.mantissa))
      )
      # Check that the function gives the same results as the original numpy
      # function
      assert bu.math.allclose(
        np.ravel(np.asarray(quadratic_matrix.mantissa)),
        ravel(np.asarray(quadratic_matrix.mantissa))
      )

    # Do the same checks for diagonal, trace and dot
    assert bu.math.allclose(diagonal(quadratic_matrix).mantissa, quadratic_matrix.diagonal().mantissa)
    assert bu.math.allclose(
      np.asarray(diagonal(quadratic_matrix).mantissa),
      diagonal(np.asarray(quadratic_matrix.mantissa))
    )
    assert bu.math.allclose(
      np.diagonal(np.asarray(quadratic_matrix.mantissa)),
      diagonal(np.asarray(quadratic_matrix.mantissa)),
    )

    assert bu.math.allclose(
      trace(quadratic_matrix).mantissa,
      quadratic_matrix.trace().mantissa
    )
    assert bu.math.allclose(
      np.asarray(trace(quadratic_matrix).mantissa),
      trace(np.asarray(quadratic_matrix.mantissa))
    )
    assert bu.math.allclose(
      np.trace(np.asarray(quadratic_matrix.mantissa)),
      trace(np.asarray(quadratic_matrix.mantissa))
    )

    assert bu.math.allclose(
      dot(quadratic_matrix, quadratic_matrix).mantissa,
      quadratic_matrix.dot(quadratic_matrix).mantissa
    )
    assert bu.math.allclose(
      np.asarray(dot(quadratic_matrix, quadratic_matrix).mantissa),
      dot(np.asarray(quadratic_matrix.mantissa),
          np.asarray(quadratic_matrix.mantissa)),
    )
    assert bu.math.allclose(
      np.dot(np.asarray(quadratic_matrix.mantissa),
             np.asarray(quadratic_matrix.mantissa)),
      dot(np.asarray(quadratic_matrix.mantissa),
          np.asarray(quadratic_matrix.mantissa)),
    )
    assert bu.math.allclose(
      np.asarray(quadratic_matrix.prod().mantissa),
      np.asarray(quadratic_matrix.mantissa).prod()
    )
    assert bu.math.allclose(
      np.asarray(quadratic_matrix.prod(axis=0).mantissa),
      np.asarray(quadratic_matrix.mantissa).prod(axis=0),
    )

    # Check for correct units
    assert have_same_dim(quadratic_matrix, ravel(quadratic_matrix))
    assert have_same_dim(quadratic_matrix, trace(quadratic_matrix))
    assert have_same_dim(quadratic_matrix, diagonal(quadratic_matrix))
    assert have_same_dim(
      quadratic_matrix[0] ** 2,
      dot(quadratic_matrix, quadratic_matrix)
    )
    assert have_same_dim(
      quadratic_matrix.prod(axis=0),
      quadratic_matrix[0] ** quadratic_matrix.shape[0]
    )

    # check the where function
    # pure numpy array
    cond = np.array([True, False, False])
    ar1 = np.array([1, 2, 3])
    ar2 = np.array([4, 5, 6])
    assert_equal(np.where(cond), where(cond))
    assert_equal(np.where(cond, ar1, ar2), where(cond, ar1, ar2))

    # dimensionless Array
    assert bu.math.allclose(
      np.where(cond, ar1, ar2),
      np.asarray(where(cond, ar1 * mV / mV, ar2 * mV / mV))
    )

    # Array with dimensions
    ar1 = ar1 * mV
    ar2 = ar2 * mV
    assert bu.math.allclose(
      np.where(cond, ar1.mantissa, ar2.mantissa),
      np.asarray(where(cond, ar1, ar2).mantissa),
    )

    # Check some error cases
    with pytest.raises(AssertionError):
      where(cond, ar1)
    with pytest.raises(TypeError):
      where(cond, ar1, ar1, ar2)
    with pytest.raises(bu.UnitMismatchError):
      where(cond, ar1, ar1 / ms)

    # Check setasflat (for numpy < 1.7)
    if hasattr(Quantity, "setasflat"):
      a = np.arange(10) * mV
      b = np.ones(10).reshape(5, 2) * volt
      c = np.ones(10).reshape(5, 2) * second
      with pytest.raises(DimensionMismatchError):
        a.setasflat(c)
      a.setasflat(b)
      assert_equal(a.flatten(), b.flatten())

    # Check cumprod
    a = np.arange(1, 10) * mV / mV
    assert bu.math.allclose(a.cumprod(), np.asarray(a).cumprod())
    (np.arange(1, 5) * mV).cumprod()

  def test_unit_discarding_functions(self):
    """
    Test functions that discard units.
    """

    values = [3 * mV, np.array([1, 2]) * mV, np.arange(12).reshape(3, 4) * mV]
    for a in values:
      assert np.allclose(np.sign(a.mantissa), np.sign(np.asarray(a.mantissa)))
      assert np.allclose(bu.math.zeros_like(a).mantissa, np.zeros_like(np.asarray(a.mantissa)))
      assert np.allclose(bu.math.ones_like(a).mantissa, np.ones_like(np.asarray(a.mantissa)))
      if a.ndim > 0:
        # Calling non-zero on a 0d array is deprecated, don't test it:
        assert np.allclose(np.nonzero(a.mantissa), np.nonzero(np.asarray(a.mantissa)))

  def test_numpy_functions_same_dimensions(self):
    values = [np.array([1, 2]), np.ones((3, 3))]
    units = [volt, second, siemens, mV, kHz]

    # Do not suopport numpy functions
    keep_dim_funcs = [
      # 'abs',
      'cumsum',
      'max',
      'mean',
      'min',
      # 'negative',
      'ptp',
      'round',
      'squeeze',
      'std',
      'sum',
      'transpose',
    ]

    for value, unit in itertools.product(values, units):
      q_ar = value * unit
      for func in keep_dim_funcs:
        test_ar = getattr(q_ar, func)()
        if bu.get_unit(test_ar) != q_ar.unit:
          raise AssertionError(
            f"'{func.__name__}' failed on {q_ar!r} -- unit was "
            f"{q_ar.unit}, is now {bu.get_unit(test_ar)}."
          )

    # Python builtins should work on one-dimensional arrays
    value = np.arange(5)
    builtins = [abs, max, min]
    for unit in units:
      q_ar = value * unit
      for func in builtins:
        test_ar = func(q_ar)
        if bu.get_unit(test_ar) != q_ar.unit:
          raise AssertionError(
            f"'{func.__name__}' failed on {q_ar!r} -- unit "
            f"was {q_ar.unit}, is now "
            f"{bu.get_unit(test_ar)}"
          )

  def test_unitsafe_functions(self):
    """
    Test the unitsafe functions wrapping their numpy counterparts.
    """
    # All functions with their numpy counterparts
    funcs = [
      (bu.math.sin, np.sin),
      (bu.math.sinh, np.sinh),
      (bu.math.arcsin, np.arcsin),
      (bu.math.arcsinh, np.arcsinh),
      (bu.math.cos, np.cos),
      (bu.math.cosh, np.cosh),
      (bu.math.arccos, np.arccos),
      (bu.math.arccosh, np.arccosh),
      (bu.math.tan, np.tan),
      (bu.math.tanh, np.tanh),
      (bu.math.arctan, np.arctan),
      (bu.math.arctanh, np.arctanh),
      (bu.math.log, np.log),
      (bu.math.exp, np.exp),
    ]

    unitless_values = [0.1 * mV / mV, np.array([0.1, 0.5]) * mV / mV, np.random.rand(3, 3) * mV / mV]
    numpy_values = [0.1, np.array([0.1, 0.5]), np.random.rand(3, 3)]
    unit_values = [0.1 * mV, np.array([0.1, 0.5]) * mV, np.random.rand(3, 3) * mV]

    for bu_fun, np_fun in funcs:
      # make sure these functions raise errors when run on values with dimensions
      for val in unit_values:
        with pytest.raises(AssertionError):
          bu_fun(val)

      for val in unitless_values:
        if hasattr(val, "mantissa"):
          assert bu.math.allclose(bu_fun(val.mantissa), np_fun(val.mantissa), equal_nan=True)
        else:
          assert bu.math.allclose(bu_fun(val), np_fun(val), equal_nan=True)

      for val in numpy_values:
        assert bu.math.allclose(bu_fun(val), np_fun(val), equal_nan=True)


class TestConstant(unittest.TestCase):

  def test_constants(self):
    import brainunit._unit_constants as constants

    # Check that the expected names exist and have the correct dimensions
    assert constants.avogadro_constant.dim == (1 / mole).dim
    assert constants.boltzmann_constant.dim == (joule / kelvin).dim
    assert constants.electric_constant.dim == (farad / meter).dim
    assert constants.electron_mass.dim == kilogram.dim
    assert constants.elementary_charge.dim == coulomb.dim
    assert constants.faraday_constant.dim == (coulomb / mole).dim
    assert constants.gas_constant.dim == (joule / mole / kelvin).dim
    assert constants.magnetic_constant.dim == (newton / amp2).dim
    assert constants.molar_mass_constant.dim == (kilogram / mole).dim
    assert constants.zero_celsius.dim == kelvin.dim

    # Check the consistency between a few constants
    assert bu.math.allclose(
      constants.gas_constant.mantissa,
      (constants.avogadro_constant * constants.boltzmann_constant).mantissa,
    )
    assert bu.math.allclose(
      constants.faraday_constant.mantissa,
      (constants.avogadro_constant * constants.elementary_charge).mantissa,
    )


class TestHelperFunctions(unittest.TestCase):

  def test_fail_for_dimension_mismatch(self):
    """
    Test the fail_for_dimension_mismatch function.
    """
    # examples that should not raise an error
    dim1, dim2 = fail_for_dimension_mismatch(3)
    assert dim1 is DIMENSIONLESS
    assert dim2 is DIMENSIONLESS
    dim1, dim2 = fail_for_dimension_mismatch(3 * volt / volt)
    assert dim1 is DIMENSIONLESS
    assert dim2 is DIMENSIONLESS
    dim1, dim2 = fail_for_dimension_mismatch(3 * volt / volt, 7)
    assert dim1 is DIMENSIONLESS
    assert dim2 is DIMENSIONLESS
    dim1, dim2 = fail_for_dimension_mismatch(3 * volt, 5 * volt)
    assert dim1 is volt.dim
    assert dim2 is volt.dim

    # examples that should raise an error
    with pytest.raises(DimensionMismatchError):
      fail_for_dimension_mismatch(6 * volt)
    with pytest.raises(DimensionMismatchError):
      fail_for_dimension_mismatch(6 * volt, 5 * second)

  def test_check_dims(self):
    """
    Test the check_units decorator
    """

    @bu.check_dims(v=volt.dim)
    def a_function(v, x):
      """
      v has to have units of volt, x can have any (or no) unit.
      """
      pass

    # Try correct units
    a_function(3 * mV, 5 * second)
    a_function(5 * volt, "something")
    a_function([1, 2, 3] * volt, None)
    # lists that can be converted should also work
    a_function([1 * volt, 2 * volt, 3 * volt], None)
    # Strings and None are also allowed to pass
    a_function("a string", None)
    a_function(None, None)

    # Try incorrect units
    with pytest.raises(DimensionMismatchError):
      a_function(5 * second, None)
    with pytest.raises(DimensionMismatchError):
      a_function(5, None)
    with pytest.raises(TypeError):
      a_function(object(), None)
    with pytest.raises(TypeError):
      a_function([1, 2 * volt, 3], None)

    @bu.check_dims(result=second.dim)
    def b_function(return_second):
      """
      Return a value in seconds if return_second is True, otherwise return
      a value in volt.
      """
      if return_second:
        return 5 * second
      else:
        return 3 * volt

    # Should work (returns second)
    b_function(True)
    # Should fail (returns volt)
    with pytest.raises(DimensionMismatchError):
      b_function(False)

    @bu.check_dims(a=bool, b=1, result=bool)
    def c_function(a, b):
      if a:
        return b > 0
      else:
        return b

    assert c_function(True, 1)
    assert not c_function(True, -1)
    with pytest.raises(TypeError):
      c_function(1, 1)
    with pytest.raises(TypeError):
      c_function(1 * mV, 1)
    with pytest.raises(TypeError):
      c_function(False, 1)

  def test_check_units(self):
    """
    Test the check_units decorator
    """

    @bu.check_units(v=volt)
    def a_function(v, x):
      """
      v has to have units of volt, x can have any (or no) unit.
      """
      pass

    # Try correct units

    with pytest.raises(bu.UnitMismatchError):
      a_function(3 * mV, 5 * second)
    a_function(3 * volt, 5 * second)
    a_function(5 * volt, "something")
    a_function([1, 2, 3] * volt, None)
    # lists that can be converted should also work
    a_function([1 * volt, 2 * volt, 3 * volt], None)
    # Strings and None are also allowed to pass
    a_function("a string", None)
    a_function(None, None)

    # Try incorrect units
    with pytest.raises(bu.UnitMismatchError):
      a_function(5 * second, None)
    with pytest.raises(bu.UnitMismatchError):
      a_function(5, None)
    with pytest.raises(TypeError):
      a_function(object(), None)
    with pytest.raises(TypeError):
      a_function([1, 2 * volt, 3], None)

    @check_units(result=second)
    def b_function(return_second):
      """
      Return a value in seconds if return_second is True, otherwise return
      a value in volt.
      """
      if return_second:
        return 5 * second
      else:
        return 3 * volt

    # Should work (returns second)
    b_function(True)
    # Should fail (returns volt)
    with pytest.raises(bu.UnitMismatchError):
      b_function(False)

    @check_units(a=bool, b=1, result=bool)
    def c_function(a, b):
      if a:
        return b > 0
      else:
        return b

    assert c_function(True, 1)
    assert not c_function(True, -1)
    with pytest.raises(TypeError):
      c_function(1, 1)
    with pytest.raises(TypeError):
      c_function(1 * mV, 1)
    with pytest.raises(TypeError):
      c_function(False, 1)


def test_str_repr():
  """
  Test that str representations do not raise any errors and that repr
  fullfills eval(repr(x)) == x.
  """

  units_which_should_exist = [
    bu.metre,
    bu.meter,
    bu.kilogram,
    bu.kilogramme,
    bu.second,
    bu.amp,
    bu.kelvin,
    bu.mole,
    bu.candle,
    bu.radian,
    bu.steradian,
    bu.hertz,
    bu.newton,
    bu.pascal,
    bu.joule,
    bu.watt,
    bu.coulomb,
    bu.volt,
    bu.farad,
    bu.ohm,
    bu.siemens,
    bu.weber,
    bu.tesla,
    bu.henry,
    bu.lumen,
    bu.lux,
    bu.becquerel,
    bu.gray,
    bu.sievert,
    bu.katal,
    bu.gram,
    bu.gramme,
    bu.molar,
    bu.liter,
    bu.litre,
  ]

  # scaled versions of all these units should exist (we just check farad as an example)
  some_scaled_units = [
    bu.Yfarad,
    bu.Zfarad,
    bu.Efarad,
    bu.Pfarad,
    bu.Tfarad,
    bu.Gfarad,
    bu.Mfarad,
    bu.kfarad,
    bu.hfarad,
    bu.dafarad,
    bu.dfarad,
    bu.cfarad,
    bu.mfarad,
    bu.ufarad,
    bu.nfarad,
    bu.pfarad,
    bu.ffarad,
    bu.afarad,
    bu.zfarad,
    bu.yfarad,
  ]

  # some powered units
  powered_units = [bu.cmetre2, bu.Yfarad3]

  # Combined units
  complex_units = [
    (bu.kgram * bu.metre2) / (bu.amp * bu.second3),
    5 * (bu.kgram * bu.metre2) / (bu.amp * bu.second3),
    bu.metre * bu.second ** -1,
    10 * bu.metre * bu.second ** -1,
    np.array([1, 2, 3]) * bu.kmetre / bu.second,
    np.ones(3) * bu.nS / bu.cm ** 2,
    # Made-up unit:
    Unit(
      dim=get_or_create_dimension(length=5, time=2),
      dispname="O",
    ),
    8000 * bu.umetre ** 3,
    [0.0001, 10000] * bu.umetre ** 3,
    1 / bu.metre,
    1 / (bu.coulomb * bu.metre ** 2),
    Unit() / second,
    3.0 * bu.mM,
    5 * bu.mole / bu.liter,
    7 * bu.liter / bu.meter3,
    1 / second ** 2,
    volt ** -2,
    (volt ** 2) ** -1,
    (1 / second) / meter,
    1 / (1 / second),
  ]

  unitless = [second / second, 5 * second / second, Unit()]
  #
  # for u in itertools.chain(
  #     units_which_should_exist,
  #     some_scaled_units,
  #     powered_units,
  #     complex_units,
  #     unitless,
  # ):
  #   assert len(str(u)) > 0
  #   print(u)
  #   v1 = bu.display_in_unit(u, python_code=False)
  #   if isinstance(u, Unit):
  #     if 'Unit(1.0)' in v1:
  #       continue
  #     v2 = eval(v1)
  #     assert v2 == u
  #     assert isinstance(u, Unit)
  #     assert bu.math.allclose(v2.value, u.value)

  # test the `DIMENSIONLESS` object
  assert str(DIMENSIONLESS) == "1"
  assert repr(DIMENSIONLESS) == "Dimension()"

  # test DimensionMismatchError (only that it works without raising an error
  for error in [
    DimensionMismatchError("A description"),
    DimensionMismatchError("A description", DIMENSIONLESS),
    DimensionMismatchError("A description", DIMENSIONLESS, second.dim),
  ]:
    assert len(str(error))
    assert len(repr(error))
