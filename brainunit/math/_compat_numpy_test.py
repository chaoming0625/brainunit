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


import unittest

import brainstate as bst
import jax.numpy as jnp
import pytest

import brainunit as bu
from brainunit import DimensionMismatchError
from brainunit._base import Quantity
from brainunit._unit_shortcuts import ms, mV
from brainunit._unit_common import second

bst.environ.set(precision=64)


def assert_quantity(q, values, unit):
  values = jnp.asarray(values)
  if isinstance(q, Quantity):
    assert q.unit == unit.unit, f"Unit mismatch: {q.unit} != {unit}"
    assert jnp.allclose(q.value, values), f"Values do not match: {q.value} != {values}"
  else:
    assert jnp.allclose(q, values), f"Values do not match: {q} != {values}"


class TestArrayCreation(unittest.TestCase):

  def test_full(self):
    result = bu.math.full(3, 4)
    self.assertEqual(result.shape, (3,))
    self.assertTrue(jnp.all(result == 4))

    q = bu.math.full(3, 4, unit=second)
    self.assertEqual(q.shape, (3,))
    assert_quantity(q, result, second)

  def test_eye(self):
    result = bu.math.eye(3)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.eye(3)))

  def test_identity(self):
    result = bu.math.identity(3)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.identity(3)))

  def test_tri(self):
    result = bu.math.tri(3)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.tri(3)))

  def test_empty(self):
    result = bu.math.empty((2, 2))
    self.assertEqual(result.shape, (2, 2))

  def test_ones(self):
    result = bu.math.ones((2, 2))
    self.assertEqual(result.shape, (2, 2))
    self.assertTrue(jnp.all(result == 1))

  def test_zeros(self):
    result = bu.math.zeros((2, 2))
    self.assertEqual(result.shape, (2, 2))
    self.assertTrue(jnp.all(result == 0))

  def test_array(self):
    result = bu.math.array([1, 2, 3])
    self.assertEqual(result.shape, (3,))
    self.assertTrue(jnp.all(result == jnp.array([1, 2, 3])))

  # with Quantity

  def test_full_like(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.full_like(array, 4)
    self.assertEqual(result.shape, array.shape)
    self.assertTrue(jnp.all(result == 4))

    q = [1, 2, 3] * bu.second
    result_q = bu.math.full_like(q, 4, unit=bu.second)
    assert_quantity(result_q, jnp.full_like(jnp.array([1, 2, 3]), 4), bu.second)

  def test_diag(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.diag(array)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.diag(array)))

    q = [1, 2, 3] * bu.second
    result_q = bu.math.diag(q, unit=bu.second)
    assert_quantity(result_q, jnp.diag(jnp.array([1, 2, 3])), bu.second)

  def test_tril(self):
    array = jnp.ones((3, 3))
    result = bu.math.tril(array)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.tril(array)))

    q = jnp.ones((3, 3)) * bu.second
    result_q = bu.math.tril(q, unit=bu.second)
    assert_quantity(result_q, jnp.tril(jnp.ones((3, 3))), bu.second)

  def test_triu(self):
    array = jnp.ones((3, 3))
    result = bu.math.triu(array)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.triu(array)))

    q = jnp.ones((3, 3)) * bu.second
    result_q = bu.math.triu(q, unit=bu.second)
    assert_quantity(result_q, jnp.triu(jnp.ones((3, 3))), bu.second)

  def test_empty_like(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.empty_like(array)
    self.assertEqual(result.shape, array.shape)

    q = [1, 2, 3] * bu.second
    result_q = bu.math.empty_like(q)
    assert_quantity(result_q, jnp.empty_like(jnp.array([1, 2, 3])), bu.second)

  def test_ones_like(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.ones_like(array)
    self.assertEqual(result.shape, array.shape)
    self.assertTrue(jnp.all(result == 1))

    q = [1, 2, 3] * bu.second
    result_q = bu.math.ones_like(q)
    assert_quantity(result_q, jnp.ones_like(jnp.array([1, 2, 3])), bu.second)

  def test_zeros_like(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.zeros_like(array)
    self.assertEqual(result.shape, array.shape)
    self.assertTrue(jnp.all(result == 0))

    q = [1, 2, 3] * bu.second
    result_q = bu.math.zeros_like(q)
    assert_quantity(result_q, jnp.zeros_like(jnp.array([1, 2, 3])), bu.second)

  def test_asarray(self):
    result = bu.math.asarray([1, 2, 3])
    self.assertEqual(result.shape, (3,))
    self.assertTrue(jnp.all(result == jnp.asarray([1, 2, 3])))

    result_q = bu.math.asarray([1 * bu.second, 2 * bu.second, 3 * bu.second])
    assert_quantity(result_q, jnp.asarray([1, 2, 3]), bu.second)

  def test_arange(self):
    result = bu.math.arange(5)
    self.assertEqual(result.shape, (5,))
    self.assertTrue(jnp.all(result == jnp.arange(5)))

    result_q = bu.math.arange(5 * bu.second, step=1 * bu.second)
    assert_quantity(result_q, jnp.arange(5, step=1), bu.second)

    result_q = bu.math.arange(3 * bu.second, 9 * bu.second, 1 * bu.second)
    assert_quantity(result_q, jnp.arange(3, 9, 1), bu.second)

  def test_linspace(self):
    result = bu.math.linspace(0, 10, 5)
    self.assertEqual(result.shape, (5,))
    self.assertTrue(jnp.all(result == jnp.linspace(0, 10, 5)))

    q = bu.math.linspace(0 * bu.second, 10 * bu.second, 5)
    assert_quantity(q, jnp.linspace(0, 10, 5), bu.second)

  def test_logspace(self):
    result = bu.math.logspace(0, 2, 5)
    self.assertEqual(result.shape, (5,))
    self.assertTrue(jnp.all(result == jnp.logspace(0, 2, 5)))

    q = bu.math.logspace(0 * bu.second, 2 * bu.second, 5)
    assert_quantity(q, jnp.logspace(0, 2, 5), bu.second)

  def test_fill_diagonal(self):
    array = jnp.zeros((3, 3))
    result = bu.math.fill_diagonal(array, 5, inplace=False)
    self.assertTrue(jnp.all(result == jnp.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])))

    q = jnp.zeros((3, 3)) * bu.second
    result_q = bu.math.fill_diagonal(q, 5 * bu.second, inplace=False)
    assert_quantity(result_q, jnp.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]), bu.second)

  def test_array_split(self):
    array = jnp.arange(9)
    result = bu.math.array_split(array, 3)
    expected = jnp.array_split(array, 3)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.all(r == e))

    q = jnp.arange(9) * bu.second
    result_q = bu.math.array_split(q, 3)
    expected_q = jnp.array_split(jnp.arange(9), 3)
    for r, e in zip(result_q, expected_q):
      assert_quantity(r, e, bu.second)

  def test_meshgrid(self):
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5])
    result = bu.math.meshgrid(x, y)
    expected = jnp.meshgrid(x, y)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.all(r == e))

    x = jnp.array([1, 2, 3]) * bu.second
    y = jnp.array([4, 5]) * bu.second
    result_q = bu.math.meshgrid(x, y)
    expected_q = jnp.meshgrid(jnp.array([1, 2, 3]), jnp.array([4, 5]))
    for r, e in zip(result_q, expected_q):
      assert_quantity(r, e, bu.second)

  def test_vander(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.vander(array)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.vander(array)))

    q = jnp.array([1, 2, 3]) * bu.second
    result_q = bu.math.vander(q)
    assert_quantity(result_q, jnp.vander(jnp.array([1, 2, 3])), bu.second)


class TestAttributeFunctions(unittest.TestCase):

  def test_ndim(self):
    array = jnp.array([[1, 2], [3, 4]])
    self.assertEqual(bu.math.ndim(array), 2)

    q = [[1, 2], [3, 4]] * ms
    self.assertEqual(bu.math.ndim(q), 2)

  def test_isreal(self):
    array = jnp.array([1.0, 2.0])
    self.assertTrue(jnp.all(bu.math.isreal(array)))

    q = [[1, 2], [3, 4]] * ms
    self.assertTrue(jnp.all(bu.math.isreal(q)))

  def test_isscalar(self):
    self.assertTrue(bu.math.isscalar(1.0))
    self.assertTrue(bu.math.isscalar(Quantity(1.0)))

  def test_isfinite(self):
    array = jnp.array([1.0, jnp.inf])
    self.assertTrue(jnp.all(bu.math.isfinite(array) == jnp.isfinite(array)))

    q = [1.0, jnp.inf] * ms
    self.assertTrue(jnp.all(bu.math.isfinite(q) == jnp.isfinite(q.value)))

  def test_isinf(self):
    array = jnp.array([1.0, jnp.inf])
    self.assertTrue(jnp.all(bu.math.isinf(array) == jnp.isinf(array)))

    q = [1.0, jnp.inf] * ms
    self.assertTrue(jnp.all(bu.math.isinf(q) == jnp.isinf(q.value)))

  def test_isnan(self):
    array = jnp.array([1.0, jnp.nan])
    self.assertTrue(jnp.all(bu.math.isnan(array) == jnp.isnan(array)))

    q = [1.0, jnp.nan] * ms
    self.assertTrue(jnp.all(bu.math.isnan(q) == jnp.isnan(q.value)))

  def test_shape(self):
    array = jnp.array([[1, 2], [3, 4]])
    self.assertEqual(bu.math.shape(array), (2, 2))

    q = [[1, 2], [3, 4]] * ms
    self.assertEqual(bu.math.shape(q), (2, 2))

  def test_size(self):
    array = jnp.array([[1, 2], [3, 4]])
    self.assertEqual(bu.math.size(array), 4)
    self.assertEqual(bu.math.size(array, 1), 2)

    q = [[1, 2], [3, 4]] * ms
    self.assertEqual(bu.math.size(q), 4)
    self.assertEqual(bu.math.size(q, 1), 2)


class TestMathFuncsKeepUnitUnary(unittest.TestCase):

  def test_real(self):
    complex_array = jnp.array([1 + 2j, 3 + 4j])
    result = bu.math.real(complex_array)
    self.assertTrue(jnp.all(result == jnp.real(complex_array)))

    q = [1 + 2j, 3 + 4j] * bu.second
    result_q = bu.math.real(q)
    self.assertTrue(jnp.all(result_q == jnp.real(q.value) * bu.second))

  def test_imag(self):
    complex_array = jnp.array([1 + 2j, 3 + 4j])
    result = bu.math.imag(complex_array)
    self.assertTrue(jnp.all(result == jnp.imag(complex_array)))

    q = [1 + 2j, 3 + 4j] * bu.second
    result_q = bu.math.imag(q)
    self.assertTrue(jnp.all(result_q == jnp.imag(q.value) * bu.second))

  def test_conj(self):
    complex_array = jnp.array([1 + 2j, 3 + 4j])
    result = bu.math.conj(complex_array)
    self.assertTrue(jnp.all(result == jnp.conj(complex_array)))

    q = [1 + 2j, 3 + 4j] * bu.second
    result_q = bu.math.conj(q)
    self.assertTrue(jnp.all(result_q == jnp.conj(q.value) * bu.second))

  def test_conjugate(self):
    complex_array = jnp.array([1 + 2j, 3 + 4j])
    result = bu.math.conjugate(complex_array)
    self.assertTrue(jnp.all(result == jnp.conjugate(complex_array)))

    q = [1 + 2j, 3 + 4j] * bu.second
    result_q = bu.math.conjugate(q)
    self.assertTrue(jnp.all(result_q == jnp.conjugate(q.value) * bu.second))

  def test_negative(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.negative(array)
    self.assertTrue(jnp.all(result == jnp.negative(array)))

    q = [1, 2, 3] * ms
    result_q = bu.math.negative(q)
    expected_q = jnp.negative(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_positive(self):
    array = jnp.array([-1, -2, -3])
    result = bu.math.positive(array)
    self.assertTrue(jnp.all(result == jnp.positive(array)))

    q = [-1, -2, -3] * ms
    result_q = bu.math.positive(q)
    expected_q = jnp.positive(jnp.array([-1, -2, -3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_abs(self):
    array = jnp.array([-1, -2, 3])
    result = bu.math.abs(array)
    self.assertTrue(jnp.all(result == jnp.abs(array)))

    q = [-1, -2, 3] * ms
    result_q = bu.math.abs(q)
    expected_q = jnp.abs(jnp.array([-1, -2, -3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_round(self):
    array = jnp.array([1.123, 2.567, 3.891])
    result = bu.math.round(array)
    self.assertTrue(jnp.all(result == jnp.round(array)))

    q = [1.123, 2.567, 3.891] * bu.second
    result_q = bu.math.round(q)
    expected_q = jnp.round(jnp.array([1.123, 2.567, 3.891])) * bu.second
    assert_quantity(result_q, expected_q.value, bu.second)

  def test_rint(self):
    array = jnp.array([1.5, 2.3, 3.8])
    result = bu.math.rint(array)
    self.assertTrue(jnp.all(result == jnp.rint(array)))

    q = [1.5, 2.3, 3.8] * bu.second
    result_q = bu.math.rint(q)
    expected_q = jnp.rint(jnp.array([1.5, 2.3, 3.8])) * bu.second
    assert_quantity(result_q, expected_q.value, bu.second)

  def test_floor(self):
    array = jnp.array([1.5, 2.3, 3.8])
    result = bu.math.floor(array)
    self.assertTrue(jnp.all(result == jnp.floor(array)))

    q = [1.5, 2.3, 3.8] * bu.second
    result_q = bu.math.floor(q)
    expected_q = jnp.floor(jnp.array([1.5, 2.3, 3.8])) * bu.second
    assert_quantity(result_q, expected_q.value, bu.second)

  def test_ceil(self):
    array = jnp.array([1.5, 2.3, 3.8])
    result = bu.math.ceil(array)
    self.assertTrue(jnp.all(result == jnp.ceil(array)))

    q = [1.5, 2.3, 3.8] * bu.second
    result_q = bu.math.ceil(q)
    expected_q = jnp.ceil(jnp.array([1.5, 2.3, 3.8])) * bu.second
    assert_quantity(result_q, expected_q.value, bu.second)

  def test_trunc(self):
    array = jnp.array([1.5, 2.3, 3.8])
    result = bu.math.trunc(array)
    self.assertTrue(jnp.all(result == jnp.trunc(array)))

    q = [1.5, 2.3, 3.8] * bu.second
    result_q = bu.math.trunc(q)
    expected_q = jnp.trunc(jnp.array([1.5, 2.3, 3.8])) * bu.second
    assert_quantity(result_q, expected_q.value, bu.second)

  def test_fix(self):
    array = jnp.array([1.5, 2.3, 3.8])
    result = bu.math.fix(array)
    self.assertTrue(jnp.all(result == jnp.fix(array)))

    q = [1.5, 2.3, 3.8] * bu.second
    result_q = bu.math.fix(q)
    expected_q = jnp.fix(jnp.array([1.5, 2.3, 3.8])) * bu.second
    assert_quantity(result_q, expected_q.value, bu.second)

  def test_sum(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.sum(array)
    self.assertTrue(result == jnp.sum(array))

    q = [1, 2, 3] * ms
    result_q = bu.math.sum(q)
    expected_q = jnp.sum(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nancumsum(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bu.math.nancumsum(array)
    self.assertTrue(jnp.all(result == jnp.nancumsum(array)))

    q = [1, jnp.nan, 3] * ms
    result_q = bu.math.nancumsum(q)
    expected_q = jnp.nancumsum(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nansum(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bu.math.nansum(array)
    self.assertTrue(result == jnp.nansum(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bu.math.nansum(q)
    expected_q = jnp.nansum(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_cumsum(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.cumsum(array)
    self.assertTrue(jnp.all(result == jnp.cumsum(array)))

    q = [1, 2, 3] * ms
    result_q = bu.math.cumsum(q)
    expected_q = jnp.cumsum(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_ediff1d(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.ediff1d(array)
    self.assertTrue(jnp.all(result == jnp.ediff1d(array)))

    q = [1, 2, 3] * ms
    result_q = bu.math.ediff1d(q)
    expected_q = jnp.ediff1d(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_absolute(self):
    array = jnp.array([-1, -2, 3])
    result = bu.math.absolute(array)
    self.assertTrue(jnp.all(result == jnp.absolute(array)))

    q = [-1, -2, 3] * ms
    result_q = bu.math.absolute(q)
    expected_q = jnp.absolute(jnp.array([-1, -2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_fabs(self):
    array = jnp.array([-1, -2, 3])
    result = bu.math.fabs(array)
    self.assertTrue(jnp.all(result == jnp.fabs(array)))

    q = [-1, -2, 3] * ms
    result_q = bu.math.fabs(q)
    expected_q = jnp.fabs(jnp.array([-1, -2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_median(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.median(array)
    self.assertTrue(result == jnp.median(array))

    q = [1, 2, 3] * ms
    result_q = bu.math.median(q)
    expected_q = jnp.median(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nanmin(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bu.math.nanmin(array)
    self.assertTrue(result == jnp.nanmin(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bu.math.nanmin(q)
    expected_q = jnp.nanmin(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nanmax(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bu.math.nanmax(array)
    self.assertTrue(result == jnp.nanmax(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bu.math.nanmax(q)
    expected_q = jnp.nanmax(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_ptp(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.ptp(array)
    self.assertTrue(result == jnp.ptp(array))

    q = [1, 2, 3] * ms
    result_q = bu.math.ptp(q)
    expected_q = jnp.ptp(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_average(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.average(array)
    self.assertTrue(result == jnp.average(array))

    q = [1, 2, 3] * ms
    result_q = bu.math.average(q)
    expected_q = jnp.average(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_mean(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.mean(array)
    self.assertTrue(result == jnp.mean(array))

    q = [1, 2, 3] * ms
    result_q = bu.math.mean(q)
    expected_q = jnp.mean(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_std(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.std(array)
    self.assertTrue(result == jnp.std(array))

    q = [1, 2, 3] * ms
    result_q = bu.math.std(q)
    expected_q = jnp.std(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nanmedian(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bu.math.nanmedian(array)
    self.assertTrue(result == jnp.nanmedian(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bu.math.nanmedian(q)
    expected_q = jnp.nanmedian(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nanmean(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bu.math.nanmean(array)
    self.assertTrue(result == jnp.nanmean(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bu.math.nanmean(q)
    expected_q = jnp.nanmean(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nanstd(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bu.math.nanstd(array)
    self.assertTrue(result == jnp.nanstd(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bu.math.nanstd(q)
    expected_q = jnp.nanstd(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_diff(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.diff(array)
    self.assertTrue(jnp.all(result == jnp.diff(array)))

    q = [1, 2, 3] * ms
    result_q = bu.math.diff(q)
    expected_q = jnp.diff(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_modf(self):
    result = bu.math.modf(jnp.array([5.5, 7.3]))
    expected = jnp.modf(jnp.array([5.5, 7.3]))
    self.assertTrue(jnp.all(result[0] == expected[0]) and jnp.all(result[1] == expected[1]))


class TestMathFuncsKeepUnitBinary(unittest.TestCase):

  def test_fmod(self):
    result = bu.math.fmod(jnp.array([5, 7]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.fmod(jnp.array([5, 7]), jnp.array([2, 3]))))

    q1 = [5, 7] * ms
    q2 = [2, 3] * ms
    result_q = bu.math.fmod(q1, q2)
    expected_q = jnp.fmod(jnp.array([5, 7]), jnp.array([2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_mod(self):
    result = bu.math.mod(jnp.array([5, 7]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.mod(jnp.array([5, 7]), jnp.array([2, 3]))))

    q1 = [5, 7] * ms
    q2 = [2, 3] * ms
    result_q = bu.math.mod(q1, q2)
    expected_q = jnp.mod(jnp.array([5, 7]), jnp.array([2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_copysign(self):
    result = bu.math.copysign(jnp.array([-1, 2]), jnp.array([1, -3]))
    self.assertTrue(jnp.all(result == jnp.copysign(jnp.array([-1, 2]), jnp.array([1, -3]))))

    q1 = [-1, 2] * ms
    q2 = [1, -3] * ms
    result_q = bu.math.copysign(q1, q2)
    expected_q = jnp.copysign(jnp.array([-1, 2]), jnp.array([1, -3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_heaviside(self):
    result = bu.math.heaviside(jnp.array([-1, 2]), jnp.array([0.5, 0.5]))
    self.assertTrue(jnp.all(result == jnp.heaviside(jnp.array([-1, 2]), jnp.array([0.5, 0.5]))))

  def test_maximum(self):
    result = bu.math.maximum(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))
    self.assertTrue(jnp.all(result == jnp.maximum(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))))

    q1 = [1, 3, 2] * ms
    q2 = [2, 1, 3] * ms
    result_q = bu.math.maximum(q1, q2)
    expected_q = jnp.maximum(jnp.array([1, 3, 2]), jnp.array([2, 1, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_minimum(self):
    result = bu.math.minimum(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))
    self.assertTrue(jnp.all(result == jnp.minimum(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))))

    q1 = [1, 3, 2] * ms
    q2 = [2, 1, 3] * ms
    result_q = bu.math.minimum(q1, q2)
    expected_q = jnp.minimum(jnp.array([1, 3, 2]), jnp.array([2, 1, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_fmax(self):
    result = bu.math.fmax(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))
    self.assertTrue(jnp.all(result == jnp.fmax(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))))

    q1 = [1, 3, 2] * ms
    q2 = [2, 1, 3] * ms
    result_q = bu.math.fmax(q1, q2)
    expected_q = jnp.fmax(jnp.array([1, 3, 2]), jnp.array([2, 1, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_fmin(self):
    result = bu.math.fmin(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))
    self.assertTrue(jnp.all(result == jnp.fmin(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))))

    q1 = [1, 3, 2] * ms
    q2 = [2, 1, 3] * ms
    result_q = bu.math.fmin(q1, q2)
    expected_q = jnp.fmin(jnp.array([1, 3, 2]), jnp.array([2, 1, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_lcm(self):
    result = bu.math.lcm(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))
    self.assertTrue(jnp.all(result == jnp.lcm(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))))

    q1 = [4, 5, 6] * bu.second
    q2 = [2, 3, 4] * bu.second
    q1 = q1.astype(jnp.int64)
    q2 = q2.astype(jnp.int64)
    result_q = bu.math.lcm(q1, q2)
    expected_q = jnp.lcm(jnp.array([4, 5, 6]), jnp.array([2, 3, 4])) * bu.second
    assert_quantity(result_q, expected_q.value, bu.second)

  def test_gcd(self):
    result = bu.math.gcd(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))
    self.assertTrue(jnp.all(result == jnp.gcd(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))))

    q1 = [4, 5, 6] * bu.second
    q2 = [2, 3, 4] * bu.second
    q1 = q1.astype(jnp.int64)
    q2 = q2.astype(jnp.int64)
    result_q = bu.math.gcd(q1, q2)
    expected_q = jnp.gcd(jnp.array([4, 5, 6]), jnp.array([2, 3, 4])) * bu.second
    assert_quantity(result_q, expected_q.value, bu.second)


class TestMathFuncsKeepUnitUnary2(unittest.TestCase):

  def test_interp(self):
    x = jnp.array([1, 2, 3])
    xp = jnp.array([0, 1, 2, 3, 4])
    fp = jnp.array([0, 1, 2, 3, 4])
    result = bu.math.interp(x, xp, fp)
    self.assertTrue(jnp.all(result == jnp.interp(x, xp, fp)))

    x = [1, 2, 3] * bu.second
    xp = [0, 1, 2, 3, 4] * bu.second
    fp = [0, 1, 2, 3, 4] * bu.second
    result_q = bu.math.interp(x, xp, fp)
    expected_q = jnp.interp(jnp.array([1, 2, 3]), jnp.array([0, 1, 2, 3, 4]), jnp.array([0, 1, 2, 3, 4])) * bu.second
    assert_quantity(result_q, expected_q.value, bu.second)

  def test_clip(self):
    array = jnp.array([1, 2, 3, 4, 5])
    result = bu.math.clip(array, 2, 4)
    self.assertTrue(jnp.all(result == jnp.clip(array, 2, 4)))

    q = [1, 2, 3, 4, 5] * ms
    result_q = bu.math.clip(q, 2 * ms, 4 * ms)
    expected_q = jnp.clip(jnp.array([1, 2, 3, 4, 5]), 2, 4) * ms
    assert_quantity(result_q, expected_q.value, ms)


class TestMathFuncsMatchUnitBinary(unittest.TestCase):

  def test_add(self):
    result = bu.math.add(jnp.array([1, 2]), jnp.array([3, 4]))
    self.assertTrue(jnp.all(result == jnp.add(jnp.array([1, 2]), jnp.array([3, 4]))))

    q1 = [1, 2] * ms
    q2 = [3, 4] * ms
    result_q = bu.math.add(q1, q2)
    expected_q = jnp.add(jnp.array([1, 2]), jnp.array([3, 4])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_subtract(self):
    result = bu.math.subtract(jnp.array([5, 6]), jnp.array([3, 2]))
    self.assertTrue(jnp.all(result == jnp.subtract(jnp.array([5, 6]), jnp.array([3, 2]))))

    q1 = [5, 6] * ms
    q2 = [3, 2] * ms
    result_q = bu.math.subtract(q1, q2)
    expected_q = jnp.subtract(jnp.array([5, 6]), jnp.array([3, 2])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nextafter(self):
    result = bu.math.nextafter(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))
    self.assertTrue(jnp.all(result == jnp.nextafter(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))


class TestMathFuncsChangeUnitUnary(unittest.TestCase):

  def test_reciprocal(self):
    array = jnp.array([1.0, 2.0, 0.5])
    result = bu.math.reciprocal(array)
    self.assertTrue(jnp.all(result == jnp.reciprocal(array)))

    q = [1.0, 2.0, 0.5] * bu.second
    result_q = bu.math.reciprocal(q)
    expected_q = jnp.reciprocal(jnp.array([1.0, 2.0, 0.5])) * (1 / bu.second)
    assert_quantity(result_q, expected_q.value, 1 / bu.second)

  def test_prod(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.prod(array)
    self.assertTrue(result == jnp.prod(array))

    q = [1, 2, 3] * ms
    result_q = bu.math.prod(q)
    expected_q = jnp.prod(jnp.array([1, 2, 3])) * (ms ** 3)
    assert_quantity(result_q, expected_q.value, ms ** 3)

  def test_nanprod(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bu.math.nanprod(array)
    self.assertTrue(result == jnp.nanprod(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bu.math.nanprod(q)
    expected_q = jnp.nanprod(jnp.array([1, jnp.nan, 3])) * (ms ** 2)
    assert_quantity(result_q, expected_q.value, ms ** 2)

  def test_cumprod(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.cumprod(array)
    self.assertTrue(jnp.all(result == jnp.cumprod(array)))

    q = [1, 2, 3] * bu.second
    result_q = bu.math.cumprod(q)
    expected_q = jnp.cumprod(jnp.array([1, 2, 3])) * (bu.second ** 3)
    assert_quantity(result_q, expected_q.value, bu.second ** 3)

  def test_nancumprod(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bu.math.nancumprod(array)
    self.assertTrue(jnp.all(result == jnp.nancumprod(array)))

    q = [1, jnp.nan, 3] * bu.second
    result_q = bu.math.nancumprod(q)
    expected_q = jnp.nancumprod(jnp.array([1, jnp.nan, 3])) * (bu.second ** 2)
    assert_quantity(result_q, expected_q.value, bu.second ** 2)

  def test_var(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.var(array)
    self.assertTrue(result == jnp.var(array))

    q = [1, 2, 3] * ms
    result_q = bu.math.var(q)
    expected_q = jnp.var(jnp.array([1, 2, 3])) * (ms ** 2)
    assert_quantity(result_q, expected_q.value, ms ** 2)

  def test_nanvar(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bu.math.nanvar(array)
    self.assertTrue(result == jnp.nanvar(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bu.math.nanvar(q)
    expected_q = jnp.nanvar(jnp.array([1, jnp.nan, 3])) * (ms ** 2)
    assert_quantity(result_q, expected_q.value, ms ** 2)

  def test_frexp(self):
    result = bu.math.frexp(jnp.array([1.0, 2.0]))
    expected = jnp.frexp(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result[0] == expected[0]) and jnp.all(result[1] == expected[1]))

  def test_sqrt(self):
    result = bu.math.sqrt(jnp.array([1.0, 4.0]))
    self.assertTrue(jnp.all(result == jnp.sqrt(jnp.array([1.0, 4.0]))))

    q = [1.0, 4.0] * (ms ** 2)
    result_q = bu.math.sqrt(q)
    expected_q = jnp.sqrt(jnp.array([1.0, 4.0])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_cbrt(self):
    result = bu.math.cbrt(jnp.array([1.0, 8.0]))
    self.assertTrue(jnp.all(result == jnp.cbrt(jnp.array([1.0, 8.0]))))

    q = [1.0, 8.0] * (ms ** 3)
    result_q = bu.math.cbrt(q)
    expected_q = jnp.cbrt(jnp.array([1.0, 8.0])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_square(self):
    result = bu.math.square(jnp.array([2.0, 3.0]))
    self.assertTrue(jnp.all(result == jnp.square(jnp.array([2.0, 3.0]))))

    q = [2.0, 3.0] * ms
    result_q = bu.math.square(q)
    expected_q = jnp.square(jnp.array([2.0, 3.0])) * (ms ** 2)
    assert_quantity(result_q, expected_q.value, ms ** 2)


class TestMathFuncsChangeUnitBinary(unittest.TestCase):

  def test_multiply(self):
    result = bu.math.multiply(jnp.array([1, 2]), jnp.array([3, 4]))
    self.assertTrue(jnp.all(result == jnp.multiply(jnp.array([1, 2]), jnp.array([3, 4]))))

    q1 = [1, 2] * ms
    q2 = [3, 4] * mV
    result_q = bu.math.multiply(q1, q2)
    expected_q = jnp.multiply(jnp.array([1, 2]), jnp.array([3, 4])) * (ms * mV)
    assert_quantity(result_q, expected_q.value, ms * mV)

  def test_divide(self):
    result = bu.math.divide(jnp.array([5, 6]), jnp.array([3, 2]))
    self.assertTrue(jnp.all(result == jnp.divide(jnp.array([5, 6]), jnp.array([3, 2]))))

    q1 = [5, 6] * ms
    q2 = [3, 2] * mV
    result_q = bu.math.divide(q1, q2)
    expected_q = jnp.divide(jnp.array([5, 6]), jnp.array([3, 2])) * (ms / mV)
    assert_quantity(result_q, expected_q.value, ms / mV)

  def test_power(self):
    result = bu.math.power(jnp.array([1, 2]), jnp.array([3, 2]))
    self.assertTrue(jnp.all(result == jnp.power(jnp.array([1, 2]), jnp.array([3, 2]))))

    q1 = [1, 2] * ms
    result_q = bu.math.power(q1, 2)
    expected_q = jnp.power(jnp.array([1, 2]), 2) * (ms ** 2)
    assert_quantity(result_q, expected_q.value, ms ** 2)

  def test_cross(self):
    result = bu.math.cross(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
    self.assertTrue(jnp.all(result == jnp.cross(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))))

  def test_ldexp(self):
    result = bu.math.ldexp(jnp.array([1.0, 2.0]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.ldexp(jnp.array([1.0, 2.0]), jnp.array([2, 3]))))

  def test_true_divide(self):
    result = bu.math.true_divide(jnp.array([5, 6]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.true_divide(jnp.array([5, 6]), jnp.array([2, 3]))))

    q1 = [5, 6] * ms
    q2 = [2, 3] * mV
    result_q = bu.math.true_divide(q1, q2)
    expected_q = jnp.true_divide(jnp.array([5, 6]), jnp.array([2, 3])) * (ms / mV)
    assert_quantity(result_q, expected_q.value, ms / mV)

  def test_floor_divide(self):
    result = bu.math.floor_divide(jnp.array([5, 6]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.floor_divide(jnp.array([5, 6]), jnp.array([2, 3]))))

    q1 = [5, 6] * ms
    q2 = [2, 3] * mV
    result_q = bu.math.floor_divide(q1, q2)
    expected_q = jnp.floor_divide(jnp.array([5, 6]), jnp.array([2, 3])) * (ms / mV)
    assert_quantity(result_q, expected_q.value, ms / mV)

  def test_float_power(self):
    result = bu.math.float_power(jnp.array([2, 3]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.float_power(jnp.array([2, 3]), jnp.array([2, 3]))))

    q1 = [2, 3] * ms
    result_q = bu.math.float_power(q1, 2)
    expected_q = jnp.float_power(jnp.array([2, 3]), 2) * (ms ** 2)
    assert_quantity(result_q, expected_q.value, ms ** 2)

  def test_divmod(self):
    result = bu.math.divmod(jnp.array([5, 6]), jnp.array([2, 3]))
    expected = jnp.divmod(jnp.array([5, 6]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result[0] == expected[0]) and jnp.all(result[1] == expected[1]))

  def test_remainder(self):
    result = bu.math.remainder(jnp.array([5, 7]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.remainder(jnp.array([5, 7]), jnp.array([2, 3]))))

    q1 = [5, 7] * (bu.second ** 2)
    q2 = [2, 3] * bu.second
    result_q = bu.math.remainder(q1, q2)
    expected_q = jnp.remainder(jnp.array([5, 7]), jnp.array([2, 3])) * bu.second
    assert_quantity(result_q, expected_q.value, bu.second)

  def test_convolve(self):
    result = bu.math.convolve(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
    self.assertTrue(jnp.all(result == jnp.convolve(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))))


class TestMathFuncsOnlyAcceptUnitlessUnary(unittest.TestCase):

  def test_exp(self):
    result = bu.math.exp(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.exp(jnp.array([1.0, 2.0]))))

    result = bu.math.exp(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.exp(jnp.array([1.0, 2.0]))))

  def test_exp2(self):
    result = bu.math.exp2(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.exp2(jnp.array([1.0, 2.0]))))

    result = bu.math.exp2(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.exp2(jnp.array([1.0, 2.0]))))

  def test_expm1(self):
    result = bu.math.expm1(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.expm1(jnp.array([1.0, 2.0]))))

    result = bu.math.expm1(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.expm1(jnp.array([1.0, 2.0]))))

  def test_log(self):
    result = bu.math.log(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.log(jnp.array([1.0, 2.0]))))

    result = bu.math.log(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.log(jnp.array([1.0, 2.0]))))

  def test_log10(self):
    result = bu.math.log10(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.log10(jnp.array([1.0, 2.0]))))

    result = bu.math.log10(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.log10(jnp.array([1.0, 2.0]))))

  def test_log1p(self):
    result = bu.math.log1p(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.log1p(jnp.array([1.0, 2.0]))))

    result = bu.math.log1p(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.log1p(jnp.array([1.0, 2.0]))))

  def test_log2(self):
    result = bu.math.log2(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.log2(jnp.array([1.0, 2.0]))))

    result = bu.math.log2(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.log2(jnp.array([1.0, 2.0]))))

  def test_arccos(self):
    result = bu.math.arccos(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.arccos(jnp.array([0.5, 1.0]))))

    result = bu.math.arccos(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.arccos(jnp.array([0.5, 1.0]))))

  def test_arccosh(self):
    result = bu.math.arccosh(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.arccosh(jnp.array([1.0, 2.0]))))

    result = bu.math.arccosh(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.arccosh(jnp.array([1.0, 2.0]))))

  def test_arcsin(self):
    result = bu.math.arcsin(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.arcsin(jnp.array([0.5, 1.0]))))

    result = bu.math.arcsin(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.arcsin(jnp.array([0.5, 1.0]))))

  def test_arcsinh(self):
    result = bu.math.arcsinh(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.arcsinh(jnp.array([0.5, 1.0]))))

    result = bu.math.arcsinh(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.arcsinh(jnp.array([0.5, 1.0]))))

  def test_arctan(self):
    result = bu.math.arctan(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.arctan(jnp.array([0.5, 1.0]))))

    result = bu.math.arctan(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.arctan(jnp.array([0.5, 1.0]))))

  def test_arctanh(self):
    result = bu.math.arctanh(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.arctanh(jnp.array([0.5, 1.0]))))

    result = bu.math.arctanh(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.arctanh(jnp.array([0.5, 1.0]))))

  def test_cos(self):
    result = bu.math.cos(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.cos(jnp.array([0.5, 1.0]))))

    result = bu.math.cos(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.cos(jnp.array([0.5, 1.0]))))

  def test_cosh(self):
    result = bu.math.cosh(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.cosh(jnp.array([0.5, 1.0]))))

    result = bu.math.cosh(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.cosh(jnp.array([0.5, 1.0]))))

  def test_sin(self):
    result = bu.math.sin(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.sin(jnp.array([0.5, 1.0]))))

    result = bu.math.sin(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.sin(jnp.array([0.5, 1.0]))))

  def test_sinc(self):
    result = bu.math.sinc(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.sinc(jnp.array([0.5, 1.0]))))

    result = bu.math.sinc(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.sinc(jnp.array([0.5, 1.0]))))

  def test_sinh(self):
    result = bu.math.sinh(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.sinh(jnp.array([0.5, 1.0]))))

    result = bu.math.sinh(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.sinh(jnp.array([0.5, 1.0]))))

  def test_tan(self):
    result = bu.math.tan(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.tan(jnp.array([0.5, 1.0]))))

    result = bu.math.tan(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.tan(jnp.array([0.5, 1.0]))))

  def test_tanh(self):
    result = bu.math.tanh(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.tanh(jnp.array([0.5, 1.0]))))

    result = bu.math.tanh(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.tanh(jnp.array([0.5, 1.0]))))

  def test_deg2rad(self):
    result = bu.math.deg2rad(jnp.array([90.0, 180.0]))
    self.assertTrue(jnp.all(result == jnp.deg2rad(jnp.array([90.0, 180.0]))))

    result = bu.math.deg2rad(Quantity(jnp.array([90.0, 180.0])))
    self.assertTrue(jnp.all(result == jnp.deg2rad(jnp.array([90.0, 180.0]))))

  def test_rad2deg(self):
    result = bu.math.rad2deg(jnp.array([jnp.pi / 2, jnp.pi]))
    self.assertTrue(jnp.all(result == jnp.rad2deg(jnp.array([jnp.pi / 2, jnp.pi]))))

    result = bu.math.rad2deg(Quantity(jnp.array([jnp.pi / 2, jnp.pi])))
    self.assertTrue(jnp.all(result == jnp.rad2deg(jnp.array([jnp.pi / 2, jnp.pi]))))

  def test_degrees(self):
    result = bu.math.degrees(jnp.array([jnp.pi / 2, jnp.pi]))
    self.assertTrue(jnp.all(result == jnp.degrees(jnp.array([jnp.pi / 2, jnp.pi]))))

    result = bu.math.degrees(Quantity(jnp.array([jnp.pi / 2, jnp.pi])))
    self.assertTrue(jnp.all(result == jnp.degrees(jnp.array([jnp.pi / 2, jnp.pi]))))

  def test_radians(self):
    result = bu.math.radians(jnp.array([90.0, 180.0]))
    self.assertTrue(jnp.all(result == jnp.radians(jnp.array([90.0, 180.0]))))

    result = bu.math.radians(Quantity(jnp.array([90.0, 180.0])))
    self.assertTrue(jnp.all(result == jnp.radians(jnp.array([90.0, 180.0]))))

  def test_angle(self):
    result = bu.math.angle(jnp.array([1.0 + 1.0j, 1.0 - 1.0j]))
    self.assertTrue(jnp.all(result == jnp.angle(jnp.array([1.0 + 1.0j, 1.0 - 1.0j]))))

    result = bu.math.angle(Quantity(jnp.array([1.0 + 1.0j, 1.0 - 1.0j])))
    self.assertTrue(jnp.all(result == jnp.angle(jnp.array([1.0 + 1.0j, 1.0 - 1.0j]))))

  def test_percentile(self):
    array = jnp.array([1, 2, 3, 4])
    result = bu.math.percentile(array, 50)
    self.assertTrue(result == jnp.percentile(array, 50))

  def test_nanpercentile(self):
    array = jnp.array([1, jnp.nan, 3, 4])
    result = bu.math.nanpercentile(array, 50)
    self.assertTrue(result == jnp.nanpercentile(array, 50))

  def test_quantile(self):
    array = jnp.array([1, 2, 3, 4])
    result = bu.math.quantile(array, 0.5)
    self.assertTrue(result == jnp.quantile(array, 0.5))

  def test_nanquantile(self):
    array = jnp.array([1, jnp.nan, 3, 4])
    result = bu.math.nanquantile(array, 0.5)
    self.assertTrue(result == jnp.nanquantile(array, 0.5))


class TestMathFuncsOnlyAcceptUnitlessBinary(unittest.TestCase):

  def test_hypot(self):
    result = bu.math.hypot(jnp.array([3.0, 4.0]), jnp.array([4.0, 3.0]))
    self.assertTrue(jnp.all(result == jnp.hypot(jnp.array([3.0, 4.0]), jnp.array([4.0, 3.0]))))

    result = bu.math.hypot(Quantity(jnp.array([3.0, 4.0])), Quantity(jnp.array([4.0, 3.0])))
    self.assertTrue(jnp.all(result == jnp.hypot(jnp.array([3.0, 4.0]), jnp.array([4.0, 3.0]))))

  def test_arctan2(self):
    result = bu.math.arctan2(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))
    self.assertTrue(jnp.all(result == jnp.arctan2(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))

    result = bu.math.arctan2(Quantity(jnp.array([1.0, 2.0])), Quantity(jnp.array([2.0, 3.0])))
    self.assertTrue(jnp.all(result == jnp.arctan2(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))

  def test_logaddexp(self):
    result = bu.math.logaddexp(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))
    self.assertTrue(jnp.all(result == jnp.logaddexp(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))

    result = bu.math.logaddexp(Quantity(jnp.array([1.0, 2.0])), Quantity(jnp.array([2.0, 3.0])))
    self.assertTrue(jnp.all(result == jnp.logaddexp(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))

  def test_logaddexp2(self):
    result = bu.math.logaddexp2(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))
    self.assertTrue(jnp.all(result == jnp.logaddexp2(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))

    result = bu.math.logaddexp2(Quantity(jnp.array([1.0, 2.0])), Quantity(jnp.array([2.0, 3.0])))
    self.assertTrue(jnp.all(result == jnp.logaddexp2(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))


class TestMathFuncsRemoveUnitUnary(unittest.TestCase):

  def test_signbit(self):
    array = jnp.array([-1.0, 2.0])
    result = bu.math.signbit(array)
    self.assertTrue(jnp.all(result == jnp.signbit(array)))

    q = [-1.0, 2.0] * bu.second
    result_q = bu.math.signbit(q)
    expected_q = jnp.signbit(jnp.array([-1.0, 2.0]))
    assert_quantity(result_q, expected_q, None)

  def test_sign(self):
    array = jnp.array([-1.0, 2.0])
    result = bu.math.sign(array)
    self.assertTrue(jnp.all(result == jnp.sign(array)))

    q = [-1.0, 2.0] * bu.second
    result_q = bu.math.sign(q)
    expected_q = jnp.sign(jnp.array([-1.0, 2.0]))
    assert_quantity(result_q, expected_q, None)

  def test_histogram(self):
    array = jnp.array([1, 2, 1])
    result, _ = bu.math.histogram(array)
    expected, _ = jnp.histogram(array)
    self.assertTrue(jnp.all(result == expected))

    q = [1, 2, 1] * bu.second
    result_q, _ = bu.math.histogram(q)
    expected_q, _ = jnp.histogram(jnp.array([1, 2, 1]))
    assert_quantity(result_q, expected_q, None)

  def test_bincount(self):
    array = jnp.array([1, 1, 2, 2, 2, 3])
    result = bu.math.bincount(array)
    self.assertTrue(jnp.all(result == jnp.bincount(array)))

    q = [1, 1, 2, 2, 2, 3] * bu.second
    q = q.astype(jnp.int64)
    result_q = bu.math.bincount(q)
    expected_q = jnp.bincount(jnp.array([1, 1, 2, 2, 2, 3]))
    assert_quantity(result_q, expected_q, None)


class TestMathFuncsRemoveUnitBinary(unittest.TestCase):

  def test_corrcoef(self):
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    result = bu.math.corrcoef(x, y)
    self.assertTrue(jnp.all(result == jnp.corrcoef(x, y)))

    x = [1, 2, 3] * bu.second
    y = [4, 5, 6] * bu.second
    result = bu.math.corrcoef(x, y)
    expected = jnp.corrcoef(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
    assert_quantity(result, expected, None)

  def test_correlate(self):
    x = jnp.array([1, 2, 3])
    y = jnp.array([0, 1, 0.5])
    result = bu.math.correlate(x, y)
    self.assertTrue(jnp.all(result == jnp.correlate(x, y)))

    x = [1, 2, 3] * bu.second
    y = [0, 1, 0.5] * bu.second
    result = bu.math.correlate(x, y)
    expected = jnp.correlate(jnp.array([1, 2, 3]), jnp.array([0, 1, 0.5]))
    assert_quantity(result, expected, None)

  def test_cov(self):
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    result = bu.math.cov(x, y)
    self.assertTrue(jnp.all(result == jnp.cov(x, y)))

    x = [1, 2, 3] * bu.second
    y = [4, 5, 6] * bu.second
    result = bu.math.cov(x, y)
    expected = jnp.cov(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
    assert_quantity(result, expected, None)

  def test_digitize(self):
    array = jnp.array([0.2, 6.4, 3.0, 1.6])
    bins = jnp.array([0.0, 1.0, 2.5, 4.0, 10.0])
    result = bu.math.digitize(array, bins)
    self.assertTrue(jnp.all(result == jnp.digitize(array, bins)))

    array = [0.2, 6.4, 3.0, 1.6] * bu.second
    bins = [0.0, 1.0, 2.5, 4.0, 10.0] * bu.second
    result = bu.math.digitize(array, bins)
    expected = jnp.digitize(jnp.array([0.2, 6.4, 3.0, 1.6]), jnp.array([0.0, 1.0, 2.5, 4.0, 10.0]))
    assert_quantity(result, expected, None)


class TestArrayManipulation(unittest.TestCase):

  def test_reshape(self):
    array = jnp.array([1, 2, 3, 4])
    result = bu.math.reshape(array, (2, 2))
    self.assertTrue(jnp.all(result == jnp.reshape(array, (2, 2))))

    q = [1, 2, 3, 4] * bu.second
    result_q = bu.math.reshape(q, (2, 2))
    expected_q = jnp.reshape(jnp.array([1, 2, 3, 4]), (2, 2))
    assert_quantity(result_q, expected_q, bu.second)

  def test_moveaxis(self):
    array = jnp.zeros((3, 4, 5))
    result = bu.math.moveaxis(array, 0, -1)
    self.assertTrue(jnp.all(result == jnp.moveaxis(array, 0, -1)))

    q = jnp.zeros((3, 4, 5)) * bu.second
    result_q = bu.math.moveaxis(q, 0, -1)
    expected_q = jnp.moveaxis(jnp.zeros((3, 4, 5)), 0, -1)
    assert_quantity(result_q, expected_q, bu.second)

  def test_transpose(self):
    array = jnp.ones((2, 3))
    result = bu.math.transpose(array)
    self.assertTrue(jnp.all(result == jnp.transpose(array)))

    q = jnp.ones((2, 3)) * bu.second
    result_q = bu.math.transpose(q)
    expected_q = jnp.transpose(jnp.ones((2, 3)))
    assert_quantity(result_q, expected_q, bu.second)

  def test_swapaxes(self):
    array = jnp.zeros((3, 4, 5))
    result = bu.math.swapaxes(array, 0, 2)
    self.assertTrue(jnp.all(result == jnp.swapaxes(array, 0, 2)))

    q = jnp.zeros((3, 4, 5)) * bu.second
    result_q = bu.math.swapaxes(q, 0, 2)
    expected_q = jnp.swapaxes(jnp.zeros((3, 4, 5)), 0, 2)
    assert_quantity(result_q, expected_q, bu.second)

  def test_row_stack(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    result = bu.math.row_stack((a, b))
    self.assertTrue(jnp.all(result == jnp.vstack((a, b))))

    q1 = [1, 2, 3] * bu.second
    q2 = [4, 5, 6] * bu.second
    result_q = bu.math.row_stack((q1, q2))
    expected_q = jnp.vstack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
    assert_quantity(result_q, expected_q, bu.second)

  def test_concatenate(self):
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[5, 6]])
    result = bu.math.concatenate((a, b), axis=0)
    self.assertTrue(jnp.all(result == jnp.concatenate((a, b), axis=0)))

    q1 = [[1, 2], [3, 4]] * bu.second
    q2 = [[5, 6]] * bu.second
    result_q = bu.math.concatenate((q1, q2), axis=0)
    expected_q = jnp.concatenate((jnp.array([[1, 2], [3, 4]]), jnp.array([[5, 6]])), axis=0)
    assert_quantity(result_q, expected_q, bu.second)

  def test_stack(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    result = bu.math.stack((a, b), axis=1)
    self.assertTrue(jnp.all(result == jnp.stack((a, b), axis=1)))

    q1 = [1, 2, 3] * bu.second
    q2 = [4, 5, 6] * bu.second
    result_q = bu.math.stack((q1, q2), axis=1)
    expected_q = jnp.stack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])), axis=1)
    assert_quantity(result_q, expected_q, bu.second)

  def test_vstack(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    result = bu.math.vstack((a, b))
    self.assertTrue(jnp.all(result == jnp.vstack((a, b))))

    q1 = [1, 2, 3] * bu.second
    q2 = [4, 5, 6] * bu.second
    result_q = bu.math.vstack((q1, q2))
    expected_q = jnp.vstack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
    assert_quantity(result_q, expected_q, bu.second)

  def test_hstack(self):
    a = jnp.array((1, 2, 3))
    b = jnp.array((4, 5, 6))
    result = bu.math.hstack((a, b))
    self.assertTrue(jnp.all(result == jnp.hstack((a, b))))

    q1 = [1, 2, 3] * bu.second
    q2 = [4, 5, 6] * bu.second
    result_q = bu.math.hstack((q1, q2))
    expected_q = jnp.hstack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
    assert_quantity(result_q, expected_q, bu.second)

  def test_dstack(self):
    a = jnp.array([[1], [2], [3]])
    b = jnp.array([[4], [5], [6]])
    result = bu.math.dstack((a, b))
    self.assertTrue(jnp.all(result == jnp.dstack((a, b))))

    q1 = [[1], [2], [3]] * bu.second
    q2 = [[4], [5], [6]] * bu.second
    result_q = bu.math.dstack((q1, q2))
    expected_q = jnp.dstack((jnp.array([[1], [2], [3]]), jnp.array([[4], [5], [6]])))
    assert_quantity(result_q, expected_q, bu.second)

  def test_column_stack(self):
    a = jnp.array((1, 2, 3))
    b = jnp.array((4, 5, 6))
    result = bu.math.column_stack((a, b))
    self.assertTrue(jnp.all(result == jnp.column_stack((a, b))))

    q1 = [1, 2, 3] * bu.second
    q2 = [4, 5, 6] * bu.second
    result_q = bu.math.column_stack((q1, q2))
    expected_q = jnp.column_stack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
    assert_quantity(result_q, expected_q, bu.second)

  def test_split(self):
    array = jnp.arange(9)
    result = bu.math.split(array, 3)
    expected = jnp.split(array, 3)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.all(r == e))

    q = jnp.arange(9) * bu.second
    result_q = bu.math.split(q, 3)
    expected_q = jnp.split(jnp.arange(9), 3)
    for r, e in zip(result_q, expected_q):
      assert_quantity(r, e, ms)

  def test_dsplit(self):
    array = jnp.arange(16.0).reshape(2, 2, 4)
    result = bu.math.dsplit(array, 2)
    expected = jnp.dsplit(array, 2)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.all(r == e))

    q = jnp.arange(16.0).reshape(2, 2, 4) * bu.second
    result_q = bu.math.dsplit(q, 2)
    expected_q = jnp.dsplit(jnp.arange(16.0).reshape(2, 2, 4), 2)
    for r, e in zip(result_q, expected_q):
      assert_quantity(r, e, bu.second)

  def test_hsplit(self):
    array = jnp.arange(16.0).reshape(4, 4)
    result = bu.math.hsplit(array, 2)
    expected = jnp.hsplit(array, 2)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.all(r == e))

    q = jnp.arange(16.0).reshape(4, 4) * bu.second
    result_q = bu.math.hsplit(q, 2)
    expected_q = jnp.hsplit(jnp.arange(16.0).reshape(4, 4), 2)
    for r, e in zip(result_q, expected_q):
      assert_quantity(r, e, bu.second)

  def test_vsplit(self):
    array = jnp.arange(16.0).reshape(4, 4)
    result = bu.math.vsplit(array, 2)
    expected = jnp.vsplit(array, 2)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.all(r == e))

    q = jnp.arange(16.0).reshape(4, 4) * bu.second
    result_q = bu.math.vsplit(q, 2)
    expected_q = jnp.vsplit(jnp.arange(16.0).reshape(4, 4), 2)
    for r, e in zip(result_q, expected_q):
      assert_quantity(r, e, bu.second)

  def test_tile(self):
    array = jnp.array([0, 1, 2])
    result = bu.math.tile(array, 2)
    self.assertTrue(jnp.all(result == jnp.tile(array, 2)))

    q = jnp.array([0, 1, 2]) * bu.second
    result_q = bu.math.tile(q, 2)
    expected_q = jnp.tile(jnp.array([0, 1, 2]), 2)
    assert_quantity(result_q, expected_q, bu.second)

  def test_repeat(self):
    array = jnp.array([0, 1, 2])
    result = bu.math.repeat(array, 2)
    self.assertTrue(jnp.all(result == jnp.repeat(array, 2)))

    q = [0, 1, 2] * bu.second
    result_q = bu.math.repeat(q, 2)
    expected_q = jnp.repeat(jnp.array([0, 1, 2]), 2)
    assert_quantity(result_q, expected_q, bu.second)

  def test_unique(self):
    array = jnp.array([0, 1, 2, 1, 0])
    result = bu.math.unique(array)
    self.assertTrue(jnp.all(result == jnp.unique(array)))

    q = [0, 1, 2, 1, 0] * bu.second
    result_q = bu.math.unique(q)
    expected_q = jnp.unique(jnp.array([0, 1, 2, 1, 0]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_append(self):
    array = jnp.array([0, 1, 2])
    result = bu.math.append(array, 3)
    self.assertTrue(jnp.all(result == jnp.append(array, 3)))

    q = [0, 1, 2] * bu.second
    result_q = bu.math.append(q, 3)
    expected_q = jnp.append(jnp.array([0, 1, 2]), 3)
    assert_quantity(result_q, expected_q, bu.second)

  def test_flip(self):
    array = jnp.array([0, 1, 2])
    result = bu.math.flip(array)
    self.assertTrue(jnp.all(result == jnp.flip(array)))

    q = [0, 1, 2] * bu.second
    result_q = bu.math.flip(q)
    expected_q = jnp.flip(jnp.array([0, 1, 2]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_fliplr(self):
    array = jnp.array([[0, 1, 2], [3, 4, 5]])
    result = bu.math.fliplr(array)
    self.assertTrue(jnp.all(result == jnp.fliplr(array)))

    q = [[0, 1, 2], [3, 4, 5]] * bu.second
    result_q = bu.math.fliplr(q)
    expected_q = jnp.fliplr(jnp.array([[0, 1, 2], [3, 4, 5]]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_flipud(self):
    array = jnp.array([[0, 1, 2], [3, 4, 5]])
    result = bu.math.flipud(array)
    self.assertTrue(jnp.all(result == jnp.flipud(array)))

    q = [[0, 1, 2], [3, 4, 5]] * bu.second
    result_q = bu.math.flipud(q)
    expected_q = jnp.flipud(jnp.array([[0, 1, 2], [3, 4, 5]]))
    assert_quantity(result_q, expected_q, ms)

  def test_roll(self):
    array = jnp.array([0, 1, 2])
    result = bu.math.roll(array, 1)
    self.assertTrue(jnp.all(result == jnp.roll(array, 1)))

    q = [0, 1, 2] * bu.second
    result_q = bu.math.roll(q, 1)
    expected_q = jnp.roll(jnp.array([0, 1, 2]), 1)
    assert_quantity(result_q, expected_q, ms)

  def test_atleast_1d(self):
    array = jnp.array(0)
    result = bu.math.atleast_1d(array)
    self.assertTrue(jnp.all(result == jnp.atleast_1d(array)))

    q = 0 * bu.second
    result_q = bu.math.atleast_1d(q)
    expected_q = jnp.atleast_1d(jnp.array(0))
    assert_quantity(result_q, expected_q, bu.second)

  def test_atleast_2d(self):
    array = jnp.array([0, 1, 2])
    result = bu.math.atleast_2d(array)
    self.assertTrue(jnp.all(result == jnp.atleast_2d(array)))

    q = [0, 1, 2] * bu.second
    result_q = bu.math.atleast_2d(q)
    expected_q = jnp.atleast_2d(jnp.array([0, 1, 2]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_atleast_3d(self):
    array = jnp.array([[0, 1, 2], [3, 4, 5]])
    result = bu.math.atleast_3d(array)
    self.assertTrue(jnp.all(result == jnp.atleast_3d(array)))

    q = [[0, 1, 2], [3, 4, 5]] * bu.second
    result_q = bu.math.atleast_3d(q)
    expected_q = jnp.atleast_3d(jnp.array([[0, 1, 2], [3, 4, 5]]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_expand_dims(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.expand_dims(array, axis=0)
    self.assertTrue(jnp.all(result == jnp.expand_dims(array, axis=0)))

    q = [1, 2, 3] * bu.second
    result_q = bu.math.expand_dims(q, axis=0)
    expected_q = jnp.expand_dims(jnp.array([1, 2, 3]), axis=0)
    assert_quantity(result_q, expected_q, bu.second)

  def test_squeeze(self):
    array = jnp.array([[[0], [1], [2]]])
    result = bu.math.squeeze(array)
    self.assertTrue(jnp.all(result == jnp.squeeze(array)))

    q = [[[0], [1], [2]]] * bu.second
    result_q = bu.math.squeeze(q)
    expected_q = jnp.squeeze(jnp.array([[[0], [1], [2]]]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_sort(self):
    array = jnp.array([2, 3, 1])
    result = bu.math.sort(array)
    self.assertTrue(jnp.all(result == jnp.sort(array)))

    q = [2, 3, 1] * bu.second
    result_q = bu.math.sort(q)
    expected_q = jnp.sort(jnp.array([2, 3, 1]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_max(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.max(array)
    self.assertTrue(result == jnp.max(array))

    q = [1, 2, 3] * bu.second
    result_q = bu.math.max(q)
    expected_q = jnp.max(jnp.array([1, 2, 3]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_min(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.min(array)
    self.assertTrue(result == jnp.min(array))

    q = [1, 2, 3] * bu.second
    result_q = bu.math.min(q)
    expected_q = jnp.min(jnp.array([1, 2, 3]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_amin(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.amin(array)
    self.assertTrue(result == jnp.min(array))

    q = [1, 2, 3] * bu.second
    result_q = bu.math.amin(q)
    expected_q = jnp.min(jnp.array([1, 2, 3]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_amax(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.amax(array)
    self.assertTrue(result == jnp.max(array))

    q = [1, 2, 3] * bu.second
    result_q = bu.math.amax(q)
    expected_q = jnp.max(jnp.array([1, 2, 3]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_block(self):
    array = jnp.array([[1, 2], [3, 4]])
    result = bu.math.block(array)
    self.assertTrue(jnp.all(result == jnp.block(array)))

    q = [[1, 2], [3, 4]] * bu.second
    result_q = bu.math.block(q)
    expected_q = jnp.block(jnp.array([[1, 2], [3, 4]]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_compress(self):
    array = jnp.array([1, 2, 3, 4])
    result = bu.math.compress(jnp.array([0, 1, 1, 0]), array)
    self.assertTrue(jnp.all(result == jnp.compress(jnp.array([0, 1, 1, 0]), array)))

    q = [1, 2, 3, 4] * bu.second
    a = [0, 1, 1, 0] * bu.second
    result_q = bu.math.compress(q, a)
    expected_q = jnp.compress(jnp.array([1, 2, 3, 4]), jnp.array([0, 1, 1, 0]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_diagflat(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.diagflat(array)
    self.assertTrue(jnp.all(result == jnp.diagflat(array)))

    q = [1, 2, 3] * bu.second
    result_q = bu.math.diagflat(q)
    expected_q = jnp.diagflat(jnp.array([1, 2, 3]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_diagonal(self):
    array = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    result = bu.math.diagonal(array)
    self.assertTrue(jnp.all(result == jnp.diagonal(array)))

    q = [[0, 1, 2], [3, 4, 5], [6, 7, 8]] * bu.second
    result_q = bu.math.diagonal(q)
    expected_q = jnp.diagonal(jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_choose(self):
    choices = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6]), jnp.array([7, 8, 9])]
    result = bu.math.choose(jnp.array([0, 1, 2]), choices)
    self.assertTrue(jnp.all(result == jnp.choose(jnp.array([0, 1, 2]), choices)))

    q = [0, 1, 2] * bu.second
    q = q.astype(jnp.int64)
    result_q = bu.math.choose(q, choices)
    expected_q = jnp.choose(jnp.array([0, 1, 2]), choices)
    assert_quantity(result_q, expected_q, bu.second)

  def test_ravel(self):
    array = jnp.array([[1, 2, 3], [4, 5, 6]])
    result = bu.math.ravel(array)
    self.assertTrue(jnp.all(result == jnp.ravel(array)))

    q = [[1, 2, 3], [4, 5, 6]] * bu.second
    result_q = bu.math.ravel(q)
    expected_q = jnp.ravel(jnp.array([[1, 2, 3], [4, 5, 6]]))
    assert_quantity(result_q, expected_q, bu.second)

  # return_quantity = False
  def test_argsort(self):
    array = jnp.array([2, 3, 1])
    result = bu.math.argsort(array)
    self.assertTrue(jnp.all(result == jnp.argsort(array)))

    q = [2, 3, 1] * bu.second
    result_q = bu.math.argsort(q)
    expected_q = jnp.argsort(jnp.array([2, 3, 1]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_argmax(self):
    array = jnp.array([2, 3, 1])
    result = bu.math.argmax(array)
    self.assertTrue(result == jnp.argmax(array))

    q = [2, 3, 1] * bu.second
    result_q = bu.math.argmax(q)
    expected_q = jnp.argmax(jnp.array([2, 3, 1]))
    assert result_q == expected_q

  def test_argmin(self):
    array = jnp.array([2, 3, 1])
    result = bu.math.argmin(array)
    self.assertTrue(result == jnp.argmin(array))

    q = [2, 3, 1] * bu.second
    result_q = bu.math.argmin(q)
    expected_q = jnp.argmin(jnp.array([2, 3, 1]))
    assert result_q == expected_q

  def test_argwhere(self):
    array = jnp.array([0, 1, 2])
    result = bu.math.argwhere(array)
    self.assertTrue(jnp.all(result == jnp.argwhere(array)))

    q = [0, 1, 2] * bu.second
    result_q = bu.math.argwhere(q)
    expected_q = jnp.argwhere(jnp.array([0, 1, 2]))
    assert jnp.all(result_q == expected_q)

  def test_nonzero(self):
    array = jnp.array([0, 1, 2])
    result = bu.math.nonzero(array)
    expected = jnp.nonzero(array)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.array_equal(r, e))

    q = [0, 1, 2] * bu.second
    result_q = bu.math.nonzero(q)
    expected_q = jnp.nonzero(jnp.array([0, 1, 2]))
    for r, e in zip(result_q, expected_q):
      assert jnp.all(r == e)

  def test_flatnonzero(self):
    array = jnp.array([0, 1, 2])
    result = bu.math.flatnonzero(array)
    self.assertTrue(jnp.all(result == jnp.flatnonzero(array)))

    q = [0, 1, 2] * bu.second
    result_q = bu.math.flatnonzero(q)
    expected_q = jnp.flatnonzero(jnp.array([0, 1, 2]))
    assert jnp.all(result_q == expected_q)

  def test_searchsorted(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.searchsorted(array, 2)
    self.assertTrue(result == jnp.searchsorted(array, 2))

    q = [0, 1, 2] * bu.second
    result_q = bu.math.searchsorted(q, 2)
    expected_q = jnp.searchsorted(jnp.array([0, 1, 2]), 2)
    assert result_q == expected_q

  def test_extract(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.extract(array > 1, array)
    self.assertTrue(jnp.all(result == jnp.extract(array > 1, array)))

    q = [1, 2, 3] * bu.second
    a = array * bu.second
    result_q = bu.math.extract(q > 1 * bu.second, a)
    expected_q = jnp.extract(jnp.array([0, 1, 2]), jnp.array([1, 2, 3]))
    assert jnp.all(result_q == expected_q)

  def test_count_nonzero(self):
    array = jnp.array([1, 0, 2, 0, 3, 0])
    result = bu.math.count_nonzero(array)
    self.assertTrue(result == jnp.count_nonzero(array))

    q = [1, 0, 2, 0, 3, 0] * bu.second
    result_q = bu.math.count_nonzero(q)
    expected_q = jnp.count_nonzero(jnp.array([1, 0, 2, 0, 3, 0]))
    assert result_q == expected_q


class TestElementwiseBitOperationsUnary(unittest.TestCase):
  def test_bitwise_not(self):
    result = bu.math.bitwise_not(jnp.array([0b1100]))
    self.assertTrue(jnp.all(result == jnp.bitwise_not(jnp.array([0b1100]))))

    with pytest.raises(ValueError):
      q = [0b1100] * bu.second
      result_q = bu.math.bitwise_not(q)

  def test_invert(self):
    result = bu.math.invert(jnp.array([0b1100]))
    self.assertTrue(jnp.all(result == jnp.invert(jnp.array([0b1100]))))

    with pytest.raises(ValueError):
      q = [0b1100] * bu.second
      result_q = bu.math.invert(q)


class TestElementwiseBitOperationsBinary(unittest.TestCase):

  def test_bitwise_and(self):
    result = bu.math.bitwise_and(jnp.array([0b1100]), jnp.array([0b1010]))
    self.assertTrue(jnp.all(result == jnp.bitwise_and(jnp.array([0b1100]), jnp.array([0b1010]))))

    with pytest.raises(ValueError):
      q1 = [0b1100] * bu.second
      q2 = [0b1010] * bu.second
      result_q = bu.math.bitwise_and(q1, q2)

  def test_bitwise_or(self):
    result = bu.math.bitwise_or(jnp.array([0b1100]), jnp.array([0b1010]))
    self.assertTrue(jnp.all(result == jnp.bitwise_or(jnp.array([0b1100]), jnp.array([0b1010]))))

    with pytest.raises(ValueError):
      q1 = [0b1100] * bu.second
      q2 = [0b1010] * bu.second
      result_q = bu.math.bitwise_or(q1, q2)

  def test_bitwise_xor(self):
    result = bu.math.bitwise_xor(jnp.array([0b1100]), jnp.array([0b1010]))
    self.assertTrue(jnp.all(result == jnp.bitwise_xor(jnp.array([0b1100]), jnp.array([0b1010]))))

    with pytest.raises(ValueError):
      q1 = [0b1100] * bu.second
      q2 = [0b1010] * bu.second
      result_q = bu.math.bitwise_xor(q1, q2)

  def test_left_shift(self):
    result = bu.math.left_shift(jnp.array([0b1100]), 2)
    self.assertTrue(jnp.all(result == jnp.left_shift(jnp.array([0b1100]), 2)))

    with pytest.raises(ValueError):
      q = [0b1100] * bu.second
      result_q = bu.math.left_shift(q, 2)

  def test_right_shift(self):
    result = bu.math.right_shift(jnp.array([0b1100]), 2)
    self.assertTrue(jnp.all(result == jnp.right_shift(jnp.array([0b1100]), 2)))

    with pytest.raises(ValueError):
      q = [0b1100] * bu.second
      result_q = bu.math.right_shift(q, 2)


class TestLogicFuncsUnary(unittest.TestCase):
  def test_all(self):
    result = bu.math.all(jnp.array([True, True, True]))
    self.assertTrue(result == jnp.all(jnp.array([True, True, True])))

    with pytest.raises(ValueError):
      q = [True, True, True] * bu.second
      result_q = bu.math.all(q)

  def test_any(self):
    result = bu.math.any(jnp.array([False, True, False]))
    self.assertTrue(result == jnp.any(jnp.array([False, True, False])))

    with pytest.raises(ValueError):
      q = [False, True, False] * bu.second
      result_q = bu.math.any(q)

  def test_logical_not(self):
    result = bu.math.logical_not(jnp.array([True, False]))
    self.assertTrue(jnp.all(result == jnp.logical_not(jnp.array([True, False]))))

    with pytest.raises(ValueError):
      q = [True, False] * bu.second
      result_q = bu.math.logical_not(q)


class TestLogicFuncsBinary(unittest.TestCase):

  def test_equal(self):
    result = bu.math.equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    self.assertTrue(jnp.all(result == jnp.equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))))

    q1 = [1, 2, 3] * bu.second
    q2 = [2, 3, 4] * bu.second
    result_q = bu.math.equal(q1, q2)
    expected_q = jnp.equal(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

    with pytest.raises(DimensionMismatchError):
      q1 = [1, 2, 3] * bu.second
      q2 = [1, 2, 4] * bu.volt
      result_q = bu.math.equal(q1, q2)

  def test_not_equal(self):
    result = bu.math.not_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 4]))
    self.assertTrue(jnp.all(result == jnp.not_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 4]))))

    q1 = [1, 2, 3] * bu.second
    q2 = [2, 3, 4] * bu.second
    result_q = bu.math.not_equal(q1, q2)
    expected_q = jnp.not_equal(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

  def test_greater(self):
    result = bu.math.greater(jnp.array([1, 2, 3]), jnp.array([0, 2, 4]))
    self.assertTrue(jnp.all(result == jnp.greater(jnp.array([1, 2, 3]), jnp.array([0, 2, 4]))))

    q1 = [1, 2, 3] * bu.second
    q2 = [2, 3, 4] * bu.second
    result_q = bu.math.greater(q1, q2)
    expected_q = jnp.greater(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

  def test_greater_equal(self):
    result = bu.math.greater_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 2]))
    self.assertTrue(jnp.all(result == jnp.greater_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 2]))))

    q1 = [1, 2, 3] * bu.second
    q2 = [2, 3, 4] * bu.second
    result_q = bu.math.greater_equal(q1, q2)
    expected_q = jnp.greater_equal(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

  def test_less(self):
    result = bu.math.less(jnp.array([1, 2, 3]), jnp.array([2, 2, 2]))
    self.assertTrue(jnp.all(result == jnp.less(jnp.array([1, 2, 3]), jnp.array([2, 2, 2]))))

    q1 = [1, 2, 3] * bu.second
    q2 = [2, 3, 4] * bu.second
    result_q = bu.math.less(q1, q2)
    expected_q = jnp.less(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

  def test_less_equal(self):
    result = bu.math.less_equal(jnp.array([1, 2, 3]), jnp.array([2, 2, 2]))
    self.assertTrue(jnp.all(result == jnp.less_equal(jnp.array([1, 2, 3]), jnp.array([2, 2, 2]))))

    q1 = [1, 2, 3] * bu.second
    q2 = [2, 3, 4] * bu.second
    result_q = bu.math.less_equal(q1, q2)
    expected_q = jnp.less_equal(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

  def test_array_equal(self):
    result = bu.math.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    self.assertTrue(result == jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3])))

    q1 = [1, 2, 3] * bu.second
    q2 = [2, 3, 4] * bu.second
    result_q = bu.math.array_equal(q1, q2)
    expected_q = jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

  def test_isclose(self):
    result = bu.math.isclose(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.1]), atol=0.2)
    self.assertTrue(jnp.all(result == jnp.isclose(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.1]), atol=0.2)))

    q1 = [1.0, 2.0] * bu.second
    q2 = [2.0, 3.0] * bu.second
    result_q = bu.math.isclose(q1, q2, atol=0.2)
    expected_q = jnp.isclose(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]), atol=0.2)
    assert_quantity(result_q, expected_q, None)

  def test_allclose(self):
    result = bu.math.allclose(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.1]), atol=0.2)
    self.assertTrue(result == jnp.allclose(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.1]), atol=0.2))

    q1 = [1.0, 2.0] * bu.second
    q2 = [2.0, 3.0] * bu.second
    result_q = bu.math.allclose(q1, q2, atol=0.2)
    expected_q = jnp.allclose(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]), atol=0.2)
    assert_quantity(result_q, expected_q, None)

  def test_logical_and(self):
    result = bu.math.logical_and(jnp.array([True, False]), jnp.array([False, False]))
    self.assertTrue(jnp.all(result == jnp.logical_and(jnp.array([True, False]), jnp.array([False, False]))))

    q1 = [True, False] * bu.second
    q2 = [False, False] * bu.second
    result_q = bu.math.logical_and(q1, q2)
    expected_q = jnp.logical_and(jnp.array([True, False]), jnp.array([False, False]))
    assert_quantity(result_q, expected_q, None)

  def test_logical_or(self):
    result = bu.math.logical_or(jnp.array([True, False]), jnp.array([False, False]))
    self.assertTrue(jnp.all(result == jnp.logical_or(jnp.array([True, False]), jnp.array([False, False]))))

    q1 = [True, False] * bu.second
    q2 = [False, False] * bu.second
    result_q = bu.math.logical_or(q1, q2)
    expected_q = jnp.logical_or(jnp.array([True, False]), jnp.array([False, False]))
    assert_quantity(result_q, expected_q, None)

  def test_logical_xor(self):
    result = bu.math.logical_xor(jnp.array([True, False]), jnp.array([False, False]))
    self.assertTrue(jnp.all(result == jnp.logical_xor(jnp.array([True, False]), jnp.array([False, False]))))

    q1 = [True, False] * bu.second
    q2 = [False, False] * bu.second
    result_q = bu.math.logical_xor(q1, q2)
    expected_q = jnp.logical_xor(jnp.array([True, False]), jnp.array([False, False]))
    assert_quantity(result_q, expected_q, None)


class TestIndexingFuncs(unittest.TestCase):

  def test_where(self):
    array = jnp.array([1, 2, 3, 4, 5])
    result = bu.math.where(array > 2, array, 0)
    self.assertTrue(jnp.all(result == jnp.where(array > 2, array, 0)))

    q = [1, 2, 3, 4, 5] * bu.second
    result_q = bu.math.where(q > 2 * bu.second, q, 0)
    expected_q = jnp.where(jnp.array([1, 2, 3, 4, 5]) > 2, jnp.array([1, 2, 3, 4, 5]), 0)
    assert_quantity(result_q, expected_q, bu.second)

  def test_tril_indices(self):
    result = bu.math.tril_indices(3)
    expected = jnp.tril_indices(3)
    for i in range(2):
      self.assertTrue(jnp.all(result[i] == expected[i]))

  def test_tril_indices_from(self):
    array = jnp.ones((3, 3))
    result = bu.math.tril_indices_from(array)
    expected = jnp.tril_indices_from(array)
    for i in range(2):
      self.assertTrue(jnp.all(result[i] == expected[i]))

  def test_triu_indices(self):
    result = bu.math.triu_indices(3)
    expected = jnp.triu_indices(3)
    for i in range(2):
      self.assertTrue(jnp.all(result[i] == expected[i]))

  def test_triu_indices_from(self):
    array = jnp.ones((3, 3))
    result = bu.math.triu_indices_from(array)
    expected = jnp.triu_indices_from(array)
    for i in range(2):
      self.assertTrue(jnp.all(result[i] == expected[i]))

  def test_take(self):
    array = jnp.array([4, 3, 5, 7, 6, 8])
    indices = jnp.array([0, 1, 4])
    result = bu.math.take(array, indices)
    self.assertTrue(jnp.all(result == jnp.take(array, indices)))

    q = [4, 3, 5, 7, 6, 8] * bu.second
    i = jnp.array([0, 1, 4])
    result_q = bu.math.take(q, i)
    expected_q = jnp.take(jnp.array([4, 3, 5, 7, 6, 8]), jnp.array([0, 1, 4]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_select(self):
    condlist = [jnp.array([True, False, True]), jnp.array([False, True, False])]
    choicelist = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])]
    result = bu.math.select(condlist, choicelist, default=0)
    self.assertTrue(jnp.all(result == jnp.select(condlist, choicelist, default=0)))

    c = [jnp.array([True, False, True]), jnp.array([False, True, False])]
    ch = [[1, 2, 3] * bu.second, [4, 5, 6] * bu.second]
    result_q = bu.math.select(c, ch, default=0)
    expected_q = jnp.select([jnp.array([True, False, True]), jnp.array([False, True, False])],
                            [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])], default=0)
    assert_quantity(result_q, expected_q, bu.second)


class TestWindowFuncs(unittest.TestCase):

  def test_bartlett(self):
    result = bu.math.bartlett(5)
    self.assertTrue(jnp.all(result == jnp.bartlett(5)))

  def test_blackman(self):
    result = bu.math.blackman(5)
    self.assertTrue(jnp.all(result == jnp.blackman(5)))

  def test_hamming(self):
    result = bu.math.hamming(5)
    self.assertTrue(jnp.all(result == jnp.hamming(5)))

  def test_hanning(self):
    result = bu.math.hanning(5)
    self.assertTrue(jnp.all(result == jnp.hanning(5)))

  def test_kaiser(self):
    result = bu.math.kaiser(5, 0.5)
    self.assertTrue(jnp.all(result == jnp.kaiser(5, 0.5)))


class TestConstants(unittest.TestCase):

  def test_constants(self):
    self.assertTrue(bu.math.e == jnp.e)
    self.assertTrue(bu.math.pi == jnp.pi)
    self.assertTrue(bu.math.inf == jnp.inf)


class TestLinearAlgebra(unittest.TestCase):

  def test_dot(self):
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    result = bu.math.dot(a, b)
    self.assertTrue(result == jnp.dot(a, b))

    q1 = [1, 2] * bu.second
    q2 = [3, 4] * bu.volt
    result_q = bu.math.dot(q1, q2)
    expected_q = jnp.dot(jnp.array([1, 2]), jnp.array([3, 4]))
    assert_quantity(result_q, expected_q, bu.second * bu.volt)

  def test_vdot(self):
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    result = bu.math.vdot(a, b)
    self.assertTrue(result == jnp.vdot(a, b))

    q1 = [1, 2] * bu.second
    q2 = [3, 4] * bu.volt
    result_q = bu.math.vdot(q1, q2)
    expected_q = jnp.vdot(jnp.array([1, 2]), jnp.array([3, 4]))
    assert_quantity(result_q, expected_q, bu.second * bu.volt)

  def test_inner(self):
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    result = bu.math.inner(a, b)
    self.assertTrue(result == jnp.inner(a, b))

    q1 = [1, 2] * bu.second
    q2 = [3, 4] * bu.volt
    result_q = bu.math.inner(q1, q2)
    expected_q = jnp.inner(jnp.array([1, 2]), jnp.array([3, 4]))
    assert_quantity(result_q, expected_q, bu.second * bu.volt)

  def test_outer(self):
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    result = bu.math.outer(a, b)
    self.assertTrue(jnp.all(result == jnp.outer(a, b)))

    q1 = [1, 2] * bu.second
    q2 = [3, 4] * bu.volt
    result_q = bu.math.outer(q1, q2)
    expected_q = jnp.outer(jnp.array([1, 2]), jnp.array([3, 4]))
    assert_quantity(result_q, expected_q, bu.second * bu.volt)

  def test_kron(self):
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    result = bu.math.kron(a, b)
    self.assertTrue(jnp.all(result == jnp.kron(a, b)))

    q1 = [1, 2] * bu.second
    q2 = [3, 4] * bu.volt
    result_q = bu.math.kron(q1, q2)
    expected_q = jnp.kron(jnp.array([1, 2]), jnp.array([3, 4]))
    assert_quantity(result_q, expected_q, bu.second * bu.volt)

  def test_matmul(self):
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[5, 6], [7, 8]])
    result = bu.math.matmul(a, b)
    self.assertTrue(jnp.all(result == jnp.matmul(a, b)))

    q1 = [[1, 2], [3, 4]] * bu.second
    q2 = [[5, 6], [7, 8]] * bu.volt
    result_q = bu.math.matmul(q1, q2)
    expected_q = jnp.matmul(jnp.array([[1, 2], [3, 4]]), jnp.array([[5, 6], [7, 8]]))
    assert_quantity(result_q, expected_q, bu.second * bu.volt)

  def test_trace(self):
    a = jnp.array([[1, 2], [3, 4]])
    result = bu.math.trace(a)
    self.assertTrue(result == jnp.trace(a))

    q = [[1, 2], [3, 4]] * bu.second
    result_q = bu.math.trace(q)
    expected_q = jnp.trace(jnp.array([[1, 2], [3, 4]]))
    assert_quantity(result_q, expected_q, bu.second)


class TestDataTypes(unittest.TestCase):

  def test_dtype(self):
    array = jnp.array([1, 2, 3])
    result = bu.math.dtype(array)
    self.assertTrue(result == jnp.dtype(array))

    q = [1, 2, 3] * bu.second
    q = q.astype(jnp.int64)
    result_q = bu.math.dtype(q)
    expected_q = jnp.dtype(jnp.array([1, 2, 3], dtype=jnp.int64))
    self.assertTrue(result_q == expected_q)

  def test_finfo(self):
    result = bu.math.finfo(jnp.float32)
    self.assertTrue(result == jnp.finfo(jnp.float32))

    q = 1 * bu.second
    q = q.astype(jnp.float64)
    result_q = bu.math.finfo(q)
    expected_q = jnp.finfo(jnp.float64)
    self.assertTrue(result_q == expected_q)

  def test_iinfo(self):
    result = bu.math.iinfo(jnp.int32)
    expected = jnp.iinfo(jnp.int32)
    self.assertEqual(result.min, expected.min)
    self.assertEqual(result.max, expected.max)
    self.assertEqual(result.dtype, expected.dtype)

    q = 1 * bu.second
    q = q.astype(jnp.int32)
    result_q = bu.math.iinfo(q)
    expected_q = jnp.iinfo(jnp.int32)
    self.assertEqual(result_q.min, expected_q.min)
    self.assertEqual(result_q.max, expected_q.max)
    self.assertEqual(result_q.dtype, expected_q.dtype)


class TestMore(unittest.TestCase):
  def test_broadcast_arrays(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([[4], [5]])
    result = bu.math.broadcast_arrays(a, b)
    self.assertTrue(jnp.all(result[0] == jnp.broadcast_arrays(a, b)[0]))
    self.assertTrue(jnp.all(result[1] == jnp.broadcast_arrays(a, b)[1]))

    q1 = [1, 2, 3] * bu.second
    q2 = [[4], [5]] * bu.second
    result_q = bu.math.broadcast_arrays(q1, q2)
    expected_q = jnp.broadcast_arrays(jnp.array([1, 2, 3]), jnp.array([[4], [5]]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_broadcast_shapes(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([[4], [5]])
    result = bu.math.broadcast_shapes(a.shape, b.shape)
    self.assertTrue(result == jnp.broadcast_shapes(a.shape, b.shape))

  def test_einsum(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5])
    result = bu.math.einsum('i,j->ij', a, b)
    self.assertTrue(jnp.all(result == jnp.einsum('i,j->ij', a, b)))

    q1 = [1, 2, 3] * bu.second
    q2 = [4, 5] * bu.volt
    result_q = bu.math.einsum('i,j->ij', q1, q2)
    expected_q = jnp.einsum('i,j->ij', jnp.array([1, 2, 3]), jnp.array([4, 5]))
    assert_quantity(result_q, expected_q, bu.second * bu.volt)

    q1 = [1, 2, 3] * bu.second
    q2 = [1, 2, 3] * bu.second
    result_q = bu.math.einsum('i,i->i', q1, q2)
    expected_q = jnp.einsum('i,i->i', jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_gradient(self):
    f = jnp.array([1, 2, 4, 7, 11, 16], dtype=float)
    result = bu.math.gradient(f)
    self.assertTrue(jnp.all(bu.math.allclose(result, jnp.gradient(f))))

    q = [1, 2, 4, 7, 11, 16] * bu.second
    result_q = bu.math.gradient(q)
    expected_q = jnp.gradient(jnp.array([1, 2, 4, 7, 11, 16]))
    assert_quantity(result_q, expected_q, bu.second)

    q1 = jnp.array([[1, 2, 6], [3, 4, 5]]) * bu.second
    dx = 2. * bu.meter
    # y = [1., 1.5, 3.5] * bu.second
    result_q = bu.math.gradient(q1, dx)
    expected_q = jnp.gradient(jnp.array([[1, 2, 6], [3, 4, 5]]), 2.)
    assert_quantity(result_q[0], expected_q[0], bu.second / bu.meter)
    assert_quantity(result_q[1], expected_q[1], bu.second / bu.meter)

  def test_intersect1d(self):
    a = jnp.array([1, 2, 3, 4, 5])
    b = jnp.array([3, 4, 5, 6, 7])
    result = bu.math.intersect1d(a, b)
    self.assertTrue(jnp.all(result == jnp.intersect1d(a, b)))

    q1 = [1, 2, 3, 4, 5] * bu.second
    q2 = [3, 4, 5, 6, 7] * bu.second
    result_q = bu.math.intersect1d(q1, q2)
    expected_q = jnp.intersect1d(jnp.array([1, 2, 3, 4, 5]), jnp.array([3, 4, 5, 6, 7]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_nan_to_num(self):
    a = jnp.array([1, 2, 3, 4, jnp.nan])
    result = bu.math.nan_to_num(a)
    self.assertTrue(jnp.all(result == jnp.nan_to_num(a)))

    q = [1, 2, 3, 4, jnp.nan] * bu.second
    result_q = bu.math.nan_to_num(q)
    expected_q = jnp.nan_to_num(jnp.array([1, 2, 3, 4, jnp.nan]))
    assert_quantity(result_q, expected_q, bu.second)

  def nanargmax(self):
    a = jnp.array([1, 2, 3, 4, jnp.nan])
    result = bu.math.nanargmax(a)
    self.assertTrue(result == jnp.nanargmax(a))

    q = [1, 2, 3, 4, jnp.nan] * bu.second
    result_q = bu.math.nanargmax(q)
    expected_q = jnp.nanargmax(jnp.array([1, 2, 3, 4, jnp.nan]))
    self.assertTrue(result_q == expected_q)

  def nanargmin(self):
    a = jnp.array([1, 2, 3, 4, jnp.nan])
    result = bu.math.nanargmin(a)
    self.assertTrue(result == jnp.nanargmin(a))

    q = [1, 2, 3, 4, jnp.nan] * bu.second
    result_q = bu.math.nanargmin(q)
    expected_q = jnp.nanargmin(jnp.array([1, 2, 3, 4, jnp.nan]))
    self.assertTrue(result_q == expected_q)

  def test_rot90(self):
    a = jnp.array([[1, 2], [3, 4]])
    result = bu.math.rot90(a)
    self.assertTrue(jnp.all(result == jnp.rot90(a)))

    q = [[1, 2], [3, 4]] * bu.second
    result_q = bu.math.rot90(q)
    expected_q = jnp.rot90(jnp.array([[1, 2], [3, 4]]))
    assert_quantity(result_q, expected_q, bu.second)

  def test_tensordot(self):
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[1, 2], [3, 4]])
    result = bu.math.tensordot(a, b)
    self.assertTrue(jnp.all(result == jnp.tensordot(a, b)))

    q1 = [[1, 2], [3, 4]] * bu.second
    q2 = [[1, 2], [3, 4]] * bu.second
    result_q = bu.math.tensordot(q1, q2)
    expected_q = jnp.tensordot(jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 2], [3, 4]]))
    assert_quantity(result_q, expected_q, bu.second ** 2)
