import unittest

import jax.numpy as jnp
import pytest

import braincore as bc
import brainunit.math as bm
import brainunit as U
from brainunit import DimensionMismatchError
from brainunit._base import Quantity
from brainunit._unit_shortcuts import ms, mV

bc.environ.set(precision=64)


def assert_quantity(q, values, unit):
  values = jnp.asarray(values)
  if isinstance(q, Quantity):
    assert q.unit == unit.unit, f"Unit mismatch: {q.unit} != {unit}"
    assert jnp.allclose(q.value, values), f"Values do not match: {q.value} != {values}"
  else:
    assert jnp.allclose(q, values), f"Values do not match: {q} != {values}"


class TestArrayCreation(unittest.TestCase):

  def test_full(self):
    result = bm.full(3, 4)
    self.assertEqual(result.shape, (3,))
    self.assertTrue(jnp.all(result == 4))

  def test_full_like(self):
    array = jnp.array([1, 2, 3])
    result = bm.full_like(array, 4)
    self.assertEqual(result.shape, array.shape)
    self.assertTrue(jnp.all(result == 4))

  def test_eye(self):
    result = bm.eye(3)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.eye(3)))

  def test_identity(self):
    result = bm.identity(3)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.identity(3)))

  def test_diag(self):
    array = jnp.array([1, 2, 3])
    result = bm.diag(array)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.diag(array)))

  def test_tri(self):
    result = bm.tri(3)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.tri(3)))

  def test_tril(self):
    array = jnp.ones((3, 3))
    result = bm.tril(array)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.tril(array)))

  def test_triu(self):
    array = jnp.ones((3, 3))
    result = bm.triu(array)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.triu(array)))

  def test_empty(self):
    result = bm.empty((2, 2))
    self.assertEqual(result.shape, (2, 2))

  def test_empty_like(self):
    array = jnp.array([1, 2, 3])
    result = bm.empty_like(array)
    self.assertEqual(result.shape, array.shape)

  def test_ones(self):
    result = bm.ones((2, 2))
    self.assertEqual(result.shape, (2, 2))
    self.assertTrue(jnp.all(result == 1))

  def test_ones_like(self):
    array = jnp.array([1, 2, 3])
    result = bm.ones_like(array)
    self.assertEqual(result.shape, array.shape)
    self.assertTrue(jnp.all(result == 1))

  def test_zeros(self):
    result = bm.zeros((2, 2))
    self.assertEqual(result.shape, (2, 2))
    self.assertTrue(jnp.all(result == 0))

  def test_zeros_like(self):
    array = jnp.array([1, 2, 3])
    result = bm.zeros_like(array)
    self.assertEqual(result.shape, array.shape)
    self.assertTrue(jnp.all(result == 0))

  def test_array(self):
    result = bm.array([1, 2, 3])
    self.assertEqual(result.shape, (3,))
    self.assertTrue(jnp.all(result == jnp.array([1, 2, 3])))

  def test_asarray(self):
    result = bm.asarray([1, 2, 3])
    self.assertEqual(result.shape, (3,))
    self.assertTrue(jnp.all(result == jnp.asarray([1, 2, 3])))

  def test_arange(self):
    result = bm.arange(3)
    self.assertEqual(result.shape, (3,))
    self.assertTrue(jnp.all(result == jnp.arange(3)))

  def test_linspace(self):
    result = bm.linspace(0, 10, 5)
    self.assertEqual(result.shape, (5,))
    self.assertTrue(jnp.all(result == jnp.linspace(0, 10, 5)))

  def test_logspace(self):
    result = bm.logspace(0, 2, 5)
    self.assertEqual(result.shape, (5,))
    self.assertTrue(jnp.all(result == jnp.logspace(0, 2, 5)))

  def test_fill_diagonal(self):
    array = jnp.zeros((3, 3))
    result = bm.fill_diagonal(array, 5, inplace=False)
    self.assertTrue(jnp.all(result == jnp.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])))

  def test_array_split(self):
    array = jnp.arange(9)
    result = bm.array_split(array, 3)
    expected = jnp.array_split(array, 3)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.all(r == e))

  def test_meshgrid(self):
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5])
    result = bm.meshgrid(x, y)
    expected = jnp.meshgrid(x, y)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.all(r == e))

  def test_vander(self):
    array = jnp.array([1, 2, 3])
    result = bm.vander(array)
    self.assertEqual(result.shape, (3, 3))
    self.assertTrue(jnp.all(result == jnp.vander(array)))


class TestAttributeFunctions(unittest.TestCase):

  def test_ndim(self):
    array = jnp.array([[1, 2], [3, 4]])
    self.assertEqual(bm.ndim(array), 2)

    q = [[1, 2], [3, 4]] * ms
    self.assertEqual(bm.ndim(q), 2)

  def test_isreal(self):
    array = jnp.array([1.0, 2.0])
    self.assertTrue(jnp.all(bm.isreal(array)))

    q = [[1, 2], [3, 4]] * ms
    self.assertTrue(jnp.all(bm.isreal(q)))

  def test_isscalar(self):
    self.assertTrue(bm.isscalar(1.0))
    self.assertTrue(bm.isscalar(Quantity(1.0)))

  def test_isfinite(self):
    array = jnp.array([1.0, jnp.inf])
    self.assertTrue(jnp.all(bm.isfinite(array) == jnp.isfinite(array)))

    q = [1.0, jnp.inf] * ms
    self.assertTrue(jnp.all(bm.isfinite(q) == jnp.isfinite(q.value)))

  def test_isinf(self):
    array = jnp.array([1.0, jnp.inf])
    self.assertTrue(jnp.all(bm.isinf(array) == jnp.isinf(array)))

    q = [1.0, jnp.inf] * ms
    self.assertTrue(jnp.all(bm.isinf(q) == jnp.isinf(q.value)))

  def test_isnan(self):
    array = jnp.array([1.0, jnp.nan])
    self.assertTrue(jnp.all(bm.isnan(array) == jnp.isnan(array)))

    q = [1.0, jnp.nan] * ms
    self.assertTrue(jnp.all(bm.isnan(q) == jnp.isnan(q.value)))

  def test_shape(self):
    array = jnp.array([[1, 2], [3, 4]])
    self.assertEqual(bm.shape(array), (2, 2))

    q = [[1, 2], [3, 4]] * ms
    self.assertEqual(bm.shape(q), (2, 2))

  def test_size(self):
    array = jnp.array([[1, 2], [3, 4]])
    self.assertEqual(bm.size(array), 4)
    self.assertEqual(bm.size(array, 1), 2)

    q = [[1, 2], [3, 4]] * ms
    self.assertEqual(bm.size(q), 4)
    self.assertEqual(bm.size(q, 1), 2)


class TestMathFuncsKeepUnitUnary(unittest.TestCase):

  def test_real(self):
    complex_array = jnp.array([1 + 2j, 3 + 4j])
    result = bm.real(complex_array)
    self.assertTrue(jnp.all(result == jnp.real(complex_array)))

    q = [1 + 2j, 3 + 4j] * U.second
    result_q = bm.real(q)
    self.assertTrue(jnp.all(result_q == jnp.real(q.value) * U.second))

  def test_imag(self):
    complex_array = jnp.array([1 + 2j, 3 + 4j])
    result = bm.imag(complex_array)
    self.assertTrue(jnp.all(result == jnp.imag(complex_array)))

    q = [1 + 2j, 3 + 4j] * U.second
    result_q = bm.imag(q)
    self.assertTrue(jnp.all(result_q == jnp.imag(q.value) * U.second))

  def test_conj(self):
    complex_array = jnp.array([1 + 2j, 3 + 4j])
    result = bm.conj(complex_array)
    self.assertTrue(jnp.all(result == jnp.conj(complex_array)))

    q = [1 + 2j, 3 + 4j] * U.second
    result_q = bm.conj(q)
    self.assertTrue(jnp.all(result_q == jnp.conj(q.value) * U.second))

  def test_conjugate(self):
    complex_array = jnp.array([1 + 2j, 3 + 4j])
    result = bm.conjugate(complex_array)
    self.assertTrue(jnp.all(result == jnp.conjugate(complex_array)))

    q = [1 + 2j, 3 + 4j] * U.second
    result_q = bm.conjugate(q)
    self.assertTrue(jnp.all(result_q == jnp.conjugate(q.value) * U.second))

  def test_negative(self):
    array = jnp.array([1, 2, 3])
    result = bm.negative(array)
    self.assertTrue(jnp.all(result == jnp.negative(array)))

    q = [1, 2, 3] * ms
    result_q = bm.negative(q)
    expected_q = jnp.negative(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_positive(self):
    array = jnp.array([-1, -2, -3])
    result = bm.positive(array)
    self.assertTrue(jnp.all(result == jnp.positive(array)))

    q = [-1, -2, -3] * ms
    result_q = bm.positive(q)
    expected_q = jnp.positive(jnp.array([-1, -2, -3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_abs(self):
    array = jnp.array([-1, -2, 3])
    result = bm.abs(array)
    self.assertTrue(jnp.all(result == jnp.abs(array)))

    q = [-1, -2, 3] * ms
    result_q = bm.abs(q)
    expected_q = jnp.abs(jnp.array([-1, -2, -3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_round(self):
    array = jnp.array([1.123, 2.567, 3.891])
    result = bm.round(array)
    self.assertTrue(jnp.all(result == jnp.round(array)))

    q = [1.123, 2.567, 3.891] * U.second
    result_q = bm.round(q)
    expected_q = jnp.round(jnp.array([1.123, 2.567, 3.891])) * U.second
    assert_quantity(result_q, expected_q.value, U.second)

  def test_rint(self):
    array = jnp.array([1.5, 2.3, 3.8])
    result = bm.rint(array)
    self.assertTrue(jnp.all(result == jnp.rint(array)))

    q = [1.5, 2.3, 3.8] * U.second
    result_q = bm.rint(q)
    expected_q = jnp.rint(jnp.array([1.5, 2.3, 3.8])) * U.second
    assert_quantity(result_q, expected_q.value, U.second)

  def test_floor(self):
    array = jnp.array([1.5, 2.3, 3.8])
    result = bm.floor(array)
    self.assertTrue(jnp.all(result == jnp.floor(array)))

    q = [1.5, 2.3, 3.8] * U.second
    result_q = bm.floor(q)
    expected_q = jnp.floor(jnp.array([1.5, 2.3, 3.8])) * U.second
    assert_quantity(result_q, expected_q.value, U.second)

  def test_ceil(self):
    array = jnp.array([1.5, 2.3, 3.8])
    result = bm.ceil(array)
    self.assertTrue(jnp.all(result == jnp.ceil(array)))

    q = [1.5, 2.3, 3.8] * U.second
    result_q = bm.ceil(q)
    expected_q = jnp.ceil(jnp.array([1.5, 2.3, 3.8])) * U.second
    assert_quantity(result_q, expected_q.value, U.second)

  def test_trunc(self):
    array = jnp.array([1.5, 2.3, 3.8])
    result = bm.trunc(array)
    self.assertTrue(jnp.all(result == jnp.trunc(array)))

    q = [1.5, 2.3, 3.8] * U.second
    result_q = bm.trunc(q)
    expected_q = jnp.trunc(jnp.array([1.5, 2.3, 3.8])) * U.second
    assert_quantity(result_q, expected_q.value, U.second)

  def test_fix(self):
    array = jnp.array([1.5, 2.3, 3.8])
    result = bm.fix(array)
    self.assertTrue(jnp.all(result == jnp.fix(array)))

    q = [1.5, 2.3, 3.8] * U.second
    result_q = bm.fix(q)
    expected_q = jnp.fix(jnp.array([1.5, 2.3, 3.8])) * U.second
    assert_quantity(result_q, expected_q.value, U.second)

  def test_sum(self):
    array = jnp.array([1, 2, 3])
    result = bm.sum(array)
    self.assertTrue(result == jnp.sum(array))

    q = [1, 2, 3] * ms
    result_q = bm.sum(q)
    expected_q = jnp.sum(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nancumsum(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bm.nancumsum(array)
    self.assertTrue(jnp.all(result == jnp.nancumsum(array)))

    q = [1, jnp.nan, 3] * ms
    result_q = bm.nancumsum(q)
    expected_q = jnp.nancumsum(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nansum(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bm.nansum(array)
    self.assertTrue(result == jnp.nansum(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bm.nansum(q)
    expected_q = jnp.nansum(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_cumsum(self):
    array = jnp.array([1, 2, 3])
    result = bm.cumsum(array)
    self.assertTrue(jnp.all(result == jnp.cumsum(array)))

    q = [1, 2, 3] * ms
    result_q = bm.cumsum(q)
    expected_q = jnp.cumsum(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_ediff1d(self):
    array = jnp.array([1, 2, 3])
    result = bm.ediff1d(array)
    self.assertTrue(jnp.all(result == jnp.ediff1d(array)))

    q = [1, 2, 3] * ms
    result_q = bm.ediff1d(q)
    expected_q = jnp.ediff1d(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_absolute(self):
    array = jnp.array([-1, -2, 3])
    result = bm.absolute(array)
    self.assertTrue(jnp.all(result == jnp.absolute(array)))

    q = [-1, -2, 3] * ms
    result_q = bm.absolute(q)
    expected_q = jnp.absolute(jnp.array([-1, -2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_fabs(self):
    array = jnp.array([-1, -2, 3])
    result = bm.fabs(array)
    self.assertTrue(jnp.all(result == jnp.fabs(array)))

    q = [-1, -2, 3] * ms
    result_q = bm.fabs(q)
    expected_q = jnp.fabs(jnp.array([-1, -2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_median(self):
    array = jnp.array([1, 2, 3])
    result = bm.median(array)
    self.assertTrue(result == jnp.median(array))

    q = [1, 2, 3] * ms
    result_q = bm.median(q)
    expected_q = jnp.median(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nanmin(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bm.nanmin(array)
    self.assertTrue(result == jnp.nanmin(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bm.nanmin(q)
    expected_q = jnp.nanmin(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nanmax(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bm.nanmax(array)
    self.assertTrue(result == jnp.nanmax(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bm.nanmax(q)
    expected_q = jnp.nanmax(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_ptp(self):
    array = jnp.array([1, 2, 3])
    result = bm.ptp(array)
    self.assertTrue(result == jnp.ptp(array))

    q = [1, 2, 3] * ms
    result_q = bm.ptp(q)
    expected_q = jnp.ptp(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_average(self):
    array = jnp.array([1, 2, 3])
    result = bm.average(array)
    self.assertTrue(result == jnp.average(array))

    q = [1, 2, 3] * ms
    result_q = bm.average(q)
    expected_q = jnp.average(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_mean(self):
    array = jnp.array([1, 2, 3])
    result = bm.mean(array)
    self.assertTrue(result == jnp.mean(array))

    q = [1, 2, 3] * ms
    result_q = bm.mean(q)
    expected_q = jnp.mean(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_std(self):
    array = jnp.array([1, 2, 3])
    result = bm.std(array)
    self.assertTrue(result == jnp.std(array))

    q = [1, 2, 3] * ms
    result_q = bm.std(q)
    expected_q = jnp.std(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nanmedian(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bm.nanmedian(array)
    self.assertTrue(result == jnp.nanmedian(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bm.nanmedian(q)
    expected_q = jnp.nanmedian(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nanmean(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bm.nanmean(array)
    self.assertTrue(result == jnp.nanmean(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bm.nanmean(q)
    expected_q = jnp.nanmean(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nanstd(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bm.nanstd(array)
    self.assertTrue(result == jnp.nanstd(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bm.nanstd(q)
    expected_q = jnp.nanstd(jnp.array([1, jnp.nan, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_diff(self):
    array = jnp.array([1, 2, 3])
    result = bm.diff(array)
    self.assertTrue(jnp.all(result == jnp.diff(array)))

    q = [1, 2, 3] * ms
    result_q = bm.diff(q)
    expected_q = jnp.diff(jnp.array([1, 2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_modf(self):
    result = bm.modf(jnp.array([5.5, 7.3]))
    expected = jnp.modf(jnp.array([5.5, 7.3]))
    self.assertTrue(jnp.all(result[0] == expected[0]) and jnp.all(result[1] == expected[1]))


class TestMathFuncsKeepUnitBinary(unittest.TestCase):

  def test_fmod(self):
    result = bm.fmod(jnp.array([5, 7]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.fmod(jnp.array([5, 7]), jnp.array([2, 3]))))

    q1 = [5, 7] * ms
    q2 = [2, 3] * ms
    result_q = bm.fmod(q1, q2)
    expected_q = jnp.fmod(jnp.array([5, 7]), jnp.array([2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_mod(self):
    result = bm.mod(jnp.array([5, 7]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.mod(jnp.array([5, 7]), jnp.array([2, 3]))))

    q1 = [5, 7] * ms
    q2 = [2, 3] * ms
    result_q = bm.mod(q1, q2)
    expected_q = jnp.mod(jnp.array([5, 7]), jnp.array([2, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_copysign(self):
    result = bm.copysign(jnp.array([-1, 2]), jnp.array([1, -3]))
    self.assertTrue(jnp.all(result == jnp.copysign(jnp.array([-1, 2]), jnp.array([1, -3]))))

    q1 = [-1, 2] * ms
    q2 = [1, -3] * ms
    result_q = bm.copysign(q1, q2)
    expected_q = jnp.copysign(jnp.array([-1, 2]), jnp.array([1, -3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_heaviside(self):
    result = bm.heaviside(jnp.array([-1, 2]), jnp.array([0.5, 0.5]))
    self.assertTrue(jnp.all(result == jnp.heaviside(jnp.array([-1, 2]), jnp.array([0.5, 0.5]))))

  def test_maximum(self):
    result = bm.maximum(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))
    self.assertTrue(jnp.all(result == jnp.maximum(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))))

    q1 = [1, 3, 2] * ms
    q2 = [2, 1, 3] * ms
    result_q = bm.maximum(q1, q2)
    expected_q = jnp.maximum(jnp.array([1, 3, 2]), jnp.array([2, 1, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_minimum(self):
    result = bm.minimum(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))
    self.assertTrue(jnp.all(result == jnp.minimum(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))))

    q1 = [1, 3, 2] * ms
    q2 = [2, 1, 3] * ms
    result_q = bm.minimum(q1, q2)
    expected_q = jnp.minimum(jnp.array([1, 3, 2]), jnp.array([2, 1, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_fmax(self):
    result = bm.fmax(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))
    self.assertTrue(jnp.all(result == jnp.fmax(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))))

    q1 = [1, 3, 2] * ms
    q2 = [2, 1, 3] * ms
    result_q = bm.fmax(q1, q2)
    expected_q = jnp.fmax(jnp.array([1, 3, 2]), jnp.array([2, 1, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_fmin(self):
    result = bm.fmin(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))
    self.assertTrue(jnp.all(result == jnp.fmin(jnp.array([1, 3, 2]), jnp.array([2, 1, 3]))))

    q1 = [1, 3, 2] * ms
    q2 = [2, 1, 3] * ms
    result_q = bm.fmin(q1, q2)
    expected_q = jnp.fmin(jnp.array([1, 3, 2]), jnp.array([2, 1, 3])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_lcm(self):
    result = bm.lcm(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))
    self.assertTrue(jnp.all(result == jnp.lcm(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))))

    q1 = [4, 5, 6] * U.second
    q2 = [2, 3, 4] * U.second
    q1 = q1.astype(jnp.int64)
    q2 = q2.astype(jnp.int64)
    result_q = bm.lcm(q1, q2)
    expected_q = jnp.lcm(jnp.array([4, 5, 6]), jnp.array([2, 3, 4])) * U.second
    assert_quantity(result_q, expected_q.value, U.second)

  def test_gcd(self):
    result = bm.gcd(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))
    self.assertTrue(jnp.all(result == jnp.gcd(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))))

    q1 = [4, 5, 6] * U.second
    q2 = [2, 3, 4] * U.second
    q1 = q1.astype(jnp.int64)
    q2 = q2.astype(jnp.int64)
    result_q = bm.gcd(q1, q2)
    expected_q = jnp.gcd(jnp.array([4, 5, 6]), jnp.array([2, 3, 4])) * U.second
    assert_quantity(result_q, expected_q.value, U.second)


class TestMathFuncsKeepUnitUnary(unittest.TestCase):

  def test_interp(self):
    x = jnp.array([1, 2, 3])
    xp = jnp.array([0, 1, 2, 3, 4])
    fp = jnp.array([0, 1, 2, 3, 4])
    result = bm.interp(x, xp, fp)
    self.assertTrue(jnp.all(result == jnp.interp(x, xp, fp)))

    x = [1, 2, 3] * U.second
    xp = [0, 1, 2, 3, 4] * U.second
    fp = [0, 1, 2, 3, 4] * U.second
    result_q = bm.interp(x, xp, fp)
    expected_q = jnp.interp(jnp.array([1, 2, 3]), jnp.array([0, 1, 2, 3, 4]), jnp.array([0, 1, 2, 3, 4])) * U.second
    assert_quantity(result_q, expected_q.value, U.second)

  def test_clip(self):
    array = jnp.array([1, 2, 3, 4, 5])
    result = bm.clip(array, 2, 4)
    self.assertTrue(jnp.all(result == jnp.clip(array, 2, 4)))

    q = [1, 2, 3, 4, 5] * ms
    result_q = bm.clip(q, 2 * ms, 4 * ms)
    expected_q = jnp.clip(jnp.array([1, 2, 3, 4, 5]), 2, 4) * ms
    assert_quantity(result_q, expected_q.value, ms)


class TestMathFuncsMatchUnitBinary(unittest.TestCase):

  def test_add(self):
    result = bm.add(jnp.array([1, 2]), jnp.array([3, 4]))
    self.assertTrue(jnp.all(result == jnp.add(jnp.array([1, 2]), jnp.array([3, 4]))))

    q1 = [1, 2] * ms
    q2 = [3, 4] * ms
    result_q = bm.add(q1, q2)
    expected_q = jnp.add(jnp.array([1, 2]), jnp.array([3, 4])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_subtract(self):
    result = bm.subtract(jnp.array([5, 6]), jnp.array([3, 2]))
    self.assertTrue(jnp.all(result == jnp.subtract(jnp.array([5, 6]), jnp.array([3, 2]))))

    q1 = [5, 6] * ms
    q2 = [3, 2] * ms
    result_q = bm.subtract(q1, q2)
    expected_q = jnp.subtract(jnp.array([5, 6]), jnp.array([3, 2])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_nextafter(self):
    result = bm.nextafter(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))
    self.assertTrue(jnp.all(result == jnp.nextafter(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))


class TestMathFuncsChangeUnitUnary(unittest.TestCase):

  def test_reciprocal(self):
    array = jnp.array([1.0, 2.0, 0.5])
    result = bm.reciprocal(array)
    self.assertTrue(jnp.all(result == jnp.reciprocal(array)))

    q = [1.0, 2.0, 0.5] * U.second
    result_q = bm.reciprocal(q)
    expected_q = jnp.reciprocal(jnp.array([1.0, 2.0, 0.5])) * (1 / U.second)
    assert_quantity(result_q, expected_q.value, 1 / U.second)

  def test_prod(self):
    array = jnp.array([1, 2, 3])
    result = bm.prod(array)
    self.assertTrue(result == jnp.prod(array))

    q = [1, 2, 3] * ms
    result_q = bm.prod(q)
    expected_q = jnp.prod(jnp.array([1, 2, 3])) * (ms ** 3)
    assert_quantity(result_q, expected_q.value, ms ** 3)

  def test_nanprod(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bm.nanprod(array)
    self.assertTrue(result == jnp.nanprod(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bm.nanprod(q)
    expected_q = jnp.nanprod(jnp.array([1, jnp.nan, 3])) * (ms ** 2)
    assert_quantity(result_q, expected_q.value, ms ** 2)

  def test_cumprod(self):
    array = jnp.array([1, 2, 3])
    result = bm.cumprod(array)
    self.assertTrue(jnp.all(result == jnp.cumprod(array)))

    q = [1, 2, 3] * U.second
    result_q = bm.cumprod(q)
    expected_q = jnp.cumprod(jnp.array([1, 2, 3])) * (U.second ** 3)
    assert_quantity(result_q, expected_q.value, U.second ** 3)

  def test_nancumprod(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bm.nancumprod(array)
    self.assertTrue(jnp.all(result == jnp.nancumprod(array)))

    q = [1, jnp.nan, 3] * U.second
    result_q = bm.nancumprod(q)
    expected_q = jnp.nancumprod(jnp.array([1, jnp.nan, 3])) * (U.second ** 2)
    assert_quantity(result_q, expected_q.value, U.second ** 2)

  def test_var(self):
    array = jnp.array([1, 2, 3])
    result = bm.var(array)
    self.assertTrue(result == jnp.var(array))

    q = [1, 2, 3] * ms
    result_q = bm.var(q)
    expected_q = jnp.var(jnp.array([1, 2, 3])) * (ms ** 2)
    assert_quantity(result_q, expected_q.value, ms ** 2)

  def test_nanvar(self):
    array = jnp.array([1, jnp.nan, 3])
    result = bm.nanvar(array)
    self.assertTrue(result == jnp.nanvar(array))

    q = [1, jnp.nan, 3] * ms
    result_q = bm.nanvar(q)
    expected_q = jnp.nanvar(jnp.array([1, jnp.nan, 3])) * (ms ** 2)
    assert_quantity(result_q, expected_q.value, ms ** 2)

  def test_frexp(self):
    result = bm.frexp(jnp.array([1.0, 2.0]))
    expected = jnp.frexp(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result[0] == expected[0]) and jnp.all(result[1] == expected[1]))

  def test_sqrt(self):
    result = bm.sqrt(jnp.array([1.0, 4.0]))
    self.assertTrue(jnp.all(result == jnp.sqrt(jnp.array([1.0, 4.0]))))

    q = [1.0, 4.0] * (ms ** 2)
    result_q = bm.sqrt(q)
    expected_q = jnp.sqrt(jnp.array([1.0, 4.0])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_cbrt(self):
    result = bm.cbrt(jnp.array([1.0, 8.0]))
    self.assertTrue(jnp.all(result == jnp.cbrt(jnp.array([1.0, 8.0]))))

    q = [1.0, 8.0] * (ms ** 3)
    result_q = bm.cbrt(q)
    expected_q = jnp.cbrt(jnp.array([1.0, 8.0])) * ms
    assert_quantity(result_q, expected_q.value, ms)

  def test_square(self):
    result = bm.square(jnp.array([2.0, 3.0]))
    self.assertTrue(jnp.all(result == jnp.square(jnp.array([2.0, 3.0]))))

    q = [2.0, 3.0] * ms
    result_q = bm.square(q)
    expected_q = jnp.square(jnp.array([2.0, 3.0])) * (ms ** 2)
    assert_quantity(result_q, expected_q.value, ms ** 2)


class TestMathFuncsChangeUnitBinary(unittest.TestCase):

  def test_multiply(self):
    result = bm.multiply(jnp.array([1, 2]), jnp.array([3, 4]))
    self.assertTrue(jnp.all(result == jnp.multiply(jnp.array([1, 2]), jnp.array([3, 4]))))

    q1 = [1, 2] * ms
    q2 = [3, 4] * mV
    result_q = bm.multiply(q1, q2)
    expected_q = jnp.multiply(jnp.array([1, 2]), jnp.array([3, 4])) * (ms * mV)
    assert_quantity(result_q, expected_q.value, ms * mV)

  def test_divide(self):
    result = bm.divide(jnp.array([5, 6]), jnp.array([3, 2]))
    self.assertTrue(jnp.all(result == jnp.divide(jnp.array([5, 6]), jnp.array([3, 2]))))

    q1 = [5, 6] * ms
    q2 = [3, 2] * mV
    result_q = bm.divide(q1, q2)
    expected_q = jnp.divide(jnp.array([5, 6]), jnp.array([3, 2])) * (ms / mV)
    assert_quantity(result_q, expected_q.value, ms / mV)

  def test_power(self):
    result = bm.power(jnp.array([1, 2]), jnp.array([3, 2]))
    self.assertTrue(jnp.all(result == jnp.power(jnp.array([1, 2]), jnp.array([3, 2]))))

    q1 = [1, 2] * ms
    result_q = bm.power(q1, 2)
    expected_q = jnp.power(jnp.array([1, 2]), 2) * (ms ** 2)
    assert_quantity(result_q, expected_q.value, ms ** 2)

  def test_cross(self):
    result = bm.cross(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
    self.assertTrue(jnp.all(result == jnp.cross(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))))

  def test_ldexp(self):
    result = bm.ldexp(jnp.array([1.0, 2.0]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.ldexp(jnp.array([1.0, 2.0]), jnp.array([2, 3]))))

  def test_true_divide(self):
    result = bm.true_divide(jnp.array([5, 6]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.true_divide(jnp.array([5, 6]), jnp.array([2, 3]))))

    q1 = [5, 6] * ms
    q2 = [2, 3] * mV
    result_q = bm.true_divide(q1, q2)
    expected_q = jnp.true_divide(jnp.array([5, 6]), jnp.array([2, 3])) * (ms / mV)
    assert_quantity(result_q, expected_q.value, ms / mV)

  def test_floor_divide(self):
    result = bm.floor_divide(jnp.array([5, 6]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.floor_divide(jnp.array([5, 6]), jnp.array([2, 3]))))

    q1 = [5, 6] * ms
    q2 = [2, 3] * mV
    result_q = bm.floor_divide(q1, q2)
    expected_q = jnp.floor_divide(jnp.array([5, 6]), jnp.array([2, 3])) * (ms / mV)
    assert_quantity(result_q, expected_q.value, ms / mV)

  def test_float_power(self):
    result = bm.float_power(jnp.array([2, 3]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.float_power(jnp.array([2, 3]), jnp.array([2, 3]))))

    q1 = [2, 3] * ms
    result_q = bm.float_power(q1, 2)
    expected_q = jnp.float_power(jnp.array([2, 3]), 2) * (ms ** 2)
    assert_quantity(result_q, expected_q.value, ms ** 2)

  def test_divmod(self):
    result = bm.divmod(jnp.array([5, 6]), jnp.array([2, 3]))
    expected = jnp.divmod(jnp.array([5, 6]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result[0] == expected[0]) and jnp.all(result[1] == expected[1]))

  def test_remainder(self):
    result = bm.remainder(jnp.array([5, 7]), jnp.array([2, 3]))
    self.assertTrue(jnp.all(result == jnp.remainder(jnp.array([5, 7]), jnp.array([2, 3]))))

    q1 = [5, 7] * (U.second ** 2)
    q2 = [2, 3] * U.second
    result_q = bm.remainder(q1, q2)
    expected_q = jnp.remainder(jnp.array([5, 7]), jnp.array([2, 3])) * U.second
    assert_quantity(result_q, expected_q.value, U.second)

  def test_convolve(self):
    result = bm.convolve(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
    self.assertTrue(jnp.all(result == jnp.convolve(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))))


class TestMathFuncsOnlyAcceptUnitlessUnary(unittest.TestCase):

  def test_exp(self):
    result = bm.exp(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.exp(jnp.array([1.0, 2.0]))))

    result = bm.exp(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.exp(jnp.array([1.0, 2.0]))))

  def test_exp2(self):
    result = bm.exp2(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.exp2(jnp.array([1.0, 2.0]))))

    result = bm.exp2(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.exp2(jnp.array([1.0, 2.0]))))

  def test_expm1(self):
    result = bm.expm1(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.expm1(jnp.array([1.0, 2.0]))))

    result = bm.expm1(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.expm1(jnp.array([1.0, 2.0]))))

  def test_log(self):
    result = bm.log(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.log(jnp.array([1.0, 2.0]))))

    result = bm.log(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.log(jnp.array([1.0, 2.0]))))

  def test_log10(self):
    result = bm.log10(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.log10(jnp.array([1.0, 2.0]))))

    result = bm.log10(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.log10(jnp.array([1.0, 2.0]))))

  def test_log1p(self):
    result = bm.log1p(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.log1p(jnp.array([1.0, 2.0]))))

    result = bm.log1p(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.log1p(jnp.array([1.0, 2.0]))))

  def test_log2(self):
    result = bm.log2(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.log2(jnp.array([1.0, 2.0]))))

    result = bm.log2(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.log2(jnp.array([1.0, 2.0]))))

  def test_arccos(self):
    result = bm.arccos(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.arccos(jnp.array([0.5, 1.0]))))

    result = bm.arccos(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.arccos(jnp.array([0.5, 1.0]))))

  def test_arccosh(self):
    result = bm.arccosh(jnp.array([1.0, 2.0]))
    self.assertTrue(jnp.all(result == jnp.arccosh(jnp.array([1.0, 2.0]))))

    result = bm.arccosh(Quantity(jnp.array([1.0, 2.0])))
    self.assertTrue(jnp.all(result == jnp.arccosh(jnp.array([1.0, 2.0]))))

  def test_arcsin(self):
    result = bm.arcsin(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.arcsin(jnp.array([0.5, 1.0]))))

    result = bm.arcsin(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.arcsin(jnp.array([0.5, 1.0]))))

  def test_arcsinh(self):
    result = bm.arcsinh(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.arcsinh(jnp.array([0.5, 1.0]))))

    result = bm.arcsinh(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.arcsinh(jnp.array([0.5, 1.0]))))

  def test_arctan(self):
    result = bm.arctan(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.arctan(jnp.array([0.5, 1.0]))))

    result = bm.arctan(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.arctan(jnp.array([0.5, 1.0]))))

  def test_arctanh(self):
    result = bm.arctanh(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.arctanh(jnp.array([0.5, 1.0]))))

    result = bm.arctanh(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.arctanh(jnp.array([0.5, 1.0]))))

  def test_cos(self):
    result = bm.cos(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.cos(jnp.array([0.5, 1.0]))))

    result = bm.cos(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.cos(jnp.array([0.5, 1.0]))))

  def test_cosh(self):
    result = bm.cosh(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.cosh(jnp.array([0.5, 1.0]))))

    result = bm.cosh(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.cosh(jnp.array([0.5, 1.0]))))

  def test_sin(self):
    result = bm.sin(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.sin(jnp.array([0.5, 1.0]))))

    result = bm.sin(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.sin(jnp.array([0.5, 1.0]))))

  def test_sinc(self):
    result = bm.sinc(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.sinc(jnp.array([0.5, 1.0]))))

    result = bm.sinc(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.sinc(jnp.array([0.5, 1.0]))))

  def test_sinh(self):
    result = bm.sinh(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.sinh(jnp.array([0.5, 1.0]))))

    result = bm.sinh(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.sinh(jnp.array([0.5, 1.0]))))

  def test_tan(self):
    result = bm.tan(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.tan(jnp.array([0.5, 1.0]))))

    result = bm.tan(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.tan(jnp.array([0.5, 1.0]))))

  def test_tanh(self):
    result = bm.tanh(jnp.array([0.5, 1.0]))
    self.assertTrue(jnp.all(result == jnp.tanh(jnp.array([0.5, 1.0]))))

    result = bm.tanh(Quantity(jnp.array([0.5, 1.0])))
    self.assertTrue(jnp.all(result == jnp.tanh(jnp.array([0.5, 1.0]))))

  def test_deg2rad(self):
    result = bm.deg2rad(jnp.array([90.0, 180.0]))
    self.assertTrue(jnp.all(result == jnp.deg2rad(jnp.array([90.0, 180.0]))))

    result = bm.deg2rad(Quantity(jnp.array([90.0, 180.0])))
    self.assertTrue(jnp.all(result == jnp.deg2rad(jnp.array([90.0, 180.0]))))

  def test_rad2deg(self):
    result = bm.rad2deg(jnp.array([jnp.pi / 2, jnp.pi]))
    self.assertTrue(jnp.all(result == jnp.rad2deg(jnp.array([jnp.pi / 2, jnp.pi]))))

    result = bm.rad2deg(Quantity(jnp.array([jnp.pi / 2, jnp.pi])))
    self.assertTrue(jnp.all(result == jnp.rad2deg(jnp.array([jnp.pi / 2, jnp.pi]))))

  def test_degrees(self):
    result = bm.degrees(jnp.array([jnp.pi / 2, jnp.pi]))
    self.assertTrue(jnp.all(result == jnp.degrees(jnp.array([jnp.pi / 2, jnp.pi]))))

    result = bm.degrees(Quantity(jnp.array([jnp.pi / 2, jnp.pi])))
    self.assertTrue(jnp.all(result == jnp.degrees(jnp.array([jnp.pi / 2, jnp.pi]))))

  def test_radians(self):
    result = bm.radians(jnp.array([90.0, 180.0]))
    self.assertTrue(jnp.all(result == jnp.radians(jnp.array([90.0, 180.0]))))

    result = bm.radians(Quantity(jnp.array([90.0, 180.0])))
    self.assertTrue(jnp.all(result == jnp.radians(jnp.array([90.0, 180.0]))))

  def test_angle(self):
    result = bm.angle(jnp.array([1.0 + 1.0j, 1.0 - 1.0j]))
    self.assertTrue(jnp.all(result == jnp.angle(jnp.array([1.0 + 1.0j, 1.0 - 1.0j]))))

    result = bm.angle(Quantity(jnp.array([1.0 + 1.0j, 1.0 - 1.0j])))
    self.assertTrue(jnp.all(result == jnp.angle(jnp.array([1.0 + 1.0j, 1.0 - 1.0j]))))

  def test_percentile(self):
    array = jnp.array([1, 2, 3, 4])
    result = bm.percentile(array, 50)
    self.assertTrue(result == jnp.percentile(array, 50))

  def test_nanpercentile(self):
    array = jnp.array([1, jnp.nan, 3, 4])
    result = bm.nanpercentile(array, 50)
    self.assertTrue(result == jnp.nanpercentile(array, 50))

  def test_quantile(self):
    array = jnp.array([1, 2, 3, 4])
    result = bm.quantile(array, 0.5)
    self.assertTrue(result == jnp.quantile(array, 0.5))

  def test_nanquantile(self):
    array = jnp.array([1, jnp.nan, 3, 4])
    result = bm.nanquantile(array, 0.5)
    self.assertTrue(result == jnp.nanquantile(array, 0.5))


class TestMathFuncsOnlyAcceptUnitlessBinary(unittest.TestCase):

  def test_hypot(self):
    result = bm.hypot(jnp.array([3.0, 4.0]), jnp.array([4.0, 3.0]))
    self.assertTrue(jnp.all(result == jnp.hypot(jnp.array([3.0, 4.0]), jnp.array([4.0, 3.0]))))

    result = bm.hypot(Quantity(jnp.array([3.0, 4.0])), Quantity(jnp.array([4.0, 3.0])))
    self.assertTrue(jnp.all(result == jnp.hypot(jnp.array([3.0, 4.0]), jnp.array([4.0, 3.0]))))

  def test_arctan2(self):
    result = bm.arctan2(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))
    self.assertTrue(jnp.all(result == jnp.arctan2(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))

    result = bm.arctan2(Quantity(jnp.array([1.0, 2.0])), Quantity(jnp.array([2.0, 3.0])))
    self.assertTrue(jnp.all(result == jnp.arctan2(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))

  def test_logaddexp(self):
    result = bm.logaddexp(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))
    self.assertTrue(jnp.all(result == jnp.logaddexp(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))

    result = bm.logaddexp(Quantity(jnp.array([1.0, 2.0])), Quantity(jnp.array([2.0, 3.0])))
    self.assertTrue(jnp.all(result == jnp.logaddexp(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))

  def test_logaddexp2(self):
    result = bm.logaddexp2(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))
    self.assertTrue(jnp.all(result == jnp.logaddexp2(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))

    result = bm.logaddexp2(Quantity(jnp.array([1.0, 2.0])), Quantity(jnp.array([2.0, 3.0])))
    self.assertTrue(jnp.all(result == jnp.logaddexp2(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))))


class TestMathFuncsRemoveUnitUnary(unittest.TestCase):

  def test_signbit(self):
    array = jnp.array([-1.0, 2.0])
    result = bm.signbit(array)
    self.assertTrue(jnp.all(result == jnp.signbit(array)))

    q = [-1.0, 2.0] * U.second
    result_q = bm.signbit(q)
    expected_q = jnp.signbit(jnp.array([-1.0, 2.0]))
    assert_quantity(result_q, expected_q, None)

  def test_sign(self):
    array = jnp.array([-1.0, 2.0])
    result = bm.sign(array)
    self.assertTrue(jnp.all(result == jnp.sign(array)))

    q = [-1.0, 2.0] * U.second
    result_q = bm.sign(q)
    expected_q = jnp.sign(jnp.array([-1.0, 2.0]))
    assert_quantity(result_q, expected_q, None)

  def test_histogram(self):
    array = jnp.array([1, 2, 1])
    result, _ = bm.histogram(array)
    expected, _ = jnp.histogram(array)
    self.assertTrue(jnp.all(result == expected))

    q = [1, 2, 1] * U.second
    result_q, _ = bm.histogram(q)
    expected_q, _ = jnp.histogram(jnp.array([1, 2, 1]))
    assert_quantity(result_q, expected_q, None)

  def test_bincount(self):
    array = jnp.array([1, 1, 2, 2, 2, 3])
    result = bm.bincount(array)
    self.assertTrue(jnp.all(result == jnp.bincount(array)))

    q = [1, 1, 2, 2, 2, 3] * U.second
    q = q.astype(jnp.int64)
    result_q = bm.bincount(q)
    expected_q = jnp.bincount(jnp.array([1, 1, 2, 2, 2, 3]))
    assert_quantity(result_q, expected_q, None)


class TestMathFuncsRemoveUnitBinary(unittest.TestCase):

  def test_corrcoef(self):
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    result = bm.corrcoef(x, y)
    self.assertTrue(jnp.all(result == jnp.corrcoef(x, y)))

    x = [1, 2, 3] * U.second
    y = [4, 5, 6] * U.second
    result = bm.corrcoef(x, y)
    expected = jnp.corrcoef(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
    assert_quantity(result, expected, None)

  def test_correlate(self):
    x = jnp.array([1, 2, 3])
    y = jnp.array([0, 1, 0.5])
    result = bm.correlate(x, y)
    self.assertTrue(jnp.all(result == jnp.correlate(x, y)))

    x = [1, 2, 3] * U.second
    y = [0, 1, 0.5] * U.second
    result = bm.correlate(x, y)
    expected = jnp.correlate(jnp.array([1, 2, 3]), jnp.array([0, 1, 0.5]))
    assert_quantity(result, expected, None)

  def test_cov(self):
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    result = bm.cov(x, y)
    self.assertTrue(jnp.all(result == jnp.cov(x, y)))

    x = [1, 2, 3] * U.second
    y = [4, 5, 6] * U.second
    result = bm.cov(x, y)
    expected = jnp.cov(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
    assert_quantity(result, expected, None)

  def test_digitize(self):
    array = jnp.array([0.2, 6.4, 3.0, 1.6])
    bins = jnp.array([0.0, 1.0, 2.5, 4.0, 10.0])
    result = bm.digitize(array, bins)
    self.assertTrue(jnp.all(result == jnp.digitize(array, bins)))

    array = [0.2, 6.4, 3.0, 1.6] * U.second
    bins = [0.0, 1.0, 2.5, 4.0, 10.0] * U.second
    result = bm.digitize(array, bins)
    expected = jnp.digitize(jnp.array([0.2, 6.4, 3.0, 1.6]), jnp.array([0.0, 1.0, 2.5, 4.0, 10.0]))
    assert_quantity(result, expected, None)


class TestArrayManipulation(unittest.TestCase):

  def test_reshape(self):
    array = jnp.array([1, 2, 3, 4])
    result = bm.reshape(array, (2, 2))
    self.assertTrue(jnp.all(result == jnp.reshape(array, (2, 2))))

    q = [1, 2, 3, 4] * U.second
    result_q = bm.reshape(q, (2, 2))
    expected_q = jnp.reshape(jnp.array([1, 2, 3, 4]), (2, 2))
    assert_quantity(result_q, expected_q, U.second)

  def test_moveaxis(self):
    array = jnp.zeros((3, 4, 5))
    result = bm.moveaxis(array, 0, -1)
    self.assertTrue(jnp.all(result == jnp.moveaxis(array, 0, -1)))

    q = jnp.zeros((3, 4, 5)) * U.second
    result_q = bm.moveaxis(q, 0, -1)
    expected_q = jnp.moveaxis(jnp.zeros((3, 4, 5)), 0, -1)
    assert_quantity(result_q, expected_q, U.second)

  def test_transpose(self):
    array = jnp.ones((2, 3))
    result = bm.transpose(array)
    self.assertTrue(jnp.all(result == jnp.transpose(array)))

    q = jnp.ones((2, 3)) * U.second
    result_q = bm.transpose(q)
    expected_q = jnp.transpose(jnp.ones((2, 3)))
    assert_quantity(result_q, expected_q, U.second)

  def test_swapaxes(self):
    array = jnp.zeros((3, 4, 5))
    result = bm.swapaxes(array, 0, 2)
    self.assertTrue(jnp.all(result == jnp.swapaxes(array, 0, 2)))

    q = jnp.zeros((3, 4, 5)) * U.second
    result_q = bm.swapaxes(q, 0, 2)
    expected_q = jnp.swapaxes(jnp.zeros((3, 4, 5)), 0, 2)
    assert_quantity(result_q, expected_q, U.second)

  def test_row_stack(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    result = bm.row_stack((a, b))
    self.assertTrue(jnp.all(result == jnp.vstack((a, b))))

    q1 = [1, 2, 3] * U.second
    q2 = [4, 5, 6] * U.second
    result_q = bm.row_stack((q1, q2))
    expected_q = jnp.vstack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
    assert_quantity(result_q, expected_q, U.second)

  def test_concatenate(self):
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[5, 6]])
    result = bm.concatenate((a, b), axis=0)
    self.assertTrue(jnp.all(result == jnp.concatenate((a, b), axis=0)))

    q1 = [[1, 2], [3, 4]] * U.second
    q2 = [[5, 6]] * U.second
    result_q = bm.concatenate((q1, q2), axis=0)
    expected_q = jnp.concatenate((jnp.array([[1, 2], [3, 4]]), jnp.array([[5, 6]])), axis=0)
    assert_quantity(result_q, expected_q, U.second)

  def test_stack(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    result = bm.stack((a, b), axis=1)
    self.assertTrue(jnp.all(result == jnp.stack((a, b), axis=1)))

    q1 = [1, 2, 3] * U.second
    q2 = [4, 5, 6] * U.second
    result_q = bm.stack((q1, q2), axis=1)
    expected_q = jnp.stack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])), axis=1)
    assert_quantity(result_q, expected_q, U.second)

  def test_vstack(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    result = bm.vstack((a, b))
    self.assertTrue(jnp.all(result == jnp.vstack((a, b))))

    q1 = [1, 2, 3] * U.second
    q2 = [4, 5, 6] * U.second
    result_q = bm.vstack((q1, q2))
    expected_q = jnp.vstack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
    assert_quantity(result_q, expected_q, U.second)

  def test_hstack(self):
    a = jnp.array((1, 2, 3))
    b = jnp.array((4, 5, 6))
    result = bm.hstack((a, b))
    self.assertTrue(jnp.all(result == jnp.hstack((a, b))))

    q1 = [1, 2, 3] * U.second
    q2 = [4, 5, 6] * U.second
    result_q = bm.hstack((q1, q2))
    expected_q = jnp.hstack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
    assert_quantity(result_q, expected_q, U.second)

  def test_dstack(self):
    a = jnp.array([[1], [2], [3]])
    b = jnp.array([[4], [5], [6]])
    result = bm.dstack((a, b))
    self.assertTrue(jnp.all(result == jnp.dstack((a, b))))

    q1 = [[1], [2], [3]] * U.second
    q2 = [[4], [5], [6]] * U.second
    result_q = bm.dstack((q1, q2))
    expected_q = jnp.dstack((jnp.array([[1], [2], [3]]), jnp.array([[4], [5], [6]])))
    assert_quantity(result_q, expected_q, U.second)

  def test_column_stack(self):
    a = jnp.array((1, 2, 3))
    b = jnp.array((4, 5, 6))
    result = bm.column_stack((a, b))
    self.assertTrue(jnp.all(result == jnp.column_stack((a, b))))

    q1 = [1, 2, 3] * U.second
    q2 = [4, 5, 6] * U.second
    result_q = bm.column_stack((q1, q2))
    expected_q = jnp.column_stack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
    assert_quantity(result_q, expected_q, U.second)

  def test_split(self):
    array = jnp.arange(9)
    result = bm.split(array, 3)
    expected = jnp.split(array, 3)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.all(r == e))

    q = jnp.arange(9) * U.second
    result_q = bm.split(q, 3)
    expected_q = jnp.split(jnp.arange(9), 3)
    for r, e in zip(result_q, expected_q):
      assert_quantity(r, e, ms)

  def test_dsplit(self):
    array = jnp.arange(16.0).reshape(2, 2, 4)
    result = bm.dsplit(array, 2)
    expected = jnp.dsplit(array, 2)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.all(r == e))

    q = jnp.arange(16.0).reshape(2, 2, 4) * U.second
    result_q = bm.dsplit(q, 2)
    expected_q = jnp.dsplit(jnp.arange(16.0).reshape(2, 2, 4), 2)
    for r, e in zip(result_q, expected_q):
      assert_quantity(r, e, U.second)

  def test_hsplit(self):
    array = jnp.arange(16.0).reshape(4, 4)
    result = bm.hsplit(array, 2)
    expected = jnp.hsplit(array, 2)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.all(r == e))

    q = jnp.arange(16.0).reshape(4, 4) * U.second
    result_q = bm.hsplit(q, 2)
    expected_q = jnp.hsplit(jnp.arange(16.0).reshape(4, 4), 2)
    for r, e in zip(result_q, expected_q):
      assert_quantity(r, e, U.second)

  def test_vsplit(self):
    array = jnp.arange(16.0).reshape(4, 4)
    result = bm.vsplit(array, 2)
    expected = jnp.vsplit(array, 2)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.all(r == e))

    q = jnp.arange(16.0).reshape(4, 4) * U.second
    result_q = bm.vsplit(q, 2)
    expected_q = jnp.vsplit(jnp.arange(16.0).reshape(4, 4), 2)
    for r, e in zip(result_q, expected_q):
      assert_quantity(r, e, U.second)

  def test_tile(self):
    array = jnp.array([0, 1, 2])
    result = bm.tile(array, 2)
    self.assertTrue(jnp.all(result == jnp.tile(array, 2)))

    q = jnp.array([0, 1, 2]) * U.second
    result_q = bm.tile(q, 2)
    expected_q = jnp.tile(jnp.array([0, 1, 2]), 2)
    assert_quantity(result_q, expected_q, U.second)

  def test_repeat(self):
    array = jnp.array([0, 1, 2])
    result = bm.repeat(array, 2)
    self.assertTrue(jnp.all(result == jnp.repeat(array, 2)))

    q = [0, 1, 2] * U.second
    result_q = bm.repeat(q, 2)
    expected_q = jnp.repeat(jnp.array([0, 1, 2]), 2)
    assert_quantity(result_q, expected_q, U.second)

  def test_unique(self):
    array = jnp.array([0, 1, 2, 1, 0])
    result = bm.unique(array)
    self.assertTrue(jnp.all(result == jnp.unique(array)))

    q = [0, 1, 2, 1, 0] * U.second
    result_q = bm.unique(q)
    expected_q = jnp.unique(jnp.array([0, 1, 2, 1, 0]))
    assert_quantity(result_q, expected_q, U.second)

  def test_append(self):
    array = jnp.array([0, 1, 2])
    result = bm.append(array, 3)
    self.assertTrue(jnp.all(result == jnp.append(array, 3)))

    q = [0, 1, 2] * U.second
    result_q = bm.append(q, 3)
    expected_q = jnp.append(jnp.array([0, 1, 2]), 3)
    assert_quantity(result_q, expected_q, U.second)

  def test_flip(self):
    array = jnp.array([0, 1, 2])
    result = bm.flip(array)
    self.assertTrue(jnp.all(result == jnp.flip(array)))

    q = [0, 1, 2] * U.second
    result_q = bm.flip(q)
    expected_q = jnp.flip(jnp.array([0, 1, 2]))
    assert_quantity(result_q, expected_q, U.second)

  def test_fliplr(self):
    array = jnp.array([[0, 1, 2], [3, 4, 5]])
    result = bm.fliplr(array)
    self.assertTrue(jnp.all(result == jnp.fliplr(array)))

    q = [[0, 1, 2], [3, 4, 5]] * U.second
    result_q = bm.fliplr(q)
    expected_q = jnp.fliplr(jnp.array([[0, 1, 2], [3, 4, 5]]))
    assert_quantity(result_q, expected_q, U.second)

  def test_flipud(self):
    array = jnp.array([[0, 1, 2], [3, 4, 5]])
    result = bm.flipud(array)
    self.assertTrue(jnp.all(result == jnp.flipud(array)))

    q = [[0, 1, 2], [3, 4, 5]] * U.second
    result_q = bm.flipud(q)
    expected_q = jnp.flipud(jnp.array([[0, 1, 2], [3, 4, 5]]))
    assert_quantity(result_q, expected_q, ms)

  def test_roll(self):
    array = jnp.array([0, 1, 2])
    result = bm.roll(array, 1)
    self.assertTrue(jnp.all(result == jnp.roll(array, 1)))

    q = [0, 1, 2] * U.second
    result_q = bm.roll(q, 1)
    expected_q = jnp.roll(jnp.array([0, 1, 2]), 1)
    assert_quantity(result_q, expected_q, ms)

  def test_atleast_1d(self):
    array = jnp.array(0)
    result = bm.atleast_1d(array)
    self.assertTrue(jnp.all(result == jnp.atleast_1d(array)))

    q = 0 * U.second
    result_q = bm.atleast_1d(q)
    expected_q = jnp.atleast_1d(jnp.array(0))
    assert_quantity(result_q, expected_q, U.second)

  def test_atleast_2d(self):
    array = jnp.array([0, 1, 2])
    result = bm.atleast_2d(array)
    self.assertTrue(jnp.all(result == jnp.atleast_2d(array)))

    q = [0, 1, 2] * U.second
    result_q = bm.atleast_2d(q)
    expected_q = jnp.atleast_2d(jnp.array([0, 1, 2]))
    assert_quantity(result_q, expected_q, U.second)

  def test_atleast_3d(self):
    array = jnp.array([[0, 1, 2], [3, 4, 5]])
    result = bm.atleast_3d(array)
    self.assertTrue(jnp.all(result == jnp.atleast_3d(array)))

    q = [[0, 1, 2], [3, 4, 5]] * U.second
    result_q = bm.atleast_3d(q)
    expected_q = jnp.atleast_3d(jnp.array([[0, 1, 2], [3, 4, 5]]))
    assert_quantity(result_q, expected_q, U.second)

  def test_expand_dims(self):
    array = jnp.array([1, 2, 3])
    result = bm.expand_dims(array, axis=0)
    self.assertTrue(jnp.all(result == jnp.expand_dims(array, axis=0)))

    q = [1, 2, 3] * U.second
    result_q = bm.expand_dims(q, axis=0)
    expected_q = jnp.expand_dims(jnp.array([1, 2, 3]), axis=0)
    assert_quantity(result_q, expected_q, U.second)

  def test_squeeze(self):
    array = jnp.array([[[0], [1], [2]]])
    result = bm.squeeze(array)
    self.assertTrue(jnp.all(result == jnp.squeeze(array)))

    q = [[[0], [1], [2]]] * U.second
    result_q = bm.squeeze(q)
    expected_q = jnp.squeeze(jnp.array([[[0], [1], [2]]]))
    assert_quantity(result_q, expected_q, U.second)

  def test_sort(self):
    array = jnp.array([2, 3, 1])
    result = bm.sort(array)
    self.assertTrue(jnp.all(result == jnp.sort(array)))

    q = [2, 3, 1] * U.second
    result_q = bm.sort(q)
    expected_q = jnp.sort(jnp.array([2, 3, 1]))
    assert_quantity(result_q, expected_q, U.second)

  def test_max(self):
    array = jnp.array([1, 2, 3])
    result = bm.max(array)
    self.assertTrue(result == jnp.max(array))

    q = [1, 2, 3] * U.second
    result_q = bm.max(q)
    expected_q = jnp.max(jnp.array([1, 2, 3]))
    assert_quantity(result_q, expected_q, U.second)

  def test_min(self):
    array = jnp.array([1, 2, 3])
    result = bm.min(array)
    self.assertTrue(result == jnp.min(array))

    q = [1, 2, 3] * U.second
    result_q = bm.min(q)
    expected_q = jnp.min(jnp.array([1, 2, 3]))
    assert_quantity(result_q, expected_q, U.second)

  def test_amin(self):
    array = jnp.array([1, 2, 3])
    result = bm.amin(array)
    self.assertTrue(result == jnp.min(array))

    q = [1, 2, 3] * U.second
    result_q = bm.amin(q)
    expected_q = jnp.min(jnp.array([1, 2, 3]))
    assert_quantity(result_q, expected_q, U.second)

  def test_amax(self):
    array = jnp.array([1, 2, 3])
    result = bm.amax(array)
    self.assertTrue(result == jnp.max(array))

    q = [1, 2, 3] * U.second
    result_q = bm.amax(q)
    expected_q = jnp.max(jnp.array([1, 2, 3]))
    assert_quantity(result_q, expected_q, U.second)

  def test_block(self):
    array = jnp.array([[1, 2], [3, 4]])
    result = bm.block(array)
    self.assertTrue(jnp.all(result == jnp.block(array)))

    q = [[1, 2], [3, 4]] * U.second
    result_q = bm.block(q)
    expected_q = jnp.block(jnp.array([[1, 2], [3, 4]]))
    assert_quantity(result_q, expected_q, U.second)

  def test_compress(self):
    array = jnp.array([1, 2, 3, 4])
    result = bm.compress(jnp.array([0, 1, 1, 0]), array)
    self.assertTrue(jnp.all(result == jnp.compress(jnp.array([0, 1, 1, 0]), array)))

    q = [1, 2, 3, 4] * U.second
    a = [0, 1, 1, 0] * U.second
    result_q = bm.compress(q, a)
    expected_q = jnp.compress(jnp.array([1, 2, 3, 4]), jnp.array([0, 1, 1, 0]))
    assert_quantity(result_q, expected_q, U.second)

  def test_diagflat(self):
    array = jnp.array([1, 2, 3])
    result = bm.diagflat(array)
    self.assertTrue(jnp.all(result == jnp.diagflat(array)))

    q = [1, 2, 3] * U.second
    result_q = bm.diagflat(q)
    expected_q = jnp.diagflat(jnp.array([1, 2, 3]))
    assert_quantity(result_q, expected_q, U.second)

  def test_diagonal(self):
    array = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    result = bm.diagonal(array)
    self.assertTrue(jnp.all(result == jnp.diagonal(array)))

    q = [[0, 1, 2], [3, 4, 5], [6, 7, 8]] * U.second
    result_q = bm.diagonal(q)
    expected_q = jnp.diagonal(jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    assert_quantity(result_q, expected_q, U.second)

  def test_choose(self):
    choices = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6]), jnp.array([7, 8, 9])]
    result = bm.choose(jnp.array([0, 1, 2]), choices)
    self.assertTrue(jnp.all(result == jnp.choose(jnp.array([0, 1, 2]), choices)))

    q = [0, 1, 2] * U.second
    q = q.astype(jnp.int64)
    result_q = bm.choose(q, choices)
    expected_q = jnp.choose(jnp.array([0, 1, 2]), choices)
    assert_quantity(result_q, expected_q, U.second)

  def test_ravel(self):
    array = jnp.array([[1, 2, 3], [4, 5, 6]])
    result = bm.ravel(array)
    self.assertTrue(jnp.all(result == jnp.ravel(array)))

    q = [[1, 2, 3], [4, 5, 6]] * U.second
    result_q = bm.ravel(q)
    expected_q = jnp.ravel(jnp.array([[1, 2, 3], [4, 5, 6]]))
    assert_quantity(result_q, expected_q, U.second)

  # return_quantity = False
  def test_argsort(self):
    array = jnp.array([2, 3, 1])
    result = bm.argsort(array)
    self.assertTrue(jnp.all(result == jnp.argsort(array)))

    q = [2, 3, 1] * U.second
    result_q = bm.argsort(q)
    expected_q = jnp.argsort(jnp.array([2, 3, 1]))
    assert jnp.all(result_q == expected_q)

  def test_argmax(self):
    array = jnp.array([2, 3, 1])
    result = bm.argmax(array)
    self.assertTrue(result == jnp.argmax(array))

    q = [2, 3, 1] * U.second
    result_q = bm.argmax(q)
    expected_q = jnp.argmax(jnp.array([2, 3, 1]))
    assert result_q == expected_q

  def test_argmin(self):
    array = jnp.array([2, 3, 1])
    result = bm.argmin(array)
    self.assertTrue(result == jnp.argmin(array))

    q = [2, 3, 1] * U.second
    result_q = bm.argmin(q)
    expected_q = jnp.argmin(jnp.array([2, 3, 1]))
    assert result_q == expected_q

  def test_argwhere(self):
    array = jnp.array([0, 1, 2])
    result = bm.argwhere(array)
    self.assertTrue(jnp.all(result == jnp.argwhere(array)))

    q = [0, 1, 2] * U.second
    result_q = bm.argwhere(q)
    expected_q = jnp.argwhere(jnp.array([0, 1, 2]))
    assert jnp.all(result_q == expected_q)

  def test_nonzero(self):
    array = jnp.array([0, 1, 2])
    result = bm.nonzero(array)
    expected = jnp.nonzero(array)
    for r, e in zip(result, expected):
      self.assertTrue(jnp.array_equal(r, e))

    q = [0, 1, 2] * U.second
    result_q = bm.nonzero(q)
    expected_q = jnp.nonzero(jnp.array([0, 1, 2]))
    for r, e in zip(result_q, expected_q):
      assert jnp.all(r == e)

  def test_flatnonzero(self):
    array = jnp.array([0, 1, 2])
    result = bm.flatnonzero(array)
    self.assertTrue(jnp.all(result == jnp.flatnonzero(array)))

    q = [0, 1, 2] * U.second
    result_q = bm.flatnonzero(q)
    expected_q = jnp.flatnonzero(jnp.array([0, 1, 2]))
    assert jnp.all(result_q == expected_q)

  def test_searchsorted(self):
    array = jnp.array([1, 2, 3])
    result = bm.searchsorted(array, 2)
    self.assertTrue(result == jnp.searchsorted(array, 2))

    q = [0, 1, 2] * U.second
    result_q = bm.searchsorted(q, 2)
    expected_q = jnp.searchsorted(jnp.array([0, 1, 2]), 2)
    assert result_q == expected_q

  def test_extract(self):
    array = jnp.array([1, 2, 3])
    result = bm.extract(array > 1, array)
    self.assertTrue(jnp.all(result == jnp.extract(array > 1, array)))

    q = [1, 2, 3] * U.second
    a = array * U.second
    result_q = bm.extract(q > 1 * U.second, a)
    expected_q = jnp.extract(jnp.array([0, 1, 2]), jnp.array([1, 2, 3]))
    assert jnp.all(result_q == expected_q)

  def test_count_nonzero(self):
    array = jnp.array([1, 0, 2, 0, 3, 0])
    result = bm.count_nonzero(array)
    self.assertTrue(result == jnp.count_nonzero(array))

    q = [1, 0, 2, 0, 3, 0] * U.second
    result_q = bm.count_nonzero(q)
    expected_q = jnp.count_nonzero(jnp.array([1, 0, 2, 0, 3, 0]))
    assert result_q == expected_q


class TestElementwiseBitOperationsUnary(unittest.TestCase):
  def test_bitwise_not(self):
    result = bm.bitwise_not(jnp.array([0b1100]))
    self.assertTrue(jnp.all(result == jnp.bitwise_not(jnp.array([0b1100]))))

    with pytest.raises(ValueError):
      q = [0b1100] * U.second
      result_q = bm.bitwise_not(q)

  def test_invert(self):
    result = bm.invert(jnp.array([0b1100]))
    self.assertTrue(jnp.all(result == jnp.invert(jnp.array([0b1100]))))

    with pytest.raises(ValueError):
      q = [0b1100] * U.second
      result_q = bm.invert(q)

  def test_left_shift(self):
    result = bm.left_shift(jnp.array([0b0100]), 2)
    self.assertTrue(jnp.all(result == jnp.left_shift(jnp.array([0b0100]), 2)))

    with pytest.raises(ValueError):
      q = [0b0100] * U.second
      result_q = bm.left_shift(q, 2)

  def test_right_shift(self):
    result = bm.right_shift(jnp.array([0b0100]), 2)
    self.assertTrue(jnp.all(result == jnp.right_shift(jnp.array([0b0100]), 2)))

    with pytest.raises(ValueError):
      q = [0b0100] * U.second
      result_q = bm.right_shift(q, 2)


class TestElementwiseBitOperationsBinary(unittest.TestCase):

  def test_bitwise_and(self):
    result = bm.bitwise_and(jnp.array([0b1100]), jnp.array([0b1010]))
    self.assertTrue(jnp.all(result == jnp.bitwise_and(jnp.array([0b1100]), jnp.array([0b1010]))))

    with pytest.raises(ValueError):
      q1 = [0b1100] * U.second
      q2 = [0b1010] * U.second
      result_q = bm.bitwise_and(q1, q2)

  def test_bitwise_or(self):
    result = bm.bitwise_or(jnp.array([0b1100]), jnp.array([0b1010]))
    self.assertTrue(jnp.all(result == jnp.bitwise_or(jnp.array([0b1100]), jnp.array([0b1010]))))

    with pytest.raises(ValueError):
      q1 = [0b1100] * U.second
      q2 = [0b1010] * U.second
      result_q = bm.bitwise_or(q1, q2)

  def test_bitwise_xor(self):
    result = bm.bitwise_xor(jnp.array([0b1100]), jnp.array([0b1010]))
    self.assertTrue(jnp.all(result == jnp.bitwise_xor(jnp.array([0b1100]), jnp.array([0b1010]))))

    with pytest.raises(ValueError):
      q1 = [0b1100] * U.second
      q2 = [0b1010] * U.second
      result_q = bm.bitwise_xor(q1, q2)


class TestLogicFuncsUnary(unittest.TestCase):
  def test_all(self):
    result = bm.all(jnp.array([True, True, True]))
    self.assertTrue(result == jnp.all(jnp.array([True, True, True])))

    with pytest.raises(ValueError):
      q = [True, True, True] * U.second
      result_q = bm.all(q)

  def test_any(self):
    result = bm.any(jnp.array([False, True, False]))
    self.assertTrue(result == jnp.any(jnp.array([False, True, False])))

    with pytest.raises(ValueError):
      q = [False, True, False] * U.second
      result_q = bm.any(q)

  def test_logical_not(self):
    result = bm.logical_not(jnp.array([True, False]))
    self.assertTrue(jnp.all(result == jnp.logical_not(jnp.array([True, False]))))

    with pytest.raises(ValueError):
      q = [True, False] * U.second
      result_q = bm.logical_not(q)


class TestLogicFuncsBinary(unittest.TestCase):

  def test_equal(self):
    result = bm.equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    self.assertTrue(jnp.all(result == jnp.equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))))

    q1 = [1, 2, 3] * U.second
    q2 = [2, 3, 4] * U.second
    result_q = bm.equal(q1, q2)
    expected_q = jnp.equal(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

    with pytest.raises(DimensionMismatchError):
      q1 = [1, 2, 3] * U.second
      q2 = [1, 2, 4] * U.volt
      result_q = bm.equal(q1, q2)

  def test_not_equal(self):
    result = bm.not_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 4]))
    self.assertTrue(jnp.all(result == jnp.not_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 4]))))

    q1 = [1, 2, 3] * U.second
    q2 = [2, 3, 4] * U.second
    result_q = bm.not_equal(q1, q2)
    expected_q = jnp.not_equal(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

  def test_greater(self):
    result = bm.greater(jnp.array([1, 2, 3]), jnp.array([0, 2, 4]))
    self.assertTrue(jnp.all(result == jnp.greater(jnp.array([1, 2, 3]), jnp.array([0, 2, 4]))))

    q1 = [1, 2, 3] * U.second
    q2 = [2, 3, 4] * U.second
    result_q = bm.greater(q1, q2)
    expected_q = jnp.greater(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

  def test_greater_equal(self):
    result = bm.greater_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 2]))
    self.assertTrue(jnp.all(result == jnp.greater_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 2]))))

    q1 = [1, 2, 3] * U.second
    q2 = [2, 3, 4] * U.second
    result_q = bm.greater_equal(q1, q2)
    expected_q = jnp.greater_equal(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

  def test_less(self):
    result = bm.less(jnp.array([1, 2, 3]), jnp.array([2, 2, 2]))
    self.assertTrue(jnp.all(result == jnp.less(jnp.array([1, 2, 3]), jnp.array([2, 2, 2]))))

    q1 = [1, 2, 3] * U.second
    q2 = [2, 3, 4] * U.second
    result_q = bm.less(q1, q2)
    expected_q = jnp.less(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

  def test_less_equal(self):
    result = bm.less_equal(jnp.array([1, 2, 3]), jnp.array([2, 2, 2]))
    self.assertTrue(jnp.all(result == jnp.less_equal(jnp.array([1, 2, 3]), jnp.array([2, 2, 2]))))

    q1 = [1, 2, 3] * U.second
    q2 = [2, 3, 4] * U.second
    result_q = bm.less_equal(q1, q2)
    expected_q = jnp.less_equal(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

  def test_array_equal(self):
    result = bm.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    self.assertTrue(result == jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3])))

    q1 = [1, 2, 3] * U.second
    q2 = [2, 3, 4] * U.second
    result_q = bm.array_equal(q1, q2)
    expected_q = jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert_quantity(result_q, expected_q, None)

  def test_isclose(self):
    result = bm.isclose(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.1]), atol=0.2)
    self.assertTrue(jnp.all(result == jnp.isclose(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.1]), atol=0.2)))

    q1 = [1.0, 2.0] * U.second
    q2 = [2.0, 3.0] * U.second
    result_q = bm.isclose(q1, q2, atol=0.2)
    expected_q = jnp.isclose(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]), atol=0.2)
    assert_quantity(result_q, expected_q, None)

  def test_allclose(self):
    result = bm.allclose(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.1]), atol=0.2)
    self.assertTrue(result == jnp.allclose(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.1]), atol=0.2))

    q1 = [1.0, 2.0] * U.second
    q2 = [2.0, 3.0] * U.second
    result_q = bm.allclose(q1, q2, atol=0.2)
    expected_q = jnp.allclose(jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]), atol=0.2)
    assert_quantity(result_q, expected_q, None)

  def test_logical_and(self):
    result = bm.logical_and(jnp.array([True, False]), jnp.array([False, False]))
    self.assertTrue(jnp.all(result == jnp.logical_and(jnp.array([True, False]), jnp.array([False, False]))))

    q1 = [True, False] * U.second
    q2 = [False, False] * U.second
    result_q = bm.logical_and(q1, q2)
    expected_q = jnp.logical_and(jnp.array([True, False]), jnp.array([False, False]))
    assert_quantity(result_q, expected_q, None)

  def test_logical_or(self):
    result = bm.logical_or(jnp.array([True, False]), jnp.array([False, False]))
    self.assertTrue(jnp.all(result == jnp.logical_or(jnp.array([True, False]), jnp.array([False, False]))))

    q1 = [True, False] * U.second
    q2 = [False, False] * U.second
    result_q = bm.logical_or(q1, q2)
    expected_q = jnp.logical_or(jnp.array([True, False]), jnp.array([False, False]))
    assert_quantity(result_q, expected_q, None)

  def test_logical_xor(self):
    result = bm.logical_xor(jnp.array([True, False]), jnp.array([False, False]))
    self.assertTrue(jnp.all(result == jnp.logical_xor(jnp.array([True, False]), jnp.array([False, False]))))

    q1 = [True, False] * U.second
    q2 = [False, False] * U.second
    result_q = bm.logical_xor(q1, q2)
    expected_q = jnp.logical_xor(jnp.array([True, False]), jnp.array([False, False]))
    assert_quantity(result_q, expected_q, None)


class TestIndexingFuncs(unittest.TestCase):

  def test_where(self):
    array = jnp.array([1, 2, 3, 4, 5])
    result = bm.where(array > 2, array, 0)
    self.assertTrue(jnp.all(result == jnp.where(array > 2, array, 0)))

    q = [1, 2, 3, 4, 5] * U.second
    result_q = bm.where(q > 2 * U.second, q, 0)
    expected_q = jnp.where(jnp.array([1, 2, 3, 4, 5]) > 2, jnp.array([1, 2, 3, 4, 5]), 0)
    assert_quantity(result_q, expected_q, U.second)

  def test_tril_indices(self):
    result = bm.tril_indices(3)
    expected = jnp.tril_indices(3)
    for i in range(2):
      self.assertTrue(jnp.all(result[i] == expected[i]))

  def test_tril_indices_from(self):
    array = jnp.ones((3, 3))
    result = bm.tril_indices_from(array)
    expected = jnp.tril_indices_from(array)
    for i in range(2):
      self.assertTrue(jnp.all(result[i] == expected[i]))

  def test_triu_indices(self):
    result = bm.triu_indices(3)
    expected = jnp.triu_indices(3)
    for i in range(2):
      self.assertTrue(jnp.all(result[i] == expected[i]))

  def test_triu_indices_from(self):
    array = jnp.ones((3, 3))
    result = bm.triu_indices_from(array)
    expected = jnp.triu_indices_from(array)
    for i in range(2):
      self.assertTrue(jnp.all(result[i] == expected[i]))

  def test_take(self):
    array = jnp.array([4, 3, 5, 7, 6, 8])
    indices = jnp.array([0, 1, 4])
    result = bm.take(array, indices)
    self.assertTrue(jnp.all(result == jnp.take(array, indices)))

    q = [4, 3, 5, 7, 6, 8] * U.second
    i = jnp.array([0, 1, 4])
    result_q = bm.take(q, i)
    expected_q = jnp.take(jnp.array([4, 3, 5, 7, 6, 8]), jnp.array([0, 1, 4]))
    assert_quantity(result_q, expected_q, U.second)

  def test_select(self):
    condlist = [jnp.array([True, False, True]), jnp.array([False, True, False])]
    choicelist = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])]
    result = bm.select(condlist, choicelist, default=0)
    self.assertTrue(jnp.all(result == jnp.select(condlist, choicelist, default=0)))

    c = [jnp.array([True, False, True]), jnp.array([False, True, False])]
    ch = [[1, 2, 3] * U.second, [4, 5, 6] * U.second]
    result_q = bm.select(c, ch, default=0)
    expected_q = jnp.select([jnp.array([True, False, True]), jnp.array([False, True, False])],
                            [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])], default=0)
    assert_quantity(result_q, expected_q, U.second)


class TestWindowFuncs(unittest.TestCase):

  def test_bartlett(self):
    result = bm.bartlett(5)
    self.assertTrue(jnp.all(result == jnp.bartlett(5)))

  def test_blackman(self):
    result = bm.blackman(5)
    self.assertTrue(jnp.all(result == jnp.blackman(5)))

  def test_hamming(self):
    result = bm.hamming(5)
    self.assertTrue(jnp.all(result == jnp.hamming(5)))

  def test_hanning(self):
    result = bm.hanning(5)
    self.assertTrue(jnp.all(result == jnp.hanning(5)))

  def test_kaiser(self):
    result = bm.kaiser(5, 0.5)
    self.assertTrue(jnp.all(result == jnp.kaiser(5, 0.5)))


class TestConstants(unittest.TestCase):

  def test_constants(self):
    self.assertTrue(bm.e == jnp.e)
    self.assertTrue(bm.pi == jnp.pi)
    self.assertTrue(bm.inf == jnp.inf)


class TestLinearAlgebra(unittest.TestCase):

  def test_dot(self):
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    result = bm.dot(a, b)
    self.assertTrue(result == jnp.dot(a, b))

    q1 = [1, 2] * U.second
    q2 = [3, 4] * U.volt
    result_q = bm.dot(q1, q2)
    expected_q = jnp.dot(jnp.array([1, 2]), jnp.array([3, 4]))
    assert_quantity(result_q, expected_q, U.second * U.volt)

  def test_vdot(self):
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    result = bm.vdot(a, b)
    self.assertTrue(result == jnp.vdot(a, b))

    q1 = [1, 2] * U.second
    q2 = [3, 4] * U.volt
    result_q = bm.vdot(q1, q2)
    expected_q = jnp.vdot(jnp.array([1, 2]), jnp.array([3, 4]))
    assert_quantity(result_q, expected_q, U.second * U.volt)

  def test_inner(self):
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    result = bm.inner(a, b)
    self.assertTrue(result == jnp.inner(a, b))

    q1 = [1, 2] * U.second
    q2 = [3, 4] * U.volt
    result_q = bm.inner(q1, q2)
    expected_q = jnp.inner(jnp.array([1, 2]), jnp.array([3, 4]))
    assert_quantity(result_q, expected_q, U.second * U.volt)

  def test_outer(self):
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    result = bm.outer(a, b)
    self.assertTrue(jnp.all(result == jnp.outer(a, b)))

    q1 = [1, 2] * U.second
    q2 = [3, 4] * U.volt
    result_q = bm.outer(q1, q2)
    expected_q = jnp.outer(jnp.array([1, 2]), jnp.array([3, 4]))
    assert_quantity(result_q, expected_q, U.second * U.volt)

  def test_kron(self):
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    result = bm.kron(a, b)
    self.assertTrue(jnp.all(result == jnp.kron(a, b)))

    q1 = [1, 2] * U.second
    q2 = [3, 4] * U.volt
    result_q = bm.kron(q1, q2)
    expected_q = jnp.kron(jnp.array([1, 2]), jnp.array([3, 4]))
    assert_quantity(result_q, expected_q, U.second * U.volt)

  def test_matmul(self):
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[5, 6], [7, 8]])
    result = bm.matmul(a, b)
    self.assertTrue(jnp.all(result == jnp.matmul(a, b)))

    q1 = [[1, 2], [3, 4]] * U.second
    q2 = [[5, 6], [7, 8]] * U.volt
    result_q = bm.matmul(q1, q2)
    expected_q = jnp.matmul(jnp.array([[1, 2], [3, 4]]), jnp.array([[5, 6], [7, 8]]))
    assert_quantity(result_q, expected_q, U.second * U.volt)

  def test_trace(self):
    a = jnp.array([[1, 2], [3, 4]])
    result = bm.trace(a)
    self.assertTrue(result == jnp.trace(a))

    q = [[1, 2], [3, 4]] * U.second
    result_q = bm.trace(q)
    expected_q = jnp.trace(jnp.array([[1, 2], [3, 4]]))
    assert_quantity(result_q, expected_q, U.second)


class TestDataTypes(unittest.TestCase):

  def test_dtype(self):
    array = jnp.array([1, 2, 3])
    result = bm.dtype(array)
    self.assertTrue(result == jnp.dtype(array))

    q = [1, 2, 3] * U.second
    q = q.astype(jnp.int64)
    result_q = bm.dtype(q)
    expected_q = jnp.dtype(jnp.array([1, 2, 3], dtype=jnp.int64))
    self.assertTrue(result_q == expected_q)

  def test_finfo(self):
    result = bm.finfo(jnp.float32)
    self.assertTrue(result == jnp.finfo(jnp.float32))

    q = 1 * U.second
    q = q.astype(jnp.float64)
    result_q = bm.finfo(q)
    expected_q = jnp.finfo(jnp.float64)
    self.assertTrue(result_q == expected_q)

  def test_iinfo(self):
    result = bm.iinfo(jnp.int32)
    expected = jnp.iinfo(jnp.int32)
    self.assertEqual(result.min, expected.min)
    self.assertEqual(result.max, expected.max)
    self.assertEqual(result.dtype, expected.dtype)

    q = 1 * U.second
    q = q.astype(jnp.int32)
    result_q = bm.iinfo(q)
    expected_q = jnp.iinfo(jnp.int32)
    self.assertEqual(result_q.min, expected_q.min)
    self.assertEqual(result_q.max, expected_q.max)
    self.assertEqual(result_q.dtype, expected_q.dtype)


class TestMore(unittest.TestCase):
  def test_broadcast_arrays(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([[4], [5]])
    result = bm.broadcast_arrays(a, b)
    self.assertTrue(jnp.all(result[0] == jnp.broadcast_arrays(a, b)[0]))
    self.assertTrue(jnp.all(result[1] == jnp.broadcast_arrays(a, b)[1]))

    q1 = [1, 2, 3] * U.second
    q2 = [[4], [5]] * U.second
    result_q = bm.broadcast_arrays(q1, q2)
    expected_q = jnp.broadcast_arrays(jnp.array([1, 2, 3]), jnp.array([[4], [5]]))
    assert_quantity(result_q, expected_q, U.second)

  def test_broadcast_shapes(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([[4], [5]])
    result = bm.broadcast_shapes(a.shape, b.shape)
    self.assertTrue(result == jnp.broadcast_shapes(a.shape, b.shape))

  def test_einsum(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5])
    result = bm.einsum('i,j->ij', a, b)
    self.assertTrue(jnp.all(result == jnp.einsum('i,j->ij', a, b)))

    q1 = [1, 2, 3] * U.second
    q2 = [4, 5] * U.volt
    result_q = bm.einsum('i,j->ij', q1, q2)
    expected_q = jnp.einsum('i,j->ij', jnp.array([1, 2, 3]), jnp.array([4, 5]))
    assert_quantity(result_q, expected_q, U.second * U.volt)

    q1 = [1, 2, 3] * U.second
    q2 = [1, 2, 3] * U.second
    result_q = bm.einsum('i,i->i', q1, q2)
    expected_q = jnp.einsum('i,i->i', jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    assert_quantity(result_q, expected_q, U.second)

  def test_gradient(self):
    f = jnp.array([1, 2, 4, 7, 11, 16], dtype=float)
    result = bm.gradient(f)
    self.assertTrue(jnp.all(bm.allclose(result, jnp.gradient(f))))

    q = [1, 2, 4, 7, 11, 16] * U.second
    result_q = bm.gradient(q)
    expected_q = jnp.gradient(jnp.array([1, 2, 4, 7, 11, 16]))
    assert_quantity(result_q, expected_q, U.second)

    q1 = jnp.array([[1, 2, 6], [3, 4, 5]]) * U.second
    dx = 2. * U.meter
    # y = [1., 1.5, 3.5] * U.second
    result_q = bm.gradient(q1, dx)
    expected_q = jnp.gradient(jnp.array([[1, 2, 6], [3, 4, 5]]), 2.)
    assert_quantity(result_q[0], expected_q[0], U.second / U.meter)
    assert_quantity(result_q[1], expected_q[1], U.second / U.meter)

  def test_intersect1d(self):
    a = jnp.array([1, 2, 3, 4, 5])
    b = jnp.array([3, 4, 5, 6, 7])
    result = bm.intersect1d(a, b)
    self.assertTrue(jnp.all(result == jnp.intersect1d(a, b)))

    q1 = [1, 2, 3, 4, 5] * U.second
    q2 = [3, 4, 5, 6, 7] * U.second
    result_q = bm.intersect1d(q1, q2)
    expected_q = jnp.intersect1d(jnp.array([1, 2, 3, 4, 5]), jnp.array([3, 4, 5, 6, 7]))
    assert_quantity(result_q, expected_q, U.second)

  def test_nan_to_num(self):
    a = jnp.array([1, 2, 3, 4, jnp.nan])
    result = bm.nan_to_num(a)
    self.assertTrue(jnp.all(result == jnp.nan_to_num(a)))

    q = [1, 2, 3, 4, jnp.nan] * U.second
    result_q = bm.nan_to_num(q)
    expected_q = jnp.nan_to_num(jnp.array([1, 2, 3, 4, jnp.nan]))
    assert_quantity(result_q, expected_q, U.second)

  def nanargmax(self):
    a = jnp.array([1, 2, 3, 4, jnp.nan])
    result = bm.nanargmax(a)
    self.assertTrue(result == jnp.nanargmax(a))

    q = [1, 2, 3, 4, jnp.nan] * U.second
    result_q = bm.nanargmax(q)
    expected_q = jnp.nanargmax(jnp.array([1, 2, 3, 4, jnp.nan]))
    self.assertTrue(result_q == expected_q)

  def nanargmin(self):
    a = jnp.array([1, 2, 3, 4, jnp.nan])
    result = bm.nanargmin(a)
    self.assertTrue(result == jnp.nanargmin(a))

    q = [1, 2, 3, 4, jnp.nan] * U.second
    result_q = bm.nanargmin(q)
    expected_q = jnp.nanargmin(jnp.array([1, 2, 3, 4, jnp.nan]))
    self.assertTrue(result_q == expected_q)

  def test_rot90(self):
    a = jnp.array([[1, 2], [3, 4]])
    result = bm.rot90(a)
    self.assertTrue(jnp.all(result == jnp.rot90(a)))

    q = [[1, 2], [3, 4]] * U.second
    result_q = bm.rot90(q)
    expected_q = jnp.rot90(jnp.array([[1, 2], [3, 4]]))
    assert_quantity(result_q, expected_q, U.second)

  def test_tensordot(self):
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[1, 2], [3, 4]])
    result = bm.tensordot(a, b)
    self.assertTrue(jnp.all(result == jnp.tensordot(a, b)))

    q1 = [[1, 2], [3, 4]] * U.second
    q2 = [[1, 2], [3, 4]] * U.second
    result_q = bm.tensordot(q1, q2)
    expected_q = jnp.tensordot(jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 2], [3, 4]]))
    assert_quantity(result_q, expected_q, U.second ** 2)
