import unittest
import brainunit as bu
import brainunit.math as bm
import inspect
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from brainunit import second, meter, DimensionMismatchError, assert_quantity

fun_accept_unitless_unary = [
  'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2',
  'deg2rad', 'rad2deg', 'degrees', 'radians', 'angle',
  'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
  'arctanh', 'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan',
  'tanh',
]
fun_accept_unitless_binary = [
  'hypot', 'arctan2', 'logaddexp', 'logaddexp2',
  'corrcoef', 'correlate', 'cov',
]
fun_accept_unitless_unary_can_return_quantity = [
  'round', 'around', 'round_', 'rint',
  'floor', 'ceil', 'trunc', 'fix',
]
fun_elementwise_bit_operation_unary = [
  'bitwise_not', 'invert',
]
fun_elementwise_bit_operation_binary = [
  'bitwise_and', 'bitwise_or', 'bitwise_xor', 'left_shift', 'right_shift',
]


class TestFunAcceptUnitless(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(TestFunAcceptUnitless, self).__init__(*args, **kwargs)

    print()

  @parameterized.product(
    value=[(1.0, 2.0), (1.23, 2.34, 3.45)]
  )
  def test_fun_accept_unitless_unary_1(self, value):
    bm_fun_list = [getattr(bm, fun) for fun in fun_accept_unitless_unary]
    jnp_fun_list = [getattr(jnp, fun) for fun in fun_accept_unitless_unary]

    for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
      print(f'fun: {bm_fun.__name__}')

      result = bm_fun(jnp.array(value))
      expected = jnp_fun(jnp.array(value))
      assert_quantity(result, expected)

      q = value * meter
      result = bm_fun(q, unit_to_scale=bu.dametre)
      expected = jnp_fun(jnp.array(value) / bu.dametre.value)
      assert_quantity(result, expected)

      with pytest.raises(AssertionError):
        result = bm_fun(q)

      with pytest.raises(DimensionMismatchError):
        result = bm_fun(q, unit_to_scale=bu.second)

  @parameterized.product(
    value=[[(1.0, 2.0), (3.0, 4.0), ],
           [(1.23, 2.34, 3.45), (4.56, 5.67, 6.78)]]
  )
  def test_func_accept_unitless_binary(self, value):
    value1, value2 = value
    bm_fun_list = [getattr(bm, fun) for fun in fun_accept_unitless_binary]
    jnp_fun_list = [getattr(jnp, fun) for fun in fun_accept_unitless_binary]

    for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
      print(f'fun: {bm_fun.__name__}')

      result = bm_fun(jnp.array(value1), jnp.array(value2))
      expected = jnp_fun(jnp.array(value1), jnp.array(value2))
      assert_quantity(result, expected)

      q1 = value1 * meter
      q2 = value2 * meter
      result = bm_fun(q1, q2, unit_to_scale=bu.dametre)
      expected = jnp_fun(jnp.array(value1) / bu.dametre.value, jnp.array(value2) / bu.dametre.value)
      assert_quantity(result, expected)

      with pytest.raises(AssertionError):
        result = bm_fun(q1, q2)

      with pytest.raises(DimensionMismatchError):
        result = bm_fun(q1, q2, unit_to_scale=bu.second)

  @parameterized.product(
    value=[(1.123, 2.567, 3.891), (1.23, 2.34, 3.45)]
  )
  def test_fun_accept_unitless_unary_can_return_quantity(self, value):
    bm_fun_list = [getattr(bm, fun) for fun in fun_accept_unitless_unary_can_return_quantity]
    jnp_fun_list = [getattr(jnp, fun) for fun in fun_accept_unitless_unary_can_return_quantity]

    for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
      print(f'fun: {bm_fun.__name__}')

      result = bm_fun(jnp.array(value))
      expected = jnp_fun(jnp.array(value))
      assert_quantity(result, expected)

      q = value * meter
      result = bm_fun(q, unit_to_scale=meter)
      expected = jnp_fun(jnp.array(value))
      assert_quantity(result, expected, meter)

      with pytest.raises(AssertionError):
        result = bm_fun(q)

      with pytest.raises(DimensionMismatchError):
        result = bm_fun(q, unit_to_scale=bu.second)

  @parameterized.product(
    value=[(1, 2), (1, 2, 3)]
  )
  def test_elementwise_bit_operation_unary(self, value):
    bm_fun_list = [getattr(bm, fun) for fun in fun_elementwise_bit_operation_unary]
    jnp_fun_list = [getattr(jnp, fun) for fun in fun_elementwise_bit_operation_unary]

    for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
      print(f'fun: {bm_fun.__name__}')

      result = bm_fun(jnp.array(value))
      expected = jnp_fun(jnp.array(value))
      assert_quantity(result, expected)

      q = value * meter
      result = bm_fun(q.to_dtype(jnp.int32))
      expected = jnp_fun(jnp.array(value))
      assert_quantity(result, expected)

      with pytest.raises(AssertionError):
        result = bm_fun(q)
        

  @parameterized.product(
    value=[[(0, 1), (1, 1)],
           [(True, False, True, False), (False, False, True, True)]]
  )
  def test_elementwise_bit_operation_binary(self, value):
    value1, value2 = value
    bm_fun_list = [getattr(bm, fun) for fun in fun_elementwise_bit_operation_binary]
    jnp_fun_list = [getattr(jnp, fun) for fun in fun_elementwise_bit_operation_binary]

    for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
      print(f'fun: {bm_fun.__name__}')

      result = bm_fun(jnp.array(value1), jnp.array(value2))
      expected = jnp_fun(jnp.array(value1), jnp.array(value2))
      assert_quantity(result, expected)

      q1 = value1 * meter
      q2 = value2 * meter
      result = bm_fun(q1.to_bool(), q2.to_bool())
      expected = jnp_fun(jnp.array(value1), jnp.array(value2))
      assert_quantity(result, expected)

      with pytest.raises(AssertionError):
        result = bm_fun(q1, q2)

  @parameterized.product(
    value=[(1.123, 2.567, 3.891), (1.23, 2.34, 3.45)]
  )
  def test_modf(self, value):

    result1, result2 = bm.modf(jnp.array(value))
    expected1, expected2 = jnp.modf(jnp.array(value))
    assert_quantity(result1, expected1)
    assert_quantity(result2, expected2)

    q = value * meter
    result1, result2 = bm.modf(q, unit_to_scale=meter)
    expected1, expected2 = jnp.modf(jnp.array(value))
    assert_quantity(result1, expected1, meter)
    assert_quantity(result2, expected2, meter)

    with pytest.raises(AssertionError):
      result1, result2 = bm.modf(q)

    with pytest.raises(DimensionMismatchError):
      result1, result2 = bm.modf(q, unit_to_scale=bu.second)

  @parameterized.product(
    value=[(1.123, 2.567, 3.891), (1.23, 2.34, 3.45)]
  )
  def test_frexp(self, value):

    result1, result2 = bm.frexp(jnp.array(value))
    expected1, expected2 = jnp.frexp(jnp.array(value))
    assert_quantity(result1, expected1)
    assert_quantity(result2, expected2)

    q = value * meter
    result1, result2 = bm.frexp(q, unit_to_scale=meter)
    expected1, expected2 = jnp.frexp(jnp.array(value))
    assert_quantity(result1, expected1)
    assert_quantity(result2, expected2)

    with pytest.raises(AssertionError):
      result1, result2 = bm.frexp(q)

    with pytest.raises(DimensionMismatchError):
      result1, result2 = bm.frexp(q, unit_to_scale=bu.second)
