import unittest
import brainunit as bu
import brainunit.math as bm
import inspect
import jax.numpy as jnp
import pytest

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


class TestFunAcceptUnitless(unittest.TestCase):
  def test_fun_accept_unitless_unary_1(self):
    bm_fun_list = [getattr(bm, fun) for fun in fun_accept_unitless_unary]
    jnp_fun_list = [getattr(jnp, fun) for fun in fun_accept_unitless_unary]

    for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
      print(f'fun: {bm_fun.__name__}')

      value = [1.0, 2.0]

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

  def test_func_accept_unitless_binary(self):
    bm_fun_list = [getattr(bm, fun) for fun in fun_accept_unitless_binary]
    jnp_fun_list = [getattr(jnp, fun) for fun in fun_accept_unitless_binary]

    for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
      print(f'fun: {bm_fun.__name__}')

      value1 = [1.0, 2.0]
      value2 = [3.0, 4.0]

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

  def test_fun_accept_unitless_unary_can_return_quantity(self):
    bm_fun_list = [getattr(bm, fun) for fun in fun_accept_unitless_unary_can_return_quantity]
    jnp_fun_list = [getattr(jnp, fun) for fun in fun_accept_unitless_unary_can_return_quantity]

    for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
      print(f'fun: {bm_fun.__name__}')

      value = [1.123, 2.567, 3.891]

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

  def test_elementwise_bit_operation_unary(self):
    fun_list = [getattr(bm, fun) for fun in fun_elementwise_bit_operation_unary]
    # TODO

  def test_elementwise_bit_operation_binary(self):
    fun_list = [getattr(bm, fun) for fun in fun_elementwise_bit_operation_binary]
    # TODO

  def test_modf(self):
    value = [1.123, 2.567, 3.891]

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

  def test_frexp(self):
    value = [1.123, 2.567, 3.891]

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