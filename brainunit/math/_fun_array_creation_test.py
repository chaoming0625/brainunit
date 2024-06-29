import unittest
import brainunit as bu
import brainunit.math as bm
import inspect
import jax.numpy as jnp

import pytest
from absl.testing import parameterized

from brainunit import second, meter, DimensionMismatchError, assert_quantity

fun_array_creation_given_shape = [
  'empty', 'ones', 'zeros',
]
fun_array_creation_given_shape_fill_value = [
  'full',
]
fun_array_creation_given_int = [
  'eye', 'identity', 'tri',
]
fun_array_creation_given_array = [
  'empty_like', 'ones_like', 'zeros_like', 'diag',
]
fun_array_creation_given_array_fill_value = [
  'full_like',
]
fun_array_creation_given_square_array = [
  'tril', 'triu', 'fill_diagonal',
]
fun_array_creation_misc = [
  'array', 'asarray', 'arange', 'linspace', 'logspace',
  'meshgrid', 'vander',
]
fun_array_creation_indexing = [
  'tril_indices', 'tril_indices_from', 'triu_indices',
  'triu_indices_from',
]
fun_array_creation_other = [
  'from_numpy',
  'as_numpy',
  'tree_ones_like',
  'tree_zeros_like',
]


class TestFunArrayCreation(parameterized.TestCase):

  @parameterized.product(
    shape=[(1,), (2, 3), (4, 5, 6)],
    unit=[second, meter]
  )
  def test_fun_array_creation_given_shape(self, shape, unit):
    bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_given_shape]
    jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_shape]

    for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
      print(f'fun: {bm_fun.__name__}')

      result = bm_fun(shape)
      expected = jnp_fun(shape)
      assert_quantity(result, expected)

      result = bm_fun(shape, unit=unit)
      expected = jnp_fun(shape)
      assert_quantity(result, expected, unit=unit)

  @parameterized.product(
    shape=[(1,), (2, 3), (4, 5, 6)],
    unit=[second, meter],
    fill_value=[-1., 1.]
  )
  def test_fun_array_creation_given_shape_fill_value(self, shape, unit, fill_value):
    bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_given_shape_fill_value]
    jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_shape_fill_value]

    for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
      print(f'fun: {bm_fun.__name__}')

      result = bm_fun(shape, fill_value=fill_value)
      expected = jnp_fun(shape, fill_value=fill_value)
      assert_quantity(result, expected)

      result = bm_fun(shape, fill_value=fill_value * unit)
      expected = jnp_fun(shape, fill_value=fill_value)
      assert_quantity(result, expected, unit=unit)


  @parameterized.product(
    array=[jnp.array([1.0, 2.0]), jnp.array([[1.0, 2.0], [3.0, 4.0]])],
    unit=[second, meter]
  )
  def test_fun_array_creation_given_array(self, array, unit):
    bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_given_array]
    jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_array]

    for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
      print(f'fun: {bm_fun.__name__}')

      result = bm_fun(array)
      expected = jnp_fun(array)
      assert_quantity(result, expected)

      result = bm_fun(array, unit=unit)
      expected = jnp_fun(array)
      assert_quantity(result, expected, unit=unit)

  @parameterized.product(
    array=[jnp.array([1.0, 2.0]), jnp.array([[1.0, 2.0], [3.0, 4.0]])],
    unit=[second, meter],
    fill_value=[-1., 1.]
  )
  def test_fun_array_creation_given_array_fill_value(self, array, unit, fill_value):
    bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_given_array_fill_value]
    jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_array_fill_value]

    for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
      print(f'fun: {bm_fun.__name__}')

      result = bm_fun(array, fill_value=fill_value)
      expected = jnp_fun(array, fill_value=fill_value)
      assert_quantity(result, expected)

      result = bm_fun(array * unit, fill_value=fill_value * unit)
      expected = jnp_fun(array, fill_value=fill_value)
      assert_quantity(result, expected, unit=unit)

      with pytest.raises(AssertionError):
        result = bm_fun(array, fill_value=fill_value * unit)

  def test_fun_array_creation_misc(self):
    fun_list = [getattr(bm, fun) for fun in fun_array_creation_misc]
    # TODO

  def test_fun_array_creation_indexing(self):
    fun_list = [getattr(bm, fun) for fun in fun_array_creation_indexing]
    # TODO

  def test_fun_array_creation_other(self):
    fun_list = [getattr(bm, fun) for fun in fun_array_creation_other]
    # TODO
