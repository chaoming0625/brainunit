import unittest
import brainunit as bu
import brainunit.math as bm
import inspect

fun_remove_unit_unary = [
  'heaviside', 'signbit', 'sign', 'bincount', 'digitize',
]

fun_remove_unit_logic_unary = [
  'all', 'any', 'logical_not',
]

fun_remove_unit_logic_binary = [
  'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
  'array_equal', 'isclose', 'allclose', 'logical_and',
  'logical_or', 'logical_xor', "alltrue", 'sometrue',
]

fun_remove_unit_indexing = [
  'argsort', 'argmax', 'argmin', 'nanargmax', 'nanargmin', 'argwhere',
  'nonzero', 'flatnonzero', 'searchsorted', 'count_nonzero',
]


class TestFunChangeUnit(unittest.TestCase):
  def test_fun_remove_unit_logic_unary(self):
    fun_list = [getattr(bm, fun) for fun in fun_remove_unit_logic_unary]
    # TODO

  def test_fun_remove_unit_logic_binary(self):
    fun_list = [getattr(bm, fun) for fun in fun_remove_unit_logic_binary]
    # TODO

  def test_fun_remove_unit_indexing(self):
    fun_list = [getattr(bm, fun) for fun in fun_remove_unit_indexing]
    # TODO

  def test_fun_remove_unit_unary(self):
    fun_list = [getattr(bm, fun) for fun in fun_remove_unit_unary]
    # TODO

