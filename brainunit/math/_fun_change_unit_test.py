import unittest
import brainunit as bu
import brainunit.math as bm
import inspect

fun_change_unit_unary = [
  'reciprocal', 'prod', 'product', 'nancumprod', 'nanprod', 'cumprod',
  'cumproduct', 'var', 'nanvar', 'cbrt', 'square', 'sqrt',
]
fun_change_unit_binary = [
  'multiply', 'divide', 'power', 'cross', 'ldexp',
  'true_divide', 'floor_divide', 'float_power',
  'divmod', 'convolve',
]
fun_change_unit_linear_algebra = [
  'dot', 'vdot', 'inner', 'outer', 'kron', 'matmul', 'tensordot',
]


class TestFunChangeUnit(unittest.TestCase):
  def test_fun_change_unit_unary(self):
    fun_list = [getattr(bm, fun) for fun in fun_change_unit_unary]
    # TODO

  def test_fun_change_unit_binary(self):
    fun_list = [getattr(bm, fun) for fun in fun_change_unit_binary]
    # TODO

  def test_fun_change_unit_linear_algebra(self):
    fun_list = [getattr(bm, fun) for fun in fun_change_unit_linear_algebra]
    # TODO
