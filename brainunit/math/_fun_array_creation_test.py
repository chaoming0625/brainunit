import unittest
import brainunit as bu
import brainunit.math as bm
import inspect

fun_array_creation_given_shape = [
  'full', 'eye', 'identity', 'tri',
  'empty', 'ones', 'zeros',
]
fun_array_creation_given_array = [
  'full_like', 'diag', 'tril', 'triu',
  'empty_like', 'ones_like', 'zeros_like', 'fill_diagonal',
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


class TestFunArrayCreation(unittest.TestCase):
  def test_fun_array_creation_given_shape(self):
    fun_list = [getattr(bm, fun) for fun in fun_array_creation_given_shape]
    # TODO

  def test_fun_array_creation_given_array(self):
    fun_list = [getattr(bm, fun) for fun in fun_array_creation_given_array]
    # TODO

  def test_fun_array_creation_misc(self):
    fun_list = [getattr(bm, fun) for fun in fun_array_creation_misc]
    # TODO

  def test_fun_array_creation_indexing(self):
    fun_list = [getattr(bm, fun) for fun in fun_array_creation_indexing]
    # TODO

  def test_fun_array_creation_other(self):
    fun_list = [getattr(bm, fun) for fun in fun_array_creation_other]
    # TODO
