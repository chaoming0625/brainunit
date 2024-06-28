import unittest
import brainunit as bu
import brainunit.math as bm
import inspect

fun_keep_unit_squence_inputs = [
  'row_stack', 'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack', 'block', 'append',
]
fun_keep_unit_squence_outputs = [
  'split', 'array_split', 'dsplit', 'hsplit', 'vsplit',
]
fun_keep_unit_broadcasting_arrays = [
  'atleast_1d', 'atleast_2d', 'atleast_3d', 'broadcast_arrays',
]
fun_keep_unit_array_manipulation = [
  'reshape', 'moveaxis', 'transpose', 'swapaxes', 'tile', 'repeat',
  'flip', 'fliplr', 'flipud', 'roll', 'expand_dims', 'squeeze',
  'sort', 'max', 'min', 'amax', 'amin', 'diagflat', 'diagonal', 'choose', 'ravel',
  'flatten', 'unflatten', 'remove_diag',
]
fun_keep_unit_math_unary = [
  'real', 'imag', 'conj', 'conjugate', 'negative', 'positive',
  'abs', 'sum', 'nancumsum', 'nansum',
  'cumsum', 'ediff1d', 'absolute', 'fabs', 'median',
  'nanmin', 'nanmax', 'ptp', 'average', 'mean', 'std',
  'nanmedian', 'nanmean', 'nanstd', 'diff', 'rot90', 'intersect1d', 'nan_to_num',
  'percentile', 'nanpercentile', 'quantile', 'nanquantile',
]

fun_keep_unit_math_binary = [
  'fmod', 'mod', 'copysign', 'remainder',
  'maximum', 'minimum', 'fmax', 'fmin', 'lcm', 'gcd', 'trace',
  'add', 'subtract', 'nextafter',
]

fun_keep_unit_selection = [
  'compress', 'extract', 'take', 'select', 'where', 'unique',
]


class TestFunKeepUnit(unittest.TestCase):
  def test_fun_keep_unit_squence_inputs(self):
    fun_list = [getattr(bm, fun) for fun in fun_keep_unit_squence_inputs]
    # TODO

  def test_fun_keep_unit_squence_outputs(self):
    fun_list = [getattr(bm, fun) for fun in fun_keep_unit_squence_outputs]
    # TODO

  def test_fun_keep_unit_broadcasting_arrays(self):
    fun_list = [getattr(bm, fun) for fun in fun_keep_unit_broadcasting_arrays]
    # TODO

  def test_fun_keep_unit_array_manipulation(self):
    fun_list = [getattr(bm, fun) for fun in fun_keep_unit_array_manipulation]
    # TODO

  def test_fun_keep_unit_math_unary(self):
    fun_list = [getattr(bm, fun) for fun in fun_keep_unit_math_unary]
    # TODO

  def test_fun_keep_unit_math_binary(self):
    fun_list = [getattr(bm, fun) for fun in fun_keep_unit_math_binary]
    # TODO

  def test_fun_keep_unit_math_other(self):
    fun_keep_unit_math_other = [
      'interp', 'clip', 'histogram',
    ]
    fun_list = [getattr(bm, fun) for fun in fun_keep_unit_math_other]
    # TODO

  def test_fun_keep_unit_selection(self):
    fun_list = [getattr(bm, fun) for fun in fun_keep_unit_selection]
    # TODO
