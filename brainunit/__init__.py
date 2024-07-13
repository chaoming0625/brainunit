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
from __future__ import annotations

__version__ = "0.0.1.1"

import importlib

from . import math
from . import _base
from . import _unit_common
from . import _unit_constants
from . import _unit_shortcuts

from ._base import *
from ._base import _default_magnitude, _siprefixes
from ._base import __all__ as _base_all
from ._unit_common import *
from ._unit_common import __all__ as _common_all
from ._unit_constants import *
from ._unit_constants import __all__ as _constants_all
from ._unit_shortcuts import *
from ._unit_shortcuts import __all__ as _std_units_all

__all__ = ['math'] + _common_all + _std_units_all + _constants_all + _base_all
del _common_all, _std_units_all, _constants_all, _base_all


def set_default_magnitude(
    magnitude: int | dict[str, int],
    # unit: Unit = None,
):
  """
  Set the default magnitude for units.

  Parameters
  ----------
  magnitude : int | dict[str, int]
      The magnitude to set. If an int is given, it will be set for all
      dimensions. If a dict is given, it will be set for the specified
      dimensions.

  Examples
  --------
  >>> set_default_magnitude('n') # Sets the default magnitude to 'nano' (10e-9)
  >>> set_default_magnitude(-9) # Alternatively, use an integer to represent the exponent of 10
  >>> set_default_magnitude({'m': -3, 'kg': -9}) # Set the default magnitude for 'metre' to 'milli' and 'kilogram' to 'nano'
  >>> set_default_magnitude({'m': 'm', 'kg': 'n'}) # Alternatively, use a string to represent the magnitude
  """
  global _default_magnitude
  if isinstance(magnitude, int):
    # if isinstance(unit, Unit):
    #   for key, dim in zip(_default_magnitude.keys(), unit.dim._dims):
    #     _default_magnitude[key] = magnitude / abs(dim) if dim != 0 else 0
    # else:
    _default_magnitude.update((key, magnitude) for key in _default_magnitude)
  elif isinstance(magnitude, str):
    # if isinstance(unit, Unit):
    #   for key, dim in zip(_default_magnitude.keys(), unit.dim._dims):
    #     _default_magnitude[key] = _siprefixes[magnitude] / abs(dim) if dim != 0 else 0
    # else:
    _default_magnitude.update((key, _siprefixes[magnitude]) for key in _default_magnitude)
  elif isinstance(magnitude, dict):
    _default_magnitude.update((key, 0) for key in _default_magnitude)
    for key, value in magnitude.items():
      if isinstance(value, int):
        _default_magnitude[key] = value
      elif isinstance(value, str):
        _default_magnitude[key] = _siprefixes[value]
      else:
        raise ValueError(f"Invalid magnitude value: {value}")
  else:
    raise ValueError(f"Invalid magnitude: {magnitude}")

  global _unit_common
  global _unit_constants
  global _unit_shortcuts
  # Reload modules
  importlib.reload(_unit_common)
  importlib.reload(_unit_constants)
  importlib.reload(_unit_shortcuts)

  from ._base import __all__ as _base_all
  from ._unit_common import __all__ as _common_all
  from ._unit_constants import __all__ as _constants_all
  from ._unit_shortcuts import __all__ as _std_units_all
  globals().update({k: getattr(_unit_common, k) for k in _common_all})
  globals().update({k: getattr(_unit_constants, k) for k in _constants_all})
  globals().update({k: getattr(_unit_shortcuts, k) for k in _std_units_all})

  global __all__
  __all__ = ['math'] + _common_all + _std_units_all + _constants_all + _base_all
  del _common_all, _std_units_all, _constants_all, _base_all
