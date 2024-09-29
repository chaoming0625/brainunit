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

from contextlib import ContextDecorator

__all__ = ["quantity_support"]

import numpy as np


def quantity_support(unit_display_name: bool = False):
  """
    Enable support for plotting `brainunit.Quantity` instances in
    matplotlib.

    May be (optionally) used with a ``with`` statement.

      >>> import matplotlib.pyplot as plt
      >>> import brainunit as u
      >>> from brainunit import visualize
      >>> with visualize.quantity_support():
      ...     plt.figure()
      ...     plt.plot([1, 2, 3] * u.meter)
      [...]
      ...     plt.plot([101, 125, 150] * u.cmeter)
      [...]
      ...     plt.draw()
    """

  from matplotlib import ticker, units

  import brainunit as u

  def rad_fn(
      x,
      pos=None
  ) -> str:
    n = int((x / np.pi) * 2.0 + 0.25)
    if n == 0:
      return "0"
    elif n == 1:
      return "π/2"
    elif n == 2:
      return "π"
    elif n % 2 == 0:
      return f"{n // 2}π"
    else:
      return f"{n}π/2"

  def get_label(
      unit_display_name: bool,
      unit: u.Unit,
  ) -> str:
    if unit_display_name:
      return unit.dispname
    else:
      return unit.name

  class MplQuantityConverter(units.ConversionInterface, ContextDecorator):
    def __init__(self):
      # Keep track of original converter in case the context manager is
      # used in a nested way.
      self._original_converter = {u.Quantity: units.registry.get(u.Quantity)}
      units.registry[u.Quantity] = self

    @staticmethod
    def axisinfo(unit, axis):
      if unit == u.radian:
        return units.AxisInfo(
          majloc=ticker.MultipleLocator(base=np.pi / 2),
          majfmt=ticker.FuncFormatter(rad_fn),
          label=get_label(unit_display_name, unit),
        )
      elif unit is not None:
        return units.AxisInfo(label=get_label(unit_display_name, unit))
      return None

    @staticmethod
    def convert(val, unit, axis):
      if isinstance(val, u.Quantity):
        return val.mantissa
      elif isinstance(val, list) and val and isinstance(val[0], u.Quantity):
        return [v.mantissa for v in val]
      else:
        return val

    @staticmethod
    def default_units(x, axis):
      if hasattr(x, "unit"):
        return x.unit
      return None

    def __enter__(self):
      return self

    def __exit__(self, type, value, tb):
      if self._original_converter[u.Quantity] is None:
        del units.registry[u.Quantity]
      else:
        units.registry[u.Quantity] = self._original_converter[u.Quantity]

  return MplQuantityConverter()
