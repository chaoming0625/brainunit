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


import brainstate as bst
from brainstate._utils import set_module_as

from ._compat_numpy_funcs_accept_unitless import funcs_only_accept_unitless_unary

__all__ = [
  'exprel',
]


@set_module_as('brainunit.math')
def exprel(x):
  return funcs_only_accept_unitless_unary(bst.math.exprel, x)
