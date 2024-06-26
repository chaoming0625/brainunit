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

from ._einops import *
from ._einops import __all__ as _einops_all
from ._numpy_accept_unitless import *
from ._numpy_accept_unitless import __all__ as _compat_funcs_accept_unitless_all
from ._numpy_array_creation import *
from ._numpy_array_creation import __all__ as _compat_array_creation_all
from ._numpy_array_manipulation import *
from ._numpy_array_manipulation import __all__ as _compat_array_manipulation_all
from ._numpy_change_unit import *
from ._numpy_change_unit import __all__ as _compat_funcs_change_unit_all
from ._numpy_get_attribute import *
from ._numpy_get_attribute import __all__ as _compat_get_attribute_all
from ._numpy_indexing import *
from ._numpy_indexing import __all__ as _compat_funcs_indexing_all
from ._numpy_keep_unit import *
from ._numpy_keep_unit import __all__ as _compat_funcs_keep_unit_all
from ._numpy_linear_algebra import *
from ._numpy_linear_algebra import __all__ as _compat_linear_algebra_all
from ._numpy_misc import *
from ._numpy_misc import __all__ as _compat_misc_all
from ._numpy_remove_unit import *
from ._numpy_remove_unit import __all__ as _compat_funcs_remove_unit_all
from ._others import *
from ._others import __all__ as _other_all

__all__ = (_compat_array_creation_all +
           _compat_array_manipulation_all +
           _compat_funcs_change_unit_all +
           _compat_funcs_keep_unit_all +
           _compat_funcs_accept_unitless_all +
           _compat_funcs_remove_unit_all +
           _compat_get_attribute_all +
           _compat_funcs_indexing_all +
           _compat_linear_algebra_all +
           _compat_misc_all +
           _other_all +
           _einops_all)

del (_compat_array_creation_all,
     _compat_array_manipulation_all,
     _compat_funcs_change_unit_all,
     _compat_funcs_keep_unit_all,
     _compat_funcs_accept_unitless_all,
     _compat_funcs_remove_unit_all,
     _compat_get_attribute_all,
     _compat_funcs_indexing_all,
     _compat_linear_algebra_all,
     _compat_misc_all,
     _other_all,
     _einops_all)
