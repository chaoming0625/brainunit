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

import jax.numpy as jnp

__all__ = [
  # window funcs
  'bartlett', 'blackman', 'hamming', 'hanning', 'kaiser',
]

# window funcs
# ------------

bartlett = jnp.bartlett
blackman = jnp.blackman
hamming = jnp.hamming
hanning = jnp.hanning
kaiser = jnp.kaiser
