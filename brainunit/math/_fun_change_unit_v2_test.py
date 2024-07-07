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


import jax

import brainunit as bu


def test_multi_dot():
  key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
  x = jax.random.normal(key1, shape=(200, 5)) * bu.mA
  y = jax.random.normal(key2, shape=(5, 100)) * bu.mV
  z = jax.random.normal(key3, shape=(100, 10)) * bu.ohm
  result1 = (x @ y) @ z
  result2 = x @ (y @ z)
  assert bu.math.allclose(result1, result2, atol=1E-4)
  result3 = bu.math.multi_dot([x, y, z])
  assert bu.math.allclose(result1, result3, atol=1E-4)
  assert jax.jit(lambda x, y, z: (x @ y) @ z).lower(x, y, z).cost_analysis()['flops'] == 600000.0
  assert jax.jit(lambda x, y, z: x @ (y @ z)).lower(x, y, z).cost_analysis()['flops'] == 30000.0
  assert jax.jit(bu.math.multi_dot).lower([x, y, z]).cost_analysis()['flops'] == 30000.0
