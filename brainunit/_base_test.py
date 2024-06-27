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

import unittest

import jax.numpy as  jnp
import numpy as np

import brainunit as bu


class TestQuantity(unittest.TestCase):
  def test_dim(self):
    a = [1, 2.] * bu.ms

    with self.assertRaises(NotImplementedError):
      a.dim = bu.mV.dim

  def test_clip(self):
    a = [1, 2.] * bu.ms
    self.assertTrue(bu.math.allclose(a.clip(1.5 * bu.ms, 2.5 * bu.ms), [1.5, 2.] * bu.ms))

    b = bu.Quantity([1, 2.])
    self.assertTrue(bu.math.allclose(b.clip(1.5, 2.5), bu.math.asarray([1.5, 2.])))

  def test_round(self):
    a = [1.1, 2.2] * bu.ms
    self.assertTrue(bu.math.allclose(a.round(unit=bu.ms), [1, 2] * bu.ms))

    b = bu.Quantity([1.1, 2.2])
    self.assertTrue(bu.math.allclose(b.round(), bu.math.asarray([1, 2])))

    with self.assertRaises(AssertionError):
      a = [1.1, 2.2] * bu.ms
      self.assertTrue(bu.math.allclose(a.round(), [1, 2] * bu.ms))

  def test_astype(self):
    a = [1, 2.] * bu.ms
    self.assertTrue(a.astype(jnp.float16).dtype == jnp.float16)

  def test_to_numpy(self):
    a = bu.Quantity([1, 2.])
    self.assertTrue(bu.math.allclose(a.to_numpy(), jnp.asarray([1, 2.])))

    with self.assertRaises(AssertionError):
      a = [1, 2.] * bu.ms
      self.assertTrue(bu.math.allclose(a.to_numpy(), jnp.asarray([1, 2.])))

  def test_to_jax(self):
    a = bu.Quantity([1, 2.])
    self.assertTrue(bu.math.allclose(a.to_jax(), jnp.asarray([1, 2.])))

    with self.assertRaises(AssertionError):
      a = [1, 2.] * bu.ms
      self.assertTrue(bu.math.allclose(a.to_jax(), jnp.asarray([1, 2.])))

  def test___array__(self):
    a = bu.Quantity([1, 2.])
    self.assertTrue(bu.math.allclose(np.asarray(a), np.asarray([1, 2.])))

    with self.assertRaises(TypeError):
      a = [1, 2.] * bu.ms
      self.assertTrue(bu.math.allclose(np.asarray(a), np.asarray([1, 2.])))

  def test__float__(self):
    a = bu.Quantity(1.)
    self.assertTrue(bu.math.allclose(float(a), 1.))

    a = bu.Quantity([1, 2.])
    with self.assertRaises(TypeError):
      self.assertTrue(bu.math.allclose(float(a), 1.5))

    with self.assertRaises(TypeError):
      a = [1, 2.] * bu.ms
      self.assertTrue(bu.math.allclose(float(a), 1.5))

  def test__int__(self):
    a = bu.Quantity(1.)
    self.assertTrue(bu.math.allclose(int(a), 1.))

    a = bu.Quantity([1, 2.])
    with self.assertRaises(TypeError):
      self.assertTrue(bu.math.allclose(int(a), 1.5))

    with self.assertRaises(TypeError):
      a = [1, 2.] * bu.ms
      self.assertTrue(bu.math.allclose(int(a), 1.5))

