from typing import Callable

import brainstate as bst
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from brainstate import init, ShortTermState, environ, surrogate
from brainstate.mixin import Mode
from brainstate.nn import exp_euler_step, Neuron
from brainstate.typing import ArrayLike, DTypeLike, Size

from brainunit import mV, ms, nA, second

bst.environ.set(dt=(0.01 * ms).value)


class LIF(Neuron):
  """Leaky integrate-and-fire neuron model."""

  def __init__(
      self,
      in_size: Size,
      keep_size: bool = False,
      tau: ArrayLike = 20. * ms,
      V_th: ArrayLike = 10. * mV,
      V_reset: ArrayLike = 0. * mV,
      V_rest: ArrayLike = 0. * mV,
      spk_fun: Callable = surrogate.ReluGrad(),
      spk_dtype: DTypeLike = None,
      spk_reset: str = 'soft',
      mode: Mode = None,
      name: str = None,
  ):
    super().__init__(in_size,
                     keep_size=keep_size,
                     name=name,
                     mode=mode,
                     spk_fun=spk_fun,
                     spk_dtype=spk_dtype,
                     spk_reset=spk_reset)

    # parameters
    self.tau = init.param(tau.value, self.varshape)
    self.V_th = init.param(V_th.value, self.varshape)
    self.V_reset = init.param(V_reset.value, self.varshape)
    self.V_rest = init.param(V_rest.value, self.varshape)

  def dv(self, v, t, x):
    x = self.sum_current_inputs(v, init=x)
    return (-v + self.V_rest + x) / self.tau

  def init_state(self, batch_size: int = None, **kwargs):
    self.V = ShortTermState(init.param(init.Constant(self.V_reset), self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.V.value = init.param(init.Constant(self.V_reset), self.varshape, batch_size)

  def get_spike(self, V=None):
    V = self.V.value if V is None else V
    v_scaled = (V - self.V_th) / self.V_th
    return self.spk_fun(v_scaled)

  def update(self, x=0.):
    last_v = self.V.value
    lst_spk = self.get_spike(last_v)
    V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
    V = last_v - (V_th - self.V_reset) * lst_spk
    # membrane potential
    V = exp_euler_step(self.dv, V, environ.get('t'), x) + self.sum_delta_inputs()
    self.V.value = V
    return self.get_spike(V)


lif = LIF(1)
bst.init_states(lif)


def run(i, inp):
  with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
    lif(inp)
  return lif.V.value


n = 100
indices = jnp.arange(n)
inp = bst.random.uniform(0., 10., n)
vs = bst.transform.for_loop(run, indices, inp,
                            pbar=bst.transform.ProgressBar(count=10))

plt.plot(indices * bst.environ.get_dt(), vs)
plt.show()
