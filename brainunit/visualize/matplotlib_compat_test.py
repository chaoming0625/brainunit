import brainunit as u
from brainunit import visualize
import pytest

try:
    import matplotlib.pyplot as plt
except ImportError:
    pytest.skip("matplotlib is not installed", allow_module_level=True)

def test_quantity_support():
    with visualize.quantity_support():
        plt.figure()
        plt.plot([1, 2, 3] * u.meter)
        plt.show()

        plt.cla()
        plt.plot([101, 125, 150] * u.cmeter)
        plt.show()

    with pytest.raises(TypeError):
        plt.figure()
        plt.plot([1, 2, 3] * u.meter)
        plt.show()

        plt.cla()
        plt.plot([101, 125, 150] * u.cmeter, [1, 2, 3] * u.kgram)
        plt.show()
