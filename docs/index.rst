``brainunit`` documentation
===========================

`brainunit <https://github.com/brainpy/brainunit>`_ implements a unit system for brain dynamics programming (BDP).

----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainunit[cpu]

    .. tab-item:: GPU (CUDA 11.0)

       .. code-block:: bash

          pip install -U brainunit[cuda11]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U brainunit[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainunit[tpu]


----


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^


- `brainpy <https://github.com/brainpy/BrainPy>`_: The solution for the general-purpose brain dynamics programming.

- `braincore <https://github.com/brainpy/braincore>`_: The core system for the next generation of BrainPy framework.

- `brainunit <https://github.com/brainpy/brainunit>`_: The tools for the brain dynamics simulation and analysis.

- `brainscale <https://github.com/brainpy/brainscale>`_: The scalable online learning for biological spiking neural networks.



.. toctree::
   :hidden:
   :maxdepth: 2

   tutorials/physical_units.rst
   api.rst

