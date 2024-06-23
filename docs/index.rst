``brainunit`` documentation
===========================

`brainunit <https://github.com/brainpy/brainunit>`_ provides physical units and unit-aware mathematical system in JAX for brain dynamics and AI4Science.




----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainunit[cpu]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U brainunit[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainunit[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html




----


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^


- `brainstate <https://github.com/brainpy/brainstate>`_: A ``State``-based transformation system for brain dynamics programming.

- `brainunit <https://github.com/brainpy/brainunit>`_: The unit system for brain dynamics programming.

- `braintaichi <https://github.com/brainpy/braintaichi>`_: Leveraging Taichi Lang to customize brain dynamics operators.

- `brainscale <https://github.com/brainpy/brainscale>`_: The scalable online learning framework for biological neural networks.

- `braintools <https://github.com/brainpy/braintools>`_: The toolbox for the brain dynamics simulation, training and analysis.



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Physical Units

   physical_units/quantity.ipynb
   physical_units/standard_units.ipynb
   physical_units/constants.ipynb
   physical_units/conversion.ipynb
   physical_units/combining_defining_displaying.ipynb
   physical_units/mechanism.ipynb



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Unit-aware Math Functions

   mathematical_functions/array_creation.ipynb
   mathematical_functions/numpy_functions.ipynb
   mathematical_functions/customize_functions.ipynb



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation

   apis/changelog.md
   apis/brainunit.rst
   apis/brainunit.math.rst



