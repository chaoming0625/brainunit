``brainunit`` documentation
===========================

`brainunit <https://github.com/chaoming0625/brainunit>`_ provides physical units and unit-aware mathematical system in JAX for brain dynamics and AI4Science.




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


Quick Start
^^^^^^^^^^^
Most users of the `brainunit` package will work with `Quantity`: the combination of a value and a unit. The most convenient way to
create a `Quantity` is to multiply or divide a value by one of the built-in
units. It works with scalars, sequences, and `numpy` or `jax.numpy` arrays.

.. code-block:: python

    import brainunit as bu
    61.8 * bu.second

.. code-block:: text

    61.8 * second

we recommend using 64-bit precision for better numerical stability

.. code-block:: python

    import brainstate as bst
    bst.environ.set(precision=64)
    61.8 * bu.second

.. code-block:: text

    61.8 * second


.. code-block:: python

    [1., 2., 3.] * bu.second

.. code-block:: text

    ArrayImpl([1. 2. 3.]) * second


.. code-block:: python
    
    import numpy as np
    np.array([1., 2., 3.]) * bu.second

.. code-block:: text
    
    ArrayImpl([1., 2., 3.]) * second


.. code-block:: python
    
    import jax.numpy as jnp
    jnp.array([1., 2., 3.]) * bu.second

.. code-block:: text

    ArrayImpl([1., 2., 3.]) * second


You can get the unit and mantissa from a `Quantity` using the unit and mantissa members:

.. code-block:: python

    q = 61.8 * bu.second
    q.mantissa

.. code-block:: text
    
    Array(61.8, dtype=float64, weak_type=True)


.. code-block:: python
    
    q.unit


From this basic building block, it is possible to start combining quantities with different units:

.. code-block:: python

    15.1 * bu.meter / (32.0 * bu.second)

.. code-block:: text

    0.471875 * meter / second


.. code-block:: python

    3.0 * bu.kmeter / (130.51 * bu.meter / bu.second)


.. code-block:: text
    
    0.022997 * (meter / second)

To create a dimensionless quantity, directly use the `Quantity` constructor:

.. code-block:: python
    
    from brainunit import Quantity
    q = Quantity(61.8)
    q.dim

.. code-block:: text
    
    Dimension()

----


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^


- `brainstate <https://github.com/chaoming0625/brainstate>`_: A ``State``-based transformation system for brain dynamics programming.

- `brainunit <https://github.com/chaoming0625/brainunit>`_: The unit system for brain dynamics programming.

- `braintaichi <https://github.com/chaoming0625/braintaichi>`_: Leveraging Taichi Lang to customize brain dynamics operators.

- `brainscale <https://github.com/chaoming0625/brainscale>`_: The scalable online learning framework for biological neural networks.

- `braintools <https://github.com/chaoming0625/braintools>`_: The toolbox for the brain dynamics simulation, training and analysis.



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Physical Units

   physical_units/quantity.ipynb
   physical_units/math_operations_with_quantity.ipynb
   physical_units/standard_units.ipynb
   physical_units/constants.ipynb
   physical_units/conversion.ipynb



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
   :caption: Advanced Tutorials

   advanced_tutorials/combining_and_defining.ipynb
   advanced_tutorials/mechanism.ipynb



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation

   apis/changelog.md
   apis/brainunit.rst
   apis/brainunit.math.rst



