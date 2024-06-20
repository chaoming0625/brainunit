``brainunit`` documentation
===========================

`brainunit <https://github.com/brainpy/brainunit>`_ provides a unit-aware system for brain dynamics programming (BDP).

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

          pip install -U brainunit[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html




----

Learn more
^^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` Installation
         :class-card: sd-text-black sd-bg-light
         :link: quickstart/installation.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` Core Concepts
         :class-card: sd-text-black sd-bg-light
         :link: core_concepts.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`science;2em` BDP Tutorials
         :class-card: sd-text-black sd-bg-light
         :link: tutorials.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`token;2em` Advanced Tutorials
         :class-card: sd-text-black sd-bg-light
         :link: advanced_tutorials.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`settings;2em` BDP Toolboxes
         :class-card: sd-text-black sd-bg-light
         :link: toolboxes.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` FAQ
         :class-card: sd-text-black sd-bg-light
         :link: FAQ.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`data_exploration;2em` API documentation
         :class-card: sd-text-black sd-bg-light
         :link: api.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`settings;2em` Examples
         :class-card: sd-text-black sd-bg-light
         :link: https://brainpy-examples.readthedocs.io/en/latest/index.html


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

   mathematical_functions/numpy_functions.ipynb
   mathematical_functions/customize_functions.ipynb



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation

   api.rst



