``brainunit.math`` module
=========================

.. currentmodule:: brainunit.math 
.. automodule:: brainunit.math 

Array Creation
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   full
   full_like
   eye
   identity
   diag
   tri
   tril
   triu
   empty
   empty_like
   ones
   ones_like
   zeros
   zeros_like
   array
   asarray
   arange
   linspace
   logspace
   fill_diagonal
   array_split
   meshgrid
   vander


Array Manipulation
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   reshape
   moveaxis
   transpose
   swapaxes
   row_stack
   concatenate
   stack
   vstack
   hstack
   dstack
   column_stack
   split
   dsplit
   hsplit
   vsplit
   tile
   repeat
   unique
   append
   flip
   fliplr
   flipud
   roll
   atleast_1d
   atleast_2d
   atleast_3d
   expand_dims
   squeeze
   sort
   argsort
   argmax
   argmin
   argwhere
   nonzero
   flatnonzero
   searchsorted
   extract
   count_nonzero
   max
   min
   amax
   amin
   block
   compress
   diagflat
   diagonal
   choose
   ravel


Functions Accepting Unitless
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   exp
   exp2
   expm1
   log
   log10
   log1p
   log2
   arccos
   arccosh
   arcsin
   arcsinh
   arctan
   arctanh
   cos
   cosh
   sin
   sinc
   sinh
   tan
   tanh
   deg2rad
   rad2deg
   degrees
   radians
   angle
   percentile
   nanpercentile
   quantile
   nanquantile
   hypot
   arctan2
   logaddexp
   logaddexp2


Functions with Bitwise Operations
---------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   bitwise_not
   invert
   bitwise_and
   bitwise_or
   bitwise_xor
   left_shift
   right_shift


Functions Changing Unit
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   reciprocal
   prod
   product
   nancumprod
   nanprod
   cumprod
   cumproduct
   var
   nanvar
   cbrt
   square
   sqrt
   multiply
   divide
   power
   cross
   ldexp
   true_divide
   floor_divide
   float_power
   divmod
   remainder
   convolve


Indexing Functions
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   where
   tril_indices
   tril_indices_from
   triu_indices
   triu_indices_from
   take
   select


Functions Keeping Unit
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   real
   imag
   conj
   conjugate
   negative
   positive
   abs
   round
   around
   round_
   rint
   floor
   ceil
   trunc
   fix
   sum
   nancumsum
   nansum
   cumsum
   ediff1d
   absolute
   fabs
   median
   nanmin
   nanmax
   ptp
   average
   mean
   std
   nanmedian
   nanmean
   nanstd
   diff
   modf
   fmod
   mod
   copysign
   heaviside
   maximum
   minimum
   fmax
   fmin
   lcm
   gcd
   interp
   clip


Logical Functions
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   all
   any
   logical_not
   equal
   not_equal
   greater
   greater_equal
   less
   less_equal
   array_equal
   isclose
   allclose
   logical_and
   logical_or
   logical_xor
   alltrue
   sometrue


Functions Matching Unit
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   add
   subtract
   nextafter


Functions Removing Unit
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   signbit
   sign
   histogram
   bincount
   corrcoef
   correlate
   cov
   digitize


Window Functions
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   bartlett
   blackman
   hamming
   hanning
   kaiser


Get Attribute Functions
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ndim
   isreal
   isscalar
   isfinite
   isinf
   isnan
   shape
   size


Linear Algebra Functions
------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   dot
   vdot
   inner
   outer
   kron
   matmul
   trace


More Functions
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   finfo
   iinfo
   broadcast_arrays
   broadcast_shapes
   einsum
   gradient
   intersect1d
   nan_to_num
   nanargmax
   nanargmin
   rot90
   tensordot
   frexp
   dtype
   e
   pi
   inf


