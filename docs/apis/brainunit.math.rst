``brainunit.math`` module
==========================

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

Getting Attribute Funcs
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

Math Funcs Keep Unit (Unary)
-----------------------------

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

Math Funcs Keep Unit (Binary)
------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

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

Math Funcs Keep Unit (N-ary)
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    interp
    clip

Math Funcs Match Unit (Binary)
-------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    add
    subtract
    nextafter

Math Funcs Change Unit (Unary)
-------------------------------

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
    frexp
    sqrt

Math Funcs Change Unit (Binary)
--------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

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

Math Funcs Only Accept Unitless (Unary)
---------------------------------------

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

Math Funcs Only Accept Unitless (Binary)
----------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    hypot
    arctan2
    logaddexp
    logaddexp2

Math Funcs Remove Unit (Unary)
-------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    signbit
    sign
    histogram
    bincount

Math Funcs Remove Unit (Binary)
--------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    corrcoef
    correlate
    cov
    digitize

Array Manipulation
-------------------

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

Elementwise Bit Operations (Unary)
----------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    bitwise_not
    invert

Elementwise Bit Operations (Binary)
-----------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    bitwise_and
    bitwise_or
    bitwise_xor
    left_shift
    right_shift

Logic Funcs (Unary)
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    all
    any
    logical_not

Logic Funcs (Binary)
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

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

Indexing Funcs
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    nonzero
    where
    tril_indices
    tril_indices_from
    triu_indices
    triu_indices_from
    take
    select

Window Funcs
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    bartlett
    blackman
    hamming
    hanning
    kaiser

Constants
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    e
    pi
    inf

Linear Algebra
---------------

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

Data Types
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    dtype
    finfo
    iinfo

More
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

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
