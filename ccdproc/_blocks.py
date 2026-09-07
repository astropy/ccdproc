# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Block downsampling and upsampling written only in terms of the array API.

`astropy.nddata.block_reduce` and `astropy.nddata.block_replicate` begin by
calling `numpy.asanyarray` on their input, so on a non-numpy array library
they silently hand back a numpy array (dask, jax) or fail outright when the
data live on a device numpy cannot reach (array-api-strict, cupy). The
functions here do the same work using only array API operations --
``reshape``, ``permute_dims``, ``repeat`` and slicing -- so the result stays
in the caller's namespace and on the caller's device.

`ccdproc.core` uses these only for non-numpy namespaces; numpy input keeps
going to `astropy.nddata`. Apart from the shared float-promotion helper and
`block_average` -- which is ccdproc's own thin wrapper, with no counterpart
in `astropy.nddata` -- this module is free of ccdproc specifics, so that
`block_reduce` and `block_replicate` can be offered upstream (astropy
#15073); once astropy's own ``blocks.py`` is array-API aware, this module
and the dispatch in `ccdproc.core` can both be deleted.

Every function is decorated with `astropy.nddata.support_nddata`, exactly
as astropy's are, so a `~astropy.nddata.CCDData` argument is unpacked to its
``.data`` and the "following attributes were set ... but will be ignored"
`~astropy.utils.exceptions.AstropyUserWarning` is emitted identically.

The deliberate differences from astropy are both about dtype, and both are
promotions of integer and boolean input to the namespace's default real
floating dtype: `block_replicate` with ``conserve_sum=True`` promotes
before dividing, because array-api-strict rejects integer true division
outright, and `block_average` promotes before averaging, because
array-api-strict rejects a non-floating ``mean``. numpy returns float64 in
both cases anyway, so these differ only for a library whose default real
dtype is not float64.
"""

import math
import operator

import array_api_compat
from astropy.nddata import support_nddata

from ._nanfuncs import _fill_doc, _promote_to_real

__all__ = ["block_average", "block_reduce", "block_replicate"]

# ``data``, ``block_size`` and ``xp`` mean the same thing for every public
# function here, so their docstring entries are written once and filled
# into each docstring's ``{params}`` placeholder by ``_fill_doc``; the
# function-specific parameter that sits between ``block_size`` and ``xp``
# is supplied as ``{extra}``.
_COMMON_PARAMS = """\
data : array
    The data to be resampled. Unlike `astropy.nddata`, this must already
    be an array of ``xp``: nothing here converts the input, which is the
    whole point of the module.
block_size : int or sequence of int
    The integer block size along each axis. A scalar is used for every
    axis. Integral floats (``2.0``) are accepted, as in `astropy.nddata`,
    but non-integral ones (``2.1``) are not.
{extra}xp : array namespace, optional
    Namespace to use. Defaults to
    ``array_api_compat.array_namespace(data)``.\
"""


def _block_size(block_size, ndim):
    """
    Validate ``block_size`` and broadcast it to one entry per axis.

    This reproduces ``astropy.nddata.blocks._process_block_inputs`` -- the
    same three checks, in the same order, with the same messages -- in pure
    Python, so that validating the block size never routes array data (or
    the block size itself) through numpy.

    Parameters
    ----------
    block_size : int or sequence of int
        The block size to validate. A scalar is broadcast to ``ndim``
        entries when ``ndim`` is greater than one.
    ndim : int
        Number of dimensions of the data the blocks will be taken from.

    Returns
    -------
    tuple of int
        One block size per axis.

    Raises
    ------
    ValueError
        If any entry is not strictly positive, if the number of entries is
        neither one nor ``ndim``, or if any entry is not an integer (an
        integral float such as ``2.0`` counts as one).
    """
    try:
        sizes = list(block_size)
    except TypeError:
        # A scalar: a Python or numpy number, or a 0-d array.
        sizes = [block_size]

    # astropy checks positivity first, then the length, then integrality,
    # and the message raised for e.g. a wrong-length non-integral block
    # size depends on that order, so keep it.
    if any(size <= 0 for size in sizes):
        raise ValueError("block_size elements must be strictly positive")

    if ndim > 1 and len(sizes) == 1:
        sizes = sizes * ndim

    if len(sizes) != ndim:
        raise ValueError(
            "block_size must be a scalar or have the same "
            "length as the number of data dimensions"
        )

    validated = []
    for size in sizes:
        try:
            validated.append(operator.index(size))
        except TypeError:
            # Not an integer type; astropy accepts a float whose value is
            # integral (2.0 yes, 2.1 no) because it compares the input
            # against its own ``astype(int)``. ``int()`` refuses NaN and
            # infinity outright, which that comparison also rejects.
            try:
                as_int = int(size)
            except (ValueError, OverflowError):
                as_int = None
            if as_int is None or as_int != size:
                raise ValueError("block_size elements must be integers") from None
            validated.append(as_int)

    return tuple(validated)


@support_nddata
@_fill_doc(
    _COMMON_PARAMS,
    extra="""\
func : callable, optional
    Reduction applied to each block, called as ``func(blocks, axis=axis)``
    with a tuple ``axis`` naming the trailing block axes, exactly as
    `astropy.nddata.block_reduce` calls it. Default is ``xp.sum``, which
    conserves the data sum.
""",
)
def block_reduce(data, block_size, func=None, *, xp=None):
    """
    Downsample a data array by applying a function to local blocks.

    An axis that ``block_size`` does not divide evenly is trimmed from the
    end, as in `astropy.nddata.block_reduce`.

    Parameters
    ----------
    {params}

    Returns
    -------
    array
        The resampled data, in the namespace and on the device of ``data``.
        Its dtype is whatever ``func`` returns; the default ``xp.sum``
        preserves a floating dtype and, for integer input, gives the
        namespace's default integer dtype.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(data)
    if func is None:
        func = xp.sum

    ndim = data.ndim
    sizes = _block_size(block_size, ndim)

    # Trim the leftover at the end of each axis so every axis divides
    # evenly into blocks.
    data = data[
        tuple(
            slice(0, (length // size) * size)
            for length, size in zip(data.shape, sizes, strict=True)
        )
    ]

    # Reshape to (n0, b0, n1, b1, ...) and permute to (n0, n1, ..., b0,
    # b1, ...) so that the block axes are last and a single reduction over
    # them collapses each block to one value.
    interleaved = tuple(
        extent
        for length, size in zip(data.shape, sizes, strict=True)
        for extent in (length // size, size)
    )
    order = tuple(range(0, 2 * ndim, 2)) + tuple(range(1, 2 * ndim, 2))
    blocks = xp.permute_dims(xp.reshape(data, interleaved), order)

    return func(blocks, axis=tuple(range(ndim, 2 * ndim)))


@support_nddata
@_fill_doc(_COMMON_PARAMS, extra="")
def block_average(data, block_size, *, xp=None):
    """
    Downsample a data array by averaging local blocks.

    `block_reduce` with ``func=xp.mean``, plus the dtype promotion that
    needs: `astropy.nddata` inherits numpy's, which turns an integer mean
    into a float, while array-API namespaces do not all agree -- jax and
    dask follow numpy, but array-api-strict rejects a non-floating
    ``mean`` outright. Promoting integer and boolean input first makes
    every namespace behave the way numpy already does.

    An axis that ``block_size`` does not divide evenly is trimmed from the
    end, as in `astropy.nddata.block_reduce`.

    Parameters
    ----------
    {params}

    Returns
    -------
    array
        The resampled data, in the namespace and on the device of ``data``.
        Always floating point: integer and boolean input is promoted to the
        namespace's default real floating dtype first.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(data)
    if xp.isdtype(data.dtype, ("integral", "bool")):
        data = _promote_to_real(data, xp, array_api_compat.device(data))
    # ``data`` is a bare array by now, so the inner ``support_nddata`` has
    # nothing left to unpack and cannot warn a second time.
    return block_reduce(data, block_size, xp.mean, xp=xp)


@support_nddata
@_fill_doc(
    _COMMON_PARAMS,
    extra="""\
conserve_sum : bool, optional
    If `True` (the default) the sum of the block-replicated data equals
    the sum of the input ``data``.
""",
)
def block_replicate(data, block_size, conserve_sum=True, *, xp=None):
    """
    Upsample a data array by block replication.

    Parameters
    ----------
    {params}

    Returns
    -------
    array
        The block-replicated data, in the namespace and on the device of
        ``data``. When ``conserve_sum`` is `True` the result is floating
        point: integer and boolean input is promoted to the namespace's
        default real floating dtype first, since array-api-strict rejects
        integer true division rather than promoting the way numpy does.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(data)
    sizes = _block_size(block_size, data.ndim)

    if conserve_sum and xp.isdtype(data.dtype, ("integral", "bool")):
        # Promote before replicating rather than after, so the cheaper
        # cast is the one that runs on the smaller array.
        data = _promote_to_real(data, xp, array_api_compat.device(data))

    for axis, size in enumerate(sizes):
        data = xp.repeat(data, size, axis=axis)

    if conserve_sum:
        # math.prod, not xp.prod: the divisor is a Python int built from
        # host-side block sizes and dividing by it keeps the dtype of
        # ``data``, while an array divisor could promote it.
        data = data / math.prod(sizes)

    return data
