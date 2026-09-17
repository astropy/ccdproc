# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Block downsampling and upsampling written only in terms of the array API.

`astropy.nddata.block_reduce` and `astropy.nddata.block_replicate` begin by
calling `numpy.asanyarray` on their input, so on a non-numpy array library
they silently hand back a numpy array (dask, jax) or fail outright when the
data live on a device numpy cannot reach (array-api-strict, cupy). The
functions here do the same work using only array API operations --
``reshape``, ``permute_dims``, ``broadcast_to`` and slicing -- so the result
stays in the caller's namespace and on the caller's device.

Every function is decorated with `astropy.nddata.support_nddata`, exactly
as astropy's are, so a `~astropy.nddata.CCDData` argument is unpacked to its
``.data`` and the "following attributes were set ... but will be ignored"
`~astropy.utils.exceptions.AstropyUserWarning` is emitted identically.
"""

import math

import array_api_compat
from astropy.nddata import support_nddata

from ._nanfuncs import _promote_to_real

__all__ = ["block_average", "block_reduce", "block_replicate"]

_UNKNOWN_SHAPE_MESSAGE = (
    "block functions need a fully known shape; on dask call "
    "compute_chunk_sizes() first"
)


def _block_size(block_size, ndim):
    """
    Validate ``block_size`` and broadcast it to one entry per axis.

    Reproduces ``astropy.nddata.blocks._process_block_inputs`` -- the same
    three checks, in the same order, because the message raised for e.g. a
    wrong-length non-integral block size depends on that order -- in pure
    Python rather than calling it, since that is private astropy API. The
    messages are the ones astropy raises today, but nothing here undertakes
    to track its wording.
    """
    try:
        sizes = list(block_size)
    except TypeError:
        # A scalar: a Python or numpy number, or a 0-d array.
        sizes = [block_size]

    if any(size <= 0 for size in sizes):
        raise ValueError("block_size elements must be strictly positive")

    if ndim > 1 and len(sizes) == 1:
        sizes = sizes * ndim

    if len(sizes) != ndim:
        raise ValueError(
            "block_size must be a scalar or have the same "
            "length as the number of data dimensions"
        )

    # astropy accepts a float whose value is integral (2.0 yes, 2.1 no)
    # because it compares the input against its own ``astype(int)``;
    # ``is_integer`` rejects NaN and infinity, which that comparison also
    # rejects.
    if not all(float(size).is_integer() for size in sizes):
        raise ValueError("block_size elements must be integers")

    return tuple(int(size) for size in sizes)


def _promote_for_division(data, xp):
    """
    Promote integer and boolean ``data`` for a true division.

    Notes
    -----
    Complex input is already floating and passes through unchanged; a
    helper that promotes anything not *real* floating would eat its
    imaginary part.
    """
    if xp.isdtype(data.dtype, "complex floating"):
        return data
    return _promote_to_real(data, xp, array_api_compat.device(data))


@support_nddata
def block_reduce(data, block_size, func=None, *, xp=None):
    """
    Downsample a data array by applying a function to local blocks.

    Parameters
    ----------
    data : array
        The data to be resampled. Unlike `astropy.nddata`, this must already
        be an array of ``xp``: nothing here converts the input.
    block_size : int or sequence of int
        The integer block size along each axis. A scalar is used for every
        axis. Integral floats (``2.0``) are accepted, non-integral ones
        (``2.1``) are not.
    func : callable, optional
        Reduction applied to each block, called as ``func(blocks, axis=axis)``
        with a tuple ``axis`` naming the trailing block axes, exactly as
        `astropy.nddata.block_reduce` calls it. Default is ``xp.sum``, which
        conserves the data sum.
    xp : array namespace, optional
        Namespace to use. Defaults to
        ``array_api_compat.array_namespace(data)``. Must be the namespace of
        ``data``; this is not checked.

    Returns
    -------
    array
        The resampled data, in the namespace and on the device of ``data``.
        Its dtype is whatever ``func`` returns; the default ``xp.sum``
        preserves a floating dtype and, for integer or boolean input, gives
        the namespace's default integer dtype.

    Notes
    -----
    An axis that ``block_size`` does not divide evenly is trimmed from the
    end, as in `astropy.nddata.block_reduce`.

    Boolean input is cast to the namespace's default integral dtype before a
    default or explicit ``xp.sum``, so that block-summing a mask counts the
    flagged pixels per block as `astropy.nddata.block_reduce` does;
    array-api-strict rejects a boolean ``sum`` outright. Data handed to any
    other ``func`` is left as it is, since promoting for an arbitrary
    reduction would be a guess.
    """
    if not all(isinstance(length, int) for length in data.shape):
        raise ValueError(_UNKNOWN_SHAPE_MESSAGE)

    if xp is None:
        xp = array_api_compat.array_namespace(data)
    if func is None or func is xp.sum:
        func = xp.sum
        if xp.isdtype(data.dtype, "bool"):
            info = xp.__array_namespace_info__()
            device = array_api_compat.device(data)
            data = xp.astype(data, info.default_dtypes(device=device)["integral"])

    ndim = data.ndim
    sizes = _block_size(block_size, ndim)

    # Trim the leftover at the end of each axis so every axis divides evenly
    # into blocks, then reshape to (n0, b0, n1, b1, ...) and permute to
    # (n0, n1, ..., b0, b1, ...) so that the block axes are last and a single
    # reduction over them collapses each block to one value.
    nblocks = tuple(n // s for n, s in zip(data.shape, sizes, strict=True))
    data = data[tuple(slice(0, n * s) for n, s in zip(nblocks, sizes, strict=True))]
    interleaved = tuple(
        extent for pair in zip(nblocks, sizes, strict=True) for extent in pair
    )
    order = tuple(range(0, 2 * ndim, 2)) + tuple(range(1, 2 * ndim, 2))
    blocks = xp.permute_dims(xp.reshape(data, interleaved), order)

    return func(blocks, axis=tuple(range(ndim, 2 * ndim)))


@support_nddata
def block_average(data, block_size, *, xp=None):
    """
    Downsample a data array by averaging local blocks.

    Parameters
    ----------
    data : array
        The data to be resampled. Unlike `astropy.nddata`, this must already
        be an array of ``xp``: nothing here converts the input.
    block_size : int or sequence of int
        The integer block size along each axis. A scalar is used for every
        axis. Integral floats (``2.0``) are accepted, non-integral ones
        (``2.1``) are not.
    xp : array namespace, optional
        Namespace to use. Defaults to
        ``array_api_compat.array_namespace(data)``. Must be the namespace of
        ``data``; this is not checked.

    Returns
    -------
    array
        The resampled data, in the namespace and on the device of ``data``.
        Always floating point.

    Notes
    -----
    `block_reduce` with ``func=xp.mean``, promoting integer and boolean input
    to the namespace's default real floating dtype first: numpy promotes an
    integer mean on its own and jax and dask follow it, but array-api-strict
    rejects a non-floating ``mean`` outright. An axis that ``block_size``
    does not divide evenly is trimmed from the end.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(data)
    data = _promote_for_division(data, xp)
    # ``data`` is a bare array by now, so the inner ``support_nddata`` has
    # nothing left to unpack and cannot warn a second time.
    return block_reduce(data, block_size, xp.mean, xp=xp)


@support_nddata
def block_replicate(data, block_size, conserve_sum=True, *, xp=None):
    """
    Upsample a data array by block replication.

    Parameters
    ----------
    data : array
        The data to be resampled. Unlike `astropy.nddata`, this must already
        be an array of ``xp``: nothing here converts the input.
    block_size : int or sequence of int
        The integer block size along each axis. A scalar is used for every
        axis. Integral floats (``2.0``) are accepted, non-integral ones
        (``2.1``) are not.
    conserve_sum : bool, optional
        If `True` (the default) the sum of the block-replicated data equals
        the sum of the input ``data``.
    xp : array namespace, optional
        Namespace to use. Defaults to
        ``array_api_compat.array_namespace(data)``. Must be the namespace of
        ``data``; this is not checked.

    Returns
    -------
    array
        The block-replicated data, in the namespace and on the device of
        ``data``. A floating input keeps its own dtype, float32 included;
        when ``conserve_sum`` is `True`, integer and boolean input is
        promoted to the namespace's default real floating dtype.

    Notes
    -----
    Integer and boolean input is promoted before the division because
    array-api-strict rejects integer true division rather than promoting the
    way numpy does. A floating input keeps its dtype, where
    `astropy.nddata.block_replicate` returns float64 for float32: it divides
    by ``numpy.prod(block_size)``, an ``int64`` scalar that NEP 50 promotes
    against, while the divisor here is a Python int (reported as
    astropy/astropy#20360).
    """
    if not all(isinstance(length, int) for length in data.shape):
        raise ValueError(_UNKNOWN_SHAPE_MESSAGE)

    if xp is None:
        xp = array_api_compat.array_namespace(data)
    sizes = _block_size(block_size, data.ndim)

    if conserve_sum:
        # Divide before replicating rather than after: the arithmetic is
        # identical and the cheaper operation is the one on the smaller
        # array. math.prod, not xp.prod: the divisor is a Python int built
        # from host-side block sizes and dividing by it keeps the dtype of
        # ``data``, while an array divisor could promote it.
        data = _promote_for_division(data, xp) / math.prod(sizes)

    # Give every axis a length-1 companion, broadcast that companion out to
    # the block size, then merge each pair back into one axis. This is the
    # exact inverse of the reshape/permute in ``block_reduce``, and unlike a
    # per-axis ``repeat`` loop it materialises the result only once.
    shape = data.shape
    pairs = tuple(zip(shape, sizes, strict=True))
    inner = tuple(extent for length in shape for extent in (length, 1))
    full = tuple(extent for pair in pairs for extent in pair)
    replicated = tuple(length * size for length, size in pairs)
    out = xp.reshape(xp.broadcast_to(xp.reshape(data, inner), full), replicated)
    if all(length == 1 or size == 1 for length, size in pairs):
        # No axis pair had to be merged, so ``reshape`` may have returned a
        # read-only view that, with ``conserve_sum=False``, aliases ``data``.
        out = xp.asarray(out, copy=True)
    return out
