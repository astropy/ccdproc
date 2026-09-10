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

The deliberate differences from `astropy.nddata` are all about dtype.
`block_average`, and `block_replicate` with ``conserve_sum=True``, promote
integer and boolean input to the namespace's default real floating dtype,
because array-api-strict rejects a non-floating ``mean`` and integer true
division outright; numpy returns float64 for both anyway, so these differ
only for a library whose default real dtype is not float64. `block_reduce`
casts boolean input to the namespace's default integral dtype before its
default ``xp.sum``, for the same reason. `block_replicate`, on the other
hand, leaves a floating dtype alone where `astropy.nddata` promotes
float32 to float64.
"""

import math

import array_api_compat
from astropy.nddata import support_nddata

from ._nanfuncs import _promote_to_real

__all__ = ["block_average", "block_reduce", "block_replicate"]


def _block_size(block_size, ndim):
    """
    Validate ``block_size`` and broadcast it to one entry per axis.

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
        neither one nor ``ndim``, or if any entry is not an integer. An
        integral float such as ``2.0`` counts as an integer; NaN and
        infinity do not.

    Notes
    -----
    This reproduces ``astropy.nddata.blocks._process_block_inputs`` -- the
    same three checks, in the same order -- in pure Python rather than
    calling it, since that is private astropy API. The messages are the
    ones astropy raises today, but nothing here undertakes to track its
    wording.
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

    # astropy accepts a float whose value is integral (2.0 yes, 2.1 no)
    # because it compares the input against its own ``astype(int)``;
    # ``is_integer`` rejects NaN and infinity, which that comparison also
    # rejects.
    if not all(float(size).is_integer() for size in sizes):
        raise ValueError("block_size elements must be integers")

    return tuple(int(size) for size in sizes)


@support_nddata
def block_reduce(data, block_size, func=None, *, xp=None):
    """
    Downsample a data array by applying a function to local blocks.

    An axis that ``block_size`` does not divide evenly is trimmed from the
    end, as in `astropy.nddata.block_reduce`.

    Parameters
    ----------
    data : array
        The data to be resampled. Unlike `astropy.nddata`, this must already
        be an array of ``xp``: nothing here converts the input, which is the
        whole point of the module.
    block_size : int or sequence of int
        The integer block size along each axis. A scalar is used for every
        axis. Integral floats (``2.0``) are accepted, as in `astropy.nddata`,
        but non-integral ones (``2.1``) are not.
    func : callable, optional
        Reduction applied to each block, called as ``func(blocks, axis=axis)``
        with a tuple ``axis`` naming the trailing block axes, exactly as
        `astropy.nddata.block_reduce` calls it. Default is ``xp.sum``, which
        conserves the data sum.
    xp : array namespace, optional
        Namespace to use. Defaults to
        ``array_api_compat.array_namespace(data)``.

    Returns
    -------
    array
        The resampled data, in the namespace and on the device of ``data``.
        Its dtype is whatever ``func`` returns; the default ``xp.sum``
        preserves a floating dtype and, for integer or boolean input, gives
        the namespace's default integer dtype.

    Notes
    -----
    Boolean input is cast to the namespace's default integral dtype before
    the default ``xp.sum``, so that block-summing a mask counts the flagged
    pixels per block as `astropy.nddata.block_reduce` does; array-api-strict
    rejects a boolean ``sum`` outright. Data handed to a caller-supplied
    ``func`` is left as it is, since promoting for an arbitrary reduction
    would be a guess: calling this with ``func=xp.mean`` on integer or
    boolean input therefore raises on array-api-strict, which is what
    `block_average` promotes to avoid.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(data)
    if func is None:
        func = xp.sum
        if xp.isdtype(data.dtype, "bool"):
            info = xp.__array_namespace_info__()
            device = array_api_compat.device(data)
            data = xp.astype(data, info.default_dtypes(device=device)["integral"])

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
def block_average(data, block_size, *, xp=None):
    """
    Downsample a data array by averaging local blocks.

    `block_reduce` with ``func=xp.mean``, promoting integer and boolean
    input to floating point first.

    Parameters
    ----------
    data : array
        The data to be resampled. Unlike `astropy.nddata`, this must already
        be an array of ``xp``: nothing here converts the input, which is the
        whole point of the module.
    block_size : int or sequence of int
        The integer block size along each axis. A scalar is used for every
        axis. Integral floats (``2.0``) are accepted, as in `astropy.nddata`,
        but non-integral ones (``2.1``) are not.
    xp : array namespace, optional
        Namespace to use. Defaults to
        ``array_api_compat.array_namespace(data)``.

    Returns
    -------
    array
        The resampled data, in the namespace and on the device of ``data``.
        Always floating point: integer and boolean input is promoted to the
        namespace's default real floating dtype first.

    Notes
    -----
    An axis that ``block_size`` does not divide evenly is trimmed from the
    end, as in `astropy.nddata.block_reduce`.

    `astropy.nddata` inherits numpy's promotion, which turns an integer mean
    into a float, while array-API namespaces do not all agree -- jax and
    dask follow numpy, but array-api-strict rejects a non-floating ``mean``
    outright. Promoting integer and boolean input first makes every
    namespace behave the way numpy already does.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(data)
    data = _promote_to_real(data, xp, array_api_compat.device(data))
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
        be an array of ``xp``: nothing here converts the input, which is the
        whole point of the module.
    block_size : int or sequence of int
        The integer block size along each axis. A scalar is used for every
        axis. Integral floats (``2.0``) are accepted, as in `astropy.nddata`,
        but non-integral ones (``2.1``) are not.
    conserve_sum : bool, optional
        If `True` (the default) the sum of the block-replicated data equals
        the sum of the input ``data``.
    xp : array namespace, optional
        Namespace to use. Defaults to
        ``array_api_compat.array_namespace(data)``.

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
    way numpy does. A floating input, on the other hand, keeps its dtype,
    where `astropy.nddata.block_replicate` returns float64 for float32 (and
    for float16, and complex128 for complex64): it divides by
    ``numpy.prod(block_size)``, an ``int64`` scalar that NEP 50 promotes
    against, while the divisor here is a Python int.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(data)
    sizes = _block_size(block_size, data.ndim)

    if conserve_sum:
        # Divide before replicating rather than after: the arithmetic is
        # identical and the cheaper operation is the one on the smaller
        # array. math.prod, not xp.prod: the divisor is a Python int built
        # from host-side block sizes and dividing by it keeps the dtype of
        # ``data``, while an array divisor could promote it.
        data = _promote_to_real(data, xp, array_api_compat.device(data)) / math.prod(
            sizes
        )

    # Give every axis a length-1 companion, broadcast that companion out to
    # the block size, then merge each pair back into one axis. This is the
    # exact inverse of the reshape/permute in ``block_reduce``, and unlike a
    # per-axis ``repeat`` loop it materialises the result only once.
    shape = data.shape
    inner = tuple(extent for length in shape for extent in (length, 1))
    full = tuple(
        extent
        for length, size in zip(shape, sizes, strict=True)
        for extent in (length, size)
    )
    replicated = tuple(length * size for length, size in zip(shape, sizes, strict=True))
    return xp.reshape(xp.broadcast_to(xp.reshape(data, inner), full), replicated)
